import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_causal_kernel (
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_HEADS,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr, # Number of query rows one program instance handles
    BLOCK_K: tl.constexpr # Number of key/value rows loaded in each iteration of the inner loop
    # Each step creates a score tile s_ij of shape [BLOCK_Q, BLOCK_K]
):
    """
    Triton implementation of FlashAttention-2 forward pass (non-causal for now)
    """
    

    # 1. Identify the query block and batch/head to be processed
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)

    batch_idx = batch_head_idx // N_HEADS
    head_idx = batch_head_idx % N_HEADS

    # 2. Initialize pointers and accumulators for the online softmax
    m_i = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    o_i = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

    # 3. Load the block of queries (Q_i)
    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q) # roughly, the role of seq_idx
    q_ptrs = Q_ptr + batch_idx * q_stride_b + head_idx * q_stride_h \
        + (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=(q_offsets[:,None] < SEQ_LEN), other=0.0)

    # PyTorch softmax is exp(x). Triton exp2 is exp2(x * log2(e)) and used for speedup; log2(e) is approx 1.44269504
    score_scale = softmax_scale * 1.44269504

    # Main loop: iterate over the blocks of Keys (K_j) and Values (V_j)
    # Iterating through sub-diagonal and diagonal blocks separately adds significant efficiency
    diag_start_idx = BLOCK_Q * q_block_idx
    diag_end_idx = BLOCK_Q * (q_block_idx + 1)

    # 5. Sub-diagonal blocks: no masking needed
    for k_start in range(0, diag_start_idx, BLOCK_K):
        # Load the block of keys (K_j)
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_ptrs = K_ptr + batch_idx * k_stride_b + head_idx * k_stride_h \
            + (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None]) # Shape [HEAD_DIM, BLOCK_K]
        k_block = tl.load(k_ptrs, mask=(k_offsets[None, :] < SEQ_LEN), other=0.0) # Transposed key block

        # Compute attention scores S_ij = Q_i * K_j^T
        scores = tl.dot(q_block, k_block) * score_scale # Shape (BLOCK_Q, BLOCK_K)
        scores = tl.where(k_offsets[None, :] < SEQ_LEN, scores, float('-inf'))

        # Load the block of values (V_j)
        v_ptrs = V_ptr + batch_idx * v_stride_b + head_idx * v_stride_h \
            + (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0)

        # ONLINE SOFTMAX UPDATE
        # Calculate the new running maximum
        m_new = tl.maximum(m_i, tl.max(scores, axis=1)) # Shape (BLOCK_Q,)
        
        # Use m_new to rescale the denominator l_i and the accumulator o_i
        rescale = tl.exp2(m_i - m_new) 
        l_i = rescale * l_i
        o_i = rescale[:, None] * o_i

        # Compute rescaled attention probabilities P^tilde_ij for the current tile
        probs = tl.exp2(scores - m_new[:, None])

        # Update l_i and o_i using P^tilde_ij and V_j
        l_i += tl.sum(probs, axis=1)
        o_i += tl.dot(probs.to(v_block.dtype), v_block)

        # Update the running maximum m_i for the next iteration
        m_i = m_new

    # 6. Diagonal blocks: apply causal mask
    for k_start in range(diag_start_idx, diag_end_idx, BLOCK_K):

        # Load the block of keys (K_j)
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_ptrs = K_ptr + batch_idx * k_stride_b + head_idx * k_stride_h \
            + (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None]) # Shape [HEAD_DIM, BLOCK_K]
        k_block = tl.load(k_ptrs, mask=(k_offsets[None, :] < SEQ_LEN), other=0.0) # Transposed key block

        # Compute attention scores S_ij = Q_i * K_j^T and apply causal mask
        scores = tl.dot(q_block, k_block) * score_scale # Shape (BLOCK_Q, BLOCK_K)
        causal_mask = q_offsets[:, None] >= k_offsets[None, :]
        scores = tl.where(causal_mask, scores, float('-inf'))

        # Load the block of values (V_j)
        v_ptrs = V_ptr + batch_idx * v_stride_b + head_idx * v_stride_h \
            + (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0)

        # ONLINE SOFTMAX UPDATE
        # Calculate the new running maximum
        m_new = tl.maximum(m_i, tl.max(scores, axis=1)) # Shape (BLOCK_Q,)
        
        # Use m_new to rescale the denominator l_i and the accumulator o_i
        rescale = tl.exp2(m_i - m_new) 
        l_i = rescale * l_i
        o_i = rescale[:, None] * o_i

        # Compute rescaled attention probabilities P^tilde_ij for the current tile
        probs = tl.exp2(scores - m_new[:, None])

        # Update l_i and o_i using P^tilde_ij and V_j
        l_i += tl.sum(probs, axis=1)
        o_i += tl.dot(probs.to(v_block.dtype), v_block)

        # Update the running maximum m_i for the next iteration
        m_i = m_new

    # 7. Normalize the accumulator o_i and save to HBM
    # Safe broadcast for denominator l_i
    l_i_safe = l_i[:, None] # + 1e-6
    o_i = o_i / l_i_safe

    # We skip computing and saving logsumexp L_i (until we write the backward pass)

    # Save o_i to the HBM
    o_block = O_ptr + batch_idx * q_stride_b + head_idx * q_stride_h \
        + (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(o_block, o_i.to(O_ptr.dtype.element_ty), mask=(q_offsets[:, None] < SEQ_LEN))

def flash_attention_forward(q, k, v, is_causal=True):
    """
    Python wrapper for the single-kernel, two-phase causal FlashAttention-2 forward pass
    """
    assert is_causal, "This implementation is for causal FlashAttention-2."
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1, "The last stride of q, k, v must all be 1."

    batch, n_heads, seq_len, head_dim = q.shape
    assert head_dim >= 16
    o = torch.empty_like(q)
    softmax_scale = 1 / math.sqrt(head_dim)
    BLOCK_Q, BLOCK_K = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_Q), batch * n_heads)

    _flash_attention_forward_causal_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale=softmax_scale,
        SEQ_LEN=seq_len,
        N_HEADS=n_heads,
        HEAD_DIM=head_dim,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K
    )
    return o
