import torch
import math

class FlashAttention2Function(torch.autograd.Function):
    """
    A pure PyTorch implementation of FlashAttention-2 forward pass.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Get dimensions from input tensors following the (B, H, N, D) convention
        B, H, N_Q, D_H = Q.shape
        _, _, N_K, _ = K.shape 

        if is_causal:
            assert N_Q == N_K, f"Causal masking assumes N_Q == N_K, got N_Q={N_Q}, N_K={N_K}"

        # Define tile sizes
        Q_TILE_SIZE = 128
        K_TILE_SIZE = 128

        # Number of tiles
        N_Q_tiles = math.ceil(N_Q / Q_TILE_SIZE)
        N_K_tiles = math.ceil(N_K / K_TILE_SIZE)

        # Initialize output O and logsumexp L
        O_final = torch.zeros_like(Q, dtype=Q.dtype)
        L_final = torch.zeros((B, H, N_Q), device=Q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(D_H)

        # Outer loop over query tiles (all batches and heads at once)
        for i in range(N_Q_tiles):
            q_start = Q_TILE_SIZE * i
            q_end = min(Q_TILE_SIZE * (i + 1), N_Q)
            Q_tile = Q[..., q_start:q_end, :]  # (B, H, Tq, D)

            # Initialize accumulators for this query tile
            o_i = torch.zeros((B, H, q_end - q_start, D_H), device=Q.device, dtype=torch.float32)
            l_i = torch.zeros((B, H, q_end - q_start), device=Q.device, dtype=torch.float32)
            m_i = torch.full((B, H, q_end - q_start), float('-inf'), device=Q.device, dtype=torch.float32)

            # Inner loop over key/value tiles
            for j in range(N_K_tiles):
                kv_start = K_TILE_SIZE * j
                kv_end = min(K_TILE_SIZE * (j + 1), N_K)

                # Skip tiles fully above the diagonal for speed-up in the causal case
                if is_causal and kv_start > q_end - 1:
                    break

                K_tile = K[..., kv_start:kv_end, :]  # (B, H, Tk, D)
                V_tile = V[..., kv_start:kv_end, :]  # (B, H, Tk, D)

                # Scaled attention scores
                S_ij = ((Q_tile * scale) @ K_tile.transpose(-1, -2)).float()  # (B, H, Tq, Tk)

                # Apply causal masking; the (Tq, Tk) mask broadcasts over (B, H)
                if is_causal and q_start < kv_end - 1:
                    q_idx = torch.arange(q_start, q_end, device=Q.device)
                    kv_idx = torch.arange(kv_start, kv_end, device=Q.device)
                    attn_mask = q_idx[:, None] < kv_idx[None, :]
                    S_ij = S_ij.masked_fill(attn_mask, float('-inf'))

                # Compute new running maximum
                m_new = torch.maximum(m_i, S_ij.amax(dim=-1))  # (B, H, Tq)
                m_safe = torch.where(m_new == float('-inf'), 0.0, m_new) # guards against NaN

                # Compute probabilities for current tile
                P_ij = torch.exp(S_ij - m_safe[..., None])

                # Update old accumulators
                alpha = torch.exp(m_i - m_safe)
                l_i = alpha * l_i + torch.sum(P_ij, dim=-1)
                o_i = alpha[..., None] * o_i + P_ij @ V_tile.float()

                # Update the running maximum for the next iteration
                m_i = m_new

            # After iterating through all tiles, normalize the output
            # Use safe division
            l_i_reciprocal = torch.where(l_i > 0, 1.0 / l_i, 0.0)
            o_i_normalized = o_i * l_i_reciprocal[..., None]

            # Compute final logsumexp
            L_tile = m_i + torch.log(l_i)

            # Write results for this query tile back to the final output tensors
            O_final[..., q_start:q_end, :] = o_i_normalized
            L_final[..., q_start:q_end] = L_tile
        
        # Cast back to the original dtype
        # O_final = O_final.to(Q.dtype)

        ctx.save_for_backward(Q, K, V, O_final, L_final)
        ctx.is_causal = is_causal
        ctx.mark_non_differentiable(L_final)

        return O_final, L_final

    @staticmethod
    def backward(ctx, grad_O, grad_L):
        raise NotImplementedError("Backward pass not yet implemented for FlashAttention2Function")