import torch
import torch.nn.functional as F
import math

def standard_multi_head_attention(Q, K, V, mask=None):
    """
    A standard PyTorch implementation of multi-head attention.
    Shapes:
    - Q, K, V: (batch_size, seq_len, embed_dim)
    - mask: (batch_size, 1, 1, seq_len) for padding, (1, 1, seq_len, seq_len) for causal (broadcasted)
    """
    
    # Assume embed_dim is divisible by num_heads
    # Below: batch_size, seq_len, num_heads, head_dim = B, S, H, D 
    batch_size, seq_len, embed_dim = Q.shape 
    num_heads = 8
    head_dim = embed_dim // num_heads 

    # 1. Reshape Q, K, V for multi-head processing
    # (batch_size, seq_len, embed_dim) -> (batch_size, num_heads, seq_len, head_dim)
    def reshape_for_heads(x: torch.Tensor):
        return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    Q_heads = reshape_for_heads(Q)
    K_heads = reshape_for_heads(K)
    V_heads = reshape_for_heads(V)

    # 2. Compute attention score with batch matmul -- THE BIG MATRIX!!
    # [B, H, S, D] x [B, H, D, S] -> [B, H, S, S]
    scores = torch.matmul(Q_heads, K_heads.transpose(-1, -2))

    # 3. Scale the scores
    scores = scores / math.sqrt(head_dim)

    # 4. Apply the mask
    # -1e9 is used in place of float('-inf') to avoid "all masked row/NaN" problem
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 5. Normalize with softmax
    attention_weights = F.softmax(scores, dim=-1)

    # 6. Compute the final output
    # [B, H, S, S] x [B, H, S, D] -> [B, H, S, D]
    output = torch.matmul(attention_weights, V_heads)

    # 7. Reshape the output back to original shape
    # [B, H, S, D] -> [B, S, H * D]
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

    return output, attention_weights