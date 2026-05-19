import torch
import torch.nn as nn

# GELU Activation Function
class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            (2.0 / torch.pi) ** 0.5 * (x + 0.044715 * x.pow(3))
        ))

# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps   = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1,  keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift
    
# MLP
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
# Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        scale    = self.head_dim ** -0.5

        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x)  .view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) * scale
        scores.masked_fill_(self.mask[:T, :T].bool(), float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(out)