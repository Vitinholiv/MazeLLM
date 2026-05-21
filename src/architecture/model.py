import torch
import torch.nn as nn
from src.architecture.components import MultiHeadAttention, LayerNorm, FeedForward
from src.data.preprocess import MazeEncoder

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att   = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff    = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.att(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

# Maze GPT Model
class MazeGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder    = MazeEncoder(cfg["vocab_size"], cfg["emb_dim"], cfg["context_length"])
        self.drop_emb   = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.name = "MazeGPT"
        self.iname = ''
        self.context_len = cfg["context_length"]

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        x = self.drop_emb(self.encoder(in_idx))
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)