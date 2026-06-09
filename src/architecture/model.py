import torch
import torch.nn as nn
from src.architecture.components import MultiHeadAttention, LayerNorm, FeedForward
from src.data.preprocess import MazeEmbedder

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
    
    def weighted_forward(self, x: torch.Tensor):
        att_out, attn_weights = self.att.weighted_forward(self.norm1(x))
        x = x + self.drop(att_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn_weights

class MazeGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedder    = MazeEmbedder(cfg["vocab_size"], cfg["emb_dim"], cfg["context_length"])
        self.drop_emb   = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.name = "MazeGPT"
        self.iname = ''
        self.context_len = cfg["context_length"]

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        x = self.drop_emb(self.embedder(in_idx))
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
    
    @torch.no_grad()
    def get_attention(self, in_idx: torch.Tensor, layer_idx: int = -1):
        x = self.drop_emb(self.embedder(in_idx))
        extracted_weights = None
        if layer_idx < 0:
            layer_idx += len(self.trf_blocks)

        for i, block in enumerate(self.trf_blocks):
            if i == layer_idx:
                x, extracted_weights = block.weighted_forward(x)
            else:
                x = block(x)
                
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits, extracted_weights