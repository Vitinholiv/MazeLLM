import torch
import torch.nn as nn
import torch.nn.functional as F

# Layers

class FeedForward(nn.Module):
    """
    Rede Neural Feed-Forward. Expande a dimensão interna por 4x e reduz novamente,
    aplicando não-linearidade GELU e regularizando com Dropout.

    **Parâmetros:**
        config (dict): Dicionário de configuração do modelo.
    """
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4*config["emb_dim"]),
            nn.GELU(approximate="tanh"), 
            nn.Linear(4*config["emb_dim"], config["emb_dim"]),
            nn.Dropout(config["drop_rate"])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# Attention

class UnmaskedSelfAttention(nn.Module):
    """
    Mecanismo de Self-Attention sem máscara causal. 
    Permite que cada token enxergue o contexto completo, sem restrição temporal.
    Usado principalmente em blocos de encoder.
    
    **Parâmetros:**
        d_in (int): Dimensão dos vetores de entrada.
        d_out (int): Dimensão dos vetores de saída.
        dropout (float): Taxa de dropout aplicada na matriz de pesos de atenção.
        num_heads (int): Número de cabeças de atenção paralelas.
        qkv_bias (bool): Se True, adiciona viés (bias) nas projeções lineares de Q, K e V.
    """
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        scale = self.head_dim ** -0.5

        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=scale,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(out)
    
    def weighted_forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        scale = self.head_dim ** -0.5

        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1))*scale
        attn_weights = torch.softmax(scores.to(torch.float32), dim=-1).to(V.dtype)

        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(out), attn_weights
    
class MaskedSelfAttention(nn.Module):
    """
    Mecanismo de Self-Attention com máscara causal.
    Bloqueia o acesso a tokens futuros usando uma máscara  triangular superior, 
    forçando o modelo a prever o futuro baseando-se apenas no passado.
    Ideal principalmente para blocos de decoder.
    
    **Parâmetros:**
        d_in (int): Dimensão dos vetores de entrada.
        d_out (int): Dimensão dos vetores de saída.
        context_length (int): Tamanho máximo do contexto.
        dropout (float): Taxa de dropout aplicada na matriz de pesos de atenção.
        num_heads (int): Número de cabeças de atenção paralelas.
        qkv_bias (bool): Se True, adiciona viés nas projeções lineares de Q, K e V.
    """
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
        scale = self.head_dim ** -0.5

        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
            scale=scale,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(out)
    
    def weighted_forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        scale = self.head_dim ** -0.5

        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1))*scale
        scores = scores.to(torch.float32)
            
        mask_bool = self.mask[:T, :T].bool()
        scores = scores.masked_fill(mask_bool, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1).to(V.dtype)

        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(out), attn_weights

class CrossAttention(nn.Module):
    """
    Mecanismo de Cross-Attention. Conecta duas sequências distintas: a matriz de Query (Q) é gerada
    pela sequência local (x),  enquanto as matrizes de Keys (K) e Values (V) vêm de um contexto externo.
    Usada principalmente nos blocos Encoder-Decoder.
    
    **Parâmetros:**
        d_in (int): Dimensão dos vetores de entrada (deve ser o mesmo para x e context).
        d_out (int): Dimensão dos vetores de saída.
        dropout (float): Taxa de dropout aplicada na matriz de pesos de atenção.
        num_heads (int): Número de cabeças de atenção paralelas.
        qkv_bias (bool): Se True, adiciona viés nas projeções lineares de Q, K e V.
    """
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
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

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        _, T_ctx, _ = context.shape
        scale = self.head_dim ** -0.5

        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(context).view(B, T_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(context).view(B, T_ctx, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=scale,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(out)
    
    def weighted_forward(self, x: torch.Tensor, context: torch.Tensor):
        B, T, _ = x.shape
        _, T_ctx, _ = context.shape
        scale = self.head_dim ** -0.5

        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(context).view(B, T_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(context).view(B, T_ctx, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1))*scale
        attn_weights = torch.softmax(scores.to(torch.float32), dim=-1).to(V.dtype)

        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(out), attn_weights
    
# Transfomer Blocks

class EncoderTransformerBlock(nn.Module):
    """
    Bloco de Transformer para o Encoder.
    Utiliza UnmaskedSelfAttention para processar toda a sequência bidirecionalmente.
    Ideal para ler o labirinto estático inteiro de uma vez.

    **Parâmetros:**
        config (dict): Dicionário de configuração do modelo.
    """
    def __init__(self, config):
        super().__init__()
        self.att = UnmaskedSelfAttention(
            d_in=config["emb_dim"], d_out=config["emb_dim"],
            dropout=config["drop_rate"], num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"],
        )
        self.ff    = FeedForward(config)
        self.norm1 = nn.LayerNorm(config["emb_dim"])
        self.norm2 = nn.LayerNorm(config["emb_dim"])
        self.drop  = nn.Dropout(config["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.att(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
    
    def weighted_forward(self, x: torch.Tensor):
        att_out, attn_weights = self.att.weighted_forward(self.norm1(x))
        x = x + self.drop(att_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn_weights

class DecoderTransformerBlock(nn.Module):
    """
    Bloco de Transformer para o Decoder.
    Utiliza MaskedSelfAttention para impedir que a sequência atual veja o futuro.
    Constroi o labirinto passo a passo.

    **Parâmetros:**
        config (dict): Dicionário de configuração do modelo.
    """
    def __init__(self, config):
        super().__init__()
        self.att = MaskedSelfAttention(
            d_in=config["emb_dim"], d_out=config["emb_dim"],
            context_length=config["context_length"],
            dropout=config["drop_rate"], num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"],
        )
        self.ff    = FeedForward(config)
        self.norm1 = nn.LayerNorm(config["emb_dim"])
        self.norm2 = nn.LayerNorm(config["emb_dim"])
        self.drop  = nn.Dropout(config["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.att(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
    
    def weighted_forward(self, x: torch.Tensor):
        att_out, attn_weights = self.att.weighted_forward(self.norm1(x))
        x = x + self.drop(att_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn_weights

class CrossDecoderTransformerBlock(nn.Module):
    """
    Bloco de Transformer para a metade Decoder de um modelo Encoder-Decoder.
    Aplica MaskedSelfAttention na rota em construção e, em seguida, CrossAttention 
    para consultar o mapa lido pelo Encoder.

    **Parâmetros:**
        config (dict): Dicionário de configuração do modelo.
    """
    def __init__(self, config):
        super().__init__()
        self.self_att  = MaskedSelfAttention(
            d_in=config["emb_dim"], d_out=config["emb_dim"],
            context_length=config["context_length"],
            dropout=config["drop_rate"], num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"],
        )
        self.cross_att = CrossAttention(
            d_in=config["emb_dim"], d_out=config["emb_dim"],
            dropout=config["drop_rate"], num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"],
        )
        self.ff    = FeedForward(config)
        self.norm1 = nn.LayerNorm(config["emb_dim"])
        self.norm2 = nn.LayerNorm(config["emb_dim"])
        self.norm3 = nn.LayerNorm(config["emb_dim"])
        self.drop  = nn.Dropout(config["drop_rate"])

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.self_att(self.norm1(x)))
        x = x + self.drop(self.cross_att(self.norm2(x), context=context))
        x = x + self.drop(self.ff(self.norm3(x)))
        return x
    
    def weighted_forward(self, x: torch.Tensor, context: torch.Tensor):
        self_out, self_weights = self.self_att.weighted_forward(self.norm1(x))
        x = x + self.drop(self_out)
        cross_out, cross_weights = self.cross_att.weighted_forward(self.norm2(x), context=context)
        x = x + self.drop(cross_out)
        x = x + self.drop(self.ff(self.norm3(x)))
        return x, {"self": self_weights, "cross": cross_weights}