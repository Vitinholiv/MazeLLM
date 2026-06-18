import torch
import torch.nn as nn
from src.architecture.components import EncoderTransformerBlock, DecoderTransformerBlock, CrossDecoderTransformerBlock
from src.data.preprocess import MazeEmbedder

class MazeEncoder(nn.Module):
    """
    Modelo Encoder para a resolução de labirintos.
    Resolve o labirinto todo de uma vez.

    **Parâmetros:**
        config (dict): Dicionário de configuração do modelo.
    """
    def __init__(self, config):
        super().__init__()
        self.embedder    = MazeEmbedder(config["vocab_size"], config["emb_dim"], config["context_length"])
        self.drop_emb    = nn.Dropout(config["drop_rate"])

        self.transformer_blocks = nn.Sequential(*[EncoderTransformerBlock(config) for _ in range(config["n_layers"])])
        
        self.final_norm  = nn.LayerNorm(config["emb_dim"])
        self.out_head    = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        self.out_head.weight = self.embedder.token_embedding.weight
        
        self.name        = "MazeEncoder"
        self.iname       = ''
        self.context_len = config["context_length"]

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.
        Recebe os tokens do labirinto incompleto e retorna os
        logits para cada token do labirinto considerando ele já preenchido.
        """
        x = self.drop_emb(self.embedder(in_idx))
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
    
    @torch.no_grad()
    def get_attention(self, in_idx: torch.Tensor, layer_idx: int = -1):
        """
        Executa a passagem do modelo extraindo os pesos de atenção de uma camada específica.
        """
        x = self.drop_emb(self.embedder(in_idx))
        extracted_weights = None
        
        if layer_idx < 0:
            layer_idx += len(self.transformer_blocks)

        for i, block in enumerate(self.transformer_blocks):
            if i == layer_idx:
                x, extracted_weights = block.weighted_forward(x)
            else:
                x = block(x)
                
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits, extracted_weights
    

class MazeDecoder(nn.Module):
    """
    Modelo Decoder para a resolução de labirintos.
    Resolve o labirinto passo a passo.

    **Parâmetros:**
        config (dict): Dicionário de configuração do modelo.
    """
    def __init__(self, config):
        super().__init__()
        self.embedder = MazeEmbedder(config["vocab_size"], config["emb_dim"], config["context_length"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.transformer_blocks = nn.Sequential(*[
            DecoderTransformerBlock(config) for _ in range(config["n_layers"])
        ])
        
        self.final_norm = nn.LayerNorm(config["emb_dim"])
        self.out_head   = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        self.out_head.weight = self.embedder.token_embedding.weight
        
        self.name        = "MazeDecoder"
        self.iname       = ''
        self.context_len = config["context_length"]

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.
        Recebe a sequência atual (labirinto e rota parcial) e retorna os logits 
        para prever o próximo passo.
        """
        x = self.drop_emb(self.embedder(in_idx))
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
    
    @torch.no_grad()
    def get_attention(self, in_idx: torch.Tensor, layer_idx: int = -1):
        """
        Executa a passagem do modelo extraindo os pesos de atenção de uma camada específica.
        """
        x = self.drop_emb(self.embedder(in_idx))
        extracted_weights = None
        
        if layer_idx < 0:
            layer_idx += len(self.transformer_blocks)

        for i, block in enumerate(self.transformer_blocks):
            if i == layer_idx:
                x, extracted_weights = block.weighted_forward(x)
            else:
                x = block(x)
                
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits, extracted_weights


class MazeDencoder(nn.Module):
    """
    Modelo Encoder-Decoder para a resolução de labirintos.
    O Encoder lê o labirinto todo obtendo um contexto global, enquanto o Decoder 
    gera a resposta consultando o contexto obtido via Cross-Attention, token a token.

    **Parâmetros:**
        config (dict): Dicionário de configuração do modelo.
    """
    def __init__(self, config):
        super().__init__()
        self.embedder = MazeEmbedder(config["vocab_size"], config["emb_dim"], config["context_length"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.encoder_blocks = nn.Sequential(*[
            EncoderTransformerBlock(config) for _ in range(config["n_layers"])
        ])
        
        self.encoder_norm = nn.LayerNorm(config["emb_dim"])
        self.decoder_blocks = nn.ModuleList([
            CrossDecoderTransformerBlock(config) for _ in range(config["n_layers"])
        ])
        
        self.final_norm = nn.LayerNorm(config["emb_dim"])
        self.out_head   = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        self.out_head.weight = self.embedder.token_embedding.weight
        
        self.name        = "MazeDencoder"
        self.iname       = ''
        self.context_len = config["context_length"]

    def encode(self, maze_idx: torch.Tensor) -> torch.Tensor:
        maze_emb = self.drop_emb(self.embedder(maze_idx))
        ctx = self.encoder_blocks(maze_emb)
        return self.encoder_norm(ctx)

    def decode_step(self, target_idx: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.drop_emb(self.embedder(target_idx))
        for block in self.decoder_blocks:
            x = block(x, context=context)
        x = self.final_norm(x)
        return self.out_head(x)

    def forward(self, maze_idx: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo Encoder-Decoder.
        Recebe o labirinto e a rota em construção,
        avançando com a construção dela.
        """
        context = self.encode(maze_idx)
        
        x = self.drop_emb(self.embedder(target_idx))
        for block in self.decoder_blocks:
            x = block(x, context=context)
            
        x = self.final_norm(x)
        return self.out_head(x)
    
    @torch.no_grad()
    def get_attention(self, maze_idx: torch.Tensor, target_idx: torch.Tensor, layer_idx: int = -1):
        """
        Extrai pesos de atenção do Decoder. Retorna um dict com
        a atenção interna e a atenção cruzada.
        """
        context = self.encode(maze_idx)
        
        x = self.drop_emb(self.embedder(target_idx))
        extracted_weights = None
        
        if layer_idx < 0:
            layer_idx += len(self.decoder_blocks)

        for i, block in enumerate(self.decoder_blocks):
            if i == layer_idx:
                x, extracted_weights = block.weighted_forward(x, context=context)
            else:
                x = block(x, context=context)
                
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits, extracted_weights