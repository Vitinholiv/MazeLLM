import torch
from src.data.preprocess import MazeTokenizer
from src.architecture.models import MazeDecoder, MazeEncoder, MazeDencoder

class ModelConfigs:

    MazeDecoder = {
        "name":           "MazeDecoder",
        "class":          MazeDecoder,
        "vocab_size":     13,
        "context_length": 842,
        "emb_dim":        256,
        "n_heads":        8,
        "n_layers":       6,
        "drop_rate":      0.1,
        "qkv_bias":       False,
    }

    MazeEncoder = {
        "name":           "MazeEncoder",
        "class":          MazeEncoder,
        "vocab_size":     13,
        "context_length": 842,
        "emb_dim":        256,
        "n_heads":        8,
        "n_layers":       6,
        "drop_rate":      0.1,
        "qkv_bias":       False,
    }
    
    MazeDencoder = {
        "name":           "MazeDencoder",
        "class":          MazeDencoder,
        "vocab_size":     13,
        "context_length": 842,
        "emb_dim":        256,
        "n_heads":        8,
        "n_layers":       6,
        "drop_rate":      0.1,
        "qkv_bias":       False,
    }

    @classmethod
    def get(cls, model_name: str) -> dict|None:
        """Gets a model dictionary by its string name, or None if not found."""
        return getattr(cls, model_name, None)

    @classmethod
    def get_all(cls) -> dict:
        """Returns the name of all models defined in the class."""
        models = {}
        for key, value in cls.__dict__.items():
            if not key.startswith('__') and isinstance(value, dict):
                models[key] = value
        return models

def init(model_name: str, seed: int = 42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    conf = ModelConfigs.get(model_name)
    if conf is None:
        raise ValueError(f"Configuração {model_name} não encontrada.")

    tokenizer = MazeTokenizer()
    model = conf["class"](conf).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    print(f"Initializing Program: {model_name}")
    print(f"- Device: {device}")
    print(f"- Parameters: {n_params:,}")

    is_dencoder = isinstance(model, MazeDencoder)
    with torch.no_grad():
        if is_dencoder:
            maze = torch.randint(0, tokenizer.vocab_size, (1, 32)).to(device)
            r_in = torch.randint(0, tokenizer.vocab_size, (1, 32)).to(device)
            logits = model(maze, r_in)
        else:
            t_in = torch.randint(0, tokenizer.vocab_size, (1, 64)).to(device)
            logits = model(t_in)
    print(f"- Forward pass OK -> logits shape: {logits.shape}\n")

    return tokenizer, model, device