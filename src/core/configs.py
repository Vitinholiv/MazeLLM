import torch
from enum import Enum
from src.core.preprocess import SimpleTokenizer
from src.architecture.models import MazeDecoder

class ModelConfigs:

    SimpleDecoder = {
        "name":           "SimpleDecoder",
        "class":          MazeDecoder,
        "tokenizer":      SimpleTokenizer,
        "tokenizer_type": "individual",
        "dataset_mode":   "decoder",
        "tasks":          ["completion", "directions"],
        "context_length": 1024,
        "emb_dim":        256,
        "n_heads":        8,
        "n_layers":       6,
        "drop_rate":      0.1,
        "qkv_bias":       False,
        "lab_size":       21
    }

    @classmethod
    def get(cls, model_name: str) -> dict:
        return getattr(cls, model_name)

    @classmethod
    def get_all(cls) -> dict:
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith('__') and isinstance(value, dict)
        }
    

def init(model_name: str, directions_task: bool = False, seed: int = 42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = ModelConfigs.get(model_name)

    tokenizer = conf["tokenizer"](conf["lab_size"],directions_task)
    conf["vocab_size"] = tokenizer.vocab_size
    model = conf["class"](conf).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    task_desc = "Directions" if directions_task else "Completion"
    tok_desc = conf["tokenizer"].__name__

    print(f"Initializing Program: {model_name}")
    print(f"- Device: {device}")
    print(f"- Tokenizer: {tok_desc} (vocab_size={tokenizer.vocab_size})")
    print(f"- Task: {task_desc}")
    print(f"- Parameters: {n_params:,}")

    return tokenizer, model, device