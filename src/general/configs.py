import torch
from enum import Enum
from src.data.preprocess import SimpleTokenizer, WallEncodedTokenizer, EdgeListTokenizer
from src.architecture.models import MazeDecoder, MazeEncoder, MazeDencoder

class OutputType(Enum):
    VARIABLE_OUTPUT = 0
    FIXED_OUTPUT = 1

TOKENIZERS = {
    "individual":   SimpleTokenizer,
    "wall_encoded": WallEncodedTokenizer,
    "free_edges":   EdgeListTokenizer,
}

class ModelConfigs:

    SimpleDecoder = {
        "name":           "SimpleDecoder",
        "class":          MazeDecoder,
        "tokenizer_type": "individual",
        "dataset_mode":   "single",
        "tasks":          {"completion": OutputType.FIXED_OUTPUT, "directions": OutputType.VARIABLE_OUTPUT},
        "vocab_size":     17,
        "context_length": 940,
        "emb_dim":        256,
        "n_heads":        8,
        "n_layers":       6,
        "drop_rate":      0.1,
        "qkv_bias":       False,
    }

    SimpleEncoder = {
        "name":           "SimpleEncoder",
        "class":          MazeEncoder,
        "tokenizer_type": "individual",
        "dataset_mode":   "single",
        "tasks":          {"completion": OutputType.FIXED_OUTPUT},
        "vocab_size":     17,
        "context_length": 940,
        "emb_dim":        256,
        "n_heads":        8,
        "n_layers":       6,
        "drop_rate":      0.1,
        "qkv_bias":       False,
    }

    SimpleDencoder = {
        "name":           "SimpleDencoder",
        "class":          MazeDencoder,
        "tokenizer_type": "individual",
        "dataset_mode":   "dencoder",
        "tasks":          {"completion": OutputType.FIXED_OUTPUT, "directions": OutputType.VARIABLE_OUTPUT},
        "vocab_size":     17,
        "context_length": 480,
        "emb_dim":        256,
        "n_heads":        8,
        "n_layers":       6,
        "drop_rate":      0.1,
        "qkv_bias":       False,
    }

    @classmethod
    def get(cls, model_name: str) -> dict | None:
        return getattr(cls, model_name, None)

    @classmethod
    def get_all(cls) -> dict:
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith('__') and isinstance(value, dict)
        }


def build_tokenizer(tokenizer_type: str):
    if tokenizer_type not in TOKENIZERS:
        raise ValueError(
            f"Tokenizador '{tokenizer_type}' inválido, escolha entre {list(TOKENIZERS.keys())}."
        )
    return TOKENIZERS[tokenizer_type]()


def is_fixed_output(conf: dict, task: str) -> bool:
    if task not in conf["tasks"]:
        raise ValueError(f"Tarefa '{task}' não suportada por '{conf['name']}'.")
    return conf["tasks"][task]


def init(model_name: str, seed: int = 42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = ModelConfigs.get(model_name)
    if conf is None:
        raise ValueError(f"Configuração {model_name} não encontrada.")

    tokenizer = build_tokenizer(conf["tokenizer_type"])
    model = conf["class"](conf).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    tasks_desc = ", ".join(f"{t} (fixed_output={f})" for t, f in conf["tasks"].items())

    print(f"Initializing Program: {model_name}")
    print(f"- Device: {device}")
    print(f"- Tokenizer: {conf['tokenizer_type']} (vocab_size={tokenizer.vocab_size})")
    print(f"- Tasks: {tasks_desc}")
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