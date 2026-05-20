import torch
from src.data.preprocess import MazeTokenizer
from src.architecture.model import MazeGPTModel

#  C++ generator settings: GRID_W = GRID_H = 20
#  Each row  = 20 chars + '\n'  = 21 tokens
#  One maze  = 20 rows × 21    = 420 tokens
#  Full seq  = 420 (unsolved) + 1 (&) + 1 (\n) + 420 (solved)
#            = 842 tokens

MAZE_GPT_CONFIG = {
    "vocab_size":     11,              # characters in MazeTokenizer
    "context_length": 842,             # max tokens the model can see at once
    "emb_dim":        256,             # hidden dimension
    "n_heads":        8,               # attention heads; head_dim = 256/8 = 32
    "n_layers":       6,               # number of stacked TransformerBlocks
    "drop_rate":      0.1,             # dropout probability
    "qkv_bias":       False,           # no bias in Q/K/V projections
}

def init(seed = 42, conf = MAZE_GPT_CONFIG):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = MazeTokenizer()
    if conf == MAZE_GPT_CONFIG:
        model = MazeGPTModel(MAZE_GPT_CONFIG)

    n_params = sum(p.numel() for p in model.parameters())

    print("Initializing Program")
    print(f"- Device: {device}")
    print(f"- Parameters: {n_params:,}")

    sample = "S  #\n#  #\n&\nR  #\n#  E\n"
    enc    = tokenizer.encode(sample)
    t_in   = torch.tensor(enc, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(t_in)
    print(f"- Forward pass OK -> logits shape: {logits.shape}\n")

    return tokenizer, model, device


