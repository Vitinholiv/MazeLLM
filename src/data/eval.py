import torch
from src.architecture.model import MazeGPTModel
from src.data.preprocess import MazeTokenizer

@torch.no_grad()
def solve_maze(model: MazeGPTModel, tokenizer: MazeTokenizer,
               unsolved_maze: str, device: torch.device,
               temperature: float = 0.0) -> str:
    """
    temperature = 0.0  →  greedy / deterministic
    temperature > 0.0  →  sampled
    """
    model.eval()
    ctx_size = model.trf_blocks[0].att.mask.shape[0]

    clean_unsolved = unsolved_maze.strip() + '\n'
    prompt = clean_unsolved + '&\n'
    
    ids    = tokenizer.encode(prompt)
    idx    = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    tokens_to_generate = len(tokenizer.encode(clean_unsolved))-1

    generated = []
    for _ in range(tokens_to_generate):
        logits  = model(idx[:, -ctx_size:])[:, -1, :]

        if temperature < 1e-8:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs   = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.pad_id:
            break

        idx = torch.cat([idx, next_id], dim=1)
        generated.append(next_id.item())
        
    return tokenizer.decode(generated)

