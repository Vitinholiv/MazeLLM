import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MazeTokenizer:
    def __init__(self):
        self.special = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<SEP>': 3}
        self.grid    = {'#': 4, ' ': 5, 'S': 6, 'E': 7, '\n': 8}
        self.dir     = {'L': 9, 'R': 10, 'U': 11, 'D': 12}
        
        self.char_to_id = {**self.special, **self.grid, **self.dir}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
        self.pad_id   = self.char_to_id['<PAD>']
        self.sep_id   = self.char_to_id['<SEP>']
        self.start_id = self.char_to_id['<START>']
        self.end_id   = self.char_to_id['<END>']
        self.vocab_size = len(self.char_to_id)

    def encode(self, text: str) -> list:
        return [self.char_to_id.get(c, self.pad_id) for c in text if c in self.char_to_id]

class MazeDataset(Dataset):
    def __init__(self, txt: str, tokenizer: MazeTokenizer, max_len: int, mode="dencoder"):
        self.tokenizer = tokenizer
        self.mode = mode
        self.samples = []

        for block in txt.split("\n\n"):
            if '<SEP>' not in block: continue
            m_part, r_part = block.split('<SEP>')
            
            m_ids = tokenizer.encode(m_part.strip())
            r_ids = tokenizer.encode(r_part.strip())
            
            if self.mode == "single":
                # [MAPA, <SEP>, <START>, ROTA, <END>]
                full = m_ids + [tokenizer.sep_id, tokenizer.start_id] + r_ids + [tokenizer.end_id]
                full = self._pad(full, max_len + 1)
                self.samples.append((torch.tensor(full[:-1]), torch.tensor(full[1:])))
            
            elif self.mode == "dencoder"
                # Encoder(MAPA) -> Decoder(START, ROTA) -> Target(ROTA, END)
                m_in = self._pad(m_ids, max_len)
                r_in = self._pad([tokenizer.start_id] + r_ids, max_len)
                r_out = self._pad(r_ids + [tokenizer.end_id], max_len)
                self.samples.append((torch.tensor(m_in), torch.tensor(r_in), torch.tensor(r_out)))

    def _pad(self, ids, length):
        return (ids + [self.tokenizer.pad_id] * length)[:length]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def build_dataloader(dsrc: str, max_length: int, batch_size: int = 4,
                     shuffle: bool = True, mode: str = "dencoder") -> DataLoader:

    with open(dsrc) as f:
        txt = f.read()
    tokenizer = MazeTokenizer()
    dataset   = MazeDataset(txt, tokenizer, max_length, mode=mode)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )

class MazeEmbedder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding   = nn.Embedding(max_len,   d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return self.token_embedding(x) + self.pos_embedding(positions)