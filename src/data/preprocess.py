import os
import math
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
    
    def decode(self, ids: list) -> str:
        return "".join(self.id_to_char.get(i, '?') for i in ids)

class MazeDataset(Dataset):
    def __init__(self, txt: str, tokenizer: MazeTokenizer, max_len: int, mode="dencoder", fixed_output=False):
        self.tokenizer, self.mode = tokenizer, mode
        self.samples = []
        
        for block in txt.split("\n\n"):
            if '&\n' not in block: 
                continue
            
            m_part, r_part = block.split('&\n')
            m_ids = tokenizer.encode(m_part.strip())
            r_ids = tokenizer.encode(r_part.strip())
            
            if self.mode == "single":
                if fixed_output:
                    full = m_ids + [tokenizer.sep_id, tokenizer.start_id] + r_ids
                else:
                    full = m_ids + [tokenizer.sep_id, tokenizer.start_id] + r_ids + [tokenizer.end_id]
                    
                full = (full + [tokenizer.pad_id] * (max_len + 1))[:max_len + 1]
                self.samples.append((torch.tensor(full[:-1]), torch.tensor(full[1:])))
            
            elif self.mode == "dencoder":
                m_in = (m_ids + [tokenizer.pad_id] * max_len)[:max_len]
                r_in = ([tokenizer.start_id] + r_ids + [tokenizer.pad_id] * max_len)[:max_len]
                
                if fixed_output:
                    r_out = (r_ids + [tokenizer.pad_id] * max_len)[:max_len]
                else:
                    r_out = (r_ids + [tokenizer.end_id] + [tokenizer.pad_id] * max_len)[:max_len]
                    
                self.samples.append((torch.tensor(m_in), torch.tensor(r_in), torch.tensor(r_out)))

    def _pad(self, ids, length):
        return (ids + [self.tokenizer.pad_id] * length)[:length]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def build_dataloader(dsrc: str, max_length: int, batch_size: int = 4,
                     shuffle: bool = True, mode: str = "dencoder", fixed_output: bool = False) -> DataLoader:
    abs_path = os.path.abspath(dsrc)
    with open(abs_path, 'r', encoding='utf-8') as f:
        txt = f.read()
    
    tokenizer = MazeTokenizer()
    dataset   = MazeDataset(txt, tokenizer, max_length, mode=mode, fixed_output=fixed_output)

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
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]]) 
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(x) + self.pe[:, :x.size(1), :]