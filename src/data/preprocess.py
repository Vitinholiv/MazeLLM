import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re

# Tokenizers

class BaseMazeTokenizer:
    def __init__(self, vocab_dict):
        self.char_to_id = vocab_dict
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.pad_id = self.char_to_id['<PAD>']
        self.vocab_size = len(self.char_to_id)

    def encode(self, text: str) -> list:
        raise NotImplementedError()
    
    def decode(self, ids: list) -> str:
        return "".join(self.id_to_char.get(i, '?') for i in ids)

class SimpleTokenizer(BaseMazeTokenizer):
    def __init__(self):
        special = {
            '<PAD>': 0, 
            '<LABYRINTH_START>': 1, '<LABYRINTH_END>': 2, 
            '<SOLUTION_START>': 3, '<SOLUTION_END>': 4,
            '<COMPLETION_START>': 5, '<COMPLETION_END>': 6
        }
        grid = {'#': 7, ' ': 8, 'S': 9, 'E': 10, '\n': 11}
        dirs = {'L': 12, 'R': 13, 'U': 14, 'D': 15, '*': 16}
        
        vocab = {**special, **grid, **dirs}
        super().__init__(vocab)

    def encode(self, text: str) -> list:
        ids = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line: continue
            
            if line.startswith('<') and line.endswith('>'):
                for t in line.split(' '):
                    if t in self.char_to_id:
                        ids.append(self.char_to_id[t])
            else:
                for t in line.split(' '):
                    if t in self.char_to_id:
                        ids.append(self.char_to_id[t])
            
            ids.append(self.char_to_id['\n'])
            
        if ids and ids[-1] == self.char_to_id['\n']:
            ids.pop()
        return ids
    
    def decode(self, ids: list) -> str:
        res = []
        for i in ids:
            char = self.id_to_char.get(i, '?')
            if char.startswith('<') and char != '<PAD>':
                res.append(f"\n{char}\n")
            elif char not in ['\n', '<PAD>']:
                res.append(char + ' ')
            else:
                res.append(char)
        return "".join(res).replace(" \n", "\n").strip()

class WallEncodedTokenizer(BaseMazeTokenizer):
    def __init__(self):
        special = {
            '<PAD>': 0, 
            '<LABYRINTH_START>': 1, '<LABYRINTH_END>': 2, 
            '<SOLUTION_START>': 3, '<SOLUTION_END>': 4,
            '<COMPLETION_START>': 5, '<COMPLETION_END>': 6
        }
        isolated = {'L': 7, 'R': 8, 'U': 9, 'D': 10, '\n': 11}
        
        grid_states = {}
        idx = 12
        for base in ['.', 'S', 'E', 'L', 'R', 'U', 'D', '*']:
            for u in range(2):
                for d in range(2):
                    for l in range(2):
                        for r in range(2):
                            grid_states[f"{base}{u}{d}{l}{r}"] = idx
                            idx += 1
                            
        vocab = {**special, **isolated, **grid_states}
        super().__init__(vocab)

    def encode(self, text: str) -> list:
        ids = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line: continue
            
            for t in line.split(' '):
                if t in self.char_to_id:
                    ids.append(self.char_to_id[t])
                    
            if not line.startswith('<'):
                ids.append(self.char_to_id['\n'])
                
        if ids and ids[-1] == self.char_to_id['\n']:
            ids.pop()
        return ids

    def decode(self, ids: list) -> str:
        res = []
        for i in ids:
            char = self.id_to_char.get(i, '?')
            if char.startswith('<') and char != '<PAD>':
                res.append(f"\n{char}\n")
            elif len(char) == 5:
                res.append(char + ' ')
            elif char in ['L', 'R', 'U', 'D', '*']:
                res.append(char + ' ')
            else:
                res.append(char)
        return "".join(res).replace(" \n", "\n").strip()

class EdgeListTokenizer(BaseMazeTokenizer):
    def __init__(self, max_grid_size=30):
        special = {
            '<PAD>': 0, 
            '<LABYRINTH_START>': 1, '<LABYRINTH_END>': 2, 
            '<ADJLIST_START>': 3, '<ADJLIST_END>': 4,
            '<ORIGIN>': 5, '<TARGET>': 6,
            '<SOLUTION_START>': 7, '<SOLUTION_END>': 8
        }
        symbols = {'<->': 9, ';': 10, '\n': 11}
        dirs    = {'L': 12, 'R': 13, 'U': 14, 'D': 15}
        
        coords = {}
        idx = 16
        for r in range(max_grid_size):
            for c in range(max_grid_size):
                coords[f"({r},{c})"] = idx
                idx += 1
        
        vocab = {**special, **symbols, **dirs, **coords}
        super().__init__(vocab)

    def encode(self, text: str) -> list:
        tokens = re.findall(r'<[A-Z_]+>|<->|[;LURD\n]|\(\d+,\d+\)', text)
        ids = [self.char_to_id[t] for t in tokens if t in self.char_to_id]
        return ids

    def decode(self, ids: list) -> str:
        res = []
        for i in ids:
            char = self.id_to_char.get(i, '?')
            if char == '<PAD>': continue
            
            if char == '<->': res.append(' <-> ')
            elif char == ';': res.append(' ; ')
            elif char.startswith('<'): res.append(f"\n{char}\n")
            else: res.append(char)
            
        return "".join(res).replace("\n\n", "\n").strip()
    
# Dataset and Dataloader

class MazeDataset(Dataset):
    def __init__(self, txt: str, tokenizer: BaseMazeTokenizer, max_len: int, mode="dencoder"):
        self.tokenizer = tokenizer
        self.mode = mode
        self.samples = []
        
        blocks = [b.strip() for b in txt.split("\n\n") if b.strip()]
        
        for block in blocks:
            if '<SOLUTION_START>' in block:
                split_idx = block.index('<SOLUTION_START>')
            elif '<COMPLETION_START>' in block:
                split_idx = block.index('<COMPLETION_START>')
            else:
                continue
                
            m_part = block[:split_idx].strip()
            r_part = block[split_idx:].strip()
            
            m_ids = tokenizer.encode(m_part)
            r_ids = tokenizer.encode(r_part)
            
            if self.mode == "single":
                full = m_ids + r_ids
                full = (full + [tokenizer.pad_id] * (max_len + 1))[:max_len + 1]
                
                x = torch.tensor(full[:-1])
                y = torch.tensor(full[1:])
                y[:len(m_ids) - 1] = tokenizer.pad_id
                
                self.samples.append((x, y))
            
            elif self.mode == "dencoder":
                m_in = (m_ids + [tokenizer.pad_id] * max_len)[:max_len]
                r_in = (r_ids + [tokenizer.pad_id] * max_len)[:max_len]
                r_out = (r_ids[1:] + [tokenizer.pad_id] * max_len)[:max_len]
                
                self.samples.append((torch.tensor(m_in), torch.tensor(r_in), torch.tensor(r_out)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def build_dataloader(dsrc: str, tokenizer_type: str, max_length: int, batch_size: int = 4,
                     shuffle: bool = True, mode: str = "dencoder") -> DataLoader:
    abs_path = os.path.abspath(dsrc)
    with open(abs_path, 'r', encoding='utf-8') as f:
        txt = f.read()
    
    if tokenizer_type == "individual":
        tokenizer = SimpleTokenizer()
    elif tokenizer_type == "wall_encoded":
        tokenizer = WallEncodedTokenizer()
    elif tokenizer_type == "free_edges":
        tokenizer = EdgeListTokenizer()
    else:
        raise ValueError("Invalid labyrinth token type, choose between 'individual', 'wall_encoded' and 'free_edges'.")
        
    dataset = MazeDataset(txt, tokenizer, max_length, mode=mode)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

# Embedder

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