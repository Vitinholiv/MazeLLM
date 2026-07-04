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
        raise NotImplementedError()

class SimpleTokenizer(BaseMazeTokenizer):
    def __init__(self, size: int, directions: bool):
        self.special = {
            '<PAD>': 0, 
            '<LABYRINTH_START>': 1, '<LABYRINTH_END>': 2, 
            '<SOLUTION_START>': 3, '<SOLUTION_END>': 4
        }
        self.grid = {'#': 5, ' ': 6, 'S': 7, 'E': 8}
        self.dirs = {'L': 9, 'R': 10, 'U': 11, 'D': 12}
        self.size = size
        self.directions = directions

        vocab = {**self.special, **self.grid, **self.dirs}
        super().__init__(vocab)

    def encode(self, text: str) -> list:
        ids = []
        for line in text.split('\n'):
            line = line.strip()
            if not line: continue
            
            if line in self.special.keys():
                ids.append(self.char_to_id[line])
            elif '<' not in line:
                for i in range(0,len(line),2):
                    ids.append(self.char_to_id[line[i]])        
        return ids
    
    def decode(self, ids: list) -> str:
        
        nx_li = ids.index(1) if 1 in ids else -1
        nx_le = ids.index(2) if 2 in ids else -1
        nx_si = ids.index(3) if 3 in ids else -1
        nx_se = ids.index(4) if 4 in ids else -1

        if nx_li == -1 or nx_le == -1 or nx_si == -1 or nx_se == -1:
            return "|".join([self.id_to_char[idd] for idd in ids])
            return f"<INVALID_LABYRINTH: {nx_li}|{nx_le} , {nx_si}|{nx_se}>"
        
        if self.directions == False:
            qt_lab = (nx_le - nx_li - 1)
            qt_sol = (nx_se - nx_si - 1)
            if qt_lab != self.size**2 or qt_sol != self.size**2:
                return f"<INVALID_LABYRINTH: {qt_lab} x {qt_sol}>"
        else:
            qt_lab = (nx_le - nx_li - 1)
            if qt_lab != self.size**2:
                return f"<INVALID_LABYRINTH: {qt_lab}!>"
            
        res = []; idx = 0
        for i in range(len(ids)):
            char = self.id_to_char.get(ids[i], '?')
            if char == '<LABYRINTH_START>' or char == '<SOLUTION_START>' or char == '<LABYRINTH_END>':
                res.append(char+'\n')
            elif char == '<SOLUTION_END>':
                if self.directions:
                    res.append('\n')
                res.append(char)
            else:
                if self.directions == False or char not in self.dirs:
                    idx += 1
                    res.append(char)
                    if idx < self.size:
                        res.append(' ')
                    else:
                        res.append('\n')
                        idx = 0
                else:
                    res.append(char)
                    if i < nx_se-1:
                        res.append(' ')

        return "".join(res)

# Dataset and Dataloader

class MazeDataset(Dataset):
    def __init__(self, txt: str, tokenizer: BaseMazeTokenizer, context_len: int, mode: str):
        self.tokenizer = tokenizer
        self.mode = mode
        self.samples = []
        
        blocks = [b for b in txt.split("\n\n")]
        
        for block in blocks:
            split_idx = block.index('<SOLUTION_START>')
                
            m_part = block[:split_idx]
            r_part = block[split_idx:]
            
            m_ids = tokenizer.encode(m_part)
            r_ids = tokenizer.encode(r_part)
            
            if self.mode == "decoder":
                full = m_ids + r_ids
                full = (full + [tokenizer.pad_id] * (context_len + 1))[:context_len + 1]
                
                x = torch.tensor(full[:-1])
                y = torch.tensor(full[1:])
                y[:len(m_ids) - 1] = tokenizer.pad_id
                
                self.samples.append((x, y))

            elif self.mode == "encoder":
                x = (m_ids + [tokenizer.pad_id] * context_len)[:context_len]
                y = (r_ids + [tokenizer.pad_id] * context_len)[:context_len]
                
                self.samples.append((torch.tensor(x), torch.tensor(y)))
            
            elif self.mode == "dencoder":
                m_in = (m_ids + [tokenizer.pad_id] * context_len)[:context_len]
                r_in = (r_ids + [tokenizer.pad_id] * context_len)[:context_len]
                r_out = (r_ids[1:] + [tokenizer.pad_id] * context_len)[:context_len]
                
                self.samples.append((torch.tensor(m_in), torch.tensor(r_in), torch.tensor(r_out)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def build_dataloader(dsrc: str, tokenizer_type: str, context_length: int, batch_size: int = 64,
                     shuffle: bool = True, mode: str = "decoder", lab_size: int = 21, directions_task: bool = False) -> DataLoader:
    abs_path = os.path.abspath(dsrc)
    with open(abs_path, 'r', encoding='utf-8') as f:
        txt = f.read()
    
    if tokenizer_type == "individual":
        tokenizer = SimpleTokenizer(lab_size,directions_task)
    else:
        raise ValueError("Invalid labyrinth token type, choose between 'individual', 'wall_encoded' and 'free_edges'.")
        
    dataset = MazeDataset(txt, tokenizer, context_length, mode=mode)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

# Embedder

class MazeEmbedder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, context_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        pe = torch.zeros(context_len, d_model)
        pos = torch.arange(0, context_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]]) 
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        return self.token_embedding(x) + self.pe[:, start_pos : start_pos + x.size(1), :]
