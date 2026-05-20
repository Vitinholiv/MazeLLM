import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Tokenizer
class MazeTokenizer:
    def __init__(self):
        self.char_to_id = {
            '<PAD>': 0,   # padding
            '#':     1,   # wall
            ' ':     2,   # open path
            'S':     3,   # start cell
            'E':     4,   # end cell
            '\n':    5,   # row separator inside maze grid
            '&':     6,   # maze separator
            'L':     7,   # direction: Left
            'R':     8,   # direction: Right
            'U':     9,   # direction: Up
            'D':    10,   # direction: Down
        }
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        self.pad_id     = self.char_to_id['<PAD>']
        self.sep_id     = self.char_to_id['&']

    def encode(self, text: str) -> list:
        tokens = []; i = 0
        while i < len(text):
            if text[i:i+5] == '<PAD>':
                tokens.append(self.char_to_id['<PAD>'])
                i += 5
            elif text[i] in self.char_to_id:
                tokens.append(self.char_to_id[text[i]])
                i += 1
            else:
                i += 1 
        return tokens

    def decode(self, ids: list) -> str:
        return "".join(self.id_to_char.get(i, '?') for i in ids)

# Dataset
class MazeDataset(Dataset):
    def __init__(self, txt: str, tokenizer: MazeTokenizer, max_length: int):
        self.input_ids  = []
        self.target_ids = []

        mazes = [m for m in txt.split("\n\n")]

        for maze in mazes:
            ids = tokenizer.encode(maze)
            if len(ids) < 2:
                continue

            if len(ids) < max_length + 1:
                ids = ids + [tokenizer.pad_id] * (max_length + 1 - len(ids))
            else:
                ids = ids[:max_length + 1]

            self.input_ids.append(torch.tensor(ids[:max_length],     dtype=torch.long))
            self.target_ids.append(torch.tensor(ids[1:max_length+1], dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Dataloader
def build_dataloader(dsrc: str, max_length: int, batch_size: int = 4,
                     shuffle: bool = True, drop_last: bool = True,
                     num_workers: int = 0) -> DataLoader:
    with open(os.path.join('datasets',dsrc)) as f:
        txt = f.read()
    tokenizer = MazeTokenizer()
    dataset   = MazeDataset(txt, tokenizer, max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

# Encoder
class MazeEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding   = nn.Embedding(max_len,   d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return self.token_embedding(x) + self.pos_embedding(positions)