from torch.utils.data import Dataset, DataLoader
import tiktoken, torch

# Tokenizer - Byte Pair Encoder
tokenizer = tiktoken.get_encoding("gpt2")

# Dataset Organizer
class GPTdataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        if len(token_ids) < max_length + 1:
            raise ValueError(f"Text is too short. Need at least {max_length + 1} tokens, but got {len(token_ids)}.")

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Dataloader Creation
def build_dataloader(txt, batch_size=4, max_length=256, stride=128,
                    shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTdataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

# Embedding Layer
def get_embedding_layer(vocab_size, output_dim):
    return torch.nn.Embedding(vocab_size, output_dim)

