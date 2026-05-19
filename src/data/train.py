import torch
import torch.nn as nn
from tqdm import tqdm
from src.architecture.model import MazeGPTModel
from src.data.preprocess import DataLoader, MazeTokenizer, build_dataloader
from src.configs import MAZE_GPT_CONFIG, init

def train(model: MazeGPTModel, src: str,
          num_epochs: int, device: torch.device, lr: float = 3e-4):
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    dataloader = build_dataloader(src, max_length=MAZE_GPT_CONFIG['context_length'], batch_size=8)

    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1:>3}/{num_epochs}")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss   = criterion(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg = total_loss / len(dataloader)
        print(f"End of Epoch {epoch + 1:>3} — Average Loss: {avg:.4f}\n")
    return model