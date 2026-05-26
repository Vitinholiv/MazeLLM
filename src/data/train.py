import os
import time
import json
import torch
import torch.nn as nn
from torch import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from src.architecture.model import MazeGPTModel
from src.data.preprocess import DataLoader, MazeTokenizer, build_dataloader
from src.general.configs import MAZE_GPT_CONFIG, init

def train(model: MazeGPTModel, src: str,
          num_epochs: int, device: torch.device, lr: float = 3e-4):
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    dataloader = build_dataloader(os.path.join('train',src), max_length=model.context_len, batch_size=8)
    name = f'{model.name}_{src[:-4]}_{time.time_ns()}'
    model.iname = name

    os.makedirs("runs", exist_ok=True)
    os.makedirs(os.path.join("runs",model.name), exist_ok=True)
    pt_path = os.path.join(os.path.join("runs",model.name), f"{name}.pt")
    json_path = os.path.join(os.path.join("runs",model.name), f"{name}.json")

    best_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1:>3}/{num_epochs}")
        
        for inputs, targets in progress_bar:
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
                loss   = criterion(logits.flatten(0, 1), targets.flatten())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg = total_loss / len(dataloader)
        print(f"Época {epoch + 1:>3}. Loss: {avg:.4f}")
        
        if avg < best_loss:
            print(f"Best Loss: ({best_loss:.4f} -> {avg:.4f}). Salvando {name}...")
            best_loss = avg
        
            torch.save(model.state_dict(), pt_path)
            info_data = {
                "model_name": name,
                "best_epoch": epoch + 1,
                "total_epochs": num_epochs,
                "best_loss": best_loss,
                "learning_rate": lr,
                "dataset_source": src,
                "architecture_config": MAZE_GPT_CONFIG
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(info_data, f, indent=4, ensure_ascii=False)
    return model
