import os
import time
import math
import json
import torch
import torch.nn as nn
from torch import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from src.architecture.models import MazeEncoder, MazeDecoder, MazeDencoder
from src.data.preprocess import build_dataloader

def train(model: nn.Module, src: str, num_epochs: int, device: torch.device, 
          lr: float = 3e-4, fixed_output: bool = False):
    
    model.to(device).train()
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

    optimizer = torch.optim.AdamW([
        {'params': decay_params,    'weight_decay': 0.01},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=lr)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    
    is_dencoder = isinstance(model, MazeDencoder)
    mode = "dencoder" if is_dencoder else "single"
    
    dataloader = build_dataloader(
        src,
        max_length=model.context_len, 
        batch_size=8, 
        mode=mode,
        fixed_output=fixed_output
    )

    total_steps = num_epochs * len(dataloader)
    warmup_steps = min(200, total_steps // 10)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    dataset_name = os.path.basename(src).replace('.txt', '')
    name = f'{model.name}_{dataset_name}_{time.time_ns()}'
    model.iname = name # type: ignore

    os.makedirs("runs", exist_ok=True)
    os.makedirs(os.path.join("runs", model.name), exist_ok=True)
    pt_path = os.path.join("runs", model.name, f"{name}.pt")
    json_path = os.path.join("runs", model.name, f"{name}.json")

    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1:>3}/{num_epochs}")
        
        for batch in progress_bar:
            if is_dencoder:
                maze, r_in, r_out = [b.to(device, non_blocking=True) for b in batch]
            else:
                inputs, targets = [b.to(device, non_blocking=True) for b in batch]

            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if is_dencoder:
                    logits = model(maze, r_in)
                    loss = criterion(logits.flatten(0, 1), r_out.flatten())
                else:
                    logits = model(inputs)
                    loss = criterion(logits.flatten(0, 1), targets.flatten())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg = total_loss / len(dataloader)
        print(f"Época {epoch + 1:>3}. Loss: {avg:.4f}")
        
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), pt_path)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model_name":   name,
                    "best_loss":    best_loss,
                    "fixed_output": fixed_output,
                    "epochs":       num_epochs,
                    "lr":           lr,
                }, f, indent=4)
                
    return model