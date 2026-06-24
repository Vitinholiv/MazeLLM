import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

from src.architecture.models import MazeDencoder
from src.data.preprocess import build_dataloader

PAD_ID = 0
def compute_loss(logits: torch.Tensor, targets: torch.Tensor, fixed_output: bool) -> torch.Tensor:
    B, T, V = logits.shape

    token_losses = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).view(B, T)

    valid_mask = targets != PAD_ID
    valid_counts = valid_mask.sum(dim=1).clamp(min=1)

    if fixed_output:
        denom = valid_counts.max().clamp(min=1)
        per_sample_loss = token_losses.sum(dim=1) / denom
    else:
        per_sample_loss = token_losses.sum(dim=1) / valid_counts

    return per_sample_loss.mean()


def train(model: nn.Module, dataset: str, epochs: int, device: torch.device,
          lr: float = 3e-4, task: str = "completion", batch_size: int = 8):

    if task not in model.tasks:
        raise ValueError(
            f"'{model.name}' não suporta a tarefa '{task}'. "
            f"Tarefas disponíveis: {list(model.tasks.keys())}."
        )
    fixed_output = model.tasks[task]

    model.to(device).train()

    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]

    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr)

    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    is_dencoder = isinstance(model, MazeDencoder)
    mode = "dencoder" if is_dencoder else "single"

    dataloader = build_dataloader(
        dataset,
        tokenizer_type=model.tokenizer_type,
        max_length=model.context_len,
        batch_size=batch_size,
        mode=mode,
    )

    total_steps = epochs * len(dataloader)
    warmup_steps = min(200, max(1, total_steps // 10))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    dataset_name = os.path.basename(dataset).rsplit(".", 1)[0]
    run_name = f"{model.name}_{task}_{dataset_name}_{time.time_ns()}"
    setattr(model,'iname',run_name)

    writer = SummaryWriter(log_dir=os.path.join("tensorboard", model.name, run_name))

    vocab_size = model.embedder.token_embedding.num_embeddings
    try:
        if is_dencoder:
            dummy_maze = torch.randint(0, vocab_size, (1, 8)).to(device)
            dummy_route = torch.randint(0, vocab_size, (1, 8)).to(device)
            writer.add_graph(model, (dummy_maze, dummy_route))
        else:
            dummy_in = torch.randint(0, vocab_size, (1, 8)).to(device)
            writer.add_graph(model, dummy_in)
    except Exception as e:
        print(f"Aviso: não foi possível registrar o grafo do modelo no TensorBoard ({e}).")

    writer.add_text("config", json.dumps({
        "model":          model.name,
        "task":           task,
        "fixed_output":   fixed_output,
        "lr":             lr,
        "epochs":         epochs,
        "batch_size":     batch_size,
        "context_length": model.context_len,
    }, indent=4))

    run_dir = os.path.join("runs", model.name)
    os.makedirs(run_dir, exist_ok=True)
    pt_path = os.path.join(run_dir, f"{run_name}.pt")
    json_path = os.path.join(run_dir, f"{run_name}.json")

    best_loss = float("inf")
    global_step = 0

    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1:>3}/{epochs}")

        for batch in progress_bar:
            if is_dencoder:
                maze, r_in, r_out = (b.to(device, non_blocking=True) for b in batch)
            else:
                inputs, targets = (b.to(device, non_blocking=True) for b in batch)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if is_dencoder:
                    logits = model(maze, r_in)
                    loss = compute_loss(logits, r_out, fixed_output)
                else:
                    logits = model(inputs)
                    loss = compute_loss(logits, targets, fixed_output)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            writer.add_scalar("Train/Loss_batch", loss.item(), global_step)
            writer.add_scalar("Train/GradNorm",   total_norm.item(), global_step)
            writer.add_scalar("Train/LR",         scheduler.get_last_lr()[0], global_step)
            global_step += 1

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar("Train/Loss_epoch", avg_loss, epoch)
        print(f"Época {epoch + 1:>3}. Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), pt_path)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model_name":   run_name,
                    "task":         task,
                    "fixed_output": fixed_output,
                    "best_loss":    best_loss,
                    "epochs":       epochs,
                    "lr":           lr,
                }, f, indent=4)

        for param_name, param in model.named_parameters():
            writer.add_histogram(f"Weights/{param_name}", param, epoch)

    writer.close()
    return model