import os
import time
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard.writer import SummaryWriter
from src.core.configs import init, ModelConfigs
from src.core.preprocess import build_dataloader
from src.general.metrics import calculate_metrics, ids_to_matrix

def train(run_id: str, config: str, dataset: str, directions_task: bool, batch_size: int = 64,
          epochs: int = 10, lr: float = 3e-4, metric_log_interval: int = 50, metric_sample_count: int = 8):

    tokenizer, model, device = init(config, directions_task)
    conf = ModelConfigs.get(config)
    task_name = 'directions' if directions_task else 'completion'
    best_loss = float('inf')
    global_step = 0

    dataloader = build_dataloader(
        dsrc=dataset,
        tokenizer_type=conf['tokenizer_type'],
        context_length=conf['context_length'],
        batch_size=batch_size,
        shuffle=True,
        mode=conf['dataset_mode'],
        lab_size=conf['lab_size'],
        directions_task=directions_task
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    print(f"\nStarting training process: {epochs} Epochs | Batch Size: {batch_size} | LR: {lr}")
    os.makedirs("runs", exist_ok=True)
    os.makedirs(f"runs/{config}", exist_ok=True)
    os.makedirs(f"runs/{config}/{task_name}", exist_ok=True)
    os.makedirs(f"runs/{config}/{task_name}/{run_id}", exist_ok=True)
    savefolder = f"runs/{config}/{task_name}/{run_id}"
    logpath = f"{savefolder}/training_logs.json"
    bestpath = f"{savefolder}/best_model.pt"
    epochpath = lambda e: f"{savefolder}/epoch_{e}.pt"

    writer = SummaryWriter(log_dir=f"{savefolder}/tensorboard")

    training_logs = {
        "metadata": {
            "run_id": run_id, "config": config, "dataset": dataset, "task": task_name,
            "batch_size": batch_size, "epochs": epochs, "learning_rate": lr,
            "parameters_count": sum(p.numel() for p in model.parameters())
        },
        "epoch_history": [], "step_loss_history": [], "lr_history": [], "metrics_history": []
    }

    def log_batch_metrics(input_ids_batch, target_ids_batch, logits, step):
        with torch.no_grad():
            pred_ids_batch = logits.argmax(dim=-1)

        n = min(metric_sample_count, input_ids_batch.size(0))
        id_to_char = tokenizer.id_to_char
        lab_size = conf['lab_size']

        score_sums, score_count = {}, 0
        for i in range(n):
            input_matrix = ids_to_matrix(input_ids_batch[i].detach().cpu(), id_to_char, lab_size, "prompt")
            target_matrix = ids_to_matrix(target_ids_batch[i].detach().cpu(), id_to_char, lab_size, "solution")
            pred_matrix = ids_to_matrix(pred_ids_batch[i].detach().cpu(), id_to_char, lab_size, "solution")

            _, score = calculate_metrics(input_matrix, target_matrix, pred_matrix, lab_size)
            for k, v in score.items():
                score_sums[k] = score_sums.get(k, 0.0) + v
            score_count += 1

        avg_scores = {k: total / score_count for k, total in score_sums.items()}
        training_logs["metrics_history"].append({"step": step, "values": avg_scores})

    def train_one_epoch(epoch):
        nonlocal best_loss, global_step
        model.train()
        total_loss = 0
        start_time = time.time()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            if conf["dataset_mode"] == "dencoder":
                m_in, r_in, r_out = [tensor.to(device) for tensor in batch]
                logits = model(m_in, r_in)
                input_ids_batch, targets = m_in, r_out
            elif conf["dataset_mode"] in ["decoder", "encoder"]:
                x, y = [tensor.to(device) for tensor in batch]
                logits = model(x)
                input_ids_batch, targets = x, y
            else:
                raise ValueError("Unknown dataset mode found during training.")

            loss = criterion(logits.view(-1, conf["vocab_size"]), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            training_logs["step_loss_history"].append(loss.item())
            writer.add_scalar("train/loss_step", loss.item(), global_step)

            if global_step % metric_log_interval == 0:
                log_batch_metrics(input_ids_batch, targets, logits, global_step)

            global_step += 1

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        new_best_sep = "\n"

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), bestpath)
            new_best_sep = " *\n"

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}", end=new_best_sep)
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        training_logs["epoch_history"].append({
            "epoch": epoch, "avg_loss": avg_loss, "time_seconds": epoch_time
        })
        training_logs["lr_history"].append(current_lr)

        with open(logpath, "w", encoding="utf-8") as f:
            json.dump(training_logs, f, indent=4)
        torch.save(model.state_dict(), epochpath(epoch))

    for epoch in range(1, epochs + 1):
        train_one_epoch(epoch)

    writer.close()

if __name__ == "__main__":
    train(
        'Testyh',
        'SimpleDecoder',
        'datasets/train/completion/Simple_example.txt',
        False,
        batch_size=32,
        epochs=5,
        lr=1e-3
    )