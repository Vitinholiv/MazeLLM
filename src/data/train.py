import os
import time
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from src.general.configs import init, ModelConfigs
from src.data.preprocess import build_dataloader

def train(config: str, dataset: str, directions_task: bool, batch_size: int = 64, epochs: int = 10, lr: float = 3e-4):

    # Globals
    run_id = time.time_ns()
    tokenizer, model, device = init(config, directions_task)
    conf = ModelConfigs.get(config)
    task_name = 'directions' if directions_task else 'completion'
    best_loss = float('inf')

    # Dataloader
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
    
    # Loss and Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # Save Folders
    print(f"\nStarting training process: {epochs} Epochs | Batch Size: {batch_size} | LR: {lr}")
    os.makedirs("runs", exist_ok=True)
    os.makedirs(f"runs/{config}", exist_ok=True)
    os.makedirs(f"runs/{config}/{task_name}", exist_ok=True)
    os.makedirs(f"runs/{config}/{task_name}/{run_id}", exist_ok=True)
    savefolder = f"runs/{config}/{task_name}/{run_id}"

    # Training Logs
    training_logs = {
        "metadata": {
            "run_id": run_id,
            "config": config,
            "dataset": dataset,
            "task": task_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "parameters_count": sum(p.numel() for p in model.parameters())
        },
        "epoch_history": [],
        "step_loss_history": [],
        "lr_history": []
    }

    # Epoch
    def train_one_epoch(epoch):
        # Variables
        nonlocal best_loss
        model.train()
        total_loss = 0
        start_time = time.time()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        # Batch Iteration
        for batch in pbar:
            # Zero Grad
            optimizer.zero_grad()

            # Forward Pass
            if conf["dataset_mode"] == "dencoder":
                m_in, r_in, r_out = [tensor.to(device) for tensor in batch]
                logits = model(m_in, r_in) 
                targets = r_out
                
            elif conf["dataset_mode"] in ["decoder", "encoder"]:
                x, y = [tensor.to(device) for tensor in batch]
                logits = model(x) 
                targets = y
            else:
                raise ValueError("Unknown dataset mode found during training.")

            # Calculate Loss
            loss = criterion(
                logits.view(-1, conf["vocab_size"]), 
                targets.view(-1)
            )
            loss.backward() 
            optimizer.step()
            
            # Show and Save Batch Loss
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            training_logs["step_loss_history"].append(loss.item())
        
        # Recalculate Variables
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        new_best_sep = "\n"

        # Update Best Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{savefolder}/best_model.pt")
            new_best_sep = " *\n"

        # Get Epoch Info
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}",end=new_best_sep)
        training_logs["epoch_history"].append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "time_seconds": epoch_time
        })
        training_logs["lr_history"].append(current_lr)

        # Saving
        logpath = f"{savefolder}/training_logs.json"
        with open(logpath, "w", encoding="utf-8") as f:
            json.dump(training_logs, f, indent=4)

        savepath = f"{savefolder}/epoch_{epoch}.pt"
        torch.save(model.state_dict(), savepath)

    for epoch in range(1, epochs+1):
        train_one_epoch(epoch)

if __name__ == "__main__":
    train(
        'SimpleDecoder',
        'datasets/train/directions/Simple_dataset.txt',
        True,
        batch_size=32,
        epochs=20,
        lr=1e-3
    )