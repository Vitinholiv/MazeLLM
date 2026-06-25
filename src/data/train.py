import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from src.general.configs import init, ModelConfigs
from src.data.preprocess import build_dataloader

def train(config: str, dataset: str, directions_task: bool, batch_size: int = 64, epochs: int = 10, lr: float = 3e-4):

    run_id = time.time_ns()
    tokenizer, model, device = init(config, directions_task)
    conf = ModelConfigs.get(config)

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

    def train_one_epoch(epoch):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()

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

            loss = criterion(
                logits.view(-1, conf["vocab_size"]), 
                targets.view(-1)
            )
            loss.backward() 
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        os.makedirs(f"runs/{config}", exist_ok=True)
        savepath = f"runs/{config}/{run_id}_epoch_{epoch}.pt"
        torch.save(model.state_dict(), savepath)

    for epoch in range(1, epochs+1):
        train_one_epoch(epoch)

if __name__ == "__main__":
    train(
        'SimpleDecoder',
        'datasets/train/completion/Simple_dataset.txt',
        False,
        batch_size=32,
        epochs=20
    )