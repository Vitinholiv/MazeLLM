import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from abc import ABC, abstractmethod
from src.architecture.models import MazeDencoder
from src.data.preprocess import build_dataloader
from src.general.helpers import prompt_options, path

class Action(ABC):
    def __init__(self, model, tokenizer, data, source, device, fixed_output=False):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.source = source
        self.device = device
        self.fixed_output = fixed_output
        self.is_dencoder = isinstance(model, MazeDencoder)

    @abstractmethod
    def run(self):
        pass

    @torch.no_grad()
    def solve_maze(self, unsolved_maze: str) -> str:
        self.model.eval()
        clean_unsolved = unsolved_maze.strip() + '\n'
        
        if self.is_dencoder:
            maze_ids = torch.tensor(self.tokenizer.encode(clean_unsolved), device=self.device).unsqueeze(0)
            target = torch.tensor([[self.tokenizer.start_id]], device=self.device)
            max_gen = len(self.tokenizer.encode(clean_unsolved)) if self.fixed_output else 256
            ctx = self.model.encode(maze_ids) 
            
            gen = []

            for _ in range(max_gen):
                logits = self.model.decode_step(target, ctx)[:, -1, :]
                next_id = logits.argmax(dim=-1).unsqueeze(0)
                if not self.fixed_output and next_id.item() == self.tokenizer.end_id:
                    break
                target = torch.cat([target, next_id], dim=1)
                gen.append(next_id.item())
            return self.tokenizer.decode(gen)
            
        else:
            maze_ids = self.tokenizer.encode(clean_unsolved)
            prompt_ids = maze_ids + [self.tokenizer.sep_id, self.tokenizer.start_id]
            idx = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
            
            gen = []
            for _ in range(128):
                logits = self.model(idx)[:, -1, :]
                next_id = logits.argmax(dim=-1).unsqueeze(0)
                
                if next_id.item() == self.tokenizer.end_id: 
                    break
                    
                idx = torch.cat([idx, next_id], dim=1)
                gen.append(next_id.item())
            return self.tokenizer.decode(gen)

class ActionSample(Action):
    def run(self):
        sample_raw = random.choice(self.data)
        parts = sample_raw.split('&\n')
        while len(parts) != 2:
            sample_raw = random.choice(self.data)
            parts = sample_raw.split('&\n')
        unsolved, expected = parts[0] + '\n', parts[1]
        pred = self.solve_maze(unsolved)
        print(f"\n[ENTRADA]\n{unsolved.strip()}\n[PREDITO]\n{pred.strip()}\n[ESPERADO]\n{expected.strip()}\n")

class ActionMetrics(Action):
    def run(self):
        print("\n[1/2] Calculando Loss...")
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        mode = "dencoder" if self.is_dencoder else "single"
        dataloader = build_dataloader(self.source, max_length=self.model.context_len, batch_size=8, shuffle=False, mode=mode, fixed_output=self.fixed_output)
        
        total_loss = 0.0
        for batch in tqdm(dataloader, desc="Loss Teste"):
            if self.is_dencoder:
                maze, r_in, r_out = [b.to(self.device) for b in batch]
                logits = self.model(maze, r_in)
                loss = criterion(logits.flatten(0, 1), r_out.flatten())
            else:
                inputs, targets = [b.to(self.device) for b in batch]
                logits = self.model(inputs)
                loss = criterion(logits.flatten(0, 1), targets.flatten())
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"\n[2/2] Avaliando métricas... Loss: {avg_loss:.4f}")
        
        stats = {
            "Total": len(self.data), "Loss": f"{avg_loss:.4f}", 
            "Exata": 0, "Parede": 0, "Desconexo": 0, 
            "Direção": 0, "Start_S": 0, "End_E": 0, "Muito Curto": 0
        }
        
        for sample in tqdm(self.data, desc="Solucionando"):
            parts = sample.split('&\n')
            if len(parts) != 2: continue
            unsolved, expected = parts[0] + '\n', parts[1]
            pred = self.solve_maze(unsolved)
            
            clean_pred = pred.strip()
            clean_exp = expected.strip()

            if clean_pred == clean_exp:
                stats["Exata"] += 1
            else:
                if 'E' not in clean_pred and len(clean_pred) < len(clean_exp):
                    stats["Muito Curto"] += 1

                p_lines = clean_pred.split('\n')
                if any(p in 'RLUD' and u == '#' for p, u in zip(pred, unsolved)): 
                    stats["Parede"] += 1
                
                idx_s = unsolved.find('S')
                idx_e = unsolved.find('E')
                if idx_s != -1 and idx_s < len(pred):
                    if pred[idx_s] not in 'RLUD': stats["Start_S"] += 1
                if idx_e != -1 and idx_e < len(pred):
                    if pred[idx_e] != 'E': stats["End_E"] += 1

                dir_inv, desconexo = False, False
                for r, row in enumerate(p_lines):
                    for c, char in enumerate(row):
                        if char in 'RLUD':
                            nr, nc = r, c
                            if char == 'D': nr += 1
                            elif char == 'U': nr -= 1
                            elif char == 'R': nc += 1
                            elif char == 'L': nc -= 1
                            try:
                                if nr < len(p_lines) and nc < len(p_lines[nr]):
                                    if p_lines[nr][nc] not in 'RLUDE': dir_inv = desconexo = True
                                else: dir_inv = desconexo = True
                            except IndexError: dir_inv = desconexo = True
                if dir_inv: stats["Direção"] += 1
                if desconexo: stats["Desconexo"] += 1
        
        table = "\n".join([f"{k:<15}: {v}" for k, v in stats.items()])
        print("\n" + table)
        
        out_dir = path(f'metrics/{self.model.name}')
        os.makedirs(out_dir, exist_ok=True)
        with open(f'{out_dir}/{self.model.iname}.txt', 'w', encoding='utf-8') as f: f.write(table)


def run_evaluation(model, tokenizer, source, device, fixed_output=False):
    with open(source, 'r', encoding='utf-8') as f: 
        data = [m for m in f.read().split("\n\n") if m.strip()]
    
    actions = {
        "Sample": ActionSample(model, tokenizer, data, source, device, fixed_output),
        "Metrics": ActionMetrics(model, tokenizer, data, source, device, fixed_output)
    }

    while True:
        action_key = prompt_options("Ação", list(actions.keys()) + ["Quit"])
        if action_key == "Quit": break
        if action_key is not None and action_key in actions:
            actions[action_key].run()