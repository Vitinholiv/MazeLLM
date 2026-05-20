import random
import torch
import torch.nn as nn
from tqdm import tqdm
from src.architecture.model import MazeGPTModel
from src.data.preprocess import MazeTokenizer, build_dataloader

@torch.no_grad()
def solve_maze(model: MazeGPTModel, tokenizer: MazeTokenizer,
               unsolved_maze: str, device: torch.device,
               temperature: float = 0.0) -> str:
    """
    temperature = 0.0  →  greedy / deterministic
    temperature > 0.0  →  sampled
    """
    model.eval()
    ctx_size = model.trf_blocks[0].att.mask.shape[0]

    clean_unsolved = unsolved_maze.strip() + '\n'
    prompt = clean_unsolved + '&\n'
    
    ids    = tokenizer.encode(prompt)
    idx    = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    tokens_to_generate = len(tokenizer.encode(clean_unsolved)) - 1

    generated = []
    for _ in range(tokens_to_generate):
        logits  = model(idx[:, -ctx_size:])[:, -1, :]

        if temperature < 1e-8:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs   = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.pad_id:
            break

        idx = torch.cat([idx, next_id], dim=1)
        generated.append(next_id.item())
        
    return tokenizer.decode(generated)

@torch.no_grad()
def evaluate(model: MazeGPTModel, tokenizer: MazeTokenizer,
             source: str, device: torch.device,
             temperature: float = 0.0):
    
    model.eval()
    with open(source, 'r', encoding='utf-8') as f:
        data = [m for m in f.read().split("\n\n") if m.strip()]

    response = {
        'Sample': ['SA','SAMPLE','SAMP','SAMPLING'],
        'Metrics': ['M','METRICS','STATISTICS','STATS','STAT','ST'],
        'Quit': ['Q','QUIT']
    }
    
    while True:
        action = input("Opção (Sample|Metrics|Quit): ").strip().upper()
    
        if action in response['Sample']:
            sample_raw = random.choice(data)
            parts = sample_raw.split('&\n')
            if len(parts) != 2:
                print("Erro de formato na amostra selecionada no arquivo.")
                return
                
            unsolved, expected = parts[0] + '\n', parts[1]
            pred = solve_maze(model, tokenizer, unsolved, device, temperature)
            
            print("Teste de Amostra:".center(40))
            print("[ENTRADA]")
            print(unsolved.strip())
            print("\n[PREDITO]")
            print(pred.strip())
            print("\n[ESPERADO]")
            print(expected.strip(),'\n')

        elif action in response['Metrics']:
            print("\n[1/2] Calculando Loss do conjunto de teste...")
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            dataloader = build_dataloader(source, max_length=model.context_len, batch_size=8, shuffle=False)
            
            total_loss = 0.0
            for inputs, targets in tqdm(dataloader, desc="Loss Teste"):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits.flatten(0, 1), targets.flatten())
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            
            print("\n[2/2] Avaliando métricas espaciais...")
            stats = {
                "Total Avaliados": len(data),
                "Loss Média": f"{avg_loss:.4f}",
                "Solução Exata": 0,
                "Violação de Parede (#)": 0,
                "Caminho Desconexo": 0,
                "Direção Inválida": 0,
                "Não Começou do Início (S)": 0,
                "Não Chegou no Final (E)": 0
            }
            
            for sample in tqdm(data, desc="Gerando Soluções"):
                parts = sample.split('&\n')
                if len(parts) != 2: continue
                
                unsolved, expected = parts[0] + '\n', parts[1]
                pred = solve_maze(model, tokenizer, unsolved, device, temperature)
                
                if pred.strip() == expected.strip():
                    stats["Solução Exata"] += 1
                else:
                    p_lines = pred.strip().split('\n')
                    parede = False
                    for p_char, u_char in zip(pred, unsolved):
                        if p_char in 'RLUD' and u_char == '#':
                            parede = True
                            break
                    if parede: stats["Violação de Parede (#)"] += 1
                
                    idx_s = unsolved.find('S')
                    idx_e = unsolved.find('E')
                    
                    if idx_s != -1 and idx_s < len(pred):
                        if pred[idx_s] not in 'RLUD':
                            stats["Não Começou do Início (S)"] += 1
                            
                    if idx_e != -1 and idx_e < len(pred):
                        if pred[idx_e] != 'E':
                            stats["Não Chegou no Final (E)"] += 1
                    
                    dir_invalida = False
                    desconexo = False
                    
                    for r, row in enumerate(p_lines):
                        for c, char in enumerate(row):
                            if char in 'RLUD':
                                nr, nc = r, c
                                if char == 'R': nc += 1
                                elif char == 'L': nc -= 1
                                elif char == 'U': nr -= 1
                                elif char == 'D': nr += 1
                                
                                try:
                                    next_char = p_lines[nr][nc]
                                    if next_char not in 'RLUDE':
                                        dir_invalida = True
                                        desconexo = True
                                except IndexError:
                                    dir_invalida = True
                                    desconexo = True
                                    
                    if dir_invalida: stats["Direção Inválida"] += 1
                    if desconexo: stats["Caminho Desconexo"] += 1

            print("\n")
            print("+" + "-"*48 + "+")
            print(f"| {'Relatório de Avaliação':^46} |")
            print("+" + "-"*35 + "+" + "-"*12 + "+")
            print(f"| {'Métrica':<33} | {'Valor':<10} |")
            print("+" + "="*35 + "+" + "="*12 + "+")
            for k, v in stats.items():
                print(f"| {k:<33} | {str(v):<10} |")
            print("+" + "-"*35 + "+" + "-"*12 + "+\n")