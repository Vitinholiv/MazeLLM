import os
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch import autocast
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from src.architecture.model import MazeGPTModel
from src.data.preprocess import MazeTokenizer, build_dataloader
from src.general.helpers import prompt_options, path



@torch.no_grad()
def solve_maze(model: MazeGPTModel, tokenizer: MazeTokenizer,
               unsolved_maze: str, device: torch.device) -> str:
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
        next_id = logits.argmax(dim=-1, keepdim=True)
      
        if next_id.item() == tokenizer.pad_id:
            break

        idx = torch.cat([idx, next_id], dim=1)
        generated.append(next_id.item())
        
    return tokenizer.decode(generated)



def action_sample(model, data, tokenizer, device):
    sample_raw = random.choice(data)
    parts = sample_raw.split('&\n')
    while len(parts) != 2:
        parts = sample_raw.split('&\n')

    unsolved, expected = parts[0] + '\n', parts[1]
    pred = solve_maze(model, tokenizer, unsolved, device)
    
    print("\nTeste de Amostra:\n")
    print("[ENTRADA]")
    print(unsolved.strip())
    print("\n[PREDITO]")
    print(pred.strip())
    print("\n[ESPERADO]")
    print(expected.strip(),'\n')



def action_metrics(model, data, source, tokenizer, device):
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
        "Total de Avaliações": len(data),
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
        pred = solve_maze(model, tokenizer, unsolved, device)
        
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

    table_str = "\n"
    table_str += "+" + "-"*48 + "+\n"
    table_str += f"| {'Relatório de Avaliação':^46} |\n"
    table_str += "+" + "-"*35 + "+" + "-"*12 + "+\n"
    table_str += f"| {'Métrica':<33} | {'Valor':<10} |\n"
    table_str += "+" + "="*35 + "+" + "="*12 + "+\n"
    for k, v in stats.items():
        table_str += f"| {k:<33} | {str(v):<10} |\n"
    table_str += "+" + "-"*35 + "+" + "-"*12 + "+\n"

    print(table_str)

    dataset_name = os.path.splitext(os.path.basename(source))[0]
    output_filename = f"{model.iname}_{dataset_name}.txt"
    output_dir = path(f'metrics/{model.name}')
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = path(f'{output_dir}/{output_filename}')
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(table_str)
        
    print(f"Resultados salvos em: {output_path}")



def render_attention(attention_history, lab, num_heads, generated_tokens, tokenizer, maze_context_size, token_to_rc):
    n = len(lab)
    m = len(lab[0]) if n > 0 else 0
    k = len(attention_history)

    n = len(lab)
    m = len(lab[0]) if n > 0 else 0
    k = len(attention_history)

    tokens_cols = m + 1 if m > 0 else 20
    tokens_rows = max(1, (k + tokens_cols - 1) // tokens_cols)

    fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.25)

    static_matrix = np.zeros((n, m))
    start_r, start_c = 0, 0
    for i in range(n):
        for j in range(m):
            static_matrix[i, j] = 0 if lab[i][j] == '#' else 1
            if lab[i][j] == 'S':
                start_r, start_c = i, j

    ax_1.imshow(static_matrix, cmap='gray', vmin=0, vmax=1)
    ax_1.set_title("Labyrinth & Path")
    ax_1.axis('off')

    path_data = []
    curr_r, curr_c = start_r, start_c
    path_data = []
    for idx, t_id in enumerate(generated_tokens):
        char = tokenizer.id_to_char.get(t_id, '?')
        r = idx // tokens_cols
        c = idx % tokens_cols
        path_data.append((r, c, char))

    text_objs = []
    for i in range(n):
        row_objs = []
        for j in range(m):
            char = lab[i][j] if lab[i][j] in ['S', 'E'] else ''
            t = ax_1.text(j, i, char, ha='center', va='center',
                          color='red', fontweight='bold', fontsize=12)
            row_objs.append(t)
        text_objs.append(row_objs)

    def update_labyrinth_path(step_idx):
        for i in range(n):
            for j in range(m):
                text_objs[i][j].set_text('')
                if lab[i][j] in ['S', 'E']:
                    text_objs[i][j].set_text(lab[i][j])
                    text_objs[i][j].set_color('red')
        for i in range(step_idx):
            if i < len(path_data):
                pr, pc, pchar = path_data[i]
                if 0 <= pr < n and 0 <= pc < m:
                    text_objs[pr][pc].set_text(pchar)
                    text_objs[pr][pc].set_color('blue')

    matrix_maze = np.zeros((n, m))
    img_maze = ax_2.imshow(matrix_maze, cmap='Greens')
    ax_2.set_title("Step: 1 | Head: Combined")
    ax_2.axis('off')

    for i in range(n):
        for j in range(m):
            ax_2.text(j, i, lab[i][j], ha='center', va='center',color='black', fontsize=10, fontweight='bold')

    matrix_tokens = np.zeros((tokens_rows, tokens_cols))
    img_tokens = ax_3.imshow(matrix_tokens, cmap='Greens')
    ax_3.set_title("Attention on Generated Path")
    ax_3.axis('off')

    text_tokens = []
    for i in range(tokens_rows):
        row_t = []
        for j in range(tokens_cols):
            t = ax_3.text(j, i, '', ha='center', va='center',
                          color='black', fontsize=10, fontweight='bold')
            row_t.append(t)
        text_tokens.append(row_t)

    def get_attention_grids(step_idx, head_idx):
        data = attention_history[step_idx]

        if head_idx >= num_heads:
            attn_slice = data.mean(axis=0)
        else:
            attn_slice = data[head_idx]
        grid_maze = np.zeros((n, m))
        for t_idx, (r_maze, c_maze) in token_to_rc.items():
            if t_idx < len(attn_slice):
                grid_maze[r_maze, c_maze] = attn_slice[t_idx]
    
        path_attn = attn_slice[maze_context_size:] if len(attn_slice) > maze_context_size else np.array([])

        grid_tokens = np.zeros((tokens_rows, tokens_cols))

        for i in range(tokens_rows):
            for j in range(tokens_cols):
                text_tokens[i][j].set_text('')

        for idx in range(step_idx):
            r_tok = idx // tokens_cols
            c_tok = idx % tokens_cols
            if r_tok >= tokens_rows:
                break
            if idx < len(path_attn):
                grid_tokens[r_tok, c_tok] = path_attn[idx]
            if idx < len(generated_tokens):
                char = tokenizer.id_to_char.get(generated_tokens[idx], '?')
                text_tokens[r_tok][c_tok].set_text(char)

        return grid_maze, grid_tokens

    ax_head = plt.axes((0.15, 0.1, 0.7, 0.03))
    slider_head = Slider(ax_head, 'Head', 1, num_heads + 1,
                         valinit=num_heads + 1, valstep=1)
    slider_head.valtext.set_text("Combined")

    ax_step = plt.axes((0.15, 0.05, 0.7, 0.03))
    slider_step = Slider(ax_step, 'Step', 1, k, valinit=1, valstep=1)

    def update(val):
        s_idx = int(slider_step.val) - 1
        h_val = int(slider_head.val)
        h_idx = h_val - 1

        head_label = "Combined" if h_val == num_heads + 1 else str(h_val)
        slider_head.valtext.set_text(head_label)
        ax_2.set_title(f"Step: {int(slider_step.val)} | Head: {head_label}")

        update_labyrinth_path(s_idx)

        g_maze, g_tokens = get_attention_grids(s_idx, h_idx)
        img_maze.set_data(g_maze)
        img_tokens.set_data(g_tokens)

        max_attention = max(g_maze.max(), g_tokens.max())
        clim_max = max_attention if max_attention > 1e-9 else 1.0

        img_maze.set_clim(vmin=0, vmax=clim_max)
        img_tokens.set_clim(vmin=0, vmax=clim_max)

        fig.canvas.draw_idle()

    slider_step.on_changed(update)
    slider_head.on_changed(update)

    update(None)
    plt.show()


@torch.no_grad()
def action_visualize_attention(model, data, source, tokenizer, device, tokens=420):
    sample_raw = random.choice(data)
    parts = sample_raw.split('&\n')
    while len(parts) != 2:
        sample_raw = random.choice(data)
        parts = sample_raw.split('&\n')

    unsolved = parts[0] + '\n'
    clean_unsolved = unsolved.strip() + '\n'
    prompt = clean_unsolved + '&\n'
    lab = [line for line in clean_unsolved.split('\n') if line]

    encoded_entry = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded_entry, dtype=torch.long, device=device).unsqueeze(0)
    maze_context_size = len(encoded_entry)

    token_to_rc = {}
    r, c = 0, 0
    n, m = len(lab), len(lab[0]) if lab else 0
    for t_idx, t_id in enumerate(encoded_entry):
        char = tokenizer.id_to_char.get(t_id, '')
        if char == '\n':
            r += 1
            c = 0
        elif char in ['#', ' ', 'S', 'E']:
            if r < n and c < m:
                token_to_rc[t_idx] = (r, c)
            c += 1

    attention_history = []
    generated_tokens = []
    model.eval()
    num_heads = model.trf_blocks[0].att.num_heads
    ctx_size = model.trf_blocks[0].att.mask.shape[0]

    for step in range(tokens):
        cond_input = input_ids[:, -ctx_size:]
        current_len = input_ids.size(1)

        logits = model(cond_input)[:, -1, :]
        next_token_id = logits.argmax(dim=-1).item()

        _, attn_matrix = model.get_attention(cond_input, layer_idx=-1)
        step_attention = attn_matrix[0, :, -1, :].cpu().to(torch.float32).numpy()

        if current_len > ctx_size:
            dropped = current_len - ctx_size
            padding = np.zeros((num_heads, dropped), dtype=np.float32)
            step_attention = np.concatenate([padding, step_attention], axis=1)

        attention_history.append(step_attention)
        generated_tokens.append(next_token_id)

        if next_token_id == tokenizer.pad_id:
            break

        next_token_tensor = torch.tensor([[next_token_id]], device=device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

    render_attention(
        attention_history, lab, num_heads,
        generated_tokens, tokenizer,
        maze_context_size, token_to_rc,
    )



@torch.no_grad()
def evaluate(model: MazeGPTModel, tokenizer: MazeTokenizer,
             source: str, device: torch.device,
             temperature: float = 0.0):
    
    model.eval()
    with open(source, 'r', encoding='utf-8') as f:
        data = [m for m in f.read().split("\n\n") if m.strip()]

    while True:
        action = prompt_options("Ação", ["Sample", "Metrics", "Attention", "Quit"])
    
        if action == "Sample":
            action_sample(model,data,tokenizer,device)
            
        elif action == "Metrics":
            action_metrics(model,data,source,tokenizer,device)

        elif action == "Attention":
            action_visualize_attention(model,data,source,tokenizer,device)
            
        elif action == "Quit":
            return