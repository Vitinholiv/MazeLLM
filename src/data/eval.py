import os
import torch
import pygame
import itertools
from typing import Union
from collections import deque
from src.general.configs import init, ModelConfigs
from src.data.preprocess import build_dataloader

# Auxiliary Functions

def parse_dim(val: Union[int, float, str], screen_w: int, screen_h: int) -> int:
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        val = val.strip().lower()
        try:
            if val.endswith('vw'):
                return int(float(val[:-2]) * screen_w / 100)
            elif val.endswith('vh'):
                return int(float(val[:-2]) * screen_h / 100)
        except ValueError:
            pass
    return 0

DIRS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
def normalize_grid(matrix, lab_size, fill='#'):
    grid = []
    for r in range(lab_size):
        row = list(matrix[r][:lab_size]) if r < len(matrix) else []
        row = row + [fill] * (lab_size - len(row))
        grid.append(row)
    return grid

def in_bounds(pos, lab_size):
    r, c = pos
    return 0 <= r < lab_size and 0 <= c < lab_size

def find_all_positions(grid, ch, lab_size):
    return [(r, c) for r in range(lab_size) for c in range(lab_size) if grid[r][c] == ch]

def check_violation(pred_grid, input_grid, ch, lab_size):
    pred_positions = find_all_positions(pred_grid, ch, lab_size)
    input_positions = find_all_positions(input_grid, ch, lab_size)
    if len(pred_positions) != 1 or not input_positions:
        return True, (pred_positions[0] if len(pred_positions) == 1 else None)
    violated = pred_positions[0] != input_positions[0]
    return violated, pred_positions[0]

def unique_directional_neighbor(grid, pos, lab_size):
    cands = []
    for dr, dc in DIRS.values():
        npos = (pos[0] + dr, pos[1] + dc)
        if in_bounds(npos, lab_size) and grid[npos[0]][npos[1]] in DIRS:
            cands.append(npos)
    return cands[0] if len(cands) == 1 else None

def find_predecessor(grid, input_grid, pos, lab_size):
    cands = []
    for ch, (dr, dc) in DIRS.items():
        npos = (pos[0] - dr, pos[1] - dc)
        if in_bounds(npos, lab_size) and grid[npos[0]][npos[1]] == ch and input_grid[npos[0]][npos[1]] != '#':
            cands.append(npos)
    return cands[0] if len(cands) == 1 else None

def walk_chain_forward(grid, start_pos, lab_size, max_steps):
    visited, seen, cur = [], set(), start_pos
    for _ in range(max_steps):
        if not in_bounds(cur, lab_size) or cur in seen:
            return visited, None
        seen.add(cur)
        ch = grid[cur[0]][cur[1]]
        if ch not in DIRS:
            return visited, cur
        visited.append(cur)
        dr, dc = DIRS[ch]
        cur = (cur[0] + dr, cur[1] + dc)
    return visited, None

def walk_chain_forward_blocked(grid, input_grid, start_pos, lab_size, max_steps):
    cur = start_pos
    for _ in range(max_steps):
        if not in_bounds(cur, lab_size):
            return cur
        ch = grid[cur[0]][cur[1]]
        if ch not in DIRS:
            return cur
        dr, dc = DIRS[ch]
        nxt = (cur[0] + dr, cur[1] + dc)
        if not in_bounds(nxt, lab_size) or input_grid[nxt[0]][nxt[1]] == '#':
            return cur
        cur = nxt
    return cur

def trace_predecessors(grid, input_grid, start_pos, lab_size, max_steps):
    positions, cur = [], start_pos
    for _ in range(max_steps):
        pred = find_predecessor(grid, input_grid, cur, lab_size)
        if pred is None:
            break
        positions.append(pred)
        cur = pred
    return positions

def is_neighbor(a, b):
    if a is None or b is None:
        return False
    return (abs(a[0] - b[0]) == 1 and a[1] == b[1]) or (abs(a[1] - b[1]) == 1 and a[0] == b[0])

def bfs_distance(input_grid, start, end, lab_size):
    if start is None or end is None:
        return -1
    if start == end:
        return 0
    visited = {start}
    q = deque([(start, 0)])
    while q:
        pos, d = q.popleft()
        for dr, dc in DIRS.values():
            npos = (pos[0] + dr, pos[1] + dc)
            if in_bounds(npos, lab_size) and npos not in visited and input_grid[npos[0]][npos[1]] != '#':
                if npos == end:
                    return d + 1
                visited.add(npos)
                q.append((npos, d + 1))
    return -1

def calculate_metrics(input_matrix, solv_matrix, pred_matrix, lab_size):
    input_grid = normalize_grid(input_matrix, lab_size)
    solv_grid = normalize_grid(solv_matrix, lab_size)
    pred_grid = normalize_grid(pred_matrix, lab_size)
    max_steps = lab_size * lab_size + 5
    results = {}

    s_input_pos = next(iter(find_all_positions(input_grid, 'S', lab_size)), None)

    # Início / Final Violado
    start_violated, s_pred_pos = check_violation(pred_grid, input_grid, 'S', lab_size)
    end_violated, e_pred_pos = check_violation(pred_grid, input_grid, 'E', lab_size)
    results['Início Violado'] = "Sim" if start_violated else "Não"
    results['Final Violado'] = "Sim" if end_violated else "Não"

    # Sequência do gabarito (forward)
    gt_first = unique_directional_neighbor(solv_grid, s_input_pos, lab_size) if s_input_pos else None
    gt_visited, gt_landing = walk_chain_forward(solv_grid, gt_first, lab_size, max_steps) if gt_first else ([], None)
    gt_dirs = [solv_grid[p[0]][p[1]] for p in gt_visited]

    # Sequência prevista (forward, completa)
    pred_first = None if start_violated else unique_directional_neighbor(pred_grid, s_pred_pos, lab_size)
    pred_visited, pred_landing = walk_chain_forward(pred_grid, pred_first, lab_size, max_steps) if pred_first else ([], None)
    pred_dirs = [pred_grid[p[0]][p[1]] for p in pred_visited]

    # Corretude / Solução
    all_free = all(input_grid[p[0]][p[1]] != '#' for p in pred_visited)
    is_correct = (not start_violated and not end_violated and pred_first is not None
                  and pred_landing == e_pred_pos and all_free)
    if is_correct:
        is_otima = (pred_dirs == gt_dirs) or (len(pred_dirs) == len(gt_dirs))
        results['Solução'] = "Ótima" if is_otima else "Correta"
    else:
        results['Solução'] = "Incorreta"

    # Tokens Alterados / Ótimos / Diferentes
    def hamming(a, b):
        return sum(1 for r in range(lab_size) for c in range(lab_size) if a[r][c] != b[r][c])
    results['Tokens Alterados'] = hamming(pred_grid, input_grid)
    results['Tokens Ótimos'] = hamming(solv_grid, input_grid)
    results['Tokens Diferentes'] = hamming(solv_grid, pred_grid)

    # Paredes / Espaços Violados
    paredes_violadas = espacos_violados = 0
    for r in range(lab_size):
        for c in range(lab_size):
            was_wall, is_wall = input_grid[r][c] == '#', pred_grid[r][c] == '#'
            if was_wall and not is_wall:
                paredes_violadas += 1
            elif not was_wall and is_wall:
                espacos_violados += 1
    results['Paredes Violadas'] = paredes_violadas
    results['Espaços Violados'] = espacos_violados

    # Progresso Direto
    if pred_first is None:
        progresso_direto = 0
    else:
        k = 0
        while k < len(gt_dirs) and k < len(pred_dirs) and pred_dirs[k] == gt_dirs[k]:
            k += 1
        bonus = (k == len(gt_dirs) == len(pred_dirs)) and (pred_landing == e_pred_pos) and not end_violated
        progresso_direto = k + (1 if bonus else 0)
    results['Progresso Direto'] = progresso_direto

    # Progresso Inverso
    gt_rev_dirs = list(reversed(gt_dirs))
    if end_violated:
        progresso_inverso = 0
        inv_positions = []
    else:
        inv_positions = trace_predecessors(pred_grid, input_grid, e_pred_pos, lab_size, max_steps)
        inv_dirs = [pred_grid[p[0]][p[1]] for p in inv_positions]
        k = 0
        while k < len(gt_rev_dirs) and k < len(inv_dirs) and inv_dirs[k] == gt_rev_dirs[k]:
            k += 1
        inv_landing = inv_positions[-1] if inv_positions else e_pred_pos
        bonus = (k == len(gt_rev_dirs) == len(inv_dirs)) and is_neighbor(inv_landing, s_pred_pos) and not start_violated
        progresso_inverso = k + (1 if bonus else 0)
    results['Progresso Inverso'] = progresso_inverso

    # Distância Direta / Inversa
    fallback_dist = lab_size ** 2
    if s_pred_pos is None or e_pred_pos is None:
        results['Distância Direta'] = fallback_dist
        results['Distância Inversa'] = fallback_dist
    else:
        dist_first = unique_directional_neighbor(pred_grid, s_pred_pos, lab_size)
        if dist_first is None:
            dd = bfs_distance(input_grid, s_pred_pos, e_pred_pos, lab_size)
        else:
            stop_pos = walk_chain_forward_blocked(pred_grid, input_grid, dist_first, lab_size, max_steps)
            dd = 0 if stop_pos == e_pred_pos else bfs_distance(input_grid, stop_pos, e_pred_pos, lab_size)
        results['Distância Direta'] = dd if dd != -1 else fallback_dist

        dist_inv_positions = trace_predecessors(pred_grid, input_grid, e_pred_pos, lab_size, max_steps)
        if not dist_inv_positions:
            di = bfs_distance(input_grid, s_pred_pos, e_pred_pos, lab_size)
        else:
            stop_pos = dist_inv_positions[-1]
            reached_start = (stop_pos == s_pred_pos) or (dist_first is not None and stop_pos == dist_first)
            di = 0 if reached_start else bfs_distance(input_grid, stop_pos, s_pred_pos, lab_size)
        results['Distância Inversa'] = di if di != -1 else fallback_dist

    # Caminho Único / Conexo
    forward_set = set(pred_visited)
    backward_set = set(inv_positions) if not end_violated else set()
    all_dir_positions = set()
    for ch in DIRS:
        all_dir_positions.update(find_all_positions(pred_grid, ch, lab_size))

    results['Caminho Único'] = "Sim" if (forward_set | backward_set) == all_dir_positions else "Não"
    results['Caminho Conexo'] = "Sim" if (forward_set == all_dir_positions or backward_set == all_dir_positions) else "Não"

    score = {
        'Solução': 1.0 if results['Solução'] == 'Ótima' else 0.8 if results['Solução'] == 'Correta' else 0.0,
        'Tokens Diferentes': 1.0 - min(abs(results['Tokens Diferentes'] / results["Tokens Ótimos"]),1.0),
        'Progresso Direto': results['Progresso Direto']/(results['Tokens Ótimos']+1),
        'Progresso Inverso': results['Progresso Inverso']/(results['Tokens Ótimos']+1),
        'Distância Direta': (1/(results['Distância Direta']+1))**0.5,
        'Distância Inversa': (1/(results['Distância Inversa']+1))**0.5,
        'Início Violado': 1.0 if results['Início Violado'] == 'Não' else 0.0,
        'Final Violado': 1.0 if results['Final Violado'] == 'Não' else 0.0,
        'Paredes Violadas': (1/(results['Paredes Violadas']+1))**0.5,
        'Espaços Violados': (1/(results['Espaços Violados']+1))**0.5,
        'Caminho Único': 0.0 if results['Caminho Único'] == 'Não' else 1.0,
        'Caminho Conexo': 0.0 if results['Caminho Conexo'] == 'Não' else 1.0
    }

    results['Corretude'] = (
        1.0 * score['Solução'] +
        0.6 * score['Tokens Diferentes'] +
        0.3 * score['Progresso Direto'] +
        0.2 * score['Progresso Inverso'] +
        1.6 * score['Distância Direta'] +
        0.3 * score['Distância Inversa'] +
        1.2 * score['Início Violado'] +
        1.2 * score['Final Violado'] +
        2.5 * score['Paredes Violadas'] +
        1.5 * score['Espaços Violados'] +
        0.8 * score['Caminho Único'] +
        0.3 * score['Caminho Conexo']
    ) / 11.5
    results['Corretude'] = max(0,min(int(results['Corretude']*1000)/1000.0,1))

    return {k: str(v) for k, v in results.items()}

def decoded_to_matrix(decoded):
    dec = decoded.split('\n')
    res = []
    for line in range(1,len(dec)-1):
        mline = []
        for i in range(0,len(dec[line]),2):
            mline.append(dec[line][i])
        res.append(mline)
    return res

def load_next_labyrinth(blocks, tokenizer, directions_task, model, device, conf, display_data):
    idx = display_data['screen_sample']['labyrinth_index']
    display_data['screen_sample']['labyrinth_index'] += 1
    if(idx + 1 == len(blocks)):
        display_data['screen_sample']['labyrinth_index'] = 0
    sample = blocks[idx]

    split_idx = sample.index('<SOLUTION_START>')
    input_str = sample[:split_idx].strip()
    solv_str = sample[split_idx:].strip()

    input_ids = tokenizer.encode(input_str)
    start_id = tokenizer.char_to_id['<SOLUTION_START>']
    end_id = tokenizer.char_to_id['<SOLUTION_END>']
    tokenizer_type = model.tokenizer_type

    if directions_task:
        if tokenizer_type == 'individual':
            yield {'matrices': [], 'metrics': {}}
            return
        elif tokenizer_type == 'wall_encoded':
            yield {'matrices': [], 'metrics': {}}
            return
        elif tokenizer_type == 'free_edges':
            yield {'matrices': [], 'metrics': {}}
            return
    else:
        if tokenizer_type == 'individual':
            steps = conf['lab_size']**2 + 2
            with torch.no_grad():
                if conf["dataset_mode"] == "decoder":
                    seq = torch.tensor(input_ids + [start_id], device=device).unsqueeze(0)
                    for i in range(steps):
                        logits = model(seq)
                        next_token = logits[:, -1:, :].argmax(dim=-1)
                        seq = torch.cat([seq, next_token], dim=1)
                        yield i / steps
                    full_ids = seq[0].tolist()

                elif conf["dataset_mode"] == "dencoder":
                    m_in = torch.tensor(input_ids).unsqueeze(0).to(device)
                    r_in = torch.tensor([start_id]).unsqueeze(0).to(device)
                    for _ in range(steps):
                        logits = model(m_in, r_in)
                        next_token = logits[0, -1, :].argmax().item()
                        
                        r_in = torch.cat([r_in, torch.tensor([[next_token]]).to(device)], dim=1)
                        if next_token == end_id:
                            break
                    full_ids = input_ids + r_in[0].tolist()

                else:
                    raise NotImplemented('EncoderNotImplemented')
                
            full_str = tokenizer.decode(full_ids)
            pred_str = '<SOLUTION_START>' + full_str.split('<SOLUTION_START>')[1]

            input_matrix = decoded_to_matrix(input_str)
            solv_matrix = decoded_to_matrix(solv_str)
            pred_matrix = decoded_to_matrix(pred_str)
            metrics = calculate_metrics(input_matrix, solv_matrix, pred_matrix, conf['lab_size'])

            yield {'matrices': [input_matrix, solv_matrix, pred_matrix], 'metrics': metrics}
            return
        elif tokenizer_type == 'wall_encoded':
            yield {'matrices': [], 'metrics': {}}
            return
        elif tokenizer_type == 'free_edges':
            yield {'matrices': [], 'metrics': {}}
            return

# General Classes

class UILoadingBar:
    def __init__(self, x, y, width, height, text: str = "Carregando...", text_color: str = "#FFFFFF", bg_color: str = "#223344", fill_color: str = "#2A93CB"):
        self.raw_x = x
        self.raw_y = y
        self.raw_w = width
        self.raw_h = height
        self.text = text
        self.text_color = pygame.Color(text_color)
        self.bg_color = pygame.Color(bg_color)
        self.fill_color = pygame.Color(fill_color)
        self.progress = 0.0

    def set_progress(self, value: float):
        self.progress = max(0.0, min(1.0, value))

    def get_rect(self, screen_w: int, screen_h: int) -> pygame.Rect:
        x = parse_dim(self.raw_x, screen_w, screen_h)
        y = parse_dim(self.raw_y, screen_w, screen_h)
        w = parse_dim(self.raw_w, screen_w, screen_h)
        h = parse_dim(self.raw_h, screen_w, screen_h)
        return pygame.Rect(x, y, w, h)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        rect = self.get_rect(screen.get_width(), screen.get_height())

        pygame.draw.rect(screen, self.bg_color, rect, border_radius=8)

        fill_rect = pygame.Rect(rect.x, rect.y, int(rect.width * self.progress), rect.height)
        if fill_rect.width > 0:
            pygame.draw.rect(screen, self.fill_color, fill_rect, border_radius=8)

        label = f"{self.text} {int(self.progress * 100)}%"
        text_surf = font.render(label, True, self.text_color)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)

class UIButton:
    def __init__(self, id_name: str, x, y, width, height, text: str, text_color: str, bg_color: str = "#223344", active_color: str = "#2A93CB"):
        self.id = id_name
        self.raw_x = x
        self.raw_y = y
        self.raw_w = width
        self.raw_h = height
        self.base_color = pygame.Color(bg_color)
        self.active_color = pygame.Color(active_color)
        self.is_selected = (self.id == "btn_screen_sample")
        self.text = text
        self.text_color = pygame.Color(text_color)
        
    def get_rect(self, screen_w: int, screen_h: int) -> pygame.Rect:
        x = parse_dim(self.raw_x, screen_w, screen_h)
        y = parse_dim(self.raw_y, screen_w, screen_h)
        w = parse_dim(self.raw_w, screen_w, screen_h)
        h = parse_dim(self.raw_h, screen_w, screen_h)
        return pygame.Rect(x, y, w, h)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, mouse_pos: tuple):
        rect = self.get_rect(screen.get_width(), screen.get_height())
        display_color = self.active_color if self.is_selected else self.base_color
        current_color = display_color
        if rect.collidepoint(mouse_pos):
            current_color = display_color.lerp(pygame.Color("white"), 0.2)
            
        pygame.draw.rect(screen, current_color, rect, border_radius=8)
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)

    def check_click(self, event_info: dict) -> bool:
        if event_info.get("mouse_clicked", False):
            rect = self.get_rect(event_info["screen_w"], event_info["screen_h"])
            return rect.collidepoint(event_info["mouse_pos"])
        return False
    
class ScreenMatrix:
    def __init__(self, x, y, width, height, k=10):
        self.raw_x, self.raw_y = x, y
        self.raw_w, self.raw_h = width, height
        self.k = k
        self.grid = [[{"char": "#", "score": 0.0} for _ in range(k)] for _ in range(k)]
        self.cell_padding = 2

    def update_from_labyrinth(self, maze_data: list):
        for r in range(self.k):
            for c in range(self.k):
                token = maze_data[r][c] if r < len(maze_data) and c < len(maze_data[0]) else "#"
                self.grid[r][c]["char"] = token

    def draw(self, screen, font, mouse_pos):
        rect = pygame.Rect(parse_dim(self.raw_x, screen.get_width(), screen.get_height()),
                           parse_dim(self.raw_y, screen.get_width(), screen.get_height()),
                           parse_dim(self.raw_w, screen.get_width(), screen.get_height()),
                           parse_dim(self.raw_h, screen.get_width(), screen.get_height()))
        
        cell_w = rect.width // self.k
        cell_h = rect.height // self.k

        for r in range(self.k):
            for c in range(self.k):
                cell = self.grid[r][c]
                token = cell["char"]
                x, y = rect.x + c * cell_w, rect.y + r * cell_h
                bg = pygame.Color("#223344")
                fg = pygame.Color("#FFFFFF")
                wall_padding = 2*self.cell_padding
                char = ""; draw_walls = None

                if token == "#": pass
                elif token == " ": bg = pygame.Color("#FFFFFF")
                elif token == "S": bg, char, fg = pygame.Color("#18D227"), "S", pygame.Color("#000000")
                elif token == "E": bg, char, fg = pygame.Color("#FF5151"), "E", pygame.Color("#000000")
                elif token in ["U", "D", "L", "R"]: bg, char, fg = pygame.Color("#FFE555"), token, pygame.Color("#000000")
                
                elif token.startswith("."):
                    bg = pygame.Color("#FFFFFF")
                    draw_walls = [b == '1' for b in token[1:5]]
                elif token[0] in "UDLR" and len(token) == 5:
                    bg, char, fg = pygame.Color("#FFFFFF"), token[0], pygame.Color("#FFD700")
                    draw_walls = [b == '1' for b in token[1:5]]
                
                elif token in ["<LABYRINTH_START>", "<LABYRINTH_END>", "<SOLUTION_START>", "<SOLUTION_END>"]:
                    bg = pygame.Color("#FFFFFF")
                    mapping = {"<LABYRINTH_START>": "LI", "<LABYRINTH_END>": "LE", "<SOLUTION_START>": "SI", "<SOLUTION_END>": "SE"}
                    char, fg = mapping.get(token, "??"), pygame.Color("#FFD700")

                pygame.draw.rect(screen, bg, (x + self.cell_padding, y + self.cell_padding, cell_w - wall_padding, cell_h - wall_padding))
                if draw_walls:
                    thick = wall_padding
                    if draw_walls[0]: pygame.draw.line(screen, (0,0,0), (x, y), (x+cell_w, y), thick) # Up
                    if draw_walls[1]: pygame.draw.line(screen, (0,0,0), (x, y+cell_h), (x+cell_w, y+cell_h), thick) # Down
                    if draw_walls[2]: pygame.draw.line(screen, (0,0,0), (x, y), (x, y+cell_h), thick) # Left
                    if draw_walls[3]: pygame.draw.line(screen, (0,0,0), (x+cell_w, y), (x+cell_w, y+cell_h), thick) # Right

                if cell["score"] > 0:
                    overlay = pygame.Surface((cell_w, cell_h), pygame.SRCALPHA)
                    overlay.fill((255, 255, 0, int(cell["score"] * 150)))
                    screen.blit(overlay, (x, y))
                if char:
                    text_surf = font.render(char, True, fg)
                    screen.blit(text_surf, text_surf.get_rect(center=(x + cell_w//2, y + cell_h//2)))

class UIMetricsTable:
    def __init__(self, x, y, width, height, col_weights, row_weights, data,
                 bg_color="#161616", text_color="#FFFFFF",
                 cell_bg_color="#223344", border_color=None, border_size=0,
                 cell_padding=6, corner_radius=8, row_alt_shade=0.05,
                 title_row=False, title_col=False,
                 title_bg_color=None, title_text_color=None,
                 title_font=None, title_border_color=None):
        self.raw_x = x
        self.raw_y = y
        self.raw_w = width
        self.raw_h = height
        self.col_weights = col_weights
        self.row_weights = row_weights
        self.data = data

        self.bg_color = pygame.Color(bg_color)
        self.text_color = pygame.Color(text_color)
        self.cell_bg_color = pygame.Color(cell_bg_color) if cell_bg_color else pygame.Color(bg_color)
        self.cell_padding = cell_padding
        self.corner_radius = corner_radius
        self.row_alt_shade = row_alt_shade
        self.border_size = border_size

        default_border = (self.cell_bg_color.lerp(pygame.Color("white"), 0.15)
                           if self.cell_bg_color else self.text_color)
        self.border_color = pygame.Color(border_color) if border_color else default_border

        self.title_row = title_row
        self.title_col = title_col

        default_title_bg = (self.cell_bg_color.lerp(pygame.Color("white"), 0.25)
                             if self.cell_bg_color else pygame.Color("#2A93CB"))
        self.title_bg_color = pygame.Color(title_bg_color) if title_bg_color else default_title_bg
        self.title_text_color = pygame.Color(title_text_color) if title_text_color else self.text_color
        self.title_border_color = (pygame.Color(title_border_color) if title_border_color
                                    else self.title_bg_color.lerp(pygame.Color("black"), 0.2))
        self.title_font = title_font

    def get_rect(self, screen_w, screen_h) -> pygame.Rect:
        return pygame.Rect(
            parse_dim(self.raw_x, screen_w, screen_h),
            parse_dim(self.raw_y, screen_w, screen_h),
            parse_dim(self.raw_w, screen_w, screen_h),
            parse_dim(self.raw_h, screen_w, screen_h)
        )

    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        rect = self.get_rect(screen.get_width(), screen.get_height())
        pygame.draw.rect(screen, self.bg_color, rect, border_radius=self.corner_radius + 4)

        total_col = sum(self.col_weights)
        total_row = sum(self.row_weights)
        col_widths = [rect.width * w / total_col for w in self.col_weights]
        row_heights = [rect.height * h / total_row for h in self.row_weights]

        title_font = self.title_font or font

        for i, row in enumerate(self.data):
            y = rect.y + sum(row_heights[:i])
            is_header_row = self.title_row and i == 0
            body_row_index = i - (1 if self.title_row else 0)

            for j, cell_text in enumerate(row):
                x = rect.x + sum(col_widths[:j])
                is_header_col = self.title_col and j == 0
                is_title_cell = is_header_row or is_header_col

                outer = pygame.Rect(x, y, col_widths[j], row_heights[i])
                cell_rect = outer.inflate(-self.cell_padding, -self.cell_padding)
                if cell_rect.width <= 0 or cell_rect.height <= 0:
                    continue

                if is_title_cell:
                    cell_color = self.title_bg_color
                    cell_text_color = self.title_text_color
                    cell_font = title_font
                    outline_color = self.title_border_color
                else:
                    cell_color = self.cell_bg_color
                    cell_text_color = self.text_color
                    cell_font = font
                    outline_color = self.border_color
                    if self.row_alt_shade and body_row_index % 2 == 1:
                        cell_color = cell_color.lerp(pygame.Color("white"), self.row_alt_shade)

                if cell_color:
                    pygame.draw.rect(screen, cell_color, cell_rect, border_radius=self.corner_radius)
                    if self.border_size > 0:
                        pygame.draw.rect(screen, outline_color, cell_rect,
                                          self.border_size, border_radius=self.corner_radius)

                text_surf = cell_font.render(str(cell_text), True, cell_text_color)
                text_rect = text_surf.get_rect(center=cell_rect.center)
                screen.blit(text_surf, text_rect)

# App Run Functions

def process_events(screen: pygame.Surface):
    running = True
    event_info = {
        "mouse_clicked": False,
        "mouse_pos": pygame.mouse.get_pos(),
        "screen_w": screen.get_width(),
        "screen_h": screen.get_height()
    }
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                event_info["mouse_clicked"] = True
        elif event.type == pygame.VIDEORESIZE:
            min_size = (800, 450)
            max_size = (1728, 972)
            new_w = max(min_size[0], min(event.w, max_size[0]))
            new_h = max(min_size[1], min(event.h, max_size[1]))
            if new_w != event.w or new_h != event.h:
                screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)
            event_info["screen_w"] = screen.get_width()
            event_info["screen_h"] = screen.get_height()
                
    return running, event_info

def iteration(blocks, model, tokenizer, task_name, conf, device, display_data, event_info):
    if event_info.get("mouse_clicked"):
        # Screen Buttons
        for btn in display_data.get("buttons", []):
            if btn.check_click(event_info):
                display_data['current_screen'] = btn.id.replace('btn_screen_', '')
                for b in display_data.get("buttons", []):
                    b.is_selected = (b.id == btn.id)

        # Sample Screen
        if display_data['current_screen'] == 'sample':
            btns = display_data['screen_sample'].get("buttons", [])
            
            # Load Labyrinth Generation
            if btns[0].check_click(event_info):
                display_data['screen_sample']['gen'] = load_next_labyrinth(blocks, tokenizer, task_name == 'directions', model, device, conf, display_data)
                display_data['screen_sample']['loading'] = True
                display_data['screen_sample']['progress'] = 0.0

                display_data['screen_sample']['current_lab_id'] = 0
                for i in range(1, 4): btns[i].is_selected = (i == 1)

            # Swap Labyrinth State
            if btns[1].check_click(event_info):
                display_data['screen_sample']['current_lab_id'] = 0
                for i in range(1, 4): btns[i].is_selected = (i == 1)
            
            if btns[2].check_click(event_info):
                display_data['screen_sample']['current_lab_id'] = 1
                for i in range(1, 4): btns[i].is_selected = (i == 2)
            
            if btns[3].check_click(event_info):
                display_data['screen_sample']['current_lab_id'] = 2
                for i in range(1, 4): btns[i].is_selected = (i == 3)

    # Generator Finished
    gen = display_data['screen_sample'].get('gen')
    if display_data['screen_sample'].get('loading') and gen is not None:
        try:
            value = next(gen)
            if isinstance(value, dict):
                matrices = value.get('matrices', [])
                if matrices and len(matrices) == 3:
                    if matrices[0] is not None: display_data['screen_sample']['labyrinths'][0].update_from_labyrinth(matrices[0])
                    if matrices[1] is not None: display_data['screen_sample']['labyrinths'][1].update_from_labyrinth(matrices[1])
                    if matrices[2] is not None: display_data['screen_sample']['labyrinths'][2].update_from_labyrinth(matrices[2])

                metrics = value.get('metrics', {})
                table = display_data['screen_sample']['table']
                for row in table.data:
                    if row[0] in metrics:
                        row[1] = metrics[row[0]]

                display_data['screen_sample']['loading'] = False
                display_data['screen_sample']['gen'] = None
            else:
                display_data['screen_sample']['loading_bar'].set_progress(value)
        except StopIteration:
            display_data['screen_sample']['loading'] = False
            display_data['screen_sample']['gen'] = None

    return display_data

def render(screen, fonts, display_data, model, task_name):
    screen.fill((30, 30, 30))

    for btn in display_data.get("buttons", []):
        btn.draw(screen, fonts[2], pygame.mouse.get_pos())

    if display_data['current_screen'] == 'sample':
        for btn in display_data['screen_sample'].get("buttons", []):
            btn.draw(screen, fonts[2], pygame.mouse.get_pos())

        if display_data['screen_sample']['loading'] == True:
            display_data['screen_sample']['loading_bar'].draw(screen, fonts[2])
        else:
            matrix = display_data['screen_sample']['labyrinths'][display_data['screen_sample']['current_lab_id']]
            matrix.draw(screen, fonts[1], pygame.mouse.get_pos())

        table = display_data['screen_sample']['table']
        table.draw(screen, fonts[2])
    elif display_data['current_screen'] == 'metrics':
        xx = 0
    elif display_data['current_screen'] == 'attention':
        xx = 0
            
    pygame.display.flip()

def evaluate(run_id: str, config: str, dataset: str, directions_task: bool, load_epoch: str = 'best', window_size: tuple = (-1, -1)):

    # Globals
    tokenizer, model, device = init(config, directions_task)
    conf = ModelConfigs.get(config)
    task_name = 'directions' if directions_task else 'completion'
    
    # Get Model
    model_path = run_id
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Data
    with open(dataset, 'r', encoding='utf-8') as f:
        blocks = [b.strip() for b in f.read().split("\n\n") if '<SOLUTION_START>' in b]

    # Initialize Interface
    pygame.init()
    if window_size == (-1, -1):
        screen = pygame.display.set_mode((1280,720), pygame.RESIZABLE)
    else:
        screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption(f"Maze Evaluation - {config} [{task_name}]")

    # Fonts
    ui_font_family = "arial, segoeui, freesansbold"
    fonts = [
        pygame.font.SysFont(ui_font_family, 24, bold=False),
        pygame.font.SysFont(ui_font_family, 20, bold=False),
        pygame.font.SysFont(ui_font_family, 16, bold=False),
        pygame.font.SysFont(ui_font_family, 13, bold=False)
    ]

    # General Objects
    display_data = {
        "current_screen": 'sample',
        "buttons": [
            UIButton(
                id_name="btn_screen_sample",
                x="3vh", y="2vh",
                width="14vh", height="5vh",
                text="Amostra",
                text_color="#FFFFFF",
                bg_color="#053911",
                active_color="#069E12"
            ),
            UIButton(
                id_name="btn_screen_metrics",
                x="19vh", y="2vh",
                width="14vh", height="5vh",
                text="Métricas",
                text_color="#FFFFFF",
                bg_color="#053911",
                active_color="#069E12"
            ),
            UIButton(
                id_name="btn_screen_attention",
                x="35vh", y="2vh",
                width="14vh", height="5vh",
                text="Atenção",
                text_color="#FFFFFF",
                bg_color="#053911",
                active_color="#069E12"
            )
        ],
        "screen_sample": {
            "labyrinth_index": 0,
            "current_lab_id": 0,
            "gen": None,
            "loading": False,
            "loading_bar": UILoadingBar(
                x="3vh", y="50vh",
                width="80vh", height="6vh",
                text="Gerando labirinto",
                text_color="#FFFFFF",
                bg_color="#112230",
                fill_color="#24DBEC"
            ),
            "buttons": [
                UIButton(
                    id_name="btn_new_sample_maze",
                    x="3vh", y="10vh",
                    width="20vh", height="5vh",
                    text="Novo Labirinto",
                    text_color="#FFFFFF",
                ),
                UIButton(
                    id_name="btn_empty_sample_mode",
                    x="28.5vh", y="10vh",
                    width="16vh", height="5vh",
                    text="Entrada",
                    text_color="#FFFFFF",
                ),
                UIButton(
                    id_name="btn_expected_sample_mode",
                    x="46.5vh", y="10vh",
                    width="16vh", height="5vh",
                    text="Esperado",
                    text_color="#FFFFFF",
                ),
                UIButton(
                    id_name="btn_obtained_sample_mode",
                    x="65vh", y="10vh",
                    width="16vh", height="5vh",
                    text="Obtido",
                    text_color="#FFFFFF",
                ),
            ],
            "labyrinths": [
                ScreenMatrix(
                    x="3vh", y="18vh",
                    width="80vh",
                    height="80vh",
                    k=21
                ),
                ScreenMatrix(
                    x="3vh", y="18vh",
                    width="80vh",
                    height="80vh",
                    k=21
                ),
                ScreenMatrix(
                    x="3vh", y="18vh",
                    width="80vh",
                    height="80vh",
                    k=21
                )
            ],
            "table": UIMetricsTable(
                x="90vh", y="10vh",
                width="80vh", height="86.5vh",
                col_weights=[3,2],
                row_weights=[1]*15,
                data=[
                    ["Corretude", ""],
                    ["Solução", ""],
                    ["Tokens Alterados", ""],
                    ["Tokens Ótimos", ""],
                    ["Tokens Diferentes", ""],
                    ["Progresso Direto", ""],
                    ["Progresso Inverso", ""],
                    ["Distância Direta", ""],
                    ["Distância Inversa", ""],
                    ["Paredes Violadas", ""],
                    ["Espaços Violados", ""],
                    ["Início Violado", ""],
                    ["Final Violado", ""],
                    ["Caminho Único", ""],
                    ["Caminho Conexo", ""],
                ],
                cell_bg_color="#112230",
                bg_color="#1E1E1E",
                text_color="#FFFFFF",
                border_size=1,
                border_color="#000000",
            )
        }
    }

    # App Loop
    running = True
    while running:
        running, event_info = process_events(screen)
        
        if running:
            display_data = iteration(
                blocks, model, tokenizer, task_name, conf, device, display_data, event_info
            )
            render(screen, fonts, display_data, model, task_name)
    pygame.quit()

if __name__ == "__main__":
    evaluate(
        run_id="runs/SimpleDecoder/1782357949889040700_epoch_20.pt",
        config="SimpleDecoder",
        dataset="datasets/train/completion/Simple_example.txt",
        directions_task=False,
        window_size=(1280,720)
    )