import os
import json
import torch
import pygame
from typing import Union
from src.core.configs import init, ModelConfigs
from src.general.metrics import calculate_metrics, calculate_direction_metrics, decoded_to_matrix, decoded_to_directions, directions_list_to_grid

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

def score_to_color(score):
    score = max(0.0, min(1.0, score))
    if score < 0.5:
        t = score / 0.5
        r, g, b = 255, int(255 * t), 0
    else:
        t = (score - 0.5) / 0.5
        r, g, b = int(255 * (1 - t)), 255, 0
    return f"#{r:02X}{g:02X}{b:02X}"

def colorize_metrics(results: dict, score: dict) -> dict:
    str_results = {}
    for key, val in results.items():
        if key == 'Corretude':
            color = score_to_color(results['Corretude'])
            str_results[key] = f"{color} {val}"
        elif key in score:
            color = score_to_color(max(0.0, min(1.0, score[key])))
            str_results[key] = f"{color} {val}"
        else:
            str_results[key] = str(val)
    return str_results

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
            steps = (conf['lab_size']**2)
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
                    m_in = torch.tensor(input_ids, device=device).unsqueeze(0)
                    r_in = torch.tensor([start_id], device=device).unsqueeze(0)
                    context, context_mask = model.encode(m_in)
                    for i in range(steps):
                        logits = model.decode_step(r_in, context, context_mask=context_mask)
                        next_token = logits[:, -1:, :].argmax(dim=-1)
                        r_in = torch.cat([r_in, next_token], dim=1)
                        yield i / steps
                    full_ids = input_ids + r_in[0].tolist()

                else:
                    raise NotImplementedError('EncoderNotImplemented')

            full_str = tokenizer.decode(full_ids)
            pred_str = '<SOLUTION_START>' + full_str.split('<SOLUTION_START>')[1]

            input_matrix = decoded_to_matrix(input_str)
            solv_directions = decoded_to_directions(solv_str)
            pred_directions = decoded_to_directions(pred_str)
            solv_matrix = directions_list_to_grid(input_matrix, solv_directions, conf['lab_size'])
            pred_matrix = directions_list_to_grid(input_matrix, pred_directions, conf['lab_size'])

            results, score = calculate_direction_metrics(input_matrix, solv_directions, pred_directions, conf['lab_size'])
            metrics = colorize_metrics(results, score)

            yield {'matrices': [input_matrix, solv_matrix, pred_matrix], 'metrics': metrics}
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
                    m_in = torch.tensor(input_ids, device=device).unsqueeze(0)
                    r_in = torch.tensor([start_id], device=device).unsqueeze(0)
                    context, context_mask = model.encode(m_in)
                    for i in range(steps):
                        logits = model.decode_step(r_in, context, context_mask=context_mask)
                        next_token = logits[:, -1:, :].argmax(dim=-1)
                        r_in = torch.cat([r_in, next_token], dim=1)
                        yield i / steps
                    full_ids = input_ids + r_in[0].tolist()

                else:
                    raise NotImplemented('EncoderNotImplemented')
            
            full_str = tokenizer.decode(full_ids)
            pred_str = '<SOLUTION_START>' + full_str.split('<SOLUTION_START>')[1]

            input_matrix = decoded_to_matrix(input_str)
            solv_matrix = decoded_to_matrix(solv_str)
            pred_matrix = decoded_to_matrix(pred_str)
            results, score = calculate_metrics(input_matrix, solv_matrix, pred_matrix, conf['lab_size'])
            metrics = colorize_metrics(results, score)

            yield {'matrices': [input_matrix, solv_matrix, pred_matrix], 'metrics': metrics}
            return
        elif tokenizer_type == 'wall_encoded':
            yield {'matrices': [], 'metrics': {}}
            return
        elif tokenizer_type == 'free_edges':
            yield {'matrices': [], 'metrics': {}}
            return
        
def generate_one_sample_metrics(sample, tokenizer, directions_task, model, device, conf):
    split_idx = sample.index('<SOLUTION_START>')
    input_str = sample[:split_idx].strip()
    solv_str = sample[split_idx:].strip()

    input_ids = tokenizer.encode(input_str)
    start_id = tokenizer.char_to_id['<SOLUTION_START>']
    tokenizer_type = model.tokenizer_type

    if tokenizer_type != 'individual':
        return {}, {}

    steps = (conf['lab_size'] ** 2) if directions_task else (conf['lab_size'] ** 2 + 2)

    with torch.no_grad():
        if conf["dataset_mode"] == "decoder":
            seq = torch.tensor(input_ids + [start_id], device=device).unsqueeze(0)
            max_steps = max(0, conf['context_length'] - seq.size(1))
            steps = min(steps, max_steps)
            for i in range(steps):
                logits = model(seq)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                seq = torch.cat([seq, next_token], dim=1)
                yield i / steps if steps > 0 else 1.0
            full_ids = seq[0].tolist()

        elif conf["dataset_mode"] == "dencoder":
            m_in = torch.tensor(input_ids, device=device).unsqueeze(0)
            r_in = torch.tensor([start_id], device=device).unsqueeze(0)
            context, context_mask = model.encode(m_in)
            max_steps = max(0, conf['context_length'] - r_in.size(1))
            steps = min(steps, max_steps)
            for i in range(steps):
                logits = model.decode_step(r_in, context, context_mask=context_mask)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                r_in = torch.cat([r_in, next_token], dim=1)
                yield i / steps if steps > 0 else 1.0
            full_ids = input_ids + r_in[0].tolist()

        else:
            raise NotImplementedError('EncoderNotImplemented')

    full_str = tokenizer.decode(full_ids)
    pred_str = '<SOLUTION_START>' + full_str.split('<SOLUTION_START>')[1]
    input_matrix = decoded_to_matrix(input_str)

    if directions_task:
        solv_directions = decoded_to_directions(solv_str)
        pred_directions = decoded_to_directions(pred_str)
        results, score = calculate_direction_metrics(input_matrix, solv_directions, pred_directions, conf['lab_size'])
    else:
        solv_matrix = decoded_to_matrix(solv_str)
        pred_matrix = decoded_to_matrix(pred_str)
        results, score = calculate_metrics(input_matrix, solv_matrix, pred_matrix, conf['lab_size'])

    return results, score


def compute_dataset_metrics_gen(blocks, tokenizer, directions_task, model, device, conf):
    total = len(blocks)
    value_history = {}
    score_history = {}
    score_sums = {}
    solucao_counts = {}
    n = 0

    if total == 0:
        return {'avg_score': {}, 'value_history': {}, 'score_history': {}, 'solucao_counts': {}, 'n': 0}

    for b_idx, sample in enumerate(blocks):
        gen = generate_one_sample_metrics(sample, tokenizer, directions_task, model, device, conf)
        results, score = {}, {}
        try:
            while True:
                inner_progress = next(gen)
                yield ((b_idx + inner_progress) / total, b_idx, total)
        except StopIteration as e:
            if e.value is not None:
                results, score = e.value

        if results:
            for k, v in results.items():
                value_history.setdefault(k, []).append(v)
            for k, v in score.items():
                score_history.setdefault(k, []).append(v)
                score_sums[k] = score_sums.get(k, 0.0) + v
            if 'Acurácia' in results:
                solucao_counts[results['Acurácia']] = solucao_counts.get(results['Acurácia'], 0) + 1
            n += 1

        yield ((b_idx + 1) / total, b_idx + 1, total)

    avg_score = {k: total_v / n for k, total_v in score_sums.items()} if n > 0 else {}
    return {
        'avg_score': avg_score, 'value_history': value_history,
        'score_history': score_history, 'solucao_counts': solucao_counts, 'n': n
    }


def build_avg_table_display(table_data, avg_score, value_history):
    colored, plain = {}, {}
    for row in table_data:
        name = row[0]
        if name in avg_score:
            v = avg_score[name]
            colored[name] = f"{score_to_color(max(0.0, min(1.0, v)))} {v:.3f}"
            plain[name] = f"{v:.3f}"
        else:
            vals = value_history.get(name, [])
            numeric_vals = [x for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool)]
            if numeric_vals:
                avgv = sum(numeric_vals) / len(numeric_vals)
                colored[name] = f"{avgv:.2f}"
                plain[name] = f"{avgv:.2f}"
            else:
                colored[name] = "-"
                plain[name] = "-"
    return colored, plain


def generate_with_attention_gen(sample, tokenizer, directions_task, model, device, conf):
    split_idx = sample.index('<SOLUTION_START>')
    input_str = sample[:split_idx].strip()
    input_ids = tokenizer.encode(input_str)
    start_id = tokenizer.char_to_id['<SOLUTION_START>']
    end_id = tokenizer.char_to_id['<SOLUTION_END>']
    tokenizer_type = model.tokenizer_type

    if tokenizer_type != 'individual':
        return None

    steps = conf['lab_size'] ** 2 if directions_task else conf['lab_size'] ** 2 + 2
    input_matrix = decoded_to_matrix(input_str)
    history = []
    predicted_ids = []

    if conf["dataset_mode"] == "decoder":
        seq = torch.tensor(input_ids + [start_id], device=device).unsqueeze(0)
        max_steps = max(0, conf['context_length'] - seq.size(1))
        steps = min(steps, max_steps)
        for i in range(steps):
            logits, attn = model.get_attention(seq, layer_idx=-1)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            history.append({'self': attn[0, :, -1, :].detach().cpu()})
            predicted_ids.append(next_token.item())
            seq = torch.cat([seq, next_token], dim=1)
            yield i / steps if steps > 0 else 1.0
        num_heads = history[0]['self'].shape[0] if history else 0
        mode = 'decoder'

    elif conf["dataset_mode"] == "dencoder":
        m_in = torch.tensor(input_ids, device=device).unsqueeze(0)
        r_in = torch.tensor([start_id], device=device).unsqueeze(0)
        with torch.no_grad():
            context, context_mask = model.encode(m_in)
        max_steps = max(0, conf['context_length'] - r_in.size(1))
        steps = min(steps, max_steps)
        for i in range(steps):
            logits, weights = model.get_attention(m_in, r_in, layer_idx=-1)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            history.append({
                'cross': weights['cross'][0, :, -1, :].detach().cpu(),
                'self': weights['self'][0, :, -1, :].detach().cpu(),
            })
            predicted_ids.append(next_token.item())
            r_in = torch.cat([r_in, next_token], dim=1)
            yield i / steps if steps > 0 else 1.0
        num_heads = history[0]['self'].shape[0] if history else 0
        mode = 'dencoder'
    else:
        raise NotImplementedError('EncoderNotImplemented')

    return {
        'input_matrix': input_matrix, 'history': history, 'predicted_ids': predicted_ids,
        'num_heads': num_heads, 'mode': mode, 'end_id': end_id, 'lab_size': conf['lab_size'],
    }


def decompose_attention_step(payload, step_idx, head_idx):
    lab_size = payload['lab_size']
    n_cells = lab_size * lab_size
    end_id = payload['end_id']
    entry = payload['history'][step_idx]
    generated_ids = payload['predicted_ids'][:step_idx]
    num_heads = payload['num_heads']

    def pick(row_tensor):
        if head_idx >= num_heads:
            return row_tensor.float().mean(dim=0).tolist()
        return row_tensor[head_idx].tolist()

    special = [0.0, 0.0, 0.0, 0.0]
    token_vals = []

    if payload['mode'] == 'decoder':
        row = pick(entry['self'])
        special[0] = row[0]
        maze = row[1:1 + n_cells]
        special[1] = row[1 + n_cells] if len(row) > 1 + n_cells else 0.0
        tail = row[1 + n_cells + 1:]
        special[2] = tail[0] if len(tail) > 0 else 0.0
        for j, gid in enumerate(generated_ids):
            v = tail[1 + j] if (1 + j) < len(tail) else 0.0
            if gid == end_id:
                special[3] = v
            else:
                token_vals.append(v)
    else:
        cross_row = pick(entry['cross'])
        self_row = pick(entry['self'])
        special[0] = cross_row[0]
        maze = cross_row[1:1 + n_cells]
        special[1] = cross_row[1 + n_cells] if len(cross_row) > 1 + n_cells else 0.0
        special[2] = self_row[0] if len(self_row) > 0 else 0.0
        for j, gid in enumerate(generated_ids):
            v = self_row[1 + j] if (1 + j) < len(self_row) else 0.0
            if gid == end_id:
                special[3] = v
            else:
                token_vals.append(v)

    return maze, special, token_vals


def save_eval_metrics(path, table_data, avg_plain, solucao_counts, n, value_history, score_history):
    data = {
        "n": n,
        "avg_plain": avg_plain,
        "solucao_counts": solucao_counts,
        "value_history": value_history,
        "score_history": score_history
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# General Classes

class UILoadingBar:
    def __init__(self, x, y, width, height, text: str = "Carregando...", text_color: str = "#FFFFFF", bg_color: str = "#223344", fill_color: str = "#2A93CB", show_time=False):
        import time
        self.raw_x = x
        self.raw_y = y
        self.raw_w = width
        self.raw_h = height
        self.text = text
        self.text_color = pygame.Color(text_color)
        self.bg_color = pygame.Color(bg_color)
        self.fill_color = pygame.Color(fill_color)
        self.progress = 0.0
        self.show_time = show_time
        self.time_text = ""
        
        self.avg_time_per_lab = 0.0
        self.last_lab_time = time.time()
        self.last_str_update = time.time()

    def set_progress(self, value: float, current_lab: int, total_labs: int):
        import time
        self.progress = max(0.0, min(1.0, value))
        
        if self.show_time and current_lab is not None and total_labs is not None:
            now = time.time()
            if current_lab > getattr(self, '_last_lab', -1):
                if current_lab > 0:
                    lab_duration = now - self.last_lab_time
                    self.avg_time_per_lab = self.avg_time_per_lab * ((current_lab - 1) / current_lab) + lab_duration / current_lab
                self.last_lab_time = now
                self._last_lab = current_lab

            if now - self.last_str_update > 2.0 and current_lab >= 5:
                rem_labs = total_labs - current_lab
                rem_sec = int(rem_labs * self.avg_time_per_lab)
                h = rem_sec // 3600
                m = (rem_sec % 3600) // 60
                s = rem_sec % 60
                self.time_text = f"({current_lab}/{total_labs})  -  Tempo Restante Estimado: {h} h {m} min {s} sec"
                self.last_str_update = now
            elif self.time_text == "" or current_lab < 5:
                self.time_text = f"({current_lab}/{total_labs})  -  Calculando..."

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
        
        if self.show_time and self.time_text:
            time_surf = font.render(self.time_text, True, self.text_color)
            time_rect = time_surf.get_rect(midtop=(rect.centerx, rect.bottom + 10))
            screen.blit(time_surf, time_rect)

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
    def __init__(self, x, y, width, height, k=10, font=None):
        self.raw_x, self.raw_y = x, y
        self.raw_w, self.raw_h = width, height
        self.k = k
        self.font = font
        self.grid = [[{"char": "#", "score": 0.0} for _ in range(k)] for _ in range(k)]
        self.cell_padding = 2
        self.highlight_pos = None
        self.highlight_color = pygame.Color("#24DBEC")

    def set_highlight(self, pos):
        self.highlight_pos = pos

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
                    overlay.fill((42, 147, 203, int(cell["score"] * 150)))
                    screen.blit(overlay, (x, y))
                if char:
                    draw_font = self.font if self.font else font
                    text_surf = draw_font.render(char, True, fg)
                    screen.blit(text_surf, text_surf.get_rect(center=(x + cell_w//2, y + cell_h//2)))
                if self.highlight_pos == (r, c):
                    pygame.draw.rect(screen, self.highlight_color, (x + self.cell_padding, y + self.cell_padding, cell_w - wall_padding, cell_h - wall_padding), 2)

    def set_scores(self, score_grid):
        for r in range(self.k):
            for c in range(self.k):
                v = score_grid[r][c] if r < len(score_grid) and c < len(score_grid[0]) else 0.0
                self.grid[r][c]["score"] = v

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

    @staticmethod
    def _parse_color_prefix(text: str):
        text = str(text)
        if text.startswith('#') and len(text) >= 8 and text[7] == ' ':
            hex_part = text[1:7]
            try:
                int(hex_part, 16)
                return pygame.Color('#' + hex_part), text[8:]
            except ValueError:
                pass
        return None, text

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

                override_color, display_text = self._parse_color_prefix(cell_text)
                final_color = override_color if override_color else cell_text_color
                text_surf = cell_font.render(display_text, True, final_color)
                text_rect = text_surf.get_rect(center=cell_rect.center)
                screen.blit(text_surf, text_rect)

class UIHistogram:
    def __init__(self, x, y, width, height, title="", bins=10, n_ticks=7,
                 bg_color="#1E1E1E", bar_color="#2A93CB", text_color="#FFFFFF",
                 title_color="#FFFFFF", corner_radius=8):
        self.raw_x = x
        self.raw_y = y
        self.raw_w = width
        self.raw_h = height
        self.title = title
        self.bins = bins
        self.n_ticks = n_ticks
        self.bg_color = pygame.Color(bg_color)
        self.bar_color = pygame.Color(bar_color)
        self.text_color = pygame.Color(text_color)
        self.title_color = pygame.Color(title_color)
        self.corner_radius = corner_radius
        self.values = []

    def set_data(self, values):
        self.values = [v for v in values if v is not None]

    def get_rect(self, screen_w, screen_h):
        return pygame.Rect(
            parse_dim(self.raw_x, screen_w, screen_h),
            parse_dim(self.raw_y, screen_w, screen_h),
            parse_dim(self.raw_w, screen_w, screen_h),
            parse_dim(self.raw_h, screen_w, screen_h)
        )

    def _is_numeric(self):
        return len(self.values) > 0 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in self.values)

    def draw(self, screen, font, title_font=None, show_grid=False):
        rect = self.get_rect(screen.get_width(), screen.get_height())
        pygame.draw.rect(screen, self.bg_color, rect, border_radius=self.corner_radius)

        title_font = title_font or font
        content_top = rect.y + 8
        if self.title:
            title_surf = title_font.render(self.title, True, self.title_color)
            screen.blit(title_surf, (rect.x + 50, content_top))
            content_top += title_surf.get_height() + 6

        plot_rect = pygame.Rect(rect.x + 10, content_top, rect.width - 20, max(0, rect.bottom - content_top - 8))
        if not self.values:
            msg = font.render("Sem dados", True, self.text_color)
            screen.blit(msg, msg.get_rect(center=plot_rect.center))
            return

        if self._is_numeric():
            self._draw_numeric(screen, font, plot_rect, show_grid)
        else:
            self._draw_categorical(screen, font, plot_rect, show_grid)

    def _draw_numeric(self, screen, font, plot_rect, show_grid=False):
        label_h = font.get_height() + 4
        y_lbl_w = font.size("1000")[0] + 6
        bar_area = pygame.Rect(plot_rect.x + y_lbl_w, plot_rect.y,
                                max(1, plot_rect.width - y_lbl_w), max(0, plot_rect.height - label_h))

        vmin, vmax = min(self.values), max(self.values)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        bin_w = (vmax - vmin) / self.bins
        counts = [0] * self.bins
        for v in self.values:
            idx = int((v - vmin) / bin_w) if bin_w > 0 else 0
            idx = min(max(idx, 0), self.bins - 1)
            counts[idx] += 1

        max_count = max(counts) if counts else 1
        bar_w = bar_area.width / self.bins

        if show_grid:
            grid_color = self.bg_color.lerp(pygame.Color("white"), 0.08)
            for i in range(self.n_ticks):
                t = i / max(1, self.n_ticks - 1)
                yp = bar_area.bottom - int(t * bar_area.height)
                pygame.draw.line(screen, grid_color, (bar_area.x, yp), (bar_area.right, yp), 1)
                xp = bar_area.x + int(t * bar_area.width)
                pygame.draw.line(screen, grid_color, (xp, bar_area.y), (xp, bar_area.bottom), 1)

        for i, c in enumerate(counts):
            h = (c / max_count) * bar_area.height if max_count > 0 else 0
            bar = pygame.Rect(int(bar_area.x + i * bar_w + 1), int(bar_area.bottom - h),
                               max(1, int(bar_w - 2)), int(h))
            pygame.draw.rect(screen, self.bar_color, bar, border_radius=3)

        pygame.draw.line(screen, self.text_color, (bar_area.x, bar_area.y), (bar_area.x, bar_area.bottom), 1)
        pygame.draw.line(screen, self.text_color, (bar_area.x, bar_area.bottom), (bar_area.right, bar_area.bottom), 1)

        for i in range(self.n_ticks):
            t = i / max(1, self.n_ticks - 1)
            val = vmin + t * (vmax - vmin)
            lbl = font.render(f"{val:.2f}", True, self.text_color)
            xp = bar_area.x + int(t * bar_area.width) - lbl.get_width() // 2
            xp = max(bar_area.x, min(xp, bar_area.right - lbl.get_width()))
            screen.blit(lbl, (xp, bar_area.bottom + 2))

        for i in range(self.n_ticks):
            t = i / max(1, self.n_ticks - 1)
            lbl = font.render(str(round(t * max_count)), True, self.text_color)
            yp = bar_area.bottom - int(t * bar_area.height) - lbl.get_height() // 2
            screen.blit(lbl, (plot_rect.x, yp))

    def _draw_categorical(self, screen, font, plot_rect, show_grid=False):
        counts = {}
        for v in self.values:
            counts[v] = counts.get(v, 0) + 1
        categories = sorted(counts.keys())
        if not categories:
            return
        label_h = font.get_height() + 4
        y_lbl_w = font.size("1000")[0] + 6
        bar_area = pygame.Rect(plot_rect.x + y_lbl_w, plot_rect.y,
                                max(1, plot_rect.width - y_lbl_w), max(0, plot_rect.height - label_h))

        max_count = max(counts.values())
        bar_w = bar_area.width / max(1, len(categories))

        if show_grid:
            grid_color = self.bg_color.lerp(pygame.Color("white"), 0.08)
            for i in range(self.n_ticks):
                t = i / max(1, self.n_ticks - 1)
                yp = bar_area.bottom - int(t * bar_area.height)
                pygame.draw.line(screen, grid_color, (bar_area.x, yp), (bar_area.right, yp), 1)
            for i in range(len(categories)):
                xp = bar_area.x + i * bar_w + bar_w / 2
                pygame.draw.line(screen, grid_color, (xp, bar_area.y), (xp, bar_area.bottom), 1)

        for i, cat in enumerate(categories):
            c = counts[cat]
            h = (c / max_count) * bar_area.height if max_count > 0 else 0
            bar = pygame.Rect(int(bar_area.x + i * bar_w + 4), int(bar_area.bottom - h),
                               max(1, int(bar_w - 8)), int(h))
            pygame.draw.rect(screen, self.bar_color, bar, border_radius=3)
            lbl = font.render(f"{cat} ({c})", True, self.text_color)
            screen.blit(lbl, lbl.get_rect(midtop=(bar.centerx, bar_area.bottom + 2)))

        pygame.draw.line(screen, self.text_color, (bar_area.x, bar_area.y), (bar_area.x, bar_area.bottom), 1)
        pygame.draw.line(screen, self.text_color, (bar_area.x, bar_area.bottom), (bar_area.right, bar_area.bottom), 1)

        for i in range(self.n_ticks):
            t = i / max(1, self.n_ticks - 1)
            lbl = font.render(str(round(t * max_count)), True, self.text_color)
            yp = bar_area.bottom - int(t * bar_area.height) - lbl.get_height() // 2
            screen.blit(lbl, (plot_rect.x, yp))

class UITokenGrid:
    def __init__(self, x, y, width, height, rows, cols, title="",
                 bg_color="#161616", dark_color="#0A0A0A", heat_color="#2A93CB",
                 text_color="#FFFFFF", title_color="#FFFFFF", highlight_color="#24DBEC",
                 corner_radius=8, cell_padding=2):
        self.raw_x, self.raw_y, self.raw_w, self.raw_h = x, y, width, height
        self.rows, self.cols = rows, cols
        self.title = title
        self.bg_color = pygame.Color(bg_color)
        self.dark_color = pygame.Color(dark_color)
        self.heat_color = pygame.Color(heat_color)
        self.text_color = pygame.Color(text_color)
        self.title_color = pygame.Color(title_color)
        self.highlight_color = pygame.Color(highlight_color)
        self.corner_radius = corner_radius
        self.cell_padding = cell_padding
        n = rows * cols
        self.chars = [None] * n
        self.attns = [0.0] * n
        self.highlight_index = None

    def set_cells(self, chars, attns, highlight_index=None):
        n = self.rows * self.cols
        self.chars = (list(chars) + [None] * n)[:n]
        self.attns = (list(attns) + [0.0] * n)[:n]
        self.highlight_index = highlight_index

    def get_rect(self, sw, sh):
        return pygame.Rect(parse_dim(self.raw_x, sw, sh), parse_dim(self.raw_y, sw, sh),
                            parse_dim(self.raw_w, sw, sh), parse_dim(self.raw_h, sw, sh))

    def draw(self, screen, font, title_font=None):
        rect = self.get_rect(screen.get_width(), screen.get_height())
        pygame.draw.rect(screen, self.bg_color, rect, border_radius=self.corner_radius)
        title_font = title_font or font
        top = rect.y + 6
        if self.title:
            t = title_font.render(self.title, True, self.title_color)
            screen.blit(t, (rect.x + 8, top))
            top += t.get_height() + 4

        grid_rect = pygame.Rect(rect.x + 6, top, rect.width - 12, max(0, rect.bottom - top - 6))
        cw = grid_rect.width / self.cols
        ch = grid_rect.height / self.rows

        for idx in range(self.rows * self.cols):
            r, c = idx // self.cols, idx % self.cols
            cx, cy = grid_rect.x + c * cw, grid_rect.y + r * ch
            cell_rect = pygame.Rect(int(cx + self.cell_padding), int(cy + self.cell_padding),
                                     max(1, int(cw - 2 * self.cell_padding)), max(1, int(ch - 2 * self.cell_padding)))
            char = self.chars[idx]
            if char is None:
                pygame.draw.rect(screen, self.dark_color, cell_rect, border_radius=3)
                continue
            a = max(0.0, min(1.0, self.attns[idx]))
            cell_color = self.dark_color.lerp(self.heat_color, a)
            pygame.draw.rect(screen, cell_color, cell_rect, border_radius=3)
            if idx == self.highlight_index:
                pygame.draw.rect(screen, self.highlight_color, cell_rect, 2, border_radius=3)
            if cw > 10 and ch > 10:
                txt = font.render(str(char), True, self.text_color)
                screen.blit(txt, txt.get_rect(center=cell_rect.center))


class UISlider:
    def __init__(self, x, y, width, height, min_value, max_value, value=0, label="",
                 bg_color="#223344", track_color="#112230", handle_color="#2A93CB", text_color="#FFFFFF"):
        self.raw_x, self.raw_y, self.raw_w, self.raw_h = x, y, width, height
        self.min_value, self.max_value = min_value, max_value
        self.value = max(min_value, min(value, max_value))
        self.label = label
        self.bg_color = pygame.Color(bg_color)
        self.track_color = pygame.Color(track_color)
        self.handle_color = pygame.Color(handle_color)
        self.text_color = pygame.Color(text_color)
        self.dragging = False

    def set_range(self, min_value, max_value, reset_value=None):
        self.min_value, self.max_value = min_value, max_value
        self.value = reset_value if reset_value is not None else max(min_value, min(self.value, max_value))

    def get_rect(self, sw, sh):
        return pygame.Rect(parse_dim(self.raw_x, sw, sh), parse_dim(self.raw_y, sw, sh),
                            parse_dim(self.raw_w, sw, sh), parse_dim(self.raw_h, sw, sh))

    def update(self, screen_w, screen_h, mouse_pos, mouse_down):
        rect = self.get_rect(screen_w, screen_h)
        track = pygame.Rect(rect.x + 10, rect.centery - 3, rect.width - 20, 6)
        if mouse_down:
            if self.dragging or track.inflate(0, 24).collidepoint(mouse_pos):
                self.dragging = True
                t = (mouse_pos[0] - track.x) / max(1, track.width)
                t = max(0.0, min(1.0, t))
                span = self.max_value - self.min_value
                self.value = self.min_value if span <= 0 else round(self.min_value + t * span)
        else:
            self.dragging = False

    def draw(self, screen, font, label_override=None):
        rect = self.get_rect(screen.get_width(), screen.get_height())
        pygame.draw.rect(screen, self.bg_color, rect, border_radius=8)
        track = pygame.Rect(rect.x + 10, rect.centery - 3, rect.width - 20, 6)
        pygame.draw.rect(screen, self.track_color, track, border_radius=3)
        span = self.max_value - self.min_value
        t = 0.0 if span <= 0 else (self.value - self.min_value) / span
        hx = track.x + int(t * track.width)
        pygame.draw.circle(screen, self.handle_color, (hx, track.centery), 9)
        text = label_override if label_override is not None else f"{self.label}: {self.value}"
        label = font.render(text, True, self.text_color)
        screen.blit(label, (rect.x + 10, rect.y + 2))

# App Run Functions

def process_events(screen: pygame.Surface):
    running = True
    event_info = {
        "mouse_clicked": False,
        "mouse_pos": pygame.mouse.get_pos(),
        "screen_w": screen.get_width(),
        "screen_h": screen.get_height(),
        "key_pressed": None
    }
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key in [pygame.K_a, pygame.K_LEFT]:
                event_info["key_pressed"] = "left"
            elif event.key in [pygame.K_d, pygame.K_RIGHT]:
                event_info["key_pressed"] = "right"
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
    display_data['_screen_w'] = event_info.get('screen_w', display_data.get('_screen_w', 1280))
    display_data['_screen_h'] = event_info.get('screen_h', display_data.get('_screen_h', 720))

    if event_info.get("mouse_clicked"):
        for btn in display_data.get("buttons", []):
            if btn.check_click(event_info):
                display_data['current_screen'] = btn.id.replace('btn_screen_', '')
                for b in display_data.get("buttons", []):
                    b.is_selected = (b.id == btn.id)
        
        sa = display_data['screen_attention']
        if display_data['current_screen'] == 'attention' and sa['gen'] is None and sa['payload'] is None and not sa['loading']:
            idx = sa['labyrinth_index']
            sa['labyrinth_index'] = (idx + 1) % len(blocks)
            sa['gen'] = generate_with_attention_gen(blocks[idx], tokenizer, task_name == 'directions', model, device, conf)
            sa['loading'] = True

        if display_data['current_screen'] == 'sample':
            btns = display_data['screen_sample'].get("buttons", [])

            if btns[0].check_click(event_info):
                display_data['screen_sample']['gen'] = load_next_labyrinth(blocks, tokenizer, task_name == 'directions', model, device, conf, display_data)
                display_data['screen_sample']['loading'] = True
                display_data['screen_sample']['progress'] = 0.0
                display_data['screen_sample']['current_lab_id'] = 0
                for i in range(1, 4): btns[i].is_selected = (i == 1)

            if btns[1].check_click(event_info):
                display_data['screen_sample']['current_lab_id'] = 0
                for i in range(1, 4): btns[i].is_selected = (i == 1)

            if btns[2].check_click(event_info):
                display_data['screen_sample']['current_lab_id'] = 1
                for i in range(1, 4): btns[i].is_selected = (i == 2)

            if btns[3].check_click(event_info):
                display_data['screen_sample']['current_lab_id'] = 2
                for i in range(1, 4): btns[i].is_selected = (i == 3)

        elif display_data['current_screen'] == 'metrics':
            sm = display_data['screen_metrics']
            btns = sm.get("buttons", [])
            max_metric = len(sm['table'].data) - 1

            if not sm['loading'] and not sm['saved']:
                save_eval_metrics(sm['save_path'], sm['table'].data, sm.get('avg_plain', {}), sm.get('solucao_counts', {}), sm.get('n', 0), sm.get('value_history', {}), sm.get('score_history', {}))
                sm['saved'] = True

            if btns[0].check_click(event_info):
                sm['current_metric'] -= 1
                if sm['current_metric'] < 0:
                    sm['current_metric'] = max_metric

            if btns[1].check_click(event_info):
                sm['current_metric'] += 1
                if sm['current_metric'] > max_metric:
                    sm['current_metric'] = 0

        elif display_data['current_screen'] == 'attention':
            sa = display_data['screen_attention']
            if sa['buttons'][0].check_click(event_info):
                idx = sa['labyrinth_index']
                sa['labyrinth_index'] = (idx + 1) % len(blocks)
                sa['gen'] = generate_with_attention_gen(blocks[idx], tokenizer, task_name == 'directions', model, device, conf)
                sa['loading'] = True
                sa['payload'] = None

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
                for row in display_data['screen_sample']['table'].data:
                    if row[0] in metrics:
                        row[1] = metrics[row[0]]
                display_data['screen_sample']['loading'] = False
                display_data['screen_sample']['gen'] = None
            else:
                display_data['screen_sample']['loading_bar'].set_progress(value, None, None)
        except StopIteration:
            display_data['screen_sample']['loading'] = False
            display_data['screen_sample']['gen'] = None

    sm = display_data['screen_metrics']
    sm_gen = sm.get('gen')
    if sm.get('loading') and sm_gen is not None:
        try:
            payload = next(sm_gen)
            if isinstance(payload, tuple) and len(payload) == 3:
                progress, current_lab, total_labs = payload
                sm['loading_bar'].set_progress(progress, current_lab, total_labs)
            else:
                sm['loading_bar'].set_progress(payload)
        except StopIteration as e:
            payload = e.value or {}
            sm['avg_score'] = payload.get('avg_score', {})
            sm['value_history'] = payload.get('value_history', {})
            sm['score_history'] = payload.get('score_history', {})
            sm['solucao_counts'] = payload.get('solucao_counts', {})
            sm['n'] = payload.get('n', 0)
            colored, plain = build_avg_table_display(sm['table'].data, sm['avg_score'], sm['value_history'])
            sm['avg_plain'] = plain
            for row in sm['table'].data:
                if row[0] in colored:
                    row[1] = colored[row[0]]
            sm['loading'] = False
            sm['gen'] = None

    sa = display_data['screen_attention']
    if sa['loading'] and sa['gen'] is not None:
        try:
            progress = next(sa['gen'])
            sa['loading_bar'].set_progress(progress, None, None)
        except StopIteration as e:
            payload = e.value
            sa['payload'] = payload
            sa['loading'] = False
            sa['gen'] = None
            if payload:
                sa['maze_matrix'].update_from_labyrinth(payload['input_matrix'])
                n_steps = len(payload['predicted_ids'])
                sa['step_slider'].set_range(0, max(0, n_steps - 1), reset_value=max(0, n_steps - 1))
                sa['head_slider'].set_range(0, payload['num_heads'])

    if display_data['current_screen'] == 'attention' and sa['payload']:
        mouse_pos = pygame.mouse.get_pos()
        mouse_down = pygame.mouse.get_pressed()[0]
        sw, sh = display_data.get('_screen_w', 1280), display_data.get('_screen_h', 720)
        sa['step_slider'].update(sw, sh, mouse_pos, mouse_down)
        sa['head_slider'].update(sw, sh, mouse_pos, mouse_down)

        if event_info.get("key_pressed") and not sa['step_slider'].dragging:
            if event_info["key_pressed"] == "left":
                sa['step_slider'].value = max(sa['step_slider'].min_value, sa['step_slider'].value - 1)
            elif event_info["key_pressed"] == "right":
                sa['step_slider'].value = min(sa['step_slider'].max_value, sa['step_slider'].value + 1)

    return display_data

    return display_data

def render(screen, fonts, display_data, tokenizer):
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

        display_data['screen_sample']['table'].draw(screen, fonts[2])

    elif display_data['current_screen'] == 'metrics':
        sm = display_data['screen_metrics']
        if sm['loading'] == True:
            sm['loading_bar'].draw(screen, fonts[2])
        else:
            for btn in sm.get("buttons", []):
                btn.draw(screen, fonts[2], pygame.mouse.get_pos())

            sm['table'].draw(screen, fonts[2])

            metric_name = sm['table'].data[sm['current_metric']][0]
            sm['hist_values'].set_data(sm['value_history'].get(metric_name, []))
            sm['hist_scores'].set_data(sm['score_history'].get(metric_name, []))
            sm['hist_values'].draw(screen, fonts[3], fonts[2], show_grid=True)
            sm['hist_scores'].draw(screen, fonts[3], fonts[2], show_grid=True)

            label = fonts[1].render(f"{metric_name}", True, pygame.Color("#FFFFFF"))
            screen.blit(label, (parse_dim('4vh', screen.get_width(), screen.get_height()),
                                 parse_dim('11vh', screen.get_width(), screen.get_height())))

    elif display_data['current_screen'] == 'attention':
        sa = display_data['screen_attention']
        for btn in sa.get('buttons', []):
            btn.draw(screen, fonts[2], pygame.mouse.get_pos())

        if sa['loading']:
            sa['loading_bar'].draw(screen, fonts[2])
        elif sa['payload']:
            payload = sa['payload']
            step_idx = sa['step_slider'].value
            head_idx = sa['head_slider'].value
            maze_vals, special_vals, token_vals = decompose_attention_step(payload, step_idx, head_idx)

            max_val = max(maze_vals + special_vals + token_vals) if (maze_vals or special_vals or token_vals) else 1e-9
            max_val = max_val if max_val > 1e-9 else 1e-9
            norm = lambda v: v / max_val

            lab_size = payload['lab_size']
            
            if step_idx < lab_size * lab_size:
                hl_pos = (step_idx // lab_size, step_idx % lab_size)
                hl_special = None
            else:
                hl_pos = None
                hl_special = 3
                
            score_grid = [[norm(maze_vals[r * lab_size + c]) for c in range(lab_size)] for r in range(lab_size)]
            sa['maze_matrix'].set_scores(score_grid)
            sa['maze_matrix'].set_highlight(hl_pos)
            sa['maze_matrix'].draw(screen, fonts[1], pygame.mouse.get_pos())

            revealed_ids = payload['predicted_ids'][:step_idx + 1]
            end_id = payload['end_id']
            se_revealed = end_id in revealed_ids
            special_chars = ['LI', 'LE', 'SI', 'SE' if se_revealed else None]
            sa['special_grid'].set_cells(special_chars, [norm(v) for v in special_vals], highlight_index=hl_special)
            sa['special_grid'].draw(screen, fonts[2], fonts[2])

            pred_display = [['#'] * lab_size for _ in range(lab_size)]
            pred_score_grid = [[0.0] * lab_size for _ in range(lab_size)]
            content_ids = [tid for tid in revealed_ids if tid != end_id]
            for i, tid in enumerate(content_ids):
                r, c = i // lab_size, i % lab_size
                if r < lab_size:
                    pred_display[r][c] = tokenizer.id_to_char.get(tid, '?')
                    pred_score_grid[r][c] = norm(token_vals[i]) if i < len(token_vals) else 0.0
            sa['predicted_matrix'].update_from_labyrinth(pred_display)
            sa['predicted_matrix'].set_scores(pred_score_grid)
            sa['predicted_matrix'].set_highlight(hl_pos)
            sa['predicted_matrix'].draw(screen, fonts[1], pygame.mouse.get_pos())

            sa['step_slider'].draw(screen, fonts[2], label_override=f"Passo: {step_idx + 1} / {len(payload['predicted_ids'])}")
            head_label = "Cabeça: Combinada" if head_idx >= payload['num_heads'] else f"Cabeça: {head_idx + 1} / {payload['num_heads']}"
            sa['head_slider'].draw(screen, fonts[2], label_override=head_label)

    pygame.display.flip()

def evaluate(run_id: str, config: str, dataset: str, directions_task: bool, load_epoch: str = 'best', window_size: tuple = (-1, -1)):

    # Globals
    tokenizer, model, device = init(config, directions_task)
    conf = ModelConfigs.get(config)
    task_name = 'directions' if directions_task else 'completion'
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]
    
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
    if directions_task:
        table_data = [
            ["Corretude", ""],
            ["Acurácia", ""],
            ["Edit Distance", ""],
            ["Tokens Ótimos", ""],
            ["Tokens Alterados", ""],
            ["Tokens Diferentes", ""],
            ["Progresso Direto", ""],
            ["Distância Direta", ""],
            ["Paredes Violadas", ""],
        ]
        table_h = "52.0vh"
        table_lines = 9
    else:
        table_data = [
            ["Corretude", ""],
            ["Acurácia", ""],
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
        ]
        table_h = "86.5vh"
        table_lines = 15    

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
                width="80vh", height=table_h,
                col_weights=[3,2],
                row_weights=[1]*table_lines,
                data=table_data,
                cell_bg_color="#112230",
                bg_color="#1E1E1E",
                text_color="#FFFFFF",
                border_size=1,
                border_color="#000000",
            )
        },
        "screen_metrics": {
            "current_metric": 0,
            "gen": None,
            "loading": True,
            "value_history": {},
            "score_history": {},
            "solucao_counts": {},
            "avg_score": {},
            "avg_plain": {},
            "n": 0,
            "saved": False,
            "save_path": os.path.join(os.path.dirname(run_id) or ".", f"eval_metrics_{dataset_name}.json"),
            "loading_bar": UILoadingBar(
                x="10vw", y="50vh",
                width="80vw", height="6vh",
                text="Gerando métricas",
                text_color="#FFFFFF",
                bg_color="#112230",
                fill_color="#4FC302",
                show_time=True
            ),
            "buttons": [
                UIButton(id_name="btn_prev_metric", x="54vh", y="10vh", width="14vh", height="5vh", text="Anterior", text_color="#FFFFFF"),
                UIButton(id_name="btn_next_metric", x="70vh", y="10vh", width="14vh", height="5vh", text="Próxima", text_color="#FFFFFF"),
            ],
            "hist_values": UIHistogram(
                x="3vh", y="18vh", width="80vh", height="39vh",
                title="Distribuição dos Valores", bins=20,
                bg_color="#1E1E1E", bar_color="#24DBEC", text_color="#FFFFFF", title_color="#FFFFFF"
            ),
            "hist_scores": UIHistogram(
                x="3vh", y="59vh", width="80vh", height="39vh",
                title="Distribuição dos Scores", bins=20,
                bg_color="#1E1E1E", bar_color="#4FC302", text_color="#FFFFFF", title_color="#FFFFFF"
            ),
            "table": UIMetricsTable(
                x="90vh", y="10vh",
                width="80vh", height=table_h,
                col_weights=[3,2],
                row_weights=[1]*table_lines,
                data=[row[:] for row in table_data],
                cell_bg_color="#112230",
                bg_color="#1E1E1E",
                text_color="#FFFFFF",
                border_size=1,
                border_color="#000000",
            )
        },
        "screen_attention": {
            "labyrinth_index": 0,
            "gen": None,
            "loading": False,
            "payload": None,
            "loading_bar": UILoadingBar(
                x="10vw", y="50vh", width="80vw", height="6vh",
                text="Calculando atenção", bg_color="#112230", fill_color="#2A93CB"
            ),
            "buttons": [
                UIButton(id_name="btn_new_attention_sample", x="3vh", y="10vh",
                         width="20vh", height="5vh", text="Nova Amostra", text_color="#FFFFFF"),
            ],
            "maze_matrix": ScreenMatrix(x="3vh", y="18vh", width="62vh", height="62vh", k=conf['lab_size'], font=fonts[3]),
            "special_grid": UITokenGrid(x="140vh", y="18vh", width="20vh", height="62vh", rows=4, cols=1, title=""),
            "predicted_matrix": ScreenMatrix(x="71.5vh", y="18vh", width="62vh", height="62vh", k=conf['lab_size'], font=fonts[3]),
            "step_slider": UISlider(x="84vh", y="90vh", width="78vh", height="6vh", min_value=0, max_value=1, value=0, label="Passo"),
            "head_slider": UISlider(x="3vh", y="90vh", width="78vh", height="6vh", min_value=0, max_value=1, value=0, label="Cabeça", handle_color="#2A93CB"),
        }
    }
    sm = display_data['screen_metrics']
    if os.path.exists(sm['save_path']):
        try:
            with open(sm['save_path'], "r", encoding="utf-8") as f:
                saved = json.load(f)
            sm['n'] = saved.get('n', 0)
            sm['avg_plain'] = saved.get('avg_plain', {})
            sm['solucao_counts'] = saved.get('solucao_counts', {})
            sm['value_history'] = saved.get('value_history', {})
            sm['score_history'] = saved.get('score_history', {})
            sm['avg_score'] = {k: sum(v)/len(v) for k, v in sm['score_history'].items() if v}
            
            colored, plain = build_avg_table_display(sm['table'].data, sm['avg_score'], sm['value_history'])
            sm['avg_plain'] = plain
            for row in sm['table'].data:
                if row[0] in colored:
                    row[1] = colored[row[0]]
            
            sm['saved'] = True
            sm['loading'] = False
            sm['gen'] = None
        except Exception:
            sm['gen'] = compute_dataset_metrics_gen(blocks, tokenizer, directions_task, model, device, conf)
    else:
        sm['gen'] = compute_dataset_metrics_gen(blocks, tokenizer, directions_task, model, device, conf)

    # App Loop
    running = True
    while running:
        running, event_info = process_events(screen)
        
        if running:
            display_data = iteration(
                blocks, model, tokenizer, task_name, conf, device, display_data, event_info
            )
            render(screen, fonts, display_data, tokenizer)
    pygame.quit()