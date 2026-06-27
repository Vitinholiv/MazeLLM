import os
import torch
import pygame
import itertools
from typing import Union
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

def load_next_labyrinth(dataloader, tokenizer, lab_size, directions_task, model, device, conf):
    if not hasattr(load_next_labyrinth, 'data_iter'):
        load_next_labyrinth.data_iter = iter(dataloader)
        
    try:
        batch = next(load_next_labyrinth.data_iter)
    except StopIteration:
        load_next_labyrinth.data_iter = iter(dataloader)
        batch = next(load_next_labyrinth.data_iter)

    is_dencoder = (len(batch) == 3)
    
    with torch.no_grad():
        if is_dencoder:
            m_in, r_in, r_out = [tensor.to(device) for tensor in batch]
            input_ids, target_ids = m_in, r_out
            logits = model(m_in, r_in)
            pred_ids = logits.argmax(dim=-1)
        else:
            inputs, targets = [tensor.to(device) for tensor in batch]
            input_ids, target_ids = inputs, targets
            logits = model(inputs)
            pred_ids = logits.argmax(dim=-1)

    def extract_matrix_from_ids(ids_tensor, extract_type="prompt"):
        token_list = ids_tensor[0].tolist()
        chars = [tokenizer.id_to_char.get(t, '') for t in token_list]
        
        sol_start, sol_end = -1, -1
        for i, c in enumerate(chars):
            if c in ['<SOLUTION_START>', '<COMPLETION_START>']:
                sol_start = i
            elif c in ['<SOLUTION_END>', '<COMPLETION_END>']:
                sol_end = i
                break
                
        if extract_type == "solution":
            if sol_end == -1:
                return [] 
            block_chars = chars[sol_start+1 : sol_end]
            grid_count = sum(1 for c in block_chars if c != '\n' and not c.startswith('<') and c != '<PAD>')
            if grid_count != (lab_size * lab_size):
                return [] 
                
        else:
            end_idx = sol_start if sol_start != -1 else len(chars)
            block_chars = chars[:end_idx]

        lines = []
        current_line = []
        
        for char in block_chars:
            if char == '\n':
                if current_line:
                    lines.append(current_line)
                    current_line = []
            elif not char.startswith('<') and char != '<PAD>':
                current_line.append(char)
                
        if current_line:
            lines.append(current_line)
            
        for i in range(len(lines)):
            lines[i] = (lines[i] + ['#'] * lab_size)[:lab_size]
        while len(lines) < lab_size:
            lines.append(['#'] * lab_size)
            
        return lines[:lab_size]

    tokenizer_type = conf.get('tokenizer_type', 'individual')
    if directions_task:
        if tokenizer_type == 'individual':
            return []
        elif tokenizer_type == 'wall_encoded':
            return []
        elif tokenizer_type == 'free_edges':
            return []
    else:
        if tokenizer_type == 'individual':
            input_matrix = extract_matrix_from_ids(input_ids, extract_type="prompt")
            target_matrix = extract_matrix_from_ids(target_ids, extract_type="solution")
            pred_matrix = extract_matrix_from_ids(pred_ids, extract_type="solution")
            return [input_matrix, target_matrix, pred_matrix]

        elif tokenizer_type == 'wall_encoded':
            return []
        elif tokenizer_type == 'free_edges':
            return []
        

# General Classes

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
                elif token == "S": bg, char, fg = pygame.Color("#FFFFFF"), "S", pygame.Color("#069E12")
                elif token == "E": bg, char, fg = pygame.Color("#FFFFFF"), "E", pygame.Color("#FF0000")
                elif token in ["U", "D", "L", "R"]: bg, char, fg = pygame.Color("#FFFFFF"), token, pygame.Color("#FFD700")
                
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
                 bg_color="#223344", text_color="#FFFFFF", border_color=None,
                 border_size=0, cell_bg_color=None):
        self.raw_x = x
        self.raw_y = y
        self.raw_w = width
        self.raw_h = height
        self.col_weights = col_weights
        self.row_weights = row_weights
        self.data = data
        self.bg_color = pygame.Color(bg_color)
        self.text_color = pygame.Color(text_color)
        self.border_color = pygame.Color(border_color) if border_color else self.text_color
        self.border_size = border_size
        self.cell_bg_color = pygame.Color(cell_bg_color) if cell_bg_color else None

    def get_rect(self, screen_w, screen_h) -> pygame.Rect:
        return pygame.Rect(
            parse_dim(self.raw_x, screen_w, screen_h),
            parse_dim(self.raw_y, screen_w, screen_h),
            parse_dim(self.raw_w, screen_w, screen_h),
            parse_dim(self.raw_h, screen_w, screen_h)
        )

    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        rect = self.get_rect(screen.get_width(), screen.get_height())
        pygame.draw.rect(screen, self.bg_color, rect)

        total_col = sum(self.col_weights)
        total_row = sum(self.row_weights)
        col_widths = [rect.width * w / total_col for w in self.col_weights]
        row_heights = [rect.height * h / total_row for h in self.row_weights]

        for i, row in enumerate(self.data):
            for j, cell_text in enumerate(row):
                x = rect.x + sum(col_widths[:j])
                y = rect.y + sum(row_heights[:i])
                inner_x = x + self.border_size
                inner_y = y + self.border_size
                inner_w = max(0, col_widths[j] - 2 * self.border_size)
                inner_h = max(0, row_heights[i] - 2 * self.border_size)
                if inner_w > 0 and inner_h > 0:
                    cell_rect = pygame.Rect(inner_x, inner_y, inner_w, inner_h)
                    if self.cell_bg_color:
                        pygame.draw.rect(screen, self.cell_bg_color, cell_rect)
                    text_surf = font.render(str(cell_text), True, self.text_color)
                    text_rect = text_surf.get_rect(center=cell_rect.center)
                    screen.blit(text_surf, text_rect)

        color = self.border_color
        x_pos = rect.x
        for j in range(len(self.col_weights) - 1):
            x_pos += col_widths[j]
            pygame.draw.line(screen, color, (x_pos, rect.y), (x_pos, rect.y + rect.height), 1)
        y_pos = rect.y
        for i in range(len(self.row_weights) - 1):
            y_pos += row_heights[i]
            pygame.draw.line(screen, color, (rect.x, y_pos), (rect.x + rect.width, y_pos), 1)
        pygame.draw.rect(screen, color, rect, 1)

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

def iteration(dataloader, model, tokenizer, task_name, conf, device, display_data, event_info):

    # Screen Buttons
    if event_info.get("mouse_clicked"):
        for btn in display_data.get("buttons", []):
            if btn.check_click(event_info):
                display_data['current_screen'] = btn.id.replace('btn_screen_', '')
                for b in display_data.get("buttons", []):
                    b.is_selected = (b.id == btn.id)

        if display_data['current_screen'] == 'sample':
            btns = display_data['screen_sample'].get("buttons")
            if btns[1].check_click(event_info):
                display_data['screen_sample']['current_lab_id'] = 0
            if btns[2].check_click(event_info):
                display_data['screen_sample']['current_lab_id'] = 1
            if btns[3].check_click(event_info):
                display_data['screen_sample']['current_lab_id'] = 2

    return display_data

def render(screen, fonts, display_data, model, task_name):
    screen.fill((30, 30, 30))

    for btn in display_data.get("buttons", []):
        btn.draw(screen, fonts[2], pygame.mouse.get_pos())

    if display_data['current_screen'] == 'sample':
        for btn in display_data['screen_sample'].get("buttons", []):
            btn.draw(screen, fonts[2], pygame.mouse.get_pos())
        
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

    # Dataloader
    dataloader = build_dataloader(
        dsrc=dataset,
        tokenizer_type=conf['tokenizer_type'],
        context_length=conf['context_length'],
        batch_size=1,
        shuffle=True,
        mode=conf['dataset_mode'],
        lab_size=conf['lab_size'],
        directions_task=directions_task
    )
    
    # Get Model
    if load_epoch == 'best':
        model_path = f"runs/{config}/{task_name}/{run_id}/best_model.pt"
    else:
        model_path = f"runs/{config}/{task_name}/{run_id}/epoch_{load_epoch}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File Not Found: {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

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
            "current_lab_id": 0,
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
                    x="28vh", y="10vh",
                    width="16vh", height="5vh",
                    text="Entrada",
                    text_color="#FFFFFF",
                ),
                UIButton(
                    id_name="btn_expected_sample_mode",
                    x="46vh", y="10vh",
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
                row_weights=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
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
                bg_color="#161616",
                border_color="#161616",
                text_color="#FFFFFF",
                border_size=1
            )
        }
    }

    # App Loop
    running = True
    while running:
        running, event_info = process_events(screen)
        
        if running:
            display_data = iteration(
                dataloader, model, tokenizer, task_name, conf, device, display_data, event_info
            )
            render(screen, fonts, display_data, model, task_name)
    pygame.quit()

if __name__ == "__main__":
    evaluate(
        run_id="1782490567257132400",
        config="SimpleDecoder",
        dataset="datasets/train/directions/Simple_example.txt",
        directions_task=True,
        window_size=(1280,720)
    )