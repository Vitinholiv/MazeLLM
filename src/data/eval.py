import os
import torch
import pygame
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

# General Classes

class UIButton:
    def __init__(self, id_name: str, x, y, width, height, bg_color: str, text: str, text_color: str):
        self.id = id_name
        self.raw_x = x
        self.raw_y = y
        self.raw_w = width
        self.raw_h = height
        self.bg_color = pygame.Color(bg_color)
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
        current_color = self.bg_color
        if rect.collidepoint(mouse_pos):
            current_color = self.bg_color.lerp(pygame.Color("white"), 0.2)
        pygame.draw.rect(screen, current_color, rect, border_radius=8)
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)

    def check_click(self, event_info: dict) -> bool:
        if event_info.get("mouse_clicked", False):
            rect = self.get_rect(event_info["screen_w"], event_info["screen_h"])
            if rect.collidepoint(event_info["mouse_pos"]):
                return True
        return False

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

    # Screen Buttons Screen State
    for btn in display_data.get("buttons", []):
        if btn.check_click(event_info):
            if btn.id == 'btn_screen_sample':
                display_data['current_screen'] = 'sample'
                btn.bg_color = pygame.Color("#069E12")
            elif btn.id == 'btn_screen_metrics':
                display_data['current_screen'] = 'metrics'
                btn.bg_color = pygame.Color("#069E12")
            elif btn.id == 'btn_screen_attention':
                display_data['current_screen'] = 'attention'
                btn.bg_color = pygame.Color("#069E12")

    # Screen Buttons Color Updates
    for btn in display_data.get("buttons", []):
        if btn.id == 'btn_screen_sample' and display_data['current_screen'] != 'sample': 
            btn.bg_color = pygame.Color("#223344")
        elif btn.id == 'btn_screen_metrics' and display_data['current_screen'] != 'metrics':
            btn.bg_color = pygame.Color("#223344")
        elif btn.id == 'btn_screen_attention' and display_data['current_screen'] != 'attention':
            btn.bg_color = pygame.Color("#223344")
                
    return display_data

def render(screen, fonts, display_data, model, task_name, event_info):
    screen.fill((30, 30, 30))

    for btn in display_data.get("buttons", []):
        btn.draw(screen, fonts[2], event_info["mouse_pos"])
            
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
                x="2vh", y="2vh",
                width="14vh", height="5vh",
                bg_color="#069E12",
                text="Amostra",
                text_color="#FFFFFF"
            ),
            UIButton(
                id_name="btn_screen_metrics",
                x="18vh", y="2vh",
                width="14vh", height="5vh",
                bg_color="#223344",
                text="Métricas",
                text_color="#FFFFFF"
            ),
            UIButton(
                id_name="btn_screen_attention",
                x="34vh", y="2vh",
                width="14vh", height="5vh",
                bg_color="#223344",
                text="Atenção",
                text_color="#FFFFFF"
            )
        ]
    }

    # App Loop
    running = True
    while running:
        running, event_info = process_events(screen)
        
        if running:
            display_data = iteration(
                dataloader, model, tokenizer, task_name, conf, device, display_data, event_info
            )
            render(screen, fonts, display_data, model, task_name, event_info)
    pygame.quit()

if __name__ == "__main__":
    evaluate(
        run_id="1782490567257132400",
        config="SimpleDecoder",
        dataset="datasets/train/directions/Simple_example.txt",
        directions_task=True
    )