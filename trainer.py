from src.general.configs import ModelConfigs
from src.architecture.models import MazeDecoder
from src.data.train import train

import torch


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

conf = ModelConfigs.SimpleDecoder

#model = MazeDecoder(conf)

#train(
#    model=model,
#    dataset="datasets/train/completion/Simple_example.txt",
#    task="completion",
#    epochs=20,
#    batch_size=128,
#    device=device,
#)

model = MazeDecoder(conf)

train(
    model=model,
    dataset="datasets/train/directions/Simple_example.txt",
    task="directions",
    epochs=20,
    batch_size=64,
    device=device,
)