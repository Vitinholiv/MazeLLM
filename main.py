from src.data.train import train
from src.configs import init, MAZE_GPT_CONFIG
import os

if __name__ == "__main__":
    tokenizer, model, device = init(42, MAZE_GPT_CONFIG)
    response = {
        'Train': ['T','TRAIN','TR','TRAINING'],
        'Eval': ['E','EVAL','EV','EVALUATION'],
        'Quit': ['Q','QUIT']
    }

    while True:
        action = input('Modo (Train|Eval|Quit): ').upper()
        if action in response['Train']:

            files = [f for f in os.listdir('./datasets/train/') if f.endswith('.txt')]
            opts = "|".join(files)
            fl = input(f'Arquivo ({opts}): ')
            if fl not in files:
                print(f'Arquivo não encontrado. Usando {files[0]}.txt')
                fl = files[0]
            epc = int(input('Épocas: '))

            train(model,f'{fl}.txt', int(input('Épocas:')), device=device)

        elif action in response['Eval']:

            break

            # Eval
            model.load_state_dict(torch.load("mazegpt.pt", map_location=device, weights_only=True))
            model.to(device)
            with open("test_maze.txt") as f:
                unsolved = f.read()
            solution = solve_maze(model, tokenizer, unsolved, device, temperature=0.0)
            print("Solution:")
            print(solution)

        elif action in response['Quit']:
            break