import torch
from src.data.train import train
from src.data.eval import evaluate
from src.general.configs import init, MAZE_GPT_CONFIG
from src.general.helpers import prompt_for_file, prompt_for_value, prompt_options, path, empty, lastname

if __name__ == "__main__":
    tokenizer, model, device = init(42, MAZE_GPT_CONFIG)

    while True:
        action = prompt_options('Modo de Execução',['Train','Eval','Quit'])
        if action == 'Train':

            file = prompt_for_file('Dados de Treino',path('datasets/train'))
            if file is None: break

            epochs = prompt_for_value('Épocas',int)

            train(model, file, epochs, device=device)

        elif action == 'Eval':
            if empty(path('runs')):
                print("Nenhuma execução encontrada. Treine um modelo primeiro.")
                continue

            run = prompt_for_file('Modelo','runs',['folder'])
            if run is None: break

            instance = prompt_for_file('Instância',run,['.pt'])
            if instance is None: break

            model.load_state_dict(torch.load(instance, map_location=device, weights_only=True))
            model.to(device)
            model.iname = lastname(instance)

            testfile = prompt_for_file('Dados de Teste',path('datasets/test'))
            if testfile is None: break

            evaluate(model, tokenizer, testfile, device, temperature=0.0)

        elif action == 'Quit':
            break