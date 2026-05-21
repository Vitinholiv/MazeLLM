import torch
from src.data.train import train
from src.data.eval import evaluate
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
        action = input('Escolha (Train|Eval|Quit): ').strip().upper()
        if action in response['Train']:

            files = [f[:-4] for f in os.listdir('./datasets/train/') if f.endswith('.txt')]
            opts = "|".join(files)
            fl = input(f'Dados de Treino ({opts}): ').strip()
            if fl not in files:
                print(f'Arquivo não encontrado. Usando {files[0]}.txt')
                fl = files[0]
            epc = int(input('Épocas: '))

            train(model, f'{fl}.txt', epc, device=device)

        elif action in response['Eval']:
            if not os.path.exists('runs') or not os.listdir('runs'):
                print("Nenhuma execução encontrada. Treine um modelo primeiro.")
                continue

            mods = [f for f in os.listdir('runs') if os.path.isdir(os.path.join('runs', f))]
            mopts = "|".join(mods)
            md = input(f'Modelo ({mopts}): ').strip()
            if md not in mods:
                print(f'Modelo não encontrado. Usando {mods[0]}')
                md = mods[0]
            
            model_folder = os.path.join('runs', md)
            inst_files = [f for f in os.listdir(model_folder) if f.endswith('.pt')]
            if not inst_files:
                print(f"Nenhum arquivo de pesos encontrado em '{model_folder}'.")
                continue

            inst_clean = [f[:-3] for f in inst_files]
            iopts = "|".join(inst_clean)
            ist = input(f'Instância ({iopts}): ').strip()
            
            if ist in inst_clean:
                true_pt_f = f"{ist}.pt"
            else:
                print(f'Instância não encontrada. Usando {inst_clean[0]}.pt')
                ist = inst_files[0][:-3]
                true_pt_f = inst_files[0]

            mdict = os.path.join(model_folder, true_pt_f)
            print(f"Carregando pesos de: {mdict}")
            model.load_state_dict(torch.load(mdict, map_location=device, weights_only=True))
            model.to(device)
            model.iname = ist

            test_dir = './datasets/test/'
            tfiles = [f[:-4] for f in os.listdir(test_dir) if f.endswith('.txt')]
            topts = "|".join(tfiles)
            tfl = input(f'Dados de Teste ({topts}): ').strip()
            if tfl not in tfiles:
                print(f'Arquivo não encontrado. Usando {tfiles[0]}.txt')
                tfl = tfiles[0]

            test_path = os.path.join(test_dir, f'{tfl}.txt')
            evaluate(model, tokenizer, test_path, device, temperature=0.0)

        elif action in response['Quit']:
            break