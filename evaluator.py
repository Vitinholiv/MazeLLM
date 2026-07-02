from src.general.helpers import prompt_for_file
from src.core.eval import evaluate
import json, os

if __name__ == "__main__":
    file_map = {}
    for root, dirs, files in os.walk("./runs"):
        if "training_logs.json" in files:
            display_name = root.replace(os.sep, '/').replace("./runs/", "")
            file_map[display_name] = root
    options = list(file_map.keys())

    errortxt = ""
    savedtxt = ""
    step = 0
    selected_run = ""

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(savedtxt,end='')
        print(errortxt,end='')

        if step == 0:
            opts_str = "\n   ".join([f"[{i+1}] {opt}" for i, opt in enumerate(options)])
            user_input = input(f"Selecione o modelo (\n   {opts_str}\n): ").strip().upper()
        
            parts = user_input.split(' E ')
            if not parts[0].isdigit() or (len(parts) > 1 and not parts[1].isdigit()):
                errortxt = "\nEntrada Inválida.\nDigite um inteiro 'x' para opção de número [x] e melhor época.\n"+"Digite 'x E y' com x,y inteiros para a opção de número [x] com época y.\n\n"
                continue
            
            idx = int(parts[0]) - 1
            if (idx < 0 or idx >= len(options)):
                errortxt = "Escolha um índice válido.\n\n"
                continue

            if len(parts) > 1:
                load_epoch = parts[1].strip()
                pt_filename = f"epoch_{load_epoch}.pt"
            else:
                pt_filename = "best_model.pt"
            
            selected_run = file_map[options[idx]]
            run_id = os.path.join(selected_run, pt_filename)
            log_path = os.path.join(selected_run, "training_logs.json")

            if not os.path.exists(run_id):
                parts = [parts[0]]
                print('\nÉpoca Inexistente. Escolhendo melhor modelo.')
                run_id = os.path.join(selected_run, "best_model.pt")
            
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)

            config = logs.get("metadata", {}).get("config", "")
            task_directions = (logs.get("metadata", {}).get("task", "") == "directions")

            step = 1
            partext = parts[0] + "" if len(parts) == 1 else parts[0] + " E " + parts[1] 
            savedtxt = f"Selecione o modelo (\n   {opts_str}\n): {partext}\n"

        if step == 1:
            dataset_map = {}
            def search_datasets(curr_path, seq=''):
                if not os.path.exists(curr_path): return
                for f in os.listdir(curr_path):
                    full_path = os.path.join(curr_path, f)
                    display_name = seq + f
                    if os.path.isdir(full_path):
                        search_datasets(full_path, display_name + ' > ')
                    elif os.path.isfile(full_path) and f.endswith('.txt'):
                        dataset_map[display_name] = full_path

            task_folder = "directions" if task_directions else "completion"
            base_dataset_path = f"./datasets/test/{task_folder}"
            
            search_datasets(base_dataset_path)
            
            if not dataset_map:
                print("\nNenhum dataset encontrado em ./datasets/test")
                break

            dataset_options = list(dataset_map.keys())
            ds_opts_str = "\n   ".join([f"[{i+1}] {opt}" for i, opt in enumerate(dataset_options)])
            
            user_input_ds = input(f"\nSelecione o dataset de teste (\n   {ds_opts_str}\n): ").strip()
            
            if user_input_ds in dataset_map:
                dataset = dataset_map[user_input_ds]
            elif user_input_ds.isdigit() and 1 <= int(user_input_ds) <= len(dataset_options):
                dataset = dataset_map[dataset_options[int(user_input_ds) - 1]]
            else:
                errortxt = "\nEscolha um índice válido para o dataset.\n"
                continue

            print()
            evaluate(run_id, config, dataset, task_directions, load_epoch if 'load_epoch' in locals() else 'best', window_size=(1280,720))
            break
        