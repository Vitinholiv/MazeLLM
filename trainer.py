from src.core.train import train
from src.core.configs import ModelConfigs
import os

if __name__ == "__main__":
    errortxt = ""
    savedtxt = ""
    step = 0

    task_directions = False
    task_name = ""
    config = ""
    dataset = ""
    run_id = ""
    batch_size = 64
    epochs = 10
    lr = 3e-4
    metric_log_interval = 50
    metric_sample_count = 8

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(savedtxt, end='')
        print(errortxt, end='')
        errortxt = ""

        if step == 0:
            user_input = input("Selecione a tarefa (\n   [1] completion\n   [2] directions\n): ").strip()

            if user_input not in ('1', '2'):
                errortxt = "\nEntrada Inválida. Escolha 1 ou 2.\n\n"
                continue

            task_directions = (user_input == '2')
            task_name = 'directions' if task_directions else 'completion'
            step = 1
            savedtxt += f"Selecione a tarefa (\n   [1] completion\n   [2] directions\n): {user_input}\n"

        if step == 1:
            # Sugere configs já utilizadas em runs anteriores, mas permite digitar outra
            known_configs = set()
            if os.path.exists("./runs"):
                for entry in os.listdir("./runs"):
                    if os.path.isdir(os.path.join("./runs", entry)):
                        known_configs.add(entry)
            config_options = sorted(known_configs)

            if config_options:
                opts_str = "\n   ".join([f"[{i+1}] {opt}" for i, opt in enumerate(config_options)])
                user_input = input(f"\nSelecione a config (\n   {opts_str}\n): ").strip()
            else:
                user_input = input("\nDigite o nome da config: ").strip()

            if not config_options:
                candidate = user_input
            elif user_input.isdigit() and 1 <= int(user_input) <= len(config_options):
                candidate = config_options[int(user_input) - 1]
            else:
                errortxt = "\nEscolha um índice válido.\n"
                continue

            try:
                ModelConfigs.get(candidate)
            except Exception:
                errortxt = f"\nConfig '{candidate}' inválida ou não encontrada.\n\n"
                continue

            config = candidate
            step = 2
            savedtxt += f"\nSelecione a config: {config}\n"

        if step == 2:
            dataset_map = {}

            def search_datasets(curr_path, seq=''):
                if not os.path.exists(curr_path):
                    return
                for f in os.listdir(curr_path):
                    full_path = os.path.join(curr_path, f)
                    display_name = seq + f
                    if os.path.isdir(full_path):
                        search_datasets(full_path, display_name + ' > ')
                    elif os.path.isfile(full_path) and f.endswith('.txt'):
                        dataset_map[display_name] = full_path

            base_dataset_path = f"./datasets/train/{task_name}"
            search_datasets(base_dataset_path)

            if not dataset_map:
                print(f"\nNenhum dataset encontrado em {base_dataset_path}")
                break

            dataset_options = list(dataset_map.keys())
            ds_opts_str = "\n   ".join([f"[{i+1}] {opt}" for i, opt in enumerate(dataset_options)])

            user_input_ds = input(f"\nSelecione o dataset de treino (\n   {ds_opts_str}\n): ").strip()

            if user_input_ds in dataset_map:
                dataset = dataset_map[user_input_ds]
            elif user_input_ds.isdigit() and 1 <= int(user_input_ds) <= len(dataset_options):
                dataset = dataset_map[dataset_options[int(user_input_ds) - 1]]
            else:
                errortxt = "\nEscolha um índice válido para o dataset.\n"
                continue

            step = 3
            savedtxt += f"\nSelecione o dataset de treino: {user_input_ds}\n"

        if step == 3:
            run_options = []
            run_base = f"./runs/{config}/{task_name}"
            if os.path.exists(run_base):
                run_options = sorted([
                    d for d in os.listdir(run_base)
                    if os.path.isdir(os.path.join(run_base, d))
                ])

            if run_options:
                opts_str = "\n   ".join([f"[{i+1}] {opt}" for i, opt in enumerate(run_options)])
                user_input = input(
                    f"\nSelecione um run existente para retomar, ou digite um novo nome (\n   {opts_str}\n): "
                ).strip()
            else:
                user_input = input("\nDigite o nome do novo run: ").strip()

            if not user_input:
                errortxt = "\nO nome do run não pode ser vazio.\n"
                continue

            if run_options and user_input.isdigit() and 1 <= int(user_input) <= len(run_options):
                run_id = run_options[int(user_input) - 1]
            elif user_input.isdigit():
                errortxt = "\nNome de run inválido: não pode ser apenas um número que não corresponda a uma opção existente.\n"
                continue
            else:
                run_id = user_input

            step = 4
            savedtxt += f"\nSelecione o run: {run_id}\n"

        if step == 4:
            def prompt_default(label, default, cast):
                raw = input(f"{label} [{default}]: ").strip()
                if raw == "":
                    return default
                try:
                    return cast(raw)
                except ValueError:
                    return None

            print()
            bs = prompt_default("Batch size", batch_size, int)
            if bs is None or bs <= 0:
                errortxt = "\nBatch size inválido.\n\n"
                continue
            batch_size = bs

            ep = prompt_default("Epochs", epochs, int)
            if ep is None or ep <= 0:
                errortxt = "\nNúmero de epochs inválido.\n\n"
                continue
            epochs = ep

            lr_val = prompt_default("Learning rate", lr, float)
            if lr_val is None or lr_val <= 0:
                errortxt = "\nLearning rate inválido.\n\n"
                continue
            lr = lr_val

            step = 5
            savedtxt += (
                f"\nBatch size: {batch_size}\n"
                f"Epochs: {epochs}\n"
                f"Learning rate: {lr}\n"
                f"Metric log interval: {metric_log_interval}\n"
                f"Metric sample count: {metric_sample_count}\n"
            )

        if step == 5:
            print()
            train(
                run_id, config, dataset, task_directions,
                batch_size=batch_size, epochs=epochs, lr=lr,
                metric_log_interval=metric_log_interval,
                metric_sample_count=metric_sample_count
            )
            break