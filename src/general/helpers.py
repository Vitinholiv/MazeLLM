import os

def prompt_for_file(message, path, extensions=None):
    if not os.path.exists(path):
        print(f"Diretório não encontrado: {path}")
        return None

    allow_folders = False
    valid_exts = tuple()
    filter_files = (extensions is not None)

    if filter_files:
        if isinstance(extensions, str):
            ext_list = [extensions]
        else:
            ext_list = list(extensions)
            
        allow_folders = 'folder' in ext_list
        valid_exts = tuple(e for e in ext_list if e != 'folder')

    files = []
    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        
        if os.path.isdir(full_path):
            if allow_folders:
                files.append(f)
                
        elif os.path.isfile(full_path):
            if not filter_files:
                files.append(f)
            elif valid_exts and f.endswith(valid_exts):
                files.append(f)

    if not files:
        print(f"Nenhum item encontrado em: {path}")
        return None

    file_map = {os.path.splitext(f)[0]: f for f in files}
    options = list(file_map.keys())
    opts_str = " | ".join([f"[{i+1}] {opt}" for i, opt in enumerate(options)])
    user_input = input(f'{message} ({opts_str}): ').strip()
    
    if user_input in file_map:
        selected_file = file_map[user_input]
    elif user_input.isdigit() and 1 <= int(user_input) <= len(options):
        idx = int(user_input) - 1
        selected_file = file_map[options[idx]]
    else:
        if user_input == '':
            print(f"Seleção automática: {options[0]}")
            selected_file = file_map[options[0]]
        else:
            print(f"Opção inválida. Usando: {options[0]}")
            selected_file = file_map[options[0]]
        
    return os.path.join(path, selected_file)

def prompt_for_value(message, expected_type):
    while True:
        user_input = input(f'{message}: ').strip()
        if not user_input and expected_type != str:
            print("Valor inválido.")
            continue

        try:
            value = expected_type(user_input)
            return value
        except ValueError:
            print(f"Formato incorreto (esperado: {expected_type}).")

def prompt_options(message, options):
    if not options:
        return None
        
    opts_str = " | ".join([f"[{i+1}] {opt}" for i, opt in enumerate(options)])
    user_input = input(f'{message} ({opts_str}): ').strip()

    opt_map = {opt.upper(): opt for opt in options}
    user_upper = user_input.upper()
    
    if user_upper in opt_map:
        selected_option = opt_map[user_upper]
    elif user_input.isdigit() and 1 <= int(user_input) <= len(options):
        idx = int(user_input) - 1
        selected_option = options[idx]
    else:
        if user_input == '':
            print(f"Seleção automática: {options[0]}")
            selected_option = options[0]
        else:
            print(f"Opção inválida. Usando: {options[0]}")
            selected_option = options[0]
            
    return selected_option

def path(x):
    parts = x.split('/')
    return os.path.join(*parts)

def empty(x):
    return not os.listdir(x)

def lastname(x):
    return os.path.splitext(os.path.basename(x))[0]