import os

class path:
    def __init__(self, x=""):
        if isinstance(x, path):
            self._val = x._val
        elif isinstance(x, str):
            self._val = os.path.join(*x.split('/')) if x else ""
        else:
            raise TypeError("path aceita apenas 'str' ou objetos 'path'")

    def __add__(self, other):
        new_path = path()
        wants_slash = isinstance(other, str) and other.endswith('/')
        
        if isinstance(other, path):
            new_path._val = os.path.join(self._val, other._val)
        elif isinstance(other, str):
            new_path._val = os.path.join(self._val, *other.split('/'))
        else:
            print(f'Warning: Path unaltered, attempted to append: {other}')
            new_path._val = self._val
            
        if wants_slash and not new_path._val.endswith(os.sep):
            new_path._val += os.sep
            
        return new_path

    def __radd__(self, other):
        new_path = path()
        wants_slash = self._val.endswith(os.sep)
        
        if isinstance(other, path):
            new_path._val = os.path.join(other._val, self._val)
        elif isinstance(other, str):
            new_path._val = os.path.join(*other.split('/'), self._val)
        else:
            print(f'Warning: Path unaltered, attempted to append: {other}')
            new_path._val = self._val

        if wants_slash and not new_path._val.endswith(os.sep):
            new_path._val += os.sep
            
        return new_path

    def __str__(self):
        return self._val

    def __repr__(self):
        return f"path('{self._val.replace(os.sep, '/')}')"
    
    def __fspath__(self):
        return self._val

def prompt_for_file(message, path, extensions=None, recurse=False) -> str|None:
    if not os.path.exists(path):
        print(f"Diretório não encontrado: {path}")
        return None

    allow_folders = False
    valid_exts = tuple()
    filter_files = (extensions is not None)

    if filter_files:
        if isinstance(extensions, str): ext_list = [extensions]
        else: ext_list = list(extensions)
        allow_folders = 'folder' in ext_list
        valid_exts = tuple(e for e in ext_list if e != 'folder')

    # Dicionário que mapeia: "Visual Name" -> "Real Full Path"
    file_map = {} 

    def search(curr_path, seq=''):
        for f in os.listdir(curr_path):
            full_path = os.path.join(curr_path, f)
            display_name = seq + f
            
            if os.path.isdir(full_path):
                if allow_folders: file_map[display_name] = full_path
                if recurse: search(full_path, display_name + ' > ')
            elif os.path.isfile(full_path):
                if not filter_files: file_map[display_name] = full_path
                elif valid_exts and f.endswith(valid_exts): file_map[display_name] = full_path

    search(path)

    if not file_map:
        print(f"Nenhum item encontrado em: {path}")
        return None

    options = list(file_map.keys())
    opts_str = "\n   ".join([f"[{i+1}] {opt}" for i, opt in enumerate(options)])
    user_input = input(f'{message} (\n   {opts_str}\n): ').strip()
    
    if user_input in file_map:
        return file_map[user_input]
    elif user_input.isdigit() and 1 <= int(user_input) <= len(options):
        return file_map[options[int(user_input) - 1]]
    else:
        print(f"Usando automático: {options[0]}")
        return file_map[options[0]]

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

def empty(x):
    return not os.listdir(x)

def lastname(x):
    return os.path.splitext(os.path.basename(x))[0]