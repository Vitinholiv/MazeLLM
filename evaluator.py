import torch
from src.general.configs import init, ModelConfigs
from src.general.helpers import prompt_for_file, prompt_options, path, empty, lastname
from src.data.eval import run_evaluation 

def eval_prompt():

    all_models = list(ModelConfigs.get_all().keys())
    model_name = prompt_options('Selecione o Modelo para avaliar', all_models) #type: ignore
    
    tokenizer, model, device = init(model_name, seed=42) #type: ignore

    if empty(path('runs')):
        print("Nenhuma execução encontrada. Treine um modelo primeiro.")
        return

    run_folder = prompt_for_file('Selecione a pasta do modelo', path('runs'), extensions=['folder'])
    if run_folder is None: return


    instance = prompt_for_file('Selecione o arquivo de checkpoint', run_folder, extensions=['.pt'])
    if instance is None: return

    model.load_state_dict(torch.load(instance, map_location=device, weights_only=True))
    model.to(device)
    model.iname = lastname(instance) # type: ignore

    testfile = prompt_for_file('Dados de Teste', path('datasets/test'), recurse=True)
    if testfile is None: return

    conf = ModelConfigs.get(model_name) #type: ignore
    fixed_output = conf.get("fixed_output", True) #type: ignore
    run_evaluation(model, tokenizer, testfile, device, fixed_output=fixed_output)

if __name__ == "__main__":
    eval_prompt()