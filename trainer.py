from src.data.train import train
from src.general.configs import init, ModelConfigs
from src.general.helpers import prompt_for_file, prompt_for_value, prompt_options, path

def train_prompt():
    all_models = list(ModelConfigs.get_all().keys())
    model_name = prompt_options('Selecione o Modelo', all_models)
    
    _, model, device = init(model_name, seed=42) # type: ignore

    train_dir = path('datasets/train')
    
    file = prompt_for_file(
        'Dados de Treino', 
        train_dir, 
        extensions=['txt'], 
        recurse=True
    )
    
    if file is None:
        print("Treino cancelado.")
        return

    epochs = prompt_for_value('Épocas', int)

    conf = ModelConfigs.get(model_name) # type: ignore
    fixed_output = conf.get("fixed_output", True) # type: ignore
    train(model, file, epochs, device=device, fixed_output=fixed_output)

if __name__ == "__main__":
    train_prompt()