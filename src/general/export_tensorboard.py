import os
import json
import shutil
from pathlib import Path

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    print("Aviso: 'tensorboard' não encontrado. Instale usando: pip install tensorboard")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


BASE_DIR = "runs"
TB_BASE_DIR = "tb_logs"


def discover_instances(base_dir: str):
    instances = []
    base = Path(base_dir)
    if not base.exists():
        return instances

    for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for instance_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                json_path = instance_dir / "training_logs.json"
                if json_path.exists():
                    instances.append((model_dir.name, task_dir.name, instance_dir.name, json_path))
    return instances


def export_instance(model: str, task: str, instance: str, json_path: Path, tb_base_dir: str):
    with open(json_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    run_name = f"{model}/{task}/{instance}"
    writer_path = os.path.join(tb_base_dir, run_name)
    writer = SummaryWriter(log_dir=writer_path)

    warnings = []

    # Loss por Batch
    step_losses = logs.get("step_loss_history", [])
    if step_losses:
        for step, loss in enumerate(step_losses):
            writer.add_scalar("Loss/Batch", loss, step)
    else:
        warnings.append("no 'step_loss_history'")

    # Loss por Época
    epoch_history = logs.get("epoch_history", [])
    if epoch_history:
        for epoch_data in epoch_history:
            writer.add_scalar("Loss/Epoch", epoch_data["avg_loss"], epoch_data["epoch"])
    else:
        warnings.append("no 'epoch_history'")

    # Learning Rate por Época
    lr_history = logs.get("lr_history", [])
    if lr_history:
        for epoch, lr in enumerate(lr_history, start=1):
            writer.add_scalar("LearningRate", lr, epoch)
    else:
        warnings.append("no 'lr_history'")

    # Métricas ao longo do treino (uma tag por métrica)
    metrics_history = logs.get("metrics_history", [])
    if metrics_history:
        for entry in metrics_history:
            step = entry.get("step")
            values = entry.get("values", {})
            if step is None:
                continue
            for metric_name, value in values.items():
                writer.add_scalar(f"Metrics/{metric_name}", value, step)
    else:
        warnings.append("no 'metrics_history'")

    writer.close()
    return warnings


def export_to_tensorboard():
    if not os.path.exists(BASE_DIR):
        print(f"Folder '{BASE_DIR}' not found: Train a model first.")
        return
    if os.path.exists(TB_BASE_DIR):
        shutil.rmtree(TB_BASE_DIR)

    instances = discover_instances(BASE_DIR)
    if not instances:
        print("No valid data found.")
        return

    models = sorted(set(m for m, _, _, _ in instances))
    tasks = sorted(set(t for _, t, _, _ in instances))
    print(f"Found {len(instances)} instances in {len(models)} models and {len(tasks)} tasks. Exporting...\n")

    for model, task, instance, json_path in tqdm(instances, desc="Exporting Instances"):
        warnings = export_instance(model, task, instance, json_path, TB_BASE_DIR)
        status = f"   + {model}/{task}/{instance}"
        if warnings:
            status += f"  (\033[93m{'; '.join(warnings)}\033[0m)"
        print(status)

    print(f"\nData exported.")
    print("Visualize your data by typing in the terminal:")
    print(f"\033[92m tensorboard --logdir={TB_BASE_DIR} \033[0m")
    print("Then, access the link http://localhost:6006 in a web browser.")

if __name__ == "__main__":
    export_to_tensorboard()