
from pathlib import Path

import numpy as np
import torch

from openood.datasets import get_dataloader
from openood.networks import get_network
from openood.utils.config import Config, parse_config


ROOT = Path(".")
NUM_CLASSES = 3

EXPERIMENTS = {
    "A_original_training_list": (
        "configs/datasets/midog/midog_expA.yml",
        "configs/pipelines/train/midog_overlap_expA.yml",
        "results/midog_resnet18_224x224_base_e100_lr0.1_midog_overlap_expA/s0/best.ckpt",
    ),
    "B_removed_three_crops": (
        "configs/datasets/midog/midog_expA.yml",
        "configs/pipelines/train/midog_overlap_expB.yml",
        "results/midog_resnet18_224x224_base_e100_lr0.1_midog_overlap_expB/s0/best.ckpt",
    ),
}

CLASS_NAMES = [
    "Background crop",
    "Mitotic figure",
    "Not mitotic figure",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Evaluation device:", device)

# Use the same validation configuration for both experiments.
dataset_cfg = "configs/datasets/midog/midog_expA.yml"
network_cfg = "configs/networks/midog_resnet18.yml"
pipeline_cfg = "configs/pipelines/train/midog_overlap_expA.yml"

config = parse_config([
    Config(dataset_cfg),
    Config(network_cfg),
    Config(pipeline_cfg),
])

# Prevent loading the original training checkpoint.
# We load each experiment's best checkpoint explicitly below.
config.network.pretrained = False

loaders = get_dataloader(config)
val_loader = loaders["val"]

print("Validation images:", len(val_loader.dataset))

net = get_network(config.network)
net = net.to(device)
net.eval()

output_dir = Path("results_midog_source_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

all_results = []

for experiment_name, (_, _, checkpoint_path) in EXPERIMENTS.items():
    checkpoint = Path(checkpoint_path)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    state_dict = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    net.load_state_dict(state_dict)
    net.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["data"].to(device)
            labels = batch["label"].to(device)

            logits = net(images)
            predictions = logits.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred):
        cm[true_label, predicted_label] += 1

    true_positive = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted_count = cm.sum(axis=0).astype(float)

    recall = np.divide(
        true_positive, support,
        out=np.zeros(NUM_CLASSES), where=support != 0
    )
    precision = np.divide(
        true_positive, predicted_count,
        out=np.zeros(NUM_CLASSES), where=predicted_count != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(NUM_CLASSES),
        where=(precision + recall) != 0
    )

    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1.mean())
    balanced_accuracy = float(recall.mean())

    print("\n" + "=" * 65)
    print("Experiment:", experiment_name)
    print("Checkpoint:", checkpoint)
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print("Classes:", CLASS_NAMES)
    print(cm)

    print("\nClass-wise metrics:")
    print(f"{'Class':25s} {'Precision':>10s} {'Recall':>10s} "
          f"{'F1':>10s} {'Support':>10s}")

    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:25s} {precision[i]:10.4f} "
            f"{recall[i]:10.4f} {f1[i]:10.4f} "
            f"{int(support[i]):10d}"
        )

    safe_name = experiment_name.lower()
    np.savetxt(
        output_dir / f"{safe_name}_confusion_matrix.csv",
        cm, delimiter=",", fmt="%d"
    )

    all_results.append({
        "experiment": experiment_name,
        "validation_accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        **{
            f"{CLASS_NAMES[i]}_precision": precision[i]
            for i in range(NUM_CLASSES)
        },
        **{
            f"{CLASS_NAMES[i]}_recall": recall[i]
            for i in range(NUM_CLASSES)
        },
        **{
            f"{CLASS_NAMES[i]}_f1": f1[i]
            for i in range(NUM_CLASSES)
        },
    })

import csv

summary_path = output_dir / "overlap_ab_validation_metrics.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

print("\nSaved summary:", summary_path)
print("Saved confusion matrices in:", output_dir.resolve())