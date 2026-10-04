import os
import yaml
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    balanced_accuracy_score
)

from openood.utils import Config
from openood.datasets import get_dataloader
from openood.networks import get_network


# =========================================================
# 1. CONFIGURATION
# =========================================================

DATASET_CONFIG = "configs/datasets/midog/midog_expA.yml"
NETWORK_CONFIG = "configs/networks/midog_resnet18.yml"

OUTPUT_DIR = "results_midog_source_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. SIX BEST CHECKPOINTS
# =========================================================

RUNS = {

    # Original dataset
    "A0": (
        "results/"
        "midog_resnet18_224x224_base_e100_lr0.1_"
        "midog_overlap_expA/s0/best.ckpt"
    ),

    "A1": (
        "results/"
        "midog_resnet18_224x224_base_e100_lr0.1_"
        "midog_overlap_expA_s1/s1/best.ckpt"
    ),

    "A2": (
        "results/"
        "midog_resnet18_224x224_base_e100_lr0.1_"
        "midog_overlap_expA_s2/s2/best.ckpt"
    ),

    # Overlap-removed dataset
    "B0": (
        "results/"
        "midog_resnet18_224x224_base_e100_lr0.1_"
        "midog_overlap_expB/s0/best.ckpt"
    ),

    "B1": (
        "results/"
        "midog_resnet18_224x224_base_e100_lr0.1_"
        "midog_overlap_expB_s1/s1/best.ckpt"
    ),

    "B2": (
        "results/"
        "midog_resnet18_224x224_base_e100_lr0.1_"
        "midog_overlap_expB_s2/s2/best.ckpt"
    ),
}


# =========================================================
# 3. DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# 4. LOAD DATASET CONFIGURATION
# =========================================================

with open(DATASET_CONFIG, "r", encoding="utf-8") as f:
    dataset_yaml = yaml.safe_load(f)

with open(
    "configs/pipelines/train/midog_overlap_expA.yml",
    "r",
    encoding="utf-8"
) as f:
    pipeline_yaml = yaml.safe_load(f)

with open(NETWORK_CONFIG, "r", encoding="utf-8") as f:
    network_yaml = yaml.safe_load(f)


# Combine configurations
combined = {}

combined.update(dataset_yaml)
combined.update(network_yaml)
combined.update(pipeline_yaml)

dataset_config = Config(combined)


# =========================================================
# 5. FORCE COMMON VALIDATION SET
# =========================================================

dataset_config.dataset.split_names = ["val"]

dataset_config.dataset.val.imglist_pth = (
    "data/benchmark_imglist/midog/valid_midog.txt"
)

dataset_config.dataset.num_workers = 0


# =========================================================
# 6. LOAD VALIDATION DATALOADER
# =========================================================

dataloader_dict = get_dataloader(dataset_config)

val_loader = dataloader_dict["val"]

print(
    "Validation samples:",
    len(val_loader.dataset)
)


# =========================================================
# 7. LOAD NETWORK CONFIGURATION SEPARATELY
# =========================================================

network_config = Config(network_yaml)

network_config.network.pretrained = False


# =========================================================
# 8. EVALUATION FUNCTION
# =========================================================

def evaluate_checkpoint(run_name, checkpoint_path):

    print("\n" + "=" * 70)
    print("Evaluating:", run_name)
    print("Checkpoint:", checkpoint_path)
    print("=" * 70)

    # -----------------------------------------------------
    # Check checkpoint
    # -----------------------------------------------------

    if not os.path.exists(checkpoint_path):

        print("ERROR: Checkpoint not found!")
        print(checkpoint_path)

        return None


    # -----------------------------------------------------
    # Create network
    # -----------------------------------------------------

    network = get_network(
        network_config.network
    )

    network = network.to(device)


    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )


    # -----------------------------------------------------
    # Determine state dictionary
    # -----------------------------------------------------

    if isinstance(checkpoint, dict):

        if "net" in checkpoint:

            state_dict = checkpoint["net"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # -----------------------------------------------------
    # Remove module. prefix if present
    # -----------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value


    # -----------------------------------------------------
    # Load model weights
    # -----------------------------------------------------

    network.load_state_dict(
        cleaned_state_dict,
        strict=True
    )


    network.eval()


    # =====================================================
    # 9. PREDICTION
    # =====================================================

    y_true = []
    y_pred = []


    with torch.no_grad():

        for batch in val_loader:

            data = batch["data"].to(device)

            target = batch["label"].to(device)

            output = network(data)

            prediction = output.argmax(
                dim=1
            )


            y_true.extend(
                target.cpu().numpy()
            )

            y_pred.extend(
                prediction.cpu().numpy()
            )


    y_true = np.array(y_true)

    y_pred = np.array(y_pred)


    # =====================================================
    # 10. METRICS
    # =====================================================

    labels = [0, 1, 2]

    class_names = [
        "Background",
        "Mitotic",
        "Not_Mitotic"
    ]


    # Accuracy

    accuracy = np.mean(
        y_true == y_pred
    )


    # Macro F1

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )


    # Balanced Accuracy

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )


    # Confusion matrix

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )


    # Classification report

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )


    # =====================================================
    # 11. PRINT RESULTS
    # =====================================================

    print()

    print(
        "Accuracy:",
        f"{accuracy * 100:.2f}%"
    )

    print(
        "Macro-F1:",
        f"{macro_f1:.4f}"
    )

    print(
        "Balanced Accuracy:",
        f"{balanced_acc:.4f}"
    )


    print("\nConfusion Matrix")
    print("Rows = True")
    print("Columns = Predicted")

    print(cm)


    print("\nClassification Report")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0
        )
    )


    # =====================================================
    # 12. SAVE CONFUSION MATRIX
    # =====================================================

    cm_df = pd.DataFrame(
        cm,
        index=[
            "True_Background",
            "True_Mitotic",
            "True_Not_Mitotic"
        ],
        columns=[
            "Pred_Background",
            "Pred_Mitotic",
            "Pred_Not_Mitotic"
        ]
    )


    cm_path = os.path.join(
        OUTPUT_DIR,
        f"{run_name}_confusion_matrix.csv"
    )


    cm_df.to_csv(
        cm_path
    )


    # =====================================================
    # 13. CREATE SUMMARY RECORD
    # =====================================================

    result = {

        "Run": run_name,

        "Accuracy": accuracy * 100,

        "Macro_F1": macro_f1,

        "Balanced_Accuracy": balanced_acc,


        # Background

        "Background_Precision":
            report["Background"]["precision"],

        "Background_Recall":
            report["Background"]["recall"],

        "Background_F1":
            report["Background"]["f1-score"],


        # Mitotic

        "Mitotic_Precision":
            report["Mitotic"]["precision"],

        "Mitotic_Recall":
            report["Mitotic"]["recall"],

        "Mitotic_F1":
            report["Mitotic"]["f1-score"],


        # Not Mitotic

        "Not_Mitotic_Precision":
            report["Not_Mitotic"]["precision"],

        "Not_Mitotic_Recall":
            report["Not_Mitotic"]["recall"],

        "Not_Mitotic_F1":
            report["Not_Mitotic"]["f1-score"]
    }


    return result


# =========================================================
# 14. EVALUATE ALL SIX RUNS
# =========================================================

results = []


for run_name, checkpoint_path in RUNS.items():

    result = evaluate_checkpoint(
        run_name,
        checkpoint_path
    )

    if result is not None:

        results.append(result)


# =========================================================
# 15. SAVE CONSOLIDATED RESULTS
# =========================================================

results_df = pd.DataFrame(
    results
)


results_path = os.path.join(
    OUTPUT_DIR,
    "all_six_runs_validation_results.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


# =========================================================
# 16. FINAL SUMMARY
# =========================================================

print("\n")

print("=" * 70)

print(
    "FINAL SIX-RUN VALIDATION SUMMARY"
)

print("=" * 70)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\nSaved consolidated results:")

print(results_path)


print("\nEvaluation completed successfully.")