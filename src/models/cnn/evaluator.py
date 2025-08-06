import torch
import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model,
    dataloader,
    device,
    class_names=None,
    model_name="cnn",
    dataset_name="mnist",
    augmentation="none",
    log_dir=".artifacts",
):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    # Compute overall metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Class-wise scores
    class_report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()

    result = {
        "model": model_name,
        "dataset": dataset_name,
        "augmentation": augmentation,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "class_report": class_report,
        "confusion_matrix": conf_matrix,
    }

    os.makedirs(log_dir, exist_ok=True)
    filename = f"eval_report_{model_name}_{dataset_name}_{augmentation}.json"
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Evaluation report saved to {filepath}")
    return result
