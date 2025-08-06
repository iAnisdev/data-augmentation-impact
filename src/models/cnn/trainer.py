import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import json


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr=1e-3,
    model_name="cnn",
    dataset_name="mnist",
    augmentation="none",
    log_dir=".artifacts",
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    log_data = {
        "model": model_name,
        "dataset": dataset_name,
        "augmentation": augmentation,
        "epochs": [],
    }

    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        epoch_duration = time.time() - start_time

        # Always validate
        model.eval()
        total_val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total

        print(
            f"[{epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )

        log_data["epochs"].append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "duration_sec": epoch_duration,
            }
        )

    # Save logs to JSON
    filename = f"train_log_{model_name}_{dataset_name}_{augmentation}.json"
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"📊 Training log saved to {filepath}")
    return model
