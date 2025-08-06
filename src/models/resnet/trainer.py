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
    model_name="resnet18",
    dataset_name="mnist",
    augmentation="none",
    log_dir=".artifacts",
):
    """
    Train ResNet model with the same interface as CNN trainer
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    log_data = {
        "model": model_name,
        "dataset": dataset_name,
        "augmentation": augmentation,
        "epochs": [],
    }

    os.makedirs(log_dir, exist_ok=True)
    
    # Overall progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc=f"🚀 Training {model_name}", unit="epoch", position=0)

    for epoch in epoch_pbar:
        start_time = time.time()
        model.train()
        running_loss = 0.0

        # Batch progress bar
        batch_pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{epochs}", 
            unit="batch", 
            position=1, 
            leave=False
        )
        
        for images, labels in batch_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Update batch progress bar with current loss
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)
        epoch_duration = time.time() - start_time

        # Validation with progress bar
        model.eval()
        total_val_loss = 0.0
        correct, total = 0, 0
        
        val_pbar = tqdm(
            val_loader, 
            desc="Validating", 
            unit="batch", 
            position=1, 
            leave=False
        )
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Update validation progress
                current_acc = correct / total if total > 0 else 0
                val_pbar.set_postfix({"val_acc": f"{current_acc:.4f}"})

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total

        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix({
            "train_loss": f"{avg_train_loss:.4f}",
            "val_loss": f"{avg_val_loss:.4f}",
            "val_acc": f"{val_accuracy:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.6f}"
        })

        log_data["epochs"].append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": scheduler.get_last_lr()[0],
                "duration_sec": epoch_duration,
            }
        )

        scheduler.step()

    # Close the epoch progress bar
    epoch_pbar.close()

    # Save logs to JSON
    filename = f"train_log_{model_name}_{dataset_name}_{augmentation}.json"
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"📊 Training log saved to {filepath}")
    return model
