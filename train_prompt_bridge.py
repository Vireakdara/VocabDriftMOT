from __future__ import annotations
import torch
import torch.nn as nn
import json
from pathlib import Path
from core.prompt_bridge import PromptBridge
from core.data_generator import PromptBridgeDataset, get_dataloader


def train(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    samples_per_pair: int = 200,
    output_dir: str = "outputs",
    device: str = "cuda",
) -> None:

    Path(output_dir).mkdir(exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Dataset
    dataset = PromptBridgeDataset(
        device=str(device),
        samples_per_pair=samples_per_pair,
    )

    # Split train/val 80/20
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {n_train}, Val samples: {n_val}")

    # Model
    model = PromptBridge().to(device)
    print(f"PromptBridge parameters: {model.count_parameters():,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # Training loop
    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            old_emb = batch["old_emb"].to(device)
            new_emb = batch["new_emb"].to(device)
            track_emb = batch["track_emb"].to(device)
            labels = batch["gate_label"].to(device)

            optimizer.zero_grad()
            gate, _ = model(old_emb, new_emb, track_emb)
            loss = criterion(gate, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (gate > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                old_emb = batch["old_emb"].to(device)
                new_emb = batch["new_emb"].to(device)
                track_emb = batch["track_emb"].to(device)
                labels = batch["gate_label"].to(device)

                gate, _ = model(old_emb, new_emb, track_emb)
                loss = criterion(gate, labels)

                val_loss += loss.item()
                preds = (gate > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        scheduler.step()

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        record = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
        }
        history.append(record)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": val_acc,
            }, f"{output_dir}/prompt_bridge_best.pt")

    # Save final model
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "val_loss": avg_val_loss,
        "val_acc": val_acc,
    }, f"{output_dir}/prompt_bridge_final.pt")

    # Save history
    with open(f"{output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final val acc: {val_acc:.4f}")
    print(f"  Model saved: {output_dir}/prompt_bridge_best.pt")
    print("=" * 50)


if __name__ == "__main__":
    train(
        epochs=100,
        batch_size=32,
        lr=1e-3,
        samples_per_pair=200,
        output_dir="outputs",
        device="cuda",
    )