"""
Train FireCNN on the fire/nofire image dataset.

Usage
-----
    python train.py                              # defaults
    python train.py --epochs 20 --batch_size 32
    python train.py --data_dir path/to/Training --output my_model.pth
"""

import argparse
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import FireCNN

matplotlib.use("Agg")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(data_dir: str, img_size: int):
    categories = ["fire", "nofire"]
    data, labels = [], []

    for label, category in enumerate(categories):
        folder = os.path.join(data_dir, category)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.resize(image, (img_size, img_size))
            data.append(image)
            labels.append(label)

    X = np.array(data, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.float32)
    return X, y, categories


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, loader, criterion, optimizer):
    model.train()
    losses, correct, total = [], 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        correct += ((out > 0.5).float() == yb).sum().item()
        total += len(yb)
    return np.mean(losses), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            losses.append(criterion(out, yb).item())
            correct += ((out > 0.5).float() == yb).sum().item()
            total += len(yb)
    return np.mean(losses), correct / total


# ── Plotting ──────────────────────────────────────────────────────────────────

def save_plots(train_losses, val_losses, train_accs, val_accs,
               all_true, all_preds, y, categories, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="o", label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot_loss.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_accs, marker="o", label="Train Accuracy")
    plt.plot(epochs, val_accs, marker="o", label="Val Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.ylim(0, 1.05); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot_accuracy.png"), dpi=150)
    plt.close()

    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(categories)
    ax.set_yticks([0, 1]); ax.set_yticklabels(categories)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot_confusion_matrix.png"), dpi=150)
    plt.close()

    counts = [np.sum(y == 0), np.sum(y == 1)]
    plt.figure(figsize=(5, 4))
    plt.bar(categories, counts, color=["tomato", "steelblue"])
    plt.title("Class Distribution in Dataset")
    plt.xlabel("Class"); plt.ylabel("Number of Images")
    for i, v in enumerate(counts):
        plt.text(i, v + 5, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot_class_distribution.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {plots_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train FireCNN fire detection model")
    parser.add_argument("--data_dir", default="dataset/Training",
                        help="Path to training data directory (default: dataset/Training)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--img_size", type=int, default=128,
                        help="Image resize dimension (default: 128)")
    parser.add_argument("--output", default="fire_model.pth",
                        help="Path to save trained model weights (default: fire_model.pth)")
    parser.add_argument("--plots_dir", default="plots",
                        help="Directory to save training plots (default: plots)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset from {args.data_dir} ...")
    X, y, categories = load_dataset(args.data_dir, args.img_size)
    print(f"  {len(X)} images loaded — fire: {int(np.sum(y == 0))}, nofire: {int(np.sum(y == 1))}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # PyTorch expects channels-first: (N, C, H, W)
    to_tensor = lambda arr, dtype: torch.tensor(arr.transpose(0, 3, 1, 2), dtype=dtype)
    X_train_t = to_tensor(X_train, torch.float32)
    X_test_t  = to_tensor(X_test,  torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t),
                              batch_size=args.batch_size)

    model = FireCNN()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    print(f"\nTraining for {args.epochs} epochs ...\n")
    for epoch in range(args.epochs):
        t_loss, t_acc = train(model, train_loader, criterion, optimizer)
        v_loss, v_acc = evaluate(model, test_loader, criterion)

        train_losses.append(t_loss); train_accs.append(t_acc)
        val_losses.append(v_loss);   val_accs.append(v_acc)

        print(f"Epoch {epoch + 1:02d}/{args.epochs}  "
              f"train_loss={t_loss:.4f}  train_acc={t_acc:.4f}  "
              f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_preds = (model(X_test_t) >= 0.5).int().numpy()
        all_true  = y_test_t.int().numpy()

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=categories))

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

    save_plots(train_losses, val_losses, train_accs, val_accs,
               all_true, all_preds, y, categories, args.plots_dir)


if __name__ == "__main__":
    main()
