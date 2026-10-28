"""Training loop.

`run_experiment` is the one function the notebook calls: it builds the data
loaders and the model, trains, keeps the checkpoint with the best validation
accuracy, and saves the training history (JSON + plot).

Can also be run from the command line, e.g. a tiny CPU smoke test:
    python -m src.train --model cnn --aug standard --epochs 1
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from .data import PV_CLASSES, get_plantvillage_loaders
from .models import count_parameters, get_model, save_checkpoint
from .utils import ensure_dir, get_device, plot_history, save_json, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """One pass over the training data. Returns (average loss, accuracy)."""
    model.train()  # enables dropout and BatchNorm batch statistics
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # Mixed precision (float16) on GPU: ~2x faster on a T4, same accuracy.
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()  # scale loss so small float16 gradients don't become 0
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += images.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()  # no gradients needed for evaluation -> faster, less memory
def evaluate_loss_acc(model, loader, criterion, device):
    model.eval()  # disables dropout, BatchNorm uses running statistics
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += images.size(0)
    return total_loss / seen, correct / seen


def fit(model, train_loader, val_loader, epochs, lr, device, ckpt_path, model_name,
        weight_decay=1e-4):
    """Train for `epochs` epochs and save the model whenever val accuracy improves."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    # Cosine schedule: learning rate smoothly decreases from lr to ~0 over training.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                device, scaler)
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)
        scheduler.step()

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        print(f"epoch {epoch:2d}/{epochs} | train loss {train_loss:.3f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.3f} acc {val_acc:.3f} | {time.time() - start:.0f}s")

        if val_acc > best_val_acc:  # keep only the best model (early-stopping style)
            best_val_acc = val_acc
            save_checkpoint(model, model_name, PV_CLASSES, ckpt_path,
                            extra={"epoch": epoch, "val_acc": val_acc})
    history["best_val_acc"] = best_val_acc
    return history


def run_experiment(name, model_name, aug, pv_dir, epochs, lr, results_dir="results",
                   ckpt_dir="results/checkpoints", batch_size=64, num_workers=2, seed=42,
                   background="original"):
    """Train one configuration, e.g. name="effnet_standard", model_name="effnet", aug="standard".

    background = "original" | "leaf_only" | "random_bg" (background-bias experiment,
    see LeafDataset in data.py). It changes the train and val images only.

    Saves:
        <ckpt_dir>/<name>.pt                 best checkpoint (by val accuracy)
        <results_dir>/history_<name>.json    loss/accuracy per epoch
        <results_dir>/plots/history_<name>.png
    Returns the path of the best checkpoint.
    """
    set_seed(seed)  # same seed for every experiment -> fair comparison
    device = get_device()
    ensure_dir(ckpt_dir)
    ensure_dir(os.path.join(results_dir, "plots"))

    train_loader, val_loader, _ = get_plantvillage_loaders(pv_dir, aug, batch_size,
                                                           num_workers, seed, background)
    model = get_model(model_name, len(PV_CLASSES)).to(device)
    print(f"\n=== {name}: model={model_name} ({count_parameters(model) / 1e6:.2f}M params), "
          f"aug={aug}, background={background}, epochs={epochs}, lr={lr}, device={device}, "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} ===")

    ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
    start = time.time()
    history = fit(model, train_loader, val_loader, epochs, lr, device, ckpt_path, model_name)
    history["train_time_sec"] = time.time() - start
    history.update(name=name, model_name=model_name, aug=aug, background=background,
                   epochs=epochs, lr=lr)

    save_json(history, os.path.join(results_dir, f"history_{name}.json"))
    plot_history(history, f"{name} (aug={aug})",
                 os.path.join(results_dir, "plots", f"history_{name}.png"))
    print(f"best val acc {history['best_val_acc']:.4f}, "
          f"time {history['train_time_sec'] / 60:.1f} min, saved {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one model on prepared PlantVillage data")
    parser.add_argument("--model", choices=["cnn", "effnet"], default="cnn")
    parser.add_argument("--aug", choices=["standard", "strong"], default="standard")
    parser.add_argument("--background", choices=["original", "leaf_only", "random_bg"],
                        default="original")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pv-dir", default="data/plantvillage")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--ckpt-dir", default="results/checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    name = f"{args.model}_{args.aug}" + ("" if args.background == "original" else f"_{args.background}")
    run_experiment(name, args.model, args.aug, args.pv_dir, args.epochs, args.lr,
                   args.results_dir, args.ckpt_dir, args.batch_size, args.num_workers,
                   background=args.background)
