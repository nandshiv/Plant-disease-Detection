"""Small helper functions used by every other module.

Nothing clever here: seeding, picking the device, reading/writing JSON,
and plotting training curves.
"""

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed=42):
    """Make runs reproducible: same seed -> same data split, same init, same shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN picks the fastest conv algorithm, which can differ between runs.
    # Turning that off makes GPU results (almost) bit-for-bit repeatable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Give each DataLoader worker process its own, but reproducible, seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    """Use the GPU if there is one (Colab T4), otherwise the CPU (laptop)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_history(history, title, out_path):
    """Plot loss and accuracy curves (train vs. validation) side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
