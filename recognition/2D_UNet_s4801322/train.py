"""Train and validate the HipMRI 2D U-Net.

All paths and training hyperparameters are configured in config.py.
Masks must contain class IDs 0..NUM_CLASSES-1 (not RGB colors).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from dataset import HipMRI_Dataset
from modules import UNet
import time

import torch.nn.functional as F


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # Standard cross-entropy
        ce_loss = self.ce(logits, targets)

        num_classes = logits.shape[1]

        # Convert predictions to probabilities
        probs = torch.softmax(logits, dim=1)

        # Convert ground truth into one-hot masks
        targets_one_hot = F.one_hot(
            targets.long(),
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()

        # Calculate Dice for each class across the batch
        dims = (0, 2, 3)

        intersection = torch.sum(
            probs * targets_one_hot, dim=dims
        )

        denominator = (
            torch.sum(probs, dim=dims)
            + torch.sum(targets_one_hot, dim=dims)
        )

        dice = (
            2 * intersection + self.smooth
        ) / (denominator + self.smooth)

        # Exclude background (class 0)
        dice_loss = 1 - dice[1:].mean()

        return ce_loss + dice_loss

def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Run one epoch; return mean pixel loss and per-class Dice scores."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_images = 0
    intersection = torch.zeros(config.NUM_CLASSES, dtype=torch.float64)
    pred_count = torch.zeros(config.NUM_CLASSES, dtype=torch.float64)
    target_count = torch.zeros(config.NUM_CLASSES, dtype=torch.float64)

    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            # CrossEntropyLoss requires masks with class IDs in [0, NUM_CLASSES).
            if masks.min() < 0 or masks.max() >= config.NUM_CLASSES:
                raise ValueError(
                    f"Mask labels must be between 0 and {config.NUM_CLASSES - 1}; "
                    f"found {masks.min().item()}..{masks.max().item()}. "
                    "Inspect/remap the dataset labels before training."
                )

            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = loss_fn(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_images += images.size(0)

            predictions = logits.argmax(dim=1)
            for cls in range(config.NUM_CLASSES):
                predicted = predictions == cls
                actual = masks == cls
                intersection[cls] += (predicted & actual).sum().item()
                pred_count[cls] += predicted.sum().item()
                target_count[cls] += actual.sum().item()

    denom = pred_count + target_count
    # Missing classes are reported as NaN, not a misleading perfect score.
    dice = torch.where(denom > 0, 2 * intersection / denom.clamp_min(1), float('nan'))
    return total_loss / total_images, dice.tolist()


def main():
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    train_dataset = HipMRI_Dataset(config.TRAIN_DIRECTORY, config.TRAIN_SEG_DIRECTORY)
    val_dataset = HipMRI_Dataset(config.VALIDATION_DIRECTORY, config.VALIDATION_SEG_DIRECTORY)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = UNet(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        base_channels=config.BASE_CHANNELS,
    ).to(device)
    loss_fn = DiceCrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    checkpoint_path = Path(config.MODEL_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = {"train": [], "val": []}
    best_val_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_dice = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_dice = run_epoch(model, val_loader, loss_fn, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | "
            f"val Dice (per class): {[round(x, 3) for x in val_dice]}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best weights: {checkpoint_path}")

    elapsed = time.perf_counter() - start_time

    print(f"Training time: {elapsed:.2f} seconds")

    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak allocated GPU memory: {memory:.2f} MiB")
    plt.plot(history["train"], label="Training")
    plt.plot(history["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Dice + Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plot_path = Path(config.LOSS_PLOT_PATH)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss plot: {plot_path}")


if __name__ == "__main__":
    main()
