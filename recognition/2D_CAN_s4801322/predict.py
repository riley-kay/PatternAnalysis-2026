"""
Evaluate the trained HipMRI 2D U-Net on the held-out test set.

Uses the same dataset preprocessing and label mapping as train.py.
All paths and model hyperparameters come from config.py.
"""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import config
from dataset import HipMRI_Dataset
from modules import UNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = HipMRI_Dataset(
        config.TEST_DIRECTORY,
        config.TEST_SEG_DIRECTORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = UNet(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        base_channels=config.BASE_CHANNELS,
    ).to(device)
    # Checkpoint was saved with torch.save(model.state_dict(), ...).
    state = torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Aggregate intersection and region sizes across ALL test images.
    intersections = torch.zeros(config.NUM_CLASSES, dtype=torch.float64)
    pred_counts = torch.zeros_like(intersections)
    true_counts = torch.zeros_like(intersections)
    example = None

    with torch.inference_mode():
        for images, masks in test_loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()

            if example is None:
                example = (images[0, 0].numpy(), masks[0].numpy(), predictions[0].numpy())

            for cls in range(config.NUM_CLASSES):
                pred = predictions == cls
                true = masks == cls
                intersections[cls] += (pred & true).sum().item()
                pred_counts[cls] += pred.sum().item()
                true_counts[cls] += true.sum().item()

    print(f"Test images: {len(test_dataset)}")
    print("Per-class test metrics (across the entire test set):")
    for cls in range(config.NUM_CLASSES):
        intersection = intersections[cls].item()
        predicted = pred_counts[cls].item()
        actual = true_counts[cls].item()
        dice = 2 * intersection / (predicted + actual) if (predicted + actual) else float("nan")
        union = predicted + actual - intersection
        iou = intersection / union if union else float("nan")
        print(f"Class {cls}: Dice={dice:.4f}, IoU={iou:.4f}, target pixels={int(actual)}")

    # Show one example with exactly the same class-color scale for both masks.
    if example is not None:
        image, actual, predicted = example
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("HipMRI input")
        axes[1].imshow(actual, cmap="tab10", vmin=0, vmax=config.NUM_CLASSES - 1, interpolation="nearest")
        axes[1].set_title("Ground truth")
        axes[2].imshow(predicted, cmap="tab10", vmin=0, vmax=config.NUM_CLASSES - 1, interpolation="nearest")
        axes[2].set_title("U-Net prediction")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
