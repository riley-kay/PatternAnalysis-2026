import torch
from torch.utils.data import DataLoader

from dataset import OASISDataset
import config

dataset = OASISDataset(
    config.TRAIN_DIRECTORY,
    config.TRAIN_SEG_DIRECTORY,
)

image, mask = dataset[0]

print("Number of training images:", len(dataset))
print("Number of test images:", len(OASISDataset(config.TEST_DIRECTORY, config.TEST_SEG_DIRECTORY)))
print("MRI shape:", image.shape)
print("Mask shape:", mask.shape)
print("MRI intensity range:", image.min().item(), image.max().item())
print("Mask class labels:", torch.unique(mask).tolist())

# Display an MRI and its corresponding segmentation mask
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].imshow(image.squeeze(), cmap="gray")
axes[0].set_title("OASIS MRI")

axes[1].imshow(mask, cmap="viridis", vmin=0, vmax=3)
axes[1].set_title("Ground Truth Segmentation")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()