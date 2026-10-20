import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import config

class OASISDataset(Dataset):
    """
    Loads preprocessed OASIS MRI PNG images.

    Each image is returned as a PyTorch tensor with shape:

        [1, 256, 256]

    ToTensor() converts normal 8-bit PNG pixel values from
    [0, 255] to floating-point values in [0, 1].
    """

    def __init__(self, image_directory):
        self.image_directory = Path(image_directory)

        print(f"Searching: {self.image_directory.resolve()}")

        # Make sure the directory actually exists.
        if not self.image_directory.exists():
            raise RuntimeError(
                f"Directory does not exist:\n"
                f"{self.image_directory.resolve()}"
            )

        # rglob searches this directory and all subdirectories.
        #
        # A filename such as:
        #
        #     image123.nii.png
        #
        # still matches "*.png".
        self.image_paths = sorted(
            self.image_directory.rglob("*.png")
        )

        print(f"Found {len(self.image_paths)} PNG images.")

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No PNG images found in:\n"
                f"{self.image_directory.resolve()}"
            )

        # The images are already preprocessed to 256x256,
        # so we do not need to resize them.
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # Open the PNG image.
        image = Image.open(image_path)

        # Convert to a [1, 256, 256] tensor in [0, 1].
        image = self.transform(image)

        return image

def main():
    print("Loading training dataset...")
    train_dataset = OASISDataset(
        config.TRAIN_DIRECTORY
    )
    print(
        f"Training images:   {len(train_dataset)}"
    )

if __name__ == "__main__":
    main()