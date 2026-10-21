
from pathlib import Path

DATA_DIRECTORY = Path(
    r"C:\Users\Riley\Desktop\Uni\COMP3710\keras_slices_data"
)

TRAIN_DIRECTORY = DATA_DIRECTORY / "keras_slices_train"
VALIDATION_DIRECTORY = DATA_DIRECTORY / "keras_slices_validate"
TEST_DIRECTORY = DATA_DIRECTORY / "keras_slices_test"

TRAIN_SEG_DIRECTORY = DATA_DIRECTORY / "keras_slices_seg_train"
VALIDATION_SEG_DIRECTORY = DATA_DIRECTORY / "keras_slices_seg_validate"
TEST_SEG_DIRECTORY = DATA_DIRECTORY / "keras_slices_seg_test"

PAD_SIZE = (256, 144)

# Model
IN_CHANNELS = 1
NUM_CLASSES = 6
BASE_CHANNELS = 32

# Training
BATCH_SIZE = 4
SEED = 4801322
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_WORKERS = 0

MODEL_PATH = Path("outputs/improved_unet.pth")
LOSS_PLOT_PATH = Path("outputs/loss.png")