
from pathlib import Path

DATA_DIRECTORY = Path(
    r"C:\Users\Riley\Desktop\Uni\COMP3710\COMP3710\keras_png_slices_data"
)

TRAIN_DIRECTORY = DATA_DIRECTORY / "keras_png_slices_train"
VALIDATION_DIRECTORY = DATA_DIRECTORY / "keras_png_slices_validate"
TEST_DIRECTORY = DATA_DIRECTORY / "keras_png_slices_test"

IMAGE_SIZE = 256
IMAGE_CHANNELS = 1

BATCH_SIZE = 32
LATENT_DIMENSION = 32
NUMBER_OF_EPOCHS = 15
LEARNING_RATE = 1e-3