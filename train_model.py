"""
train_model.py  (CNN version)

Trains an actual convolutional neural network for parking-spot
occupancy detection, sized so that every intermediate feature map
fits comfortably inside the MSP430F5529's 8 KB of SRAM.

PIPELINE:

    original image
        -> grayscale
        -> 64x64 (LANCZOS)
        -> normalize /255
        -> 4x4 average pool -> 16x16
        -> 16x16x1 CNN input

MODEL:

    Input (16,16,1)
      -> Conv2D(4 filters, 3x3, valid)   -> 14x14x4
      -> ReLU
      -> MaxPool2D(2x2)                  -> 7x7x4
      -> Conv2D(8 filters, 3x3, valid)   -> 5x5x8
      -> ReLU
      -> MaxPool2D(2x2)                  -> 2x2x8
      -> Flatten                         -> 32
      -> Dense(16, ReLU)
      -> Dense(2, Softmax)

BoardImage.jpg is NOT used for training.
It is processed separately after training as a final demonstration
of the trained model's prediction.

Training output is intentionally simplified so that only one clean
line is printed per epoch:

    Epoch  1/30 | Train Acc: 68.42% | Train Loss: 0.6124 | Val Acc: 71.33% | Val Loss: 0.5812
"""


import os
import random

import numpy as np
from PIL import Image

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = (
    r"C:\Users\ishan\Smart-City-Project\PKLot"
    r"\PKLotSegmented\PUCPR"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOARD_IMAGE_PATH = os.path.join(SCRIPT_DIR, "BoardImage.jpg")

IMG_SIZE = 64
DOWNSAMPLE = 16
POOL_FACTOR = IMG_SIZE // DOWNSAMPLE

EPOCHS = 30
BATCH_SIZE = 32
RANDOM_SEED = 41

TOTAL_IMAGES = 10000
TEST_IMAGES = 1000
TEST_IMAGES_PER_CLASS = TEST_IMAGES // 2

TRAIN_IMAGES = TOTAL_IMAGES - TEST_IMAGES
TRAIN_IMAGES_PER_CLASS = TRAIN_IMAGES // 2

VALIDATION_SPLIT = 0.15

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# ============================================================
# Clean training logger
# ============================================================

class CleanTrainingLogger(keras.callbacks.Callback):
    """
    Prints exactly one clean line at the end of each epoch.

    Example:

    Epoch  1/30 | Train Acc: 68.42% | Train Loss: 0.6124 |
    Val Acc: 71.33% | Val Loss: 0.5812
    """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        train_acc = logs.get("accuracy", 0.0) * 100
        train_loss = logs.get("loss", 0.0)

        val_acc = logs.get("val_accuracy", 0.0) * 100
        val_loss = logs.get("val_loss", 0.0)

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"Train Acc: {train_acc:6.2f}% | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f}"
        )


# ============================================================
# Dataset discovery
# ============================================================

def find_dataset_images(dataset_root):
    dataset_root = os.path.abspath(dataset_root)

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(
            f"Dataset directory does not exist:\n{dataset_root}"
        )

    empty_images = []
    occupied_images = []

    for root, _dirs, files in os.walk(dataset_root):

        folder_name = os.path.basename(root).lower()

        if folder_name == "empty":
            bucket = empty_images

        elif folder_name == "occupied":
            bucket = occupied_images

        else:
            continue

        for fname in files:

            if fname.lower().endswith(IMAGE_EXTENSIONS):
                bucket.append(os.path.join(root, fname))

    required_per_class = (
        TEST_IMAGES_PER_CLASS +
        TRAIN_IMAGES_PER_CLASS
    )

    if (
        len(empty_images) < required_per_class
        or len(occupied_images) < required_per_class
    ):
        raise RuntimeError(
            f"Not enough images. "
            f"Need {required_per_class} per class, "
            f"found Empty={len(empty_images)} "
            f"Occupied={len(occupied_images)}"
        )

    rng = random.Random(RANDOM_SEED)

    selected_empty = rng.sample(
        empty_images,
        required_per_class
    )

    selected_occupied = rng.sample(
        occupied_images,
        required_per_class
    )

    # --------------------------------------------------------
    # Test split
    # --------------------------------------------------------

    test_empty = selected_empty[
        :TEST_IMAGES_PER_CLASS
    ]

    train_empty = selected_empty[
        TEST_IMAGES_PER_CLASS:
    ]

    test_occupied = selected_occupied[
        :TEST_IMAGES_PER_CLASS
    ]

    train_occupied = selected_occupied[
        TEST_IMAGES_PER_CLASS:
    ]

    # --------------------------------------------------------
    # Create paths and labels
    # --------------------------------------------------------

    test_paths = test_empty + test_occupied

    test_labels = (
        [0] * len(test_empty)
        + [1] * len(test_occupied)
    )

    train_paths = train_empty + train_occupied

    train_labels = (
        [0] * len(train_empty)
        + [1] * len(train_occupied)
    )

    # --------------------------------------------------------
    # Shuffle training set
    # --------------------------------------------------------

    train_combined = list(
        zip(train_paths, train_labels)
    )

    rng.shuffle(train_combined)

    train_paths, train_labels = zip(
        *train_combined
    )

    train_paths = list(train_paths)

    train_labels = np.array(
        train_labels,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Shuffle test set
    # --------------------------------------------------------

    test_combined = list(
        zip(test_paths, test_labels)
    )

    rng.shuffle(test_combined)

    test_paths, test_labels = zip(
        *test_combined
    )

    test_paths = list(test_paths)

    test_labels = np.array(
        test_labels,
        dtype=np.int64
    )

    print(f"      Train images: {len(train_paths)}")
    print(f"      Test images:  {len(test_paths)}")

    return (
        train_paths,
        train_labels,
        test_paths,
        test_labels
    )


# ============================================================
# Image preprocessing
# ============================================================

def load_images(paths, labels):

    imgs = np.zeros(
        (
            len(paths),
            IMG_SIZE,
            IMG_SIZE
        ),
        dtype=np.float32
    )

    for i, p in enumerate(paths):

        img = Image.open(p).convert("L")

        # Resize to 64x64
        img = img.resize(
            (IMG_SIZE, IMG_SIZE),
            Image.Resampling.LANCZOS
        )

        # Convert to numpy and normalize
        arr = np.array(
            img,
            dtype=np.float32
        ) / 255.0

        imgs[i] = arr

    return imgs, labels


def average_pool(images_64):
    """
    Convert:

        (N, 64, 64)

    into:

        (N, 16, 16)

    using 4x4 average pooling.
    """

    n = images_64.shape[0]

    reshaped = images_64.reshape(
        n,
        DOWNSAMPLE,
        POOL_FACTOR,
        DOWNSAMPLE,
        POOL_FACTOR
    )

    return reshaped.mean(axis=(2, 4))


# ============================================================
# Board image
# ============================================================

def load_board_image():

    if not os.path.exists(BOARD_IMAGE_PATH):
        raise FileNotFoundError(
            f"BoardImage.jpg was not found:\n"
            f"{BOARD_IMAGE_PATH}"
        )

    img = Image.open(
        BOARD_IMAGE_PATH
    ).convert("L")

    # Resize to 64x64
    img = img.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.LANCZOS
    )

    # Normalize 0-255 -> 0.0-1.0
    arr = np.array(
        img,
        dtype=np.float32
    ) / 255.0

    return arr


# ============================================================
# Model
# ============================================================

def build_model():

    model = keras.Sequential([

        layers.Input(
            shape=(
                DOWNSAMPLE,
                DOWNSAMPLE,
                1
            )
        ),

        # 16x16x1 -> 14x14x4
        layers.Conv2D(
            4,
            3,
            activation="relu",
            padding="valid",
            name="conv1"
        ),

        # 14x14x4 -> 7x7x4
        layers.MaxPooling2D(
            2,
            name="pool1"
        ),

        # 7x7x4 -> 5x5x8
        layers.Conv2D(
            8,
            3,
            activation="relu",
            padding="valid",
            name="conv2"
        ),

        # 5x5x8 -> 2x2x8
        layers.MaxPooling2D(
            2,
            name="pool2"
        ),

        # 2x2x8 -> 32
        layers.Flatten(),

        # 32 -> 16
        layers.Dense(
            16,
            activation="relu",
            name="fc1"
        ),

        # 16 -> 2
        layers.Dense(
            2,
            activation="softmax",
            name="fc2"
        ),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, X_test, y_test):

    print("\n" + "=" * 60)
    print("                 TEST SET EVALUATION")
    print("=" * 60)

    loss, accuracy_value = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # Calculate metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        f"\nTest Loss:      {loss:.4f}"
    )

    print(
        f"Test Accuracy:  {accuracy * 100:.2f}%"
    )

    print(
        f"Precision:      {precision * 100:.2f}%"
    )

    print(
        f"Recall:         {recall * 100:.2f}%"
    )

    print(
        f"F1 Score:       {f1 * 100:.2f}%"
    )

    print("\nConfusion Matrix:")
    print(
        "                 Predicted"
    )
    print(
        "               VACANT  OCCUPIED"
    )
    print(
        f"Actual VACANT  {cm[0][0]:6d}  {cm[0][1]:8d}"
    )
    print(
        f"Actual OCCUPIED{cm[1][0]:6d}  {cm[1][1]:8d}"
    )

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "VACANT",
                "OCCUPIED"
            ],
            digits=4,
            zero_division=0
        )
    )

    return probabilities, predictions


# ============================================================
# Board image prediction
# ============================================================

def test_board_image(model):

    print("\n" + "=" * 60)
    print("                BOARD IMAGE TEST")
    print("=" * 60)

    print("\nLoading BoardImage.jpg...")

    board = load_board_image()

    print("      Grayscale:       Yes")
    print("      Original resize: 64x64")
    print("      Normalization:   0.0 - 1.0")
    print("      Average pooling: 4x4")
    print("      CNN input:       16x16x1")

    # 64x64 -> 16x16
    board_small = average_pool(
        board[np.newaxis, ...]
    )[0]

    # Add batch and channel dimensions
    board_input = board_small.reshape(
        1,
        DOWNSAMPLE,
        DOWNSAMPLE,
        1
    ).astype(np.float32)

    print("\nRunning CNN prediction...")

    probs = model.predict(
        board_input,
        verbose=0
    )[0]

    pred = int(
        np.argmax(probs)
    )

    label_names = {
        0: "VACANT",
        1: "OCCUPIED"
    }

    print("\nPrediction:")
    print(
        f"      Result:   {label_names[pred]}"
    )

    print(
        f"      Vacant:   {probs[0] * 100:.2f}%"
    )

    print(
        f"      Occupied: {probs[1] * 100:.2f}%"
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Initialization
    # ========================================================

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    keras.utils.set_random_seed(
        RANDOM_SEED
    )

    print("\n" + "=" * 60)
    print("             PARKING OCCUPANCY CNN")
    print("=" * 60)

    print("\nRandom seed:", RANDOM_SEED)
    print("Epochs:     ", EPOCHS)
    print("Batch size: ", BATCH_SIZE)

    # ========================================================
    # Step 1: Dataset
    # ========================================================

    print("\n" + "=" * 60)
    print("[1/6] DATASET")
    print("=" * 60)

    (
        train_paths,
        train_labels,
        test_paths,
        test_labels
    ) = find_dataset_images(
        DATASET_ROOT
    )

    # ========================================================
    # Step 2: Load and preprocess
    # ========================================================

    print("\n" + "=" * 60)
    print("[2/6] IMAGE PREPROCESSING")
    print("=" * 60)

    print("\n      Loading training images...")

    X_train_raw, y_train = load_images(
        train_paths,
        train_labels
    )

    print("      Loading test images...")

    X_test_raw, y_test = load_images(
        test_paths,
        test_labels
    )

    print("\n      Preprocessing:")
    print("      - Grayscale")
    print("      - Resize: 64x64")
    print("      - Normalize: /255")
    print("      - 4x4 average pooling")
    print("      - Final size: 16x16")

    X_train_small = average_pool(
        X_train_raw
    )

    X_test_small = average_pool(
        X_test_raw
    )

    # ========================================================
    # Step 3: Prepare CNN input
    # ========================================================

    print("\n" + "=" * 60)
    print("[3/6] CNN INPUT PREPARATION")
    print("=" * 60)

    X_train = X_train_small.reshape(
        -1,
        DOWNSAMPLE,
        DOWNSAMPLE,
        1
    )

    X_test = X_test_small.reshape(
        -1,
        DOWNSAMPLE,
        DOWNSAMPLE,
        1
    )

    print(
        f"\n      Training input shape: {X_train.shape}"
    )

    print(
        f"      Test input shape:     {X_test.shape}"
    )

    # Free unnecessary memory
    del (
        X_train_raw,
        X_test_raw,
        X_train_small,
        X_test_small
    )

    # ========================================================
    # Step 4: Build CNN
    # ========================================================

    print("\n" + "=" * 60)
    print("[4/6] BUILDING CNN")
    print("=" * 60)

    model = build_model()

    print("\n      Architecture:")
    print("      Input:       16x16x1")
    print("      Conv1:       4 filters, 3x3")
    print("      Pool1:       2x2")
    print("      Conv2:       8 filters, 3x3")
    print("      Pool2:       2x2")
    print("      Flatten:     32")
    print("      Dense:       16")
    print("      Output:      2 classes")

    print("\n      Model summary:\n")

    model.summary()

    # ========================================================
    # Step 5: Train
    # ========================================================

    print("\n" + "=" * 60)
    print("[5/6] CNN TRAINING")
    print("=" * 60)

    print(
        f"\nTraining for {EPOCHS} epochs..."
    )

    print(
        "Validation split: "
        f"{VALIDATION_SPLIT * 100:.0f}%"
    )

    print("\n" + "-" * 100)

    model.fit(
        X_train,
        y_train,

        validation_split=VALIDATION_SPLIT,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        shuffle=True,

        # IMPORTANT:
        # Disable Keras' normal batch-by-batch progress bar.
        verbose=0,

        # Our own clean one-line-per-epoch logger.
        callbacks=[
            CleanTrainingLogger()
        ],
    )

    print("-" * 100)

    print("\nTraining finished.")

    # ========================================================
    # Save model
    # ========================================================

    model_path = os.path.join(
        SCRIPT_DIR,
        "parking_model.keras"
    )

    model.save(model_path)

    print(
        f"\nModel saved to:"
        f"\n{model_path}"
    )

    # ========================================================
    # Test model
    # ========================================================

    print("\nRunning evaluation on the held-out test set...")

    evaluate_model(
        model,
        X_test,
        y_test
    )

    # ========================================================
    # Step 6: Board image
    # ========================================================

    print("\n" + "=" * 60)
    print("[6/6] BOARD IMAGE")
    print("=" * 60)

    test_board_image(
        model
    )

    # ========================================================
    # Complete
    # ========================================================

    print("\n" + "=" * 60)
    print("                 TRAINING COMPLETE")
    print("=" * 60)

    print("\nNext step:")
    print("Run quantize_and_export.py")


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":
    main()
