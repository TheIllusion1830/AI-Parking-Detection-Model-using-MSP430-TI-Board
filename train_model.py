"""
train_model.py

Trains a tiny neural network for parking-spot occupancy detection.

PKLot structure:

C:/Users/ishan/Smart-City-Project/PKLot/PKLotSegmented/PUCPR/
├── Cloudy/
├── Rainy/
└── Sunny/

Labels:

    Empty     -> 0 (VACANT)
    Occupied  -> 1 (OCCUPIED)


Pipeline:

    64x64 grayscale image
            ↓
      16x16 average pooling
            ↓
        256 inputs
            ↓
      Dense(16, ReLU)
            ↓
      Dense(2, Softmax)
            ↓
     VACANT / OCCUPIED


Dataset split:

    10,000 randomly selected PKLot images
              |
              +---- 9,000 training images
              |       |
              |       +---- 15% validation
              |
              +---- 1,000 completely unseen test images


Board image:

    BoardImage.jpg
          ↓
      grayscale
          ↓
        64x64
          ↓
        16x16
          ↓
       256 values
          ↓
      trained model
          ↓
    VACANT/OCCUPIED


IMPORTANT:

BoardImage.jpg is NOT part of the PKLot training/test dataset.

Place BoardImage.jpg in the same directory as this Python file.

The board image is tested directly in this script.

No NumPy file is generated for BoardImage.

Generated files:

    parking_model.keras
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

# ------------------------------------------------------------
# Board image
# ------------------------------------------------------------

# BoardImage.jpg must be in the same folder as this script.
SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BOARD_IMAGE_PATH = os.path.join(
    SCRIPT_DIR,
    "BoardImage.jpg"
)

# ------------------------------------------------------------
# Image processing
# ------------------------------------------------------------

IMG_SIZE = 64
DOWNSAMPLE = 16

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

EPOCHS = 30
BATCH_SIZE = 32

RANDOM_SEED = 41

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

TOTAL_IMAGES = 10000

TEST_IMAGES = 1000

TEST_IMAGES_PER_CLASS = (
    TEST_IMAGES // 2
)

TRAIN_IMAGES = (
    TOTAL_IMAGES - TEST_IMAGES
)

TRAIN_IMAGES_PER_CLASS = (
    TRAIN_IMAGES // 2
)

VALIDATION_SPLIT = 0.15

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
)


# ============================================================
# Find dataset images
# ============================================================

def find_dataset_images(dataset_root):

    dataset_root = os.path.abspath(
        dataset_root
    )

    if not os.path.exists(dataset_root):

        raise FileNotFoundError(
            f"Dataset directory does not exist:\n"
            f"{dataset_root}"
        )

    empty_images = []
    occupied_images = []

    print("\n========================================")
    print("Searching PKLot dataset")
    print("========================================")

    print(
        f"Dataset root:\n{dataset_root}"
    )

    # --------------------------------------------------------
    # Recursively search
    # --------------------------------------------------------

    for root, dirs, files in os.walk(
        dataset_root
    ):

        folder_name = os.path.basename(
            root
        ).lower()

        if folder_name == "empty":

            image_list = empty_images

        elif folder_name == "occupied":

            image_list = occupied_images

        else:

            continue

        for fname in files:

            if fname.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                image_list.append(
                    os.path.join(
                        root,
                        fname
                    )
                )

    # --------------------------------------------------------
    # Dataset size
    # --------------------------------------------------------

    print("\nImages found:")

    print(
        f"  Empty    : "
        f"{len(empty_images)}"
    )

    print(
        f"  Occupied : "
        f"{len(occupied_images)}"
    )

    required_per_class = (
        TEST_IMAGES_PER_CLASS
        + TRAIN_IMAGES_PER_CLASS
    )

    if len(empty_images) < required_per_class:

        raise RuntimeError(
            f"Not enough Empty images.\n"
            f"Required: {required_per_class}\n"
            f"Found: {len(empty_images)}"
        )

    if len(occupied_images) < required_per_class:

        raise RuntimeError(
            f"Not enough Occupied images.\n"
            f"Required: {required_per_class}\n"
            f"Found: {len(occupied_images)}"
        )

    # --------------------------------------------------------
    # Reproducible selection
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    selected_empty = rng.sample(
        empty_images,
        required_per_class
    )

    selected_occupied = rng.sample(
        occupied_images,
        required_per_class
    )

    # --------------------------------------------------------
    # Test set
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

    test_paths = (
        test_empty +
        test_occupied
    )

    test_labels = (
        [0] * len(test_empty) +
        [1] * len(test_occupied)
    )

    # --------------------------------------------------------
    # Training set
    # --------------------------------------------------------

    train_paths = (
        train_empty +
        train_occupied
    )

    train_labels = (
        [0] * len(train_empty) +
        [1] * len(train_occupied)
    )

    # --------------------------------------------------------
    # Shuffle training set
    # --------------------------------------------------------

    train_combined = list(
        zip(
            train_paths,
            train_labels
        )
    )

    rng.shuffle(
        train_combined
    )

    train_paths, train_labels = zip(
        *train_combined
    )

    train_paths = list(
        train_paths
    )

    train_labels = np.array(
        train_labels,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Shuffle test set
    # --------------------------------------------------------

    test_combined = list(
        zip(
            test_paths,
            test_labels
        )
    )

    rng.shuffle(
        test_combined
    )

    test_paths, test_labels = zip(
        *test_combined
    )

    test_paths = list(
        test_paths
    )

    test_labels = np.array(
        test_labels,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n========================================")
    print("Dataset split")
    print("========================================")

    print("\nTraining:")

    print(
        f"  Total    : {len(train_paths)}"
    )

    print(
        f"  Empty    : "
        f"{np.sum(train_labels == 0)}"
    )

    print(
        f"  Occupied : "
        f"{np.sum(train_labels == 1)}"
    )

    print("\nFinal test:")

    print(
        f"  Total    : {len(test_paths)}"
    )

    print(
        f"  Empty    : "
        f"{np.sum(test_labels == 0)}"
    )

    print(
        f"  Occupied : "
        f"{np.sum(test_labels == 1)}"
    )

    return (
        train_paths,
        train_labels,
        test_paths,
        test_labels,
    )


# ============================================================
# Load dataset images
# ============================================================

def load_images(
    image_paths,
    labels
):

    X = []
    valid_labels = []

    print(
        f"\nLoading "
        f"{len(image_paths)} images..."
    )

    for i, image_path in enumerate(
        image_paths
    ):

        try:

            img = Image.open(
                image_path
            )

            img = img.convert("L")

            img = img.resize(
                (IMG_SIZE, IMG_SIZE),
                Image.Resampling.LANCZOS
            )

            img_array = np.array(
                img,
                dtype=np.float32
            )

            img_array /= 255.0

            X.append(
                img_array
            )

            valid_labels.append(
                labels[i]
            )

        except Exception as e:

            print(
                "\nWARNING: Could not load:"
            )

            print(
                f"  {image_path}"
            )

            print(
                f"  Reason: {e}"
            )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        valid_labels,
        dtype=np.int64
    )

    print(
        f"Successfully loaded "
        f"{len(X)} images."
    )

    return X, y


# ============================================================
# Load BoardImage.jpg
# ============================================================

def load_board_image():

    print("\n========================================")
    print("Loading BoardImage.jpg")
    print("========================================")

    print(
        f"Path:\n{BOARD_IMAGE_PATH}"
    )

    if not os.path.exists(
        BOARD_IMAGE_PATH
    ):

        raise FileNotFoundError(
            "\nBoardImage.jpg was not found.\n\n"
            "Put an image named:\n"
            "    BoardImage.jpg\n\n"
            "in the same folder as train_model.py.\n\n"
            f"Expected location:\n"
            f"{BOARD_IMAGE_PATH}"
        )

    try:

        img = Image.open(
            BOARD_IMAGE_PATH
        )

        print(
            f"Original image size: "
            f"{img.size}"
        )

        # Convert to grayscale.
        img = img.convert("L")

        # Resize exactly like PKLot images.
        img = img.resize(
            (IMG_SIZE, IMG_SIZE),
            Image.Resampling.LANCZOS
        )

        # Convert to NumPy.
        image_array = np.array(
            img,
            dtype=np.float32
        )

        # Normalize.
        image_array /= 255.0

    except Exception as e:

        raise RuntimeError(
            f"Could not load BoardImage.jpg:\n"
            f"{e}"
        )

    print(
        "Board image converted to "
        "64x64 grayscale."
    )

    return image_array


# ============================================================
# Downsample
# ============================================================

def downsample(
    images,
    size=DOWNSAMPLE
):

    if IMG_SIZE % size != 0:

        raise ValueError(
            f"IMG_SIZE ({IMG_SIZE}) must be "
            f"divisible by DOWNSAMPLE ({size})."
        )

    n = images.shape[0]

    block = IMG_SIZE // size

    reshaped = images.reshape(
        n,
        size,
        block,
        size,
        block
    )

    result = reshaped.mean(
        axis=(2, 4)
    )

    return result.astype(
        np.float32
    )


# ============================================================
# Downsample single BoardImage
# ============================================================

def downsample_board_image(
    image
):

    image_batch = np.expand_dims(
        image,
        axis=0
    )

    small = downsample(
        image_batch
    )

    return small[0]


# ============================================================
# Build model
# ============================================================

def build_model(
    input_dim
):

    model = keras.Sequential([

        layers.Input(
            shape=(input_dim,)
        ),

        layers.Dense(
            16,
            activation="relu"
        ),

        layers.Dense(
            2,
            activation="softmax"
        ),
    ])

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"],
    )

    return model


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n========================================")
    print("Final evaluation")
    print("========================================")

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy_value = accuracy_score(
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

    print("\nFinal test metrics")
    print("----------------------------------------")

    print(
        f"Test loss       : "
        f"{loss:.4f}"
    )

    print(
        f"Accuracy        : "
        f"{accuracy_value * 100:.2f}%"
    )

    print(
        f"Precision       : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall          : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 score        : "
        f"{f1 * 100:.2f}%"
    )

    print("\nConfusion matrix")

    print(
        "                 Predicted"
    )

    print(
        "                 Vacant  Occupied"
    )

    print(
        f"Actual Vacant   "
        f"{cm[0, 0]:7d}"
        f"{cm[0, 1]:10d}"
    )

    print(
        f"Actual Occupied "
        f"{cm[1, 0]:7d}"
        f"{cm[1, 1]:10d}"
    )

    print("\nClassification report")

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

    return (
        probabilities,
        predictions
    )


# ============================================================
# Test BoardImage.jpg
# ============================================================

def test_board_image(
    model
):

    print("\n========================================")
    print("Testing BoardImage.jpg")
    print("========================================")

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    board_raw = load_board_image()

    print(
        f"64x64 shape: "
        f"{board_raw.shape}"
    )

    # --------------------------------------------------------
    # Downsample
    # --------------------------------------------------------

    board_small = downsample_board_image(
        board_raw
    )

    print(
        f"16x16 shape: "
        f"{board_small.shape}"
    )

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    board_input = board_small.reshape(
        256
    ).astype(
        np.float32
    )

    print(
        f"Flattened shape: "
        f"{board_input.shape}"
    )

    # --------------------------------------------------------
    # Run TensorFlow prediction
    # --------------------------------------------------------

    probabilities = model.predict(
        np.expand_dims(
            board_input,
            axis=0
        ),
        verbose=0
    )[0]

    prediction = int(
        np.argmax(
            probabilities
        )
    )

    label_names = {
        0: "VACANT",
        1: "OCCUPIED",
    }

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("\nBoardImage result")
    print("----------------------------------------")

    print(
        f"Prediction : "
        f"{label_names[prediction]}"
    )

    print(
        f"Vacant     : "
        f"{probabilities[0]:.6f}"
    )

    print(
        f"Occupied   : "
        f"{probabilities[1]:.6f}"
    )

    return (
        board_input,
        prediction,
        probabilities
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Seeds
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    keras.utils.set_random_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    (
        train_paths,
        train_labels,
        test_paths,
        test_labels,
    ) = find_dataset_images(
        DATASET_ROOT
    )

    # --------------------------------------------------------
    # Load training images
    # --------------------------------------------------------

    X_train_raw, y_train = load_images(
        train_paths,
        train_labels
    )

    # --------------------------------------------------------
    # Load final test images
    #
    # These 1000 images are never used by model.fit().
    # --------------------------------------------------------

    X_test_raw, y_test = load_images(
        test_paths,
        test_labels
    )

    # --------------------------------------------------------
    # Downsample
    # --------------------------------------------------------

    X_train_small = downsample(
        X_train_raw
    )

    X_test_small = downsample(
        X_test_raw
    )

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    X_train = X_train_small.reshape(
        len(X_train_small),
        -1
    )

    X_test = X_test_small.reshape(
        len(X_test_small),
        -1
    )

    print("\n========================================")
    print("Neural network input")
    print("========================================")

    print(
        f"Training : "
        f"{X_train.shape}"
    )

    print(
        f"Testing  : "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # Free unnecessary arrays
    # --------------------------------------------------------

    del X_train_raw
    del X_test_raw
    del X_train_small
    del X_test_small

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        X_train.shape[1]
    )

    print("\n========================================")
    print("Model")
    print("========================================")

    model.summary()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n========================================")
    print("Training")
    print("========================================")

    model.fit(

        X_train,
        y_train,

        validation_split=VALIDATION_SPLIT,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        shuffle=True,

        verbose=1,
    )

    # --------------------------------------------------------
    # Evaluate ONLY on the 1000 unseen images
    # --------------------------------------------------------

    (
        probabilities,
        predictions
    ) = evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    model_path = os.path.join(
        SCRIPT_DIR,
        "parking_model.keras"
    )

    model.save(
        model_path
    )

    print(
        f"\nSaved model:\n"
        f"{model_path}"
    )

    # --------------------------------------------------------
    # Test BoardImage.jpg separately
    #
    # No NumPy files are created here.
    #
    # quantize_and_export.py will independently load
    # BoardImage.jpg and perform the same preprocessing
    # before generating BOARD_IMAGE[256].
    # --------------------------------------------------------

    test_board_image(
        model
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")

    print(
        "\nThe 1000-image test metrics above are "
        "the proper model evaluation."
    )

    print(
        "\nBoardImage.jpg was tested separately."
    )

    print(
        "\nThe BoardImage NumPy files are NOT "
        "generated."
    )

    print("\nFiles generated:")

    print(
        "  parking_model.keras"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
