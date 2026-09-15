"""
quantize_and_export.py

Converts the trained Keras model into int8 C arrays for the
MSP430F5529.

Input:

    parking_model.keras
    BoardImage.jpg

Output:

    model_weights.h

BoardImage.jpg preprocessing is exactly the same as in
train_model.py:

    BoardImage.jpg
          ↓
      grayscale
          ↓
        64x64
          ↓
        16x16
          ↓
       256 float32
          ↓
       fixed int8
          ↓
     BOARD_IMAGE[256]


Quantization:

    real_value ~= int8_value * scale

The neural-network weights use symmetric per-tensor
quantization.

The input uses a FIXED scale:

    INPUT_SCALE = 1 / 127

because the training pipeline normalizes pixels to [0,1].

Therefore:

    0.0 -> 0
    0.5 -> ~64
    1.0 -> 127
"""


import os

import numpy as np
from PIL import Image
from tensorflow import keras


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_FILE = os.path.join(
    SCRIPT_DIR,
    "parking_model.keras"
)

# BoardImage.jpg is located in the same
# folder as this script.
BOARD_IMAGE_FILE = os.path.join(
    SCRIPT_DIR,
    "BoardImage.jpg"
)

OUTPUT_FILE = os.path.join(
    SCRIPT_DIR,
    "model_weights.h"
)

# ------------------------------------------------------------
# Image processing
# ------------------------------------------------------------

IMG_SIZE = 64
DOWNSAMPLE = 16

# Fixed input scale because training inputs
# are normalized to [0,1].
INPUT_SCALE = 1.0 / 127.0


# ============================================================
# Weight quantization
# ============================================================

def quantize_tensor(tensor):
    """
    Symmetric per-tensor int8 quantization.

    real_value ~= int8_value * scale
    """

    max_val = np.max(
        np.abs(tensor)
    )

    if max_val > 0:

        scale = (
            max_val / 127.0
        )

    else:

        scale = 1.0

    q = np.round(
        tensor / scale
    )

    q = np.clip(
        q,
        -127,
        127
    )

    q = q.astype(
        np.int8
    )

    return q, scale


# ============================================================
# Fixed input quantization
# ============================================================

def quantize_input(sample):
    """
    Quantizes a [0,1] float input using the fixed
    deployment scale:

        INPUT_SCALE = 1/127

    Therefore:

        q = round(real / INPUT_SCALE)

    and:

        real ~= q * INPUT_SCALE
    """

    q = np.round(
        sample / INPUT_SCALE
    )

    q = np.clip(
        q,
        -127,
        127
    )

    return q.astype(
        np.int8
    )


# ============================================================
# Load BoardImage.jpg
# ============================================================

def load_board_image():
    """
    Loads BoardImage.jpg and performs exactly the
    same preprocessing used by train_model.py:

        BoardImage.jpg
              ↓
          grayscale
              ↓
            64x64
              ↓
          normalize [0,1]

    Returns:

        float32 array with shape (64, 64)
    """

    print("\n========================================")
    print("Loading BoardImage.jpg")
    print("========================================")

    print(
        f"Path:\n{BOARD_IMAGE_FILE}"
    )

    if not os.path.exists(
        BOARD_IMAGE_FILE
    ):

        raise FileNotFoundError(
            "\nBoardImage.jpg was not found.\n\n"
            "Put an image named:\n"
            "    BoardImage.jpg\n\n"
            "in the same folder as "
            "quantize_and_export.py.\n\n"
            f"Expected location:\n"
            f"{BOARD_IMAGE_FILE}"
        )

    try:

        img = Image.open(
            BOARD_IMAGE_FILE
        )

        print(
            f"Original image size: "
            f"{img.size}"
        )

        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        img = img.convert("L")

        # ----------------------------------------------------
        # Resize to 64x64
        #
        # ----------------------------------------------------

        img = img.resize(
            (IMG_SIZE, IMG_SIZE),
            Image.Resampling.LANCZOS
        )

        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        image_array = np.array(
            img,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Normalize 0-255 -> 0-1
        #
        # ----------------------------------------------------

        image_array /= 255.0

    except Exception as e:

        raise RuntimeError(
            f"Could not load BoardImage.jpg:\n"
            f"{e}"
        )

    print(
        "BoardImage converted to "
        "64x64 grayscale."
    )

    print(
        f"Image shape: "
        f"{image_array.shape}"
    )

    print(
        f"Pixel range: "
        f"{image_array.min():.6f} "
        f"to "
        f"{image_array.max():.6f}"
    )

    return image_array


# ============================================================
# Downsample
# ============================================================

def downsample(
    images,
    size=DOWNSAMPLE
):
    """
    Performs exactly the same 64x64 -> 16x16
    average pooling used by train_model.py.

    Each output pixel is the average of
    a 4x4 block.
    """

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
# Prepare BoardImage for the MSP430
# ============================================================

def prepare_board_image():
    """
    Performs the complete BoardImage preprocessing
    used during training.

    Pipeline:

        BoardImage.jpg
              ↓
          grayscale
              ↓
            64x64
              ↓
            16x16
              ↓
          flatten
              ↓
          256 values
    """

    # --------------------------------------------------------
    # Load and normalize image
    # --------------------------------------------------------

    board_raw = load_board_image()

    # --------------------------------------------------------
    # Add batch dimension
    #
    # 64x64
    #
    # becomes:
    #
    # 1x64x64
    # --------------------------------------------------------

    board_batch = np.expand_dims(
        board_raw,
        axis=0
    )

    # --------------------------------------------------------
    # Average-pool 64x64 -> 16x16
    # --------------------------------------------------------

    board_small = downsample(
        board_batch
    )

    # Remove batch dimension.
    board_small = board_small[0]

    print(
        f"\nAfter average pooling: "
        f"{board_small.shape}"
    )

    # --------------------------------------------------------
    # Flatten 16x16 -> 256
    # --------------------------------------------------------

    board_input = board_small.reshape(
        256
    ).astype(
        np.float32
    )

    print(
        f"Flattened input shape: "
        f"{board_input.shape}"
    )

    print(
        f"Input range: "
        f"{board_input.min():.6f} "
        f"to "
        f"{board_input.max():.6f}"
    )

    return board_input


# ============================================================
# Convert NumPy array to C
# ============================================================

def array_to_c(
    name,
    arr,
    dtype="int8_t"
):

    flat = arr.flatten()

    values = ", ".join(
        str(int(v))
        for v in flat
    )

    if arr.ndim > 1:

        dims = "".join(
            f"[{d}]"
            for d in arr.shape
        )

    else:

        dims = (
            f"[{arr.shape[0]}]"
        )

    return (
        f"const {dtype} "
        f"{name}{dims} = "
        f"{{{values}}};\n\n"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n========================================")
    print("Quantizing parking model")
    print("========================================")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{MODEL_FILE}\n\n"
            "Run train_model.py first."
        )

    # --------------------------------------------------------
    # Check BoardImage
    # --------------------------------------------------------

    if not os.path.exists(
        BOARD_IMAGE_FILE
    ):

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{BOARD_IMAGE_FILE}\n\n"
            "Place BoardImage.jpg in the "
            "same folder as this script."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        f"Loading model:\n"
        f"{MODEL_FILE}"
    )

    model = keras.models.load_model(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Get weights
    # --------------------------------------------------------

    w1, b1 = (
        model.layers[0].get_weights()
    )

    w2, b2 = (
        model.layers[1].get_weights()
    )

    print("\nOriginal weights:")

    print(
        f"  W1: {w1.shape}"
    )

    print(
        f"  B1: {b1.shape}"
    )

    print(
        f"  W2: {w2.shape}"
    )

    print(
        f"  B2: {b2.shape}"
    )

    # --------------------------------------------------------
    # Quantize weights
    # --------------------------------------------------------

    w1_q, w1_scale = (
        quantize_tensor(w1)
    )

    b1_q, b1_scale = (
        quantize_tensor(b1)
    )

    w2_q, w2_scale = (
        quantize_tensor(w2)
    )

    b2_q, b2_scale = (
        quantize_tensor(b2)
    )

    # --------------------------------------------------------
    # Load and preprocess BoardImage.jpg
    #
    # IMPORTANT:
    #
    # The image is now converted directly from JPEG.
    #
    # No board_image_input.npy is required.
    # --------------------------------------------------------

    sample = prepare_board_image()

    print(
        "\nBoardImage input:"
    )

    print(
        f"  Values: {sample.size}"
    )

    print(
        f"  Min:    {sample.min():.6f}"
    )

    print(
        f"  Max:    {sample.max():.6f}"
    )

    # --------------------------------------------------------
    # Quantize BoardImage
    # --------------------------------------------------------

    sample_q = quantize_input(
        sample
    )

    print(
        "\nQuantized BoardImage:"
    )

    print(
        f"  Min:    {sample_q.min()}"
    )

    print(
        f"  Max:    {sample_q.max()}"
    )

    # --------------------------------------------------------
    # Generate model_weights.h
    # --------------------------------------------------------

    print(
        f"\nWriting:\n"
        f"{OUTPUT_FILE}"
    )

    with open(
        OUTPUT_FILE,
        "w"
    ) as f:

        f.write(
            "/*\n"
            " * Auto-generated model weights.\n"
            " *\n"
            " * Generated by quantize_and_export.py\n"
            " * DO NOT HAND EDIT.\n"
            " */\n\n"
        )

        f.write(
            "#ifndef MODEL_WEIGHTS_H\n"
        )

        f.write(
            "#define MODEL_WEIGHTS_H\n\n"
        )

        f.write(
            "#include <stdint.h>\n\n"
        )

        # ----------------------------------------------------
        # Network dimensions
        # ----------------------------------------------------

        f.write(
            "#define INPUT_SIZE 256\n"
        )

        f.write(
            "#define HIDDEN_SIZE 16\n"
        )

        f.write(
            "#define OUTPUT_SIZE 2\n\n"
        )

        # ----------------------------------------------------
        # Quantization scales
        # ----------------------------------------------------

        f.write(
            f"#define INPUT_SCALE "
            f"{INPUT_SCALE:.12f}f\n"
        )

        f.write(
            f"#define W1_SCALE "
            f"{w1_scale:.12f}f\n"
        )

        f.write(
            f"#define B1_SCALE "
            f"{b1_scale:.12f}f\n"
        )

        f.write(
            f"#define W2_SCALE "
            f"{w2_scale:.12f}f\n"
        )

        f.write(
            f"#define B2_SCALE "
            f"{b2_scale:.12f}f\n\n"
        )

        # ----------------------------------------------------
        # First layer weights
        #
        # Keras:
        #
        # W1 = [256][16]
        #
        # C:
        #
        # W1 = [16][256]
        # ----------------------------------------------------

        f.write(
            array_to_c(
                "W1",
                w1_q.T
            )
        )

        # ----------------------------------------------------
        # First layer biases
        # ----------------------------------------------------

        f.write(
            array_to_c(
                "B1",
                b1_q
            )
        )

        # ----------------------------------------------------
        # Second layer weights
        #
        # Keras:
        #
        # W2 = [16][2]
        #
        # C:
        #
        # W2 = [2][16]
        # ----------------------------------------------------

        f.write(
            array_to_c(
                "W2",
                w2_q.T
            )
        )

        # ----------------------------------------------------
        # Second layer biases
        # ----------------------------------------------------

        f.write(
            array_to_c(
                "B2",
                b2_q
            )
        )

        # ----------------------------------------------------
        # BoardImage
        # ----------------------------------------------------

        f.write(
            "/*\n"
            " * BoardImage.jpg after:\n"
            " *\n"
            " *   grayscale\n"
            " *   64x64 resize\n"
            " *   16x16 average pooling\n"
            " *   flatten\n"
            " *   fixed int8 quantization\n"
            " */\n\n"
        )

        f.write(
            array_to_c(
                "BOARD_IMAGE",
                sample_q
            )
        )

        f.write(
            "#endif\n"
        )

    # --------------------------------------------------------
    # Calculate flash size
    # --------------------------------------------------------

    total_bytes = (
        w1_q.nbytes
        + b1_q.nbytes
        + w2_q.nbytes
        + b2_q.nbytes
        + sample_q.nbytes
    )

    print("\n========================================")
    print("Export complete")
    print("========================================")

    print(
        f"Generated:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nApproximate int8 data size:"
        f" {total_bytes} bytes"
    )

    print("\nContents:")

    print(
        "  W1[16][256]"
    )

    print(
        "  B1[16]"
    )

    print(
        "  W2[2][16]"
    )

    print(
        "  B2[2]"
    )

    print(
        "  BOARD_IMAGE[256]"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
