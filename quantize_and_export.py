"""
quantize_and_export.py  (CNN version)

Loads the trained CNN (parking_model.keras), quantizes every layer
to int8, derives INTEGER requantization constants for each layer
boundary, preprocesses BoardImage.jpg exactly like training, and
writes model_weights.h.

IMPORTANT:
    BoardImage.jpg is NOT used to train the CNN.

    It is used here for:
      1. Activation calibration
      2. Creating the quantized BOARD_IMAGE array
      3. Checking the trained Keras model's prediction

The calibration can also use additional images placed in:

    calibration_images/

next to this script.

Recommended:
    20-50 representative Empty/Occupied images.

With only BoardImage.jpg available, the script still works, but
the activation calibration is based on a single image.

OUTPUT:
    model_weights.h containing:

    - int8 weights for conv1, conv2, fc1, fc2
    - int32 biases
    - integer requantization multipliers and shifts
    - quantized BoardImage as BOARD_IMAGE[16][16]
"""


import os
import glob

import numpy as np
from PIL import Image
from tensorflow import keras


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "parking_model.keras"
)

BOARD_IMAGE_PATH = os.path.join(
    SCRIPT_DIR,
    "BoardImage.jpg"
)

CALIB_DIR = os.path.join(
    SCRIPT_DIR,
    "calibration_images"
)

OUT_HEADER = os.path.join(
    SCRIPT_DIR,
    "model_weights.h"
)


# ------------------------------------------------------------
# Image dimensions
# ------------------------------------------------------------

IMG_SIZE = 64
DOWNSAMPLE = 16

POOL_FACTOR = (
    IMG_SIZE // DOWNSAMPLE
)


# ------------------------------------------------------------
# Input quantization
# ------------------------------------------------------------

INPUT_SCALE = 1.0 / 127.0


# ============================================================
# Preprocessing
# Must exactly match train_model.py
# ============================================================

def preprocess_image(path):
    """
    Load an image and convert it to the same 16x16 format
    used during CNN training.

    Pipeline:

        image
          ↓
        grayscale
          ↓
        64x64 LANCZOS
          ↓
        normalize /255
          ↓
        4x4 average pooling
          ↓
        16x16
    """

    img = Image.open(
        path
    ).convert("L")

    img = img.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.LANCZOS
    )

    arr = (
        np.array(
            img,
            dtype=np.float32
        )
        / 255.0
    )

    reshaped = arr.reshape(
        DOWNSAMPLE,
        POOL_FACTOR,
        DOWNSAMPLE,
        POOL_FACTOR
    )

    small = reshaped.mean(
        axis=(1, 3)
    )

    return small


# ============================================================
# Input quantization
# ============================================================

def quantize_input(image_16x16):
    """
    Convert floating-point 16x16 image values into int8.

    Floating-point range:

        0.0 -> 1.0

    Integer range:

        0 -> 127
    """

    q = np.round(
        image_16x16 / INPUT_SCALE
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
# Per-tensor symmetric int8 weight quantization
# ============================================================

def quantize_tensor(tensor):
    """
    Symmetric int8 quantization.

    Returns:

        quantized_tensor
        scale
    """

    max_val = float(
        np.max(
            np.abs(tensor)
        )
    )

    if max_val == 0:
        max_val = 1e-8

    scale = (
        max_val / 127.0
    )

    q = np.round(
        tensor / scale
    )

    q = np.clip(
        q,
        -127,
        127
    ).astype(
        np.int8
    )

    return q, scale


# ============================================================
# Integer requantization multiplier
# ============================================================

def compute_multiplier_shift(
    real_multiplier,
    shift_bits=20
):
    """
    Calculate integer constants so that:

        (acc * M0) >> shift

    approximates:

        acc * real_multiplier

    Returns:

        M0
        shift
    """

    shift = shift_bits

    m0 = round(
        real_multiplier
        * (1 << shift)
    )

    # Keep M0 safely below signed 32-bit range.
    while (
        abs(m0) > (1 << 30)
        and shift > 0
    ):
        shift -= 1

        m0 = round(
            real_multiplier
            * (1 << shift)
        )

    return (
        int(m0),
        int(shift)
    )


# ============================================================
# Calibration image discovery
# ============================================================

def get_calibration_images():

    print(
        "\n[CALIBRATION] Finding calibration images..."
    )

    paths = []

    # --------------------------------------------------------
    # BoardImage.jpg is always included
    # --------------------------------------------------------

    if os.path.exists(
        BOARD_IMAGE_PATH
    ):
        paths.append(
            BOARD_IMAGE_PATH
        )

    else:
        raise FileNotFoundError(
            "BoardImage.jpg was not found:\n"
            f"{BOARD_IMAGE_PATH}"
        )

    # --------------------------------------------------------
    # Optional calibration directory
    # --------------------------------------------------------

    if os.path.isdir(
        CALIB_DIR
    ):

        paths += sorted(
            glob.glob(
                os.path.join(
                    CALIB_DIR,
                    "*.jpg"
                )
            )
        )

        paths += sorted(
            glob.glob(
                os.path.join(
                    CALIB_DIR,
                    "*.jpeg"
                )
            )
        )

        paths += sorted(
            glob.glob(
                os.path.join(
                    CALIB_DIR,
                    "*.png"
                )
            )
        )

        paths += sorted(
            glob.glob(
                os.path.join(
                    CALIB_DIR,
                    "*.bmp"
                )
            )
        )

    print(
        f"[CALIBRATION] Images found: {len(paths)}"
    )

    for path in paths:
        print(
            f"                {os.path.basename(path)}"
        )

    imgs = np.stack(
        [
            preprocess_image(path)
            for path in paths
        ],
        axis=0
    )

    return imgs.reshape(
        -1,
        DOWNSAMPLE,
        DOWNSAMPLE,
        1
    ).astype(
        np.float32
    )


# ============================================================
# Activation calibration
# ============================================================

def calibrate_activation_scale(
    model,
    layer_name,
    calib_batch
):
    """
    Run calibration images through the requested intermediate
    Keras layer and calculate a symmetric int8 activation scale.

    IMPORTANT KERAS FIX:

    Older code used:

        model.input

    With newer Keras versions, a loaded Sequential model can
    report that model.input is undefined.

    We therefore use:

        model.inputs

    which correctly references the model's input tensor.
    """

    layer = model.get_layer(
        layer_name
    )

    sub = keras.Model(
        inputs=model.inputs,
        outputs=layer.output
    )

    acts = sub.predict(
        calib_batch,
        verbose=0
    )

    max_val = float(
        np.max(
            np.abs(acts)
        )
    )

    if max_val == 0:
        max_val = 1e-8

    scale = (
        max_val / 127.0
    )

    print(
        f"      {layer_name:<6} "
        f"max activation = {max_val:.6f}   "
        f"scale = {scale:.10f}"
    )

    return scale


# ============================================================
# C array formatting helpers
# ============================================================

def fmt_1d(
    name,
    arr,
    dtype="int8_t"
):

    vals = ", ".join(
        str(int(v))
        for v in arr
    )

    return (
        f"static const {dtype} "
        f"{name}[{len(arr)}] = "
        f"{{{vals}}};\n"
    )


def fmt_conv_weights(
    name,
    w
):
    """
    w shape:

        (out_ch, in_ch, kh, kw)
    """

    (
        out_ch,
        in_ch,
        kh,
        kw
    ) = w.shape

    lines = [
        f"static const int8_t "
        f"{name}"
        f"[{out_ch}]"
        f"[{in_ch}]"
        f"[{kh}]"
        f"[{kw}] = {{"
    ]

    for o in range(
        out_ch
    ):

        lines.append(
            "  {"
        )

        for c in range(
            in_ch
        ):

            row = ", ".join(
                str(int(v))
                for v in w[o, c].flatten()
            )

            lines.append(
                f"    {{{row}}},"
            )

        lines.append(
            "  },"
        )

    lines.append(
        "};\n"
    )

    return "\n".join(
        lines
    )


def fmt_dense_weights(
    name,
    w
):
    """
    w shape:

        (out, in)
    """

    (
        out_n,
        in_n
    ) = w.shape

    lines = [
        f"static const int8_t "
        f"{name}"
        f"[{out_n}]"
        f"[{in_n}] = {{"
    ]

    for o in range(
        out_n
    ):

        row = ", ".join(
            str(int(v))
            for v in w[o]
        )

        lines.append(
            f"  {{{row}}},"
        )

    lines.append(
        "};\n"
    )

    return "\n".join(
        lines
    )


def fmt_board_image(
    name,
    img16
):

    lines = [
        f"static const int8_t "
        f"{name}[16][16] = {{"
    ]

    for r in range(16):

        row = ", ".join(
            str(int(v))
            for v in img16[r]
        )

        lines.append(
            f"  {{{row}}},"
        )

    lines.append(
        "};\n"
    )

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 64)
    print("              CNN INT8 QUANTIZATION")
    print("=" * 64)

    # ========================================================
    # Step 1: Load model
    # ========================================================

    print("\n[1/7] LOADING TRAINED MODEL")

    if not os.path.exists(
        MODEL_PATH
    ):
        raise FileNotFoundError(
            "Trained model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run train_model.py first."
        )

    print(
        f"      Model: "
        f"{os.path.basename(MODEL_PATH)}"
    )

    model = keras.models.load_model(
        MODEL_PATH
    )

    print(
        "      Model loaded successfully."
    )

    # --------------------------------------------------------
    # Display model input information
    # --------------------------------------------------------

    print(
        f"      Input shape: "
        f"{model.inputs[0].shape}"
    )

    # ========================================================
    # Step 2: Get layers
    # ========================================================

    print("\n[2/7] READING MODEL WEIGHTS")

    conv1 = model.get_layer(
        "conv1"
    )

    conv2 = model.get_layer(
        "conv2"
    )

    fc1 = model.get_layer(
        "fc1"
    )

    fc2 = model.get_layer(
        "fc2"
    )

    print(
        "      conv1  -> Conv2D"
    )

    print(
        "      conv2  -> Conv2D"
    )

    print(
        "      fc1    -> Dense"
    )

    print(
        "      fc2    -> Dense"
    )

    # ========================================================
    # Step 3: Convert Keras weight layouts to C layouts
    # ========================================================

    print(
        "\n[3/7] CONVERTING WEIGHT LAYOUTS"
    )

    # --------------------------------------------------------
    # Conv2D
    #
    # Keras:
    #
    #   (kh, kw, in_ch, out_ch)
    #
    # C:
    #
    #   (out_ch, in_ch, kh, kw)
    # --------------------------------------------------------

    w_conv1, b_conv1 = (
        conv1.get_weights()
    )

    w_conv1_c = np.transpose(
        w_conv1,
        (3, 2, 0, 1)
    )

    w_conv2, b_conv2 = (
        conv2.get_weights()
    )

    w_conv2_c = np.transpose(
        w_conv2,
        (3, 2, 0, 1)
    )

    # --------------------------------------------------------
    # Dense
    #
    # Keras:
    #
    #   (in, out)
    #
    # C:
    #
    #   (out, in)
    # --------------------------------------------------------

    w_fc1, b_fc1 = (
        fc1.get_weights()
    )

    w_fc1_c = w_fc1.T

    w_fc2, b_fc2 = (
        fc2.get_weights()
    )

    w_fc2_c = w_fc2.T

    print(
        "      Weight layouts converted."
    )

    # ========================================================
    # Step 4: Quantize weights
    # ========================================================

    print(
        "\n[4/7] QUANTIZING WEIGHTS TO INT8"
    )

    q_w_conv1, s_w_conv1 = (
        quantize_tensor(
            w_conv1_c
        )
    )

    q_w_conv2, s_w_conv2 = (
        quantize_tensor(
            w_conv2_c
        )
    )

    q_w_fc1, s_w_fc1 = (
        quantize_tensor(
            w_fc1_c
        )
    )

    q_w_fc2, s_w_fc2 = (
        quantize_tensor(
            w_fc2_c
        )
    )

    print(
        f"      conv1 scale: "
        f"{s_w_conv1:.10f}"
    )

    print(
        f"      conv2 scale: "
        f"{s_w_conv2:.10f}"
    )

    print(
        f"      fc1   scale: "
        f"{s_w_fc1:.10f}"
    )

    print(
        f"      fc2   scale: "
        f"{s_w_fc2:.10f}"
    )

    # ========================================================
    # Step 5: Activation calibration
    # ========================================================

    print(
        "\n[5/7] CALIBRATING ACTIVATION SCALES"
    )

    print(
        "\n      Calibration uses:"
    )

    print(
        "      - BoardImage.jpg"
    )

    if os.path.isdir(
        CALIB_DIR
    ):
        print(
            "      - Additional images "
            "from calibration_images/"
        )
    else:
        print(
            "      - No calibration_images/ "
            "folder found"
        )

    calib_batch = (
        get_calibration_images()
    )

    print(
        "\n      Activation ranges:"
    )

    s_act_conv1 = (
        calibrate_activation_scale(
            model,
            "conv1",
            calib_batch
        )
    )

    s_act_conv2 = (
        calibrate_activation_scale(
            model,
            "conv2",
            calib_batch
        )
    )

    s_act_fc1 = (
        calibrate_activation_scale(
            model,
            "fc1",
            calib_batch
        )
    )

    # ========================================================
    # Step 6: Calculate biases and requantization
    # ========================================================

    print(
        "\n[6/7] CALCULATING INTEGER PARAMETERS"
    )

    # --------------------------------------------------------
    # Biases
    #
    # Bias is converted into the corresponding accumulator
    # units for each layer.
    # --------------------------------------------------------

    # Layer 1:
    #
    # accumulator units:
    #
    # INPUT_SCALE * s_w_conv1

    b1_acc = np.round(
        b_conv1
        / (
            INPUT_SCALE
            * s_w_conv1
        )
    ).astype(
        np.int32
    )

    # Layer 2:
    #
    # accumulator units:
    #
    # s_act_conv1 * s_w_conv2

    b2_acc = np.round(
        b_conv2
        / (
            s_act_conv1
            * s_w_conv2
        )
    ).astype(
        np.int32
    )

    # FC1:
    #
    # accumulator units:
    #
    # s_act_conv2 * s_w_fc1

    b_fc1_acc = np.round(
        b_fc1
        / (
            s_act_conv2
            * s_w_fc1
        )
    ).astype(
        np.int32
    )

    # FC2:
    #
    # accumulator units:
    #
    # s_act_fc1 * s_w_fc2

    b_fc2_acc = np.round(
        b_fc2
        / (
            s_act_fc1
            * s_w_fc2
        )
    ).astype(
        np.int32
    )

    # --------------------------------------------------------
    # Requantization
    # --------------------------------------------------------

    # Conv1 accumulator
    #
    # INPUT_SCALE * s_w_conv1
    #
    # ->
    #
    # s_act_conv1

    m1, sh1 = (
        compute_multiplier_shift(
            (
                INPUT_SCALE
                * s_w_conv1
            )
            / s_act_conv1
        )
    )

    # Conv2 accumulator
    #
    # s_act_conv1 * s_w_conv2
    #
    # ->
    #
    # s_act_conv2

    m2, sh2 = (
        compute_multiplier_shift(
            (
                s_act_conv1
                * s_w_conv2
            )
            / s_act_conv2
        )
    )

    # FC1 accumulator
    #
    # s_act_conv2 * s_w_fc1
    #
    # ->
    #
    # s_act_fc1

    m3, sh3 = (
        compute_multiplier_shift(
            (
                s_act_conv2
                * s_w_fc1
            )
            / s_act_fc1
        )
    )

    print(
        "\n      Requantization constants:"
    )

    print(
        f"      Conv1 -> Conv2: "
        f"M={m1}, Shift={sh1}"
    )

    print(
        f"      Conv2 -> FC1:   "
        f"M={m2}, Shift={sh2}"
    )

    print(
        f"      FC1 -> FC2:     "
        f"M={m3}, Shift={sh3}"
    )

    # ========================================================
    # BoardImage
    # ========================================================

    print(
        "\n      Processing BoardImage.jpg..."
    )

    board16 = preprocess_image(
        BOARD_IMAGE_PATH
    )

    board_q = quantize_input(
        board16
    )

    print(
        "      Board image converted "
        "to 16x16 int8."
    )

    # ========================================================
    # Keras prediction sanity check
    # ========================================================

    print(
        "\n      Checking Keras prediction..."
    )

    board_batch = (
        board16
        .reshape(
            1,
            16,
            16,
            1
        )
        .astype(
            np.float32
        )
    )

    keras_probs = model.predict(
        board_batch,
        verbose=0
    )[0]

    keras_pred = int(
        np.argmax(
            keras_probs
        )
    )

    label = (
        "OCCUPIED"
        if keras_pred == 1
        else "VACANT"
    )

    print(
        f"\n      Keras prediction:"
    )

    print(
        f"      Result:   {label}"
    )

    print(
        f"      Vacant:   "
        f"{keras_probs[0] * 100:.2f}%"
    )

    print(
        f"      Occupied: "
        f"{keras_probs[1] * 100:.2f}%"
    )

    # ========================================================
    # Step 7: Write model_weights.h
    # ========================================================

    print(
        "\n[7/7] WRITING model_weights.h"
    )

    with open(
        OUT_HEADER,
        "w"
    ) as f:

        f.write(
            "/* AUTO-GENERATED by "
            "quantize_and_export.py. "
            "Do not edit by hand. */\n"
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
        # Input scale
        # ----------------------------------------------------

        f.write(
            f"#define INPUT_SCALE "
            f"{INPUT_SCALE:.10f}f\n\n"
        )

        # ----------------------------------------------------
        # Conv1
        # ----------------------------------------------------

        f.write(
            fmt_conv_weights(
                "CONV1_W",
                q_w_conv1
            )
        )

        f.write(
            fmt_1d(
                "CONV1_B",
                b1_acc,
                dtype="int32_t"
            )
        )

        f.write(
            f"#define "
            f"CONV1_REQUANT_MULT "
            f"{m1}\n"
        )

        f.write(
            f"#define "
            f"CONV1_REQUANT_SHIFT "
            f"{sh1}\n\n"
        )

        # ----------------------------------------------------
        # Conv2
        # ----------------------------------------------------

        f.write(
            fmt_conv_weights(
                "CONV2_W",
                q_w_conv2
            )
        )

        f.write(
            fmt_1d(
                "CONV2_B",
                b2_acc,
                dtype="int32_t"
            )
        )

        f.write(
            f"#define "
            f"CONV2_REQUANT_MULT "
            f"{m2}\n"
        )

        f.write(
            f"#define "
            f"CONV2_REQUANT_SHIFT "
            f"{sh2}\n\n"
        )

        # ----------------------------------------------------
        # FC1
        # ----------------------------------------------------

        f.write(
            fmt_dense_weights(
                "FC1_W",
                q_w_fc1
            )
        )

        f.write(
            fmt_1d(
                "FC1_B",
                b_fc1_acc,
                dtype="int32_t"
            )
        )

        f.write(
            f"#define "
            f"FC1_REQUANT_MULT "
            f"{m3}\n"
        )

        f.write(
            f"#define "
            f"FC1_REQUANT_SHIFT "
            f"{sh3}\n\n"
        )

        # ----------------------------------------------------
        # FC2
        # ----------------------------------------------------

        f.write(
            fmt_dense_weights(
                "FC2_W",
                q_w_fc2
            )
        )

        f.write(
            fmt_1d(
                "FC2_B",
                b_fc2_acc,
                dtype="int32_t"
            )
        )

        f.write(
            "\n"
        )

        # ----------------------------------------------------
        # BoardImage
        # ----------------------------------------------------

        f.write(
            fmt_board_image(
                "BOARD_IMAGE",
                board_q
            )
        )

        f.write(
            "\n#endif "
            "/* MODEL_WEIGHTS_H */\n"
        )

    # ========================================================
    # Finished
    # ========================================================

    print(
        "\n" + "=" * 64
    )

    print(
        "             QUANTIZATION COMPLETE"
    )

    print(
        "=" * 64
    )

    print(
        f"\nGenerated:"
    )

    print(
        f"    {OUT_HEADER}"
    )

    print(
        "\nContains:"
    )

    print(
        "    - Conv1 int8 weights"
    )

    print(
        "    - Conv2 int8 weights"
    )

    print(
        "    - FC1 int8 weights"
    )

    print(
        "    - FC2 int8 weights"
    )

    print(
        "    - int32 biases"
    )

    print(
        "    - requantization multipliers"
    )

    print(
        "    - requantization shifts"
    )

    print(
        "    - quantized BoardImage"
    )

    print(
        "\nNext step:"
    )

    print(
        "    Copy model_weights.h into "
        "your CCS/MSP430 project."
    )


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":
    main()
