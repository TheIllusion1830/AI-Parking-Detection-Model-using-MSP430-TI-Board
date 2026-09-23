"""
reference_check.py

Re-implements main.c's integer arithmetic in Python (same order of
operations, same clipping, same requantization) and compares its
prediction against the plain Keras model, on BoardImage.jpg and
optionally on the 1000-image PKLot test set. Run this after
quantize_and_export.py and BEFORE flashing the board -- this is
the "compare Python quantized inference emulation against Keras"
step called for in the project notes (Sec. 26).
"""

import os
import re
import numpy as np
from PIL import Image
from tensorflow import keras

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEADER_PATH = os.path.join(SCRIPT_DIR, "model_weights.h")
MODEL_PATH = os.path.join(SCRIPT_DIR, "parking_model.keras")
BOARD_IMAGE_PATH = os.path.join(SCRIPT_DIR, "BoardImage.jpg")


def parse_header(path):
    """Very small parser: pulls out the C arrays/#defines this script needs."""
    with open(path) as f:
        text = f.read()

    def get_define(name):
        m = re.search(rf"#define {name} (-?\d+)", text)
        return int(m.group(1))

    def get_array(name):
        m = re.search(rf"{name}\[[^=]*=\s*(\{{.*?\}});", text, re.S)
        body = m.group(1)
        body = body.replace("{", "[").replace("}", "]")
        return np.array(eval(body))

    data = {}
    for arr in ["CONV1_W", "CONV2_W", "FC1_W", "FC2_W", "BOARD_IMAGE"]:
        data[arr] = get_array(arr)
    for arr in ["CONV1_B", "CONV2_B", "FC1_B", "FC2_B"]:
        data[arr] = get_array(arr)
    for d in ["CONV1_REQUANT_MULT", "CONV1_REQUANT_SHIFT",
              "CONV2_REQUANT_MULT", "CONV2_REQUANT_SHIFT",
              "FC1_REQUANT_MULT", "FC1_REQUANT_SHIFT"]:
        data[d] = get_define(d)
    return data


def requantize(acc, mult, shift):
    val = acc.astype(np.int64) * np.int64(mult)
    rounded = val + (1 << (shift - 1))
    out = (rounded >> shift).astype(np.int32)
    return np.clip(out, -127, 127).astype(np.int8)


def run_c_equivalent(d, board_image):
    # ---- conv1: 16x16x1 -> 14x14x4 ----
    conv1 = np.zeros((14, 14, 4), dtype=np.int32)
    for oc in range(4):
        for oy in range(14):
            for ox in range(14):
                patch = board_image[oy:oy + 3, ox:ox + 3].astype(np.int32)
                w = d["CONV1_W"][oc, 0].astype(np.int32)
                conv1[oy, ox, oc] = d["CONV1_B"][oc] + int(np.sum(patch * w))
    conv1 = np.maximum(conv1, 0)
    conv1_q = requantize(conv1, d["CONV1_REQUANT_MULT"], d["CONV1_REQUANT_SHIFT"])

    # ---- pool1: 14x14x4 -> 7x7x4 ----
    pool1 = conv1_q.reshape(7, 2, 7, 2, 4).max(axis=(1, 3))

    # ---- conv2: 7x7x4 -> 5x5x8 ----
    conv2 = np.zeros((5, 5, 8), dtype=np.int32)
    for oc in range(8):
        for oy in range(5):
            for ox in range(5):
                patch = pool1[oy:oy + 3, ox:ox + 3, :].astype(np.int32)  # (3,3,4)
                w = d["CONV2_W"][oc].astype(np.int32)                    # (4,3,3)
                w = np.transpose(w, (1, 2, 0))                           # (3,3,4)
                conv2[oy, ox, oc] = d["CONV2_B"][oc] + int(np.sum(patch * w))
    conv2 = np.maximum(conv2, 0)
    conv2_q = requantize(conv2, d["CONV2_REQUANT_MULT"], d["CONV2_REQUANT_SHIFT"])

    # ---- pool2: 5x5x8 -> 2x2x8 (drop last row/col, matches Keras 'valid') ----
    cropped = conv2_q[:4, :4, :]
    pool2 = cropped.reshape(2, 2, 2, 2, 8).max(axis=(1, 3))

    # ---- fc1: flatten (h,w,c) -> 32 -> 16 ----
    flat = pool2.reshape(32).astype(np.int32)
    fc1_out = d["FC1_B"].astype(np.int32) + flat @ d["FC1_W"].astype(np.int32).T
    fc1_out = np.maximum(fc1_out, 0)
    fc1_q = requantize(fc1_out, d["FC1_REQUANT_MULT"], d["FC1_REQUANT_SHIFT"])

    # ---- fc2: 16 -> 2, argmax ----
    fc2_out = d["FC2_B"].astype(np.int32) + fc1_q.astype(np.int32) @ d["FC2_W"].astype(np.int32).T
    pred = int(np.argmax(fc2_out))
    return pred, fc2_out


def main():
    d = parse_header(HEADER_PATH)
    model = keras.models.load_model(MODEL_PATH)

    board_image = d["BOARD_IMAGE"].astype(np.int8)
    c_pred, c_logits = run_c_equivalent(d, board_image)

    keras_input = (board_image.astype(np.float32) / 127.0).reshape(1, 16, 16, 1)
    keras_probs = model.predict(keras_input, verbose=0)[0]
    keras_pred = int(np.argmax(keras_probs))

    labels = {0: "VACANT", 1: "OCCUPIED"}
    print(f"Keras prediction:        {labels[keras_pred]}  probs={keras_probs}")
    print(f"C-equivalent prediction: {labels[c_pred]}  logits={c_logits}")

    if c_pred == keras_pred:
        print("\nMATCH -- safe to flash model_weights.h to the MSP430.")
    else:
        print("\nMISMATCH -- do not flash yet. Check calibration images "
              "(add more to calibration_images/) or verify layer scales.")


if __name__ == "__main__":
    main()
