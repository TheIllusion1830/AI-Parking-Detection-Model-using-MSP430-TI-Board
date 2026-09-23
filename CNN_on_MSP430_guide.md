# Running a Real CNN On the MSP430F5529 — Project Guide

## 0. What changed and why it now satisfies "a CNN must run on the board"

The old design trained a network on a PC, flattened everything to a 256-vector,
and the MSP430 only ever computed two dense-layer matrix multiplies. Functionally
that's indistinguishable from evaluating a giant precomputed lookup/weighted-sum
table — there's no weight sharing, no receptive field, nothing that is
structurally a "convolution."

The new design:

- Keeps the image as a **2D 16×16 map** instead of flattening it.
- Trains **real `Conv2D` layers** in Keras (3×3 kernels, weight-shared across
  spatial positions) plus `MaxPooling2D`.
- Quantizes every layer (not just the dense ones) to int8.
- Reimplements the **exact same sliding-window convolution, max-pooling, and
  dense arithmetic in `main.c`**, using nested loops over the kernel window.

The MSP430 therefore performs the actual multiply-accumulate sliding-window
convolutions for every image, using the trained (quantized) weights — it is
not classifying via a decision tree or a pre-baked boundary. This is exactly
how real embedded-CNN runtimes (e.g. TensorFlow Lite Micro) work: the weights
are fixed ahead of time, but the *computation graph* (conv → relu → pool →
dense) is executed fresh on-device for every input.

## 1. Architecture chosen

```
Input (16×16×1, int8)
  → Conv2D(4 filters, 3×3, valid)   → 14×14×4
  → ReLU
  → MaxPool2D(2×2)                  → 7×7×4
  → Conv2D(8 filters, 3×3, valid)   → 5×5×8
  → ReLU
  → MaxPool2D(2×2)                  → 2×2×8   (Keras 'valid' pooling drops the
                                                 odd leftover row/col — main.c
                                                 replicates this exactly)
  → Flatten (row-major h,w,c, same order as Keras Flatten())  → 32
  → Dense(16) → ReLU
  → Dense(2)  → argmax → VACANT / OCCUPIED
```

Sized deliberately small (~4,150 weights total) so every intermediate feature
map and all quantized weights fit in SRAM simultaneously — see §3 below.

## 2. Files produced

| File | Role |
|---|---|
| `train_model.py` | Trains the real CNN on PKLot; preprocessing unchanged (grayscale → 64×64 LANCZOS → /255 → 4×4 avg-pool → 16×16), just no longer flattened. Saves `parking_model.keras`. |
| `quantize_and_export.py` | Loads the trained model + `BoardImage.jpg`, quantizes every layer to int8, calibrates and generates **integer** requantization constants per layer boundary, writes `model_weights.h`. |
| `reference_check.py` | Re-implements `main.c`'s exact integer arithmetic in NumPy and checks it agrees with the plain Keras prediction — run this **before** flashing. |
| `main.c` | The on-device CNN: hand-written conv/pool/dense loops in integer arithmetic, LED + UART output unchanged from the original firmware. |

## 3. Concepts used, and where each one came from

**From the memory-hierarchy paper you supplied** (Banerjee et al., *"Memory-aware
Efficient Deep Learning Mechanism for IoT Devices,"* ASAP 2021):

- The MCU memory model — two-level hierarchy of small, fast **SRAM** vs. larger,
  slower **Flash**, with the paper's own reference numbers for the MSP430F5529
  (8 KB SRAM / 128 KB Flash) — used directly to size this project's SRAM budget.
- The four dataflow strategies (**Perc**, **OaaT**, **MM**, **Hybrid**) the paper
  proposes for when a CNN's feature maps/weights don't fit in SRAM at once.
- The paper's own finding that **OaaT (One-at-a-Time — materialize whole
  input/weights/output per layer, no Flash write-backs mid-layer) is the most
  energy- and time-efficient option when SRAM is large enough to hold it**
  (paper Sec. V-B / Table IV) — this is the justification cited in `main.c` for
  why we simply keep full `conv1_act`, `pool1`, `conv2_act`, `pool2`, etc.
  arrays in SRAM instead of implementing the paper's tiled Perc/MM/Hybrid logic.
  Total resident activation storage here is **~1.24 KB**, and the paper's own
  Table IV shows their smallest benchmark (HAR) still needs 13,184 bytes of
  SRAM under OaaT — ours fits because the network itself is much smaller, not
  because we invented a new dataflow trick.
- The paper explicitly does **not** cover quantization (it states pruning/
  quantization are "orthogonal" to its contributions) — so all of the
  quantization work below is standard ML/embedded-inference practice, not
  from the paper.

**Standard practice, not from the paper** (used because the project's own
notes in the earlier context document — Sec. 12, 13, 16, 25, 26, 36 —
required it):

- **Per-tensor symmetric int8 quantization** of weights (`scale = max(|w|)/127`,
  round, clip to [-127,127]) — this was already the approach in the original
  `quantize_and_export.py`; kept.
- **Bias folding into each layer's own accumulator units** (`bias_acc =
  round(bias_real / (input_scale * weight_scale))`) instead of the old
  `bias * 256` placeholder — this directly fixes the bug flagged in the
  earlier project notes (Sec. 16/36: *"acc + B1[i]*256 ... is not
  mathematically equivalent to the trained Keras network"*).
- **Integer-only requantization** (`(acc * M0) >> shift`) between layer
  boundaries — the standard technique used by quantized-inference runtimes
  (e.g. TFLite's reference kernels) to convert a layer's large int32
  accumulator back into the small int8 range the next layer's weights expect,
  without ever using floating point on the MCU. This was necessary because
  chaining raw un-rescaled accumulators layer-to-layer overflows a 32-bit
  accumulator by the 3rd/4th layer on this network (checked by hand during
  design — conv2's accumulator alone would already approach ~7×10⁸ without
  requantizing conv1's output first).
- **Post-training calibration** of each activation's int8 scale, by running
  representative images through the (float) Keras model and taking
  `max(|activation|)/127` — standard post-training quantization calibration.
  The provided script calibrates off `BoardImage.jpg` alone by default (works,
  but fragile); it looks for an optional `calibration_images/` folder so you
  can calibrate on a proper sample of Empty/Occupied crops instead — do this
  before your final flash for a much more reliable model.
- **Two-class softmax → argmax shortcut**: since softmax is monotonic, the
  final class decision only needs `logit[1] > logit[0]`, so no exponential is
  computed on-device. (This was already noted in the earlier project
  documentation, Sec. 17 — kept, not new.)

## 4. Full workflow

1. Put `BoardImage.jpg` next to the Python scripts (and, ideally, a handful of
   calibration images in `calibration_images/`).
2. `python train_model.py` → trains the CNN on PKLot, saves
   `parking_model.keras`, reports the same 1000-image held-out metrics as
   before, and prints the CNN's own prediction on `BoardImage.jpg`.
3. `python quantize_and_export.py` → quantizes every layer, calibrates
   requantization constants, writes `model_weights.h`.
4. `python reference_check.py` → re-runs `main.c`'s exact integer arithmetic
   in Python and confirms it agrees with the Keras (float) prediction. **Do
   not flash the board until this prints MATCH.**
5. Copy `model_weights.h` next to `main.c` in the CCS project.
6. Build the CCS project, flash the MSP430F5529.
7. Board runs the full conv→pool→conv→pool→dense→dense pipeline itself on
   power-up; LEDs + UART report VACANT/OCCUPIED exactly as before.

## 5. SRAM budget actually used on-device

| Buffer | Size |
|---|---|
| `conv1_act[14][14][4]` | 784 B |
| `pool1[7][7][4]` | 196 B |
| `conv2_act[5][5][8]` | 200 B |
| `pool2[2][2][8]` | 32 B |
| `fc1_act[16]` | 16 B |
| `fc2_out[2]` (int32) | 8 B |
| **Total activations** | **~1.24 KB** |
| Weights (Flash, not SRAM) | CONV1_W 36 B + CONV2_W 288 B + FC1_W 512 B + FC2_W 32 B + biases (int32) ~120 B ≈ **~1 KB** |

Well inside the MSP430F5529's 8 KB SRAM / 128 KB Flash, with plenty of margin
for the stack and UART buffers — which is exactly why the paper's tiled
dataflows (Perc/MM/Hybrid) aren't needed here; see §3.

## 6. If your professor wants the paper's dataflow techniques explicitly demonstrated

Right now the model is small enough that OaaT-style full materialization is
sufficient and is the paper's own recommended choice when SRAM allows it. If
you want to additionally *demonstrate* one of the more advanced dataflows
(e.g. to show you engaged with the paper's Move-the-Minimum or Hybrid method,
not just cite OaaT), the cleanest way is to **scale the CNN up** (e.g. more
filters, or a 32×32 input) until it genuinely no longer fits in 8 KB SRAM at
once, and then implement MM-style tiling for `conv2_layer()` (load only the
3×3×in_ch input window needed for one output pixel at a time, per the paper's
Eq. 14–15, instead of materializing the whole `pool1` array first). Say the
word and this can be built out as a follow-up.

---

## Compiled context (paste this into a new conversation if credits run low)

```
PROJECT: Smart parking occupancy CNN on TI MSP-EXP430F5529LP (CCS, no TFLite
Micro, hand-written C inference, no filesystem/JPEG decode on device).

STATUS: Just redesigned from an MLP (flatten 16x16->256->Dense16->Dense2) to
an ACTUAL small CNN so the board performs real sliding-window convolutions,
not a lookup table. Four files were produced:
  - train_model.py: trains Conv2D(4,3x3)->MaxPool2D(2)->Conv2D(8,3x3)->
    MaxPool2D(2)->Flatten->Dense(16)->Dense(2,softmax) on 16x16x1 grayscale
    (preprocessing unchanged: grayscale->64x64 LANCZOS->/255->4x4 avgpool->
    16x16, just no longer flattened before the network). Saves
    parking_model.keras.
  - quantize_and_export.py: per-tensor symmetric int8 weight quantization;
    biases folded directly into each layer's own accumulator units
    (bias_acc = round(bias_real/(input_scale*weight_scale))) instead of the
    old broken "bias*256" hack; calibrates each activation's int8 scale from
    real images (BoardImage.jpg + optional calibration_images/ folder);
    computes an INTEGER requantization multiplier+shift (acc*M0)>>shift for
    each layer boundary (conv1->conv2, conv2->fc1, fc1->fc2) so the MCU never
    needs floats or overflowing accumulators; writes model_weights.h.
  - reference_check.py: replicates main.c's exact int arithmetic in NumPy,
    compares against Keras, must print MATCH before flashing.
  - main.c: hand-written conv1/pool1/conv2/pool2/fc1/fc2 functions using
    nested loops (real sliding-window convolution + 2x2 max pooling), int32
    accumulators, requantize() helper doing (acc*mult)>>shift with rounding
    and clip to [-127,127]. All activation buffers are ~1.24 KB total,
    weights ~1 KB, fits easily in the MSP430F5529's 8 KB SRAM / 128 KB Flash.
    LED (P1.0 vacant / P4.7 occupied) and UART (P4.4 TX/P4.5 RX, 9600 baud)
    logic unchanged from the original firmware.

PAPER USED FOR JUSTIFICATION: Banerjee et al., "Memory-aware Efficient Deep
Learning Mechanism for IoT Devices" (ASAP 2021). Cited concepts: the
SRAM/Flash two-level memory model and MSP430F5529 numbers (8KB SRAM/128KB
Flash) from paper Sec. II-A/Table III; the paper's four dataflow strategies
Perc/OaaT/MM/Hybrid (Sec. IV); and specifically the paper's finding that OaaT
(full materialization, no mid-layer Flash write-back) is the most energy/time
efficient option when SRAM is large enough to hold it (Sec. V-B/Table IV) --
used to justify why this project's tiny CNN (whose full activation set is
~1.24KB) uses straightforward full-buffer OaaT-style computation rather than
the paper's more complex tiled Perc/MM/Hybrid dataflows, which exist in the
paper specifically for larger CNNs (MNIST/HAR/TrafficSign, needing 13KB-225KB
SRAM under OaaT per the paper's Table IV) that don't fit an MSP430's SRAM.
Quantization itself is explicitly NOT covered by the paper (paper states
pruning/quantization are "orthogonal" to its contributions) -- all
quantization/requantization design (per-tensor symmetric int8, bias folding
into accumulator units, integer multiply+shift requantization, post-training
calibration) is standard embedded-inference practice, not from the paper.

LABELS: 0 = VACANT (red LED, P1.0), 1 = OCCUPIED (green LED, P4.7).
DATASET: PKLot, 9000 train / 1000 held-out test, seed=41, unchanged from
original project. BoardImage.jpg stays separate from PKLot, same directory
as the Python scripts, processed only by quantize_and_export.py now.

OPEN FOLLOW-UP OFFERED: if the professor wants an explicit demonstration of
the paper's Move-the-Minimum or Hybrid tiled dataflow (not just OaaT), scale
the CNN up until it no longer fits 8KB SRAM at once, then tile conv2_layer()
per the paper's Eq. 14-15 (load only the 3x3xin_ch window needed for one
output pixel at a time). Not yet built.

NEXT STEPS FOR USER: run train_model.py -> quantize_and_export.py ->
reference_check.py (must show MATCH) -> copy model_weights.h into CCS ->
build/flash main.c.
```
