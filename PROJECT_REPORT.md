# Project Report: Real-Time Facial Affect Analysis

## 1. Introduction

This project implements a real-time facial affect analysis system that performs two complementary forms of emotion inference from a live webcam feed:

1. **Categorical Emotion Classification** — Assigns each detected face one of seven discrete emotion labels (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using a mini Xception network trained on FER-2013.
2. **Dimensional Affect Regression** — Predicts continuous Valence (pleasure–displeasure) and Arousal (activation–deactivation) values using a MobileNetV2 network trained on AffectNet.
3. **Affect Fusion** — Cross-validates both outputs using Russell's Circumplex Model and Plutchik's Wheel of Emotions to infer intensity levels, compound emotions (dyads), and prediction reliability.

Both models execute in parallel per detected face using multithreaded inference, and fused results are overlaid on the video feed in real time.

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
Webcam Frame (720×480 BGR, mirrored)
        │
        ▼
   RGB Conversion
        │
        ▼
   MediaPipe BlazeFace Detection
        │
        ▼
   Face ROI Extraction
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
   Emotion Branch                    VA Branch
   (grayscale, 64×64)               (RGB, 224×224)
        │                                  │
        ▼                                  ▼
   Mini Xception (TFLite)           MobileNetV2 (TFLite)
        │                                  │
        ▼                                  ▼
   7-class softmax                  [valence, arousal]
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              AffectFusionEngine
              (Russell + Plutchik)
                       │
                       ▼
              Frame Overlay + Display
```

### 2.2 Threading Model

Both inference branches run concurrently via `concurrent.futures.ThreadPoolExecutor` with `max_workers=2`. For each detected face in a frame:

1. The main thread extracts the face ROI (both grayscale and color crops)
2. Two futures are submitted simultaneously — one per model
3. The main thread blocks on `future.result()` for both
4. Results are passed to the `AffectFusionEngine` for cross-validation and enrichment
5. The fused result drives the overlay rendering

This avoids sequential bottlenecking where the larger VA model would delay the faster emotion model.

### 2.3 Inference Runtime

Both models were converted from Keras HDF5 format to **TensorFlow Lite** (FlatBuffer `.tflite` format) for optimized CPU inference. TFLite uses the **XNNPACK delegate** for accelerated floating-point operations on x86/ARM CPUs.

| Metric | HDF5 (Keras) | TFLite |
|---|---|---|
| Runtime overhead | Full TensorFlow graph executor | Lightweight interpreter |
| Cold start | ~2–4 seconds | ~100 ms |
| Per-call overhead | `model.predict()` Python/session overhead | Direct tensor dispatch |
| Memory footprint | Full TF runtime in RAM | Interpreter + model buffer only |

---

## 3. Face Detection

### 3.1 Detector

| Property | Value |
|---|---|
| Library | MediaPipe 0.10.33 |
| Model | BlazeFace short-range |
| Model size | 225 KB |
| Method | SSD-based single-shot detector optimized for faces < 2m from camera |
| Input | RGB frame |
| Output | List of detections with bounding boxes (normalized coordinates) |
| Min detection confidence | 0.5 |

### 3.2 Migration from dlib

The original implementation used dlib's HOG + Linear SVM detector with a 68-point landmark predictor (95.1 MB). The landmarks were extracted but never used beyond bounding box computation. MediaPipe BlazeFace replaced both components with a single 225 KB model that runs faster and handles a wider range of face angles and occlusions.

---

## 4. Emotion Classification Model

### 4.1 Architecture — Mini Xception

A compact variant of the Xception architecture using depthwise separable convolutions with residual connections. Designed for efficient inference on resource-constrained devices.

```
Input (64×64×1 grayscale)
    │
    ├─ Conv2D(8, 3×3)  →  BN  →  ReLU
    ├─ Conv2D(8, 3×3)  →  BN  →  ReLU
    │
    ├─ Residual Block 1: SepConv(16) → BN → ReLU → SepConv(16) → BN → MaxPool(3×3)
    │   └─ Skip: Conv2D(16, 1×1) → BN
    │
    ├─ Residual Block 2: SepConv(32) → BN → ReLU → SepConv(32) → BN → MaxPool(3×3)
    │   └─ Skip: Conv2D(32, 1×1) → BN
    │
    ├─ Residual Block 3: SepConv(64) → BN → ReLU → SepConv(64) → BN → MaxPool(3×3)
    │   └─ Skip: Conv2D(64, 1×1) → BN
    │
    ├─ Residual Block 4: SepConv(128) → BN → ReLU → SepConv(128) → BN → MaxPool(3×3)
    │   └─ Skip: Conv2D(128, 1×1) → BN
    │
    ├─ Conv2D(7, 1×1)  →  GlobalAveragePooling2D
    └─ Softmax Activation → Output (7 classes)
```

### 4.2 Layer-by-Layer Parameter Count

| Layer | Output Shape | Parameters |
|---|---|---|
| Input | (64, 64, 1) | 0 |
| Conv2D (8 filters, 3×3) | (62, 62, 8) | 72 |
| Conv2D (8 filters, 3×3) | (60, 60, 8) | 576 |
| SeparableConv2D (16) | (60, 60, 16) | 168 |
| SeparableConv2D (16) | (60, 60, 16) | 400 |
| Conv2D skip (16, 1×1) | (30, 30, 16) | 128 |
| MaxPooling2D (3×3) | (30, 30, 16) | 0 |
| SeparableConv2D (32) | (30, 30, 32) | 656 |
| SeparableConv2D (32) | (30, 30, 32) | 1,184 |
| Conv2D skip (32, 1×1) | (15, 15, 32) | 512 |
| MaxPooling2D (3×3) | (15, 15, 32) | 0 |
| SeparableConv2D (64) | (15, 15, 64) | 2,336 |
| SeparableConv2D (64) | (15, 15, 64) | 4,672 |
| Conv2D skip (64, 1×1) | (8, 8, 64) | 2,048 |
| MaxPooling2D (3×3) | (8, 8, 64) | 0 |
| SeparableConv2D (128) | (8, 8, 128) | 8,768 |
| SeparableConv2D (128) | (8, 8, 128) | 17,536 |
| Conv2D skip (128, 1×1) | (4, 4, 128) | 8,192 |
| MaxPooling2D (3×3) | (4, 4, 128) | 0 |
| Conv2D (7, 1×1) | (4, 4, 7) | 8,071 |
| GlobalAveragePooling2D | (7,) | 0 |
| Softmax | (7,) | 0 |

### 4.3 Model Metrics

| Metric | Value |
|---|---|
| Total parameters | 58,423 |
| Trainable parameters | 56,951 |
| Non-trainable parameters (BN) | 1,472 |
| Original model size (HDF5) | 852.4 KB |
| Converted model size (TFLite) | 235.6 KB |
| Size reduction | 72.4% |
| Input shape | (1, 64, 64, 1) |
| Input dtype | float32 |
| Output shape | (1, 7) |
| Output dtype | float32 |
| Confidence threshold | 0.36 |

### 4.4 Training Dataset — FER-2013

| Property | Value |
|---|---|
| Dataset | Facial Expression Recognition 2013 |
| Total images | 35,887 |
| Original image size | 48×48 grayscale |
| Classes | 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) |
| Class distribution | Imbalanced (Happy overrepresented, Disgust underrepresented) |
| Transfer learning base | Xception architecture (adapted to mini variant) |

### 4.5 Preprocessing Pipeline

```python
# 1. Extract grayscale face ROI from frame
grayFace = grayFrame[y:y+h, x:x+w]

# 2. Resize to model input dimensions
grayFace = cv2.resize(grayFace, (64, 64))

# 3. Cast to float32
grayFace = grayFace.astype('float32')

# 4. Normalize to [0, 1]
grayFace = grayFace / 255.0

# 5. Center and scale to [-1, 1]
grayFace = (grayFace - 0.5) * 2.0

# 6. Add batch dimension → (1, 64, 64)
grayFace = np.expand_dims(grayFace, 0)

# 7. Add channel dimension → (1, 64, 64, 1)
grayFace = np.expand_dims(grayFace, -1)
```

### 4.6 Output Classes

| Index | Emotion |
|---|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

---

## 5. Valence-Arousal Regression Model

### 5.1 Architecture — MobileNetV2 + Regression Head

MobileNetV2 is a lightweight CNN architecture designed for mobile and edge deployment. It uses **inverted residual blocks** with **depthwise separable convolutions** and **linear bottlenecks** to minimize computational cost while maintaining accuracy.

```
Input (224×224×3 RGB)
    │
    ├─ Conv2D(32, 3×3, stride=2) → BN → ReLU6        # 112×112×32
    │
    ├─ Inverted Residual Block ×1  (t=1, c=16, s=1)   # 112×112×16
    ├─ Inverted Residual Block ×2  (t=6, c=24, s=2)   # 56×56×24
    ├─ Inverted Residual Block ×3  (t=6, c=32, s=2)   # 28×28×32
    ├─ Inverted Residual Block ×4  (t=6, c=64, s=2)   # 14×14×64
    ├─ Inverted Residual Block ×3  (t=6, c=96, s=1)   # 14×14×96
    ├─ Inverted Residual Block ×3  (t=6, c=160, s=2)  # 7×7×160
    ├─ Inverted Residual Block ×1  (t=6, c=320, s=1)  # 7×7×320
    │
    ├─ Conv2D(1280, 1×1) → BN → ReLU6                 # 7×7×1280
    │
    ├─ GlobalAveragePooling2D                          # 1280
    ├─ Dense(1024, activation='relu')                  # 1024
    └─ Dense(2, activation='linear')                   # 2 [valence, arousal]
```

Where `t` = expansion factor, `c` = output channels, `s` = stride.

### 5.2 Inverted Residual Block Structure

Each block follows this pattern:

```
Input (h×w×k)
    │
    ├─ Expand:    Conv2D(1×1, tk filters) → BN → ReLU6    # Expand channels
    ├─ Depthwise: DepthwiseConv2D(3×3)    → BN → ReLU6    # Spatial filtering
    ├─ Project:   Conv2D(1×1, c filters)  → BN             # Compress (linear, no ReLU)
    │
    └─ Residual connection (Add) if stride=1 and input_channels == output_channels
```

### 5.3 Model Metrics

| Metric | Value |
|---|---|
| Total parameters | 3,571,778 |
| Trainable parameters | 3,537,666 |
| Non-trainable parameters (BN) | 34,112 |
| Original weights size (H5) | 14,325.9 KB (14.0 MB) |
| Converted model size (TFLite) | 13,795.7 KB (13.5 MB) |
| Size reduction | 3.7% |
| Input shape | (1, 224, 224, 3) |
| Input dtype | float32 |
| Output shape | (1, 2) |
| Output dtype | float32 |
| MobileNetV2 backbone params | 2,257,984 |
| Regression head params | 1,313,794 |

### 5.4 Regression Head Breakdown

| Layer | Input → Output | Parameters | Computation |
|---|---|---|---|
| GlobalAveragePooling2D | (7, 7, 1280) → (1280,) | 0 | Spatial mean reduction |
| Dense (ReLU) | (1280,) → (1024,) | 1,311,744 | 1280×1024 weights + 1024 biases |
| Dense (Linear) | (1024,) → (2,) | 2,050 | 1024×2 weights + 2 biases |

### 5.5 Training Dataset — AffectNet

| Property | Value |
|---|---|
| Dataset | AffectNet |
| Total images | 400,000+ (manually annotated subset) |
| Annotation type | Both categorical (8 emotions) and dimensional (V/A) |
| Valence range | [-1, 1] (negative to positive) |
| Arousal range | [-1, 1] (calm to excited) |
| Images used | Manually annotated only |

### 5.6 Reported Performance (from source project)

| Metric | Valence | Arousal |
|---|---|---|
| RMSE | 0.39 | 0.38 |
| Baseline RMSE | 0.39 | 0.38 |

The model matches the AffectNet paper baseline while using a compact MobileNetV2 backbone occupying only 14 MB.

### 5.7 Preprocessing Pipeline

```python
# 1. Extract color face ROI from BGR frame
face = frame[y:y+h, x:x+w]

# 2. Resize to model input dimensions
face = cv2.resize(face, (224, 224))

# 3. Convert BGR (OpenCV default) to RGB
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

# 4. Cast to float32
face = face.astype('float32')

# 5. Normalize to [0, 1]
face = face / 255.0

# 6. Center and scale to [-1, 1]
face = (face - 0.5) * 2.0

# 7. Add batch dimension → (1, 224, 224, 3)
face = np.expand_dims(face, 0)
```

### 5.8 Output Interpretation

| Output Index | Dimension | Range | Semantic Meaning |
|---|---|---|---|
| 0 | Valence | [-1, 1] | -1 = very negative (disgust, anger) → +1 = very positive (joy, elation) |
| 1 | Arousal | [-1, 1] | -1 = very calm (sleepy, bored) → +1 = very excited (rage, ecstasy) |

**Valence-Arousal Circumplex Mapping:**

```
                    High Arousal (+1)
                         │
            Angry/Fear   │   Excited/Happy
                         │
   Negative (-1) ────────┼──────── Positive (+1)  ← Valence
                         │
           Sad/Bored     │   Calm/Relaxed
                         │
                    Low Arousal (-1)
```

---

## 6. Affect Fusion Engine

### 6.1 Overview

The `AffectFusionEngine` (in `affect_fusion.py`) integrates the outputs of both models using two established psychological frameworks to produce enriched, cross-validated emotion inferences.

### 6.2 Russell's Circumplex Model — Sanity Check

Each of the 7 categorical predictions has an expected position in VA space, derived from AffectNet empirical data (Mollahosseini et al. 2017):

| Xception Class | Expected Valence | Expected Arousal |
|---|---|---|
| Angry | -0.45 | +0.31 |
| Disgust | -0.48 | +0.22 |
| Fear | -0.36 | +0.38 |
| Happy | +0.55 | +0.28 |
| Sad | -0.41 | -0.12 |
| Surprise | +0.13 | +0.45 |
| Neutral | +0.02 | -0.05 |

A **conflict score** (Euclidean distance between predicted VA and expected VA centroid) flags disagreements. If `CS > 0.5`, the prediction is marked unreliable.

### 6.3 Plutchik's Wheel of Emotions — Petal Mapping

VA coordinates are mapped to Plutchik's 8 primary emotions using **nearest-centroid** with empirically validated coordinates from Mehrabian's PAD model (1980/1995):

| Petal | Valence | Arousal | Mild | Basic | Intense |
|---|---|---|---|---|---|
| Joy | +0.76 | +0.48 | Serenity | Joy | Ecstasy |
| Trust | +0.52 | +0.20 | Acceptance | Trust | Admiration |
| Fear | -0.64 | +0.60 | Apprehension | Fear | Terror |
| Surprise | +0.14 | +0.67 | Distraction | Surprise | Amazement |
| Sadness | -0.63 | -0.27 | Pensiveness | Sadness | Grief |
| Disgust | -0.60 | +0.35 | Boredom | Disgust | Loathing |
| Anger | -0.51 | +0.59 | Annoyance | Anger | Rage |
| Anticipation | +0.22 | +0.62 | Interest | Anticipation | Vigilance |

**Why nearest-centroid instead of angular mapping:** Fear (-0.64, +0.60) and Anger (-0.51, +0.59) are nearly identical in VA space but are opposites on Plutchik's wheel (180 degrees apart). Angular mapping from VA to Plutchik is therefore psychologically invalid (Semeraro et al. 2021, Buechel & Hahn 2018).

**Categorical disambiguation:** When the two closest VA centroids are within 0.15 distance of each other, the categorical model's prediction is used as a tiebreaker. This is psychologically justified because facial expressions can distinguish fear from anger even when felt arousal/valence are similar.

**Intensity** is determined by VA magnitude (`r = sqrt(V² + A²)`): `r < 0.33` = mild, `0.33–0.66` = basic, `r ≥ 0.66` = intense.

### 6.4 Dyad Detection

When the classifier's top two softmax probabilities are close (gap < 0.15, top < 0.5), the system checks for compound emotions:

**Primary Dyads (adjacent petals):**

| Components | Dyad |
|---|---|
| Joy + Trust | Love |
| Trust + Fear | Submission |
| Fear + Surprise | Awe |
| Surprise + Sadness | Disapproval |
| Sadness + Disgust | Remorse |
| Disgust + Anger | Contempt |
| Anger + Anticipation | Aggressiveness |
| Anticipation + Joy | Optimism |

**Secondary Dyads (2 petals apart):**

| Components | Dyad |
|---|---|
| Joy + Fear | Guilt |
| Trust + Surprise | Curiosity |
| Fear + Sadness | Despair |
| Surprise + Disgust | Unbelief |
| Sadness + Anger | Envy |
| Disgust + Anticipation | Cynicism |
| Anger + Joy | Pride |
| Anticipation + Trust | Hope |

Dyads are only emitted when the Russell sanity check passes (reliable prediction).

### 6.5 Temporal Smoothing

An exponential moving average (EMA) with α=0.3 smooths the V and A streams to reduce frame-to-frame jitter. The EMA resets when no face is detected (prevents stale values from bleeding into a new face).

### 6.6 Known Limitations

- **Trust and Anticipation** have no corresponding FER-2013 class. These petals are inferred purely from VA proximity to Mehrabian centroids and cannot be reached via categorical disambiguation.
- **Fear/Anger ambiguity** in VA space is mitigated but not fully resolved. The Dominance dimension (absent from our VA model) is the true discriminator.
- **Neutral** has no Plutchik equivalent. When the categorical model predicts Neutral and VA magnitude is low (< 0.2), no petal is assigned.

---

## 7. TFLite Conversion

### 7.1 Emotion Model Conversion

Straightforward — the HDF5 file contains the full model (architecture + weights):

```python
model = tf.keras.models.load_model('emotionModel.hdf5', compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

### 7.2 VA Model Conversion

The H5 file contains **weights only** (no architecture). The MobileNetV2 + regression head must be reconstructed before conversion:

```python
base = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base.output)
x = Dense(1024, activation='relu')(x)
x = Dense(2, activation='linear')(x)
model = Model(inputs=base.input, outputs=x)
model.load_weights('regressor_weights.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

### 7.3 Conversion Results

| Model | Original Format | Original Size | TFLite Size | Reduction |
|---|---|---|---|---|
| Emotion (Xception) | HDF5 (full model) | 852.4 KB | 235.6 KB | 72.4% |
| VA (MobileNetV2) | H5 (weights only) | 14,325.9 KB | 13,795.7 KB | 3.7% |

The emotion model sees a larger reduction because the HDF5 format stores architecture metadata, optimizer state, and training config alongside weights. TFLite strips all of this. The VA model's H5 file is weights-only, so the overhead was already minimal.

---

## 8. Model Comparison

| Property | Emotion Model | VA Model |
|---|---|---|
| Architecture | Mini Xception | MobileNetV2 |
| Task | Classification (7 classes) | Regression (2 values) |
| Dataset | FER-2013 | AffectNet |
| Input resolution | 64×64 | 224×224 |
| Color channels | 1 (grayscale) | 3 (RGB) |
| Parameters | 58,423 | 3,571,778 |
| Parameter ratio | 1x | 61x |
| TFLite size | 236 KB | 13.5 MB |
| Size ratio | 1x | 57x |
| FLOPs (approx.) | ~2M | ~300M |
| Convolution type | Depthwise separable | Depthwise separable |
| Normalization | BatchNorm | BatchNorm |
| Skip connections | Yes (residual add) | Yes (inverted residual add) |
| Output activation | Softmax | Linear |
| Loss function (training) | Categorical cross-entropy | Mean squared error |

---

## 9. Software Dependencies

| Package | Version | Purpose |
|---|---|---|
| tensorflow | 2.21.0 | TFLite interpreter, model conversion |
| keras | 3.13.2 | Model loading (conversion only) |
| opencv-python | 4.13.0.92 | Webcam capture, image processing, display |
| mediapipe | >=0.10.14 | Face detection (BlazeFace) |
| numpy | 2.4.4 | Array operations, preprocessing |
| h5py | 3.14.0 | HDF5 file I/O |

---

## 10. File Inventory

| File | Size | Purpose |
|---|---|---|
| `facecombined.py` | — | Combined parallel inference + affect fusion |
| `affect_fusion.py` | — | AffectFusionEngine (Russell + Plutchik integration) |
| `facemot.py` | — | Standalone emotion classification |
| `faceva.py` | — | Standalone VA regression |
| `convert_to_tflite.py` | — | Emotion HDF5 → TFLite conversion |
| `convert_va_to_tflite.py` | — | VA H5 weights → TFLite conversion |
| `models/emotionModel.tflite` | 235.6 KB | Emotion classifier (production) |
| `models/vaModel.tflite` | 13.5 MB | VA regressor (production) |
| `models/blaze_face_short_range.tflite` | 225 KB | MediaPipe face detection model |
| `models/emotionModel.hdf5` | 852.4 KB | Emotion classifier (original) |
| `models/regressor_weights.h5` | 14.0 MB | VA regressor weights (original) |

---

## 11. Visualization Details

### 11.1 Combined Mode (`facecombined.py`)

For each detected face, the overlay renders:

1. **Bounding box** — Color-coded by Plutchik petal (dimmed if prediction is unreliable)
2. **Base emotion label** (first row below box) — Direct model output: emotion name + confidence percentage (e.g., `Happy 80%`). An asterisk (`*`) indicates the Russell conflict score exceeds the reliability threshold.
3. **Plutchik intensity word** (second row, conditional) — Only shown when the Plutchik-inferred label differs from the base emotion (e.g., `~ Ecstasy` for intense joy). Omitted when redundant.
4. **Dyad label** (third row, conditional) — Compound emotion when detected (e.g., `+ Love`)
5. **VA HUD** (top-right corner, 150×150) — Real-time Valence-Arousal grid with:
   - Quadrant labels: Pleasant+, Unpleasant+, Pleasant-, Unpleasant-
   - Colored dot tracking smoothed VA position
   - Numeric readout: `V:+0.52  A:+0.31`
   - Semi-transparent dark background

### 11.2 Display Parameters

| Parameter | Value |
|---|---|
| Frame resolution | 720×480 |
| Camera | Mirrored (selfie view) |
| Font | `cv2.FONT_HERSHEY_SIMPLEX` |
| Font scale | 0.45 (labels), 0.38 (secondary), 0.25–0.35 (HUD) |
| Text color | White (labels), light gray (secondary/HUD) |
| Line type | `cv2.LINE_AA` (anti-aliased) |
| Box thickness | 2px |
| Label width | Auto-sized to text content |
| HUD opacity | 70% background |
| Exit key | Escape (key code 27) |

---

## 12. Limitations

- **Face detector** — MediaPipe BlazeFace short-range is optimized for faces within ~2 meters of the camera. Performance degrades at longer distances or extreme profile angles.
- **Emotion model** — FER-2013 dataset has known label noise (~10% of images are incorrectly labeled per multiple studies). The 0.36 confidence threshold was empirically chosen and may not generalize across demographics.
- **VA model** — AffectNet annotations are subjective; inter-annotator agreement varies. The model outputs can exceed the [-1, 1] range slightly since the final layer has no bounded activation.
- **Affect fusion** — The Plutchik mapping relies on Mehrabian PAD centroids which represent population-level averages. Individual emotional experiences may deviate significantly. Trust and Anticipation are unreachable via the categorical model. Fear/Anger disambiguation is imperfect without the Dominance dimension.
- **Threading** — Python's GIL limits true parallelism for CPU-bound NumPy operations. The threading benefit comes primarily from TFLite's internal C++ execution releasing the GIL during `interpreter.invoke()`.
- **Temporal smoothing** — EMA is scoped to a single face. Multi-face scenarios lack face tracking, so the EMA state may blend across different faces if detection order changes between frames.
- **Latency** — The VA model (224×224 input, 3.6M params) is significantly slower per inference than the emotion model (64×64 input, 58K params). In the combined script, the VA model is the bottleneck.

---

## 13. References

1. Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2019). AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild. *IEEE Transactions on Affective Computing*, 10(1), 18–31.
2. Goodfellow, I. J., et al. (2013). Challenges in Representation Learning: A report on three machine learning contests. *ICML Workshop on Challenges in Representation Learning*.
3. Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. *CVPR 2017*.
4. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR 2018*.
5. Plutchik, R. (1980). *Emotion: A Psychoevolutionary Synthesis*. Harper & Row.
6. Russell, J. A. (1980). A Circumplex Model of Affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178.
7. Mehrabian, A. (1995). Framework for a comprehensive description and measurement of emotional states. *Genetic, Social, and General Psychology Monographs*, 121(3), 339–361.
8. Semeraro, A., et al. (2021). PyPlutchik: Visualising and comparing emotion-annotated corpora. *PLOS ONE*, 16(9).
9. Buechel, S. & Hahn, U. (2018). Representation Mapping: A Novel Approach to Generate High-Quality Multi-Lingual Emotion Lexicons. *LREC 2018*.
10. Batic, D. Facial Affect Analysis on Mobile Devices Using Convolutional Neural Networks. *Bachelor thesis, University of Novi Sad*.
