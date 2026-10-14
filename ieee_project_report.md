# Real-Time Dual-Model Facial Affect Analysis via Fusion of Categorical and Dimensional Predictions Using Russell's Circumplex, Plutchik's Wheel of Emotions, and Mehrabian's PAD Framework

**lemon**
Department of Computer Science
October 14, 2026

---

## Abstract

*Automatic facial expression analysis systems typically commit to a single representation of affect: either discrete emotion categories (e.g., happy, sad, angry) or continuous dimensional scores (valence and arousal). Categorical models offer interpretability but discard intensity and blend information; dimensional models capture graded affect but produce coordinates that are difficult to communicate to end users. This project presents a real-time webcam-based affect analysis system that runs both representations concurrently and fuses them under classical psychological structure. A Mini Xception convolutional network trained on FER-2013 performs 7-class expression classification, while a MobileNetV2 regressor trained on AffectNet predicts continuous valence and arousal. Both models execute in parallel per detected face as TensorFlow Lite interpreters, and MediaPipe BlazeFace supplies face detection. An Affect Fusion Engine cross-validates the two model outputs using Russell's Circumplex Model of Affect, maps fused coordinates onto the eight petals of Plutchik's Wheel of Emotions using Mehrabian's empirically measured Pleasure-Arousal-Dominance (PAD) word coordinates as the numeric bridge, and in doing so expands the petal vocabulary into sixteen compound-emotion labels (dyads) and three intensity tiers per petal. The engine additionally emits a per-frame reliability signal derived from cross-representation agreement. The complete pipeline, including a 58K-parameter emotion classifier and a 3.6M-parameter valence-arousal regressor, runs entirely on commodity CPU hardware at interactive frame rates, with total model footprint under 14.2 MB.*

**Index Terms** — affective computing, facial expression recognition, basic emotions, valence-arousal, pleasure-arousal-dominance, compound emotions, dyads, microexpressions, emotion classification, TensorFlow Lite, real-time systems, convolutional neural networks

---

## I. Introduction

### A. Motivation: Faces

The human face is a remarkably dense affective signal. A single expression can carry numerous co-occurring micro-expressions; the expressive repertoire has expanded substantially over evolutionary time; and portrayals still differ widely across individuals and cultures. This project takes that density seriously: it builds a system that observes a face and reports something richer than a single categorical label.

Both halves of that observation have scientific footing, and they pull the design in opposite directions.

**The face is combinatorially deep.** Darwin opened the comparative study of expression, arguing that our displays are evolved, functional behaviors continuous with those of other species [1]. The modern ethological view sharpens this: facial displays are adaptive social instruments, and the value of a social instrument grows with the granularity of the signal it carries, which is a plausible account of why the human expressive repertoire expanded over evolutionary time [2]. At the mechanical level, Ekman and Friesen showed that the face leaks concealed information in sub-second fragments, the phenomena now called micro-expressions [3], and their Facial Action Coding System (FACS) decomposed the visible repertoire into 44 anatomically grounded action units [5] whose combinations run into the thousands of distinguishable configurations. Against a signal space of that size, a seven-way classifier is coarse by orders of magnitude, and the density of the medium is itself the case against categorical-only analysis.

**And yet portrayals differ.** The same tradition established that a small set of expressions is recognized pan-culturally [4], which is why the six basic emotions became the standard vocabulary. But recognition is reliably better within one's own cultural group: expressions carry "dialects," subtle stylistic group differences layered over universals [6]. The second half of the opening observation, that portrayals remain strongly individualized despite a universal substrate, is equally documented.

A system that takes faces seriously therefore has to hold both facts at once: a universal, low-dimensional structure underneath enormous surface variation and individual style. That tension, more than any benchmark, motivated this project.

### B. Why Categorical-Only Analysis Is Not Enough

The dominant applied paradigm, FER-2013-style classification into seven labels [7], inherits the categorical half of the tradition and discards the rest. It has three structural failure modes:

1. **Intensity is discarded.** A classifier that reports "Happy 82%" cannot distinguish serenity from ecstasy. The softmax probability is a property of the model's decision margin, not of the person's face.
2. **Blends are forced into single labels.** Faces routinely mix categories (fear with surprise, anger with disgust). Argmax erases the mixture.
3. **No self-knowledge of failure.** A single model has no principled mechanism for detecting when its own prediction is wrong. Cross-representation agreement, by contrast, is a natural error signal: if the classifier says "Sad" while a regressor places the same face firmly in the positive-valence quadrant, at least one of them is wrong.

Meanwhile, affect science itself has moved past categorical reductionism: large-scale self-report recovers on the order of 27 distinct emotion categories, bridged by continuous gradients rather than partitioned by boundaries [9]. The modern data supports the assessment that the six-category vocabulary is too small, and it adds a structural claim: the remainder is not another flat list but a continuous space.

### C. Theoretical Itinerary

The project assembled its analytical vocabulary from the classical dimensional and combinatorial theories, in the following order (Fig. 1):

```
 Darwin (1872) [1]
      │
 Tomkins (1962) [21] ─── Izard (1971) [22]     Wundt (1896) [23]
      │                       │                      │
 Ekman & Friesen                                     │
 (1969-1978) [3],[4],[5]       Schlosberg (1952) [24]
      │                       │                      │
      │              Mehrabian & Russell (1974) [13]
      │                       │                      │
 SIX BASIC EMOTIONS       RUSSELL CIRCUMPLEX (1980) [10]
      │                       │
      │              PLUTCHIK WHEEL (1980) [11],[12]
      │                       │
      └───────────┬────────────┘
                  ▼
     PAD word-coordinates (1974/1995) [13],[14],[15]
     as the metric bridge ──────────────► AffectFusionEngine
                  │
    standard CV backbone: BlazeFace [17], MediaPipe [18],
    Mini XCEPTION [8],[19], MobileNetV2 [20], TFLite [27]
```

**Fig. 1.** Lineage of the frameworks used, and where this project sits at the confluence.

- **Russell's Circumplex Model (1980)** [10] is the basis of the analysis. It plotted emotions on a geometric scale, the valence-arousal plane, rather than a categorical one: affect words occupy approximately fixed angular positions around a circle, and distance from the origin tracks intensity. This supplies the continuous substrate.
- **Ekman's six basic emotions** [4] were the categorical starting point, and proved too coarse a vocabulary for the granularity this project targets.
- **Plutchik's Wheel of Emotions (1980)** [11], [12] answers that objection combinatorially: eight basic emotions arranged as petals with opposites across, each petal subdivided into three intensity tiers, plus a compositional algebra of "dyads" in which adjacent petals combine into compound emotions (Joy + Trust = Love). Eight basic petals plus eight primary dyads form sixteen distinct emotion labels, with secondary dyads extending further.
- **Mehrabian's PAD model (1974/1995)** [13], [14], [15] is perhaps the most important piece. It supplies empirically measured Pleasure-Arousal-Dominance coordinates for emotion words: the numeric bridge needed to convert continuous geometry back into discrete Plutchik labels without arbitrary hand-tuning.
- **The Affect Fusion Engine** (`emot_recog/affect_fusion.py`) is the software statement of that itinerary. The rest is standard computer vision: MediaPipe BlazeFace [17], [18], Mini Xception [8], [19], and MobileNetV2 regression [20], [22], all under TensorFlow Lite [27], trained on FER-2013 [7] and AffectNet [16].

### D. Contributions

1. A **parallel dual-model inference architecture** in which a Mini Xception classifier [8], [19] and a MobileNetV2 valence-arousal regressor [20], [22] run simultaneously per detected face via a thread pool, each as an independent TensorFlow Lite interpreter.
2. An **Affect Fusion Engine** that:
   - smooths valence and arousal with an exponential moving average,
   - computes a **conflict score** between categorical and dimensional predictions against AffectNet-derived centroids [16], gating downstream inference on their agreement,
   - maps valence-arousal coordinates onto the eight Plutchik petals via nearest-centroid classification over PAD-derived coordinates [14], [15],
   - disambiguates near-ties (notably Fear/Anger, which are nearly coincident in two dimensions) using the categorical prediction as a tiebreaker,
   - infers a three-tier **intensity label** (mild / basic / intense) from the polar magnitude of the valence-arousal vector,
   - detects **compound emotions (dyads)** from the classifier's softmax distribution, following Plutchik's primary and secondary dyad definitions [11], [12].
3. A **quantified system description**: per-layer parameter counts, dataset provenance, preprocessing pipelines, all hyperparameters, and TFLite conversion results, verified against the source code.
4. A documented account of the **psychometric limits of two-dimensional fusion**: the Fear/Anger collinearity in valence-arousal space [15], [25], its origin in the truncation of PAD's Dominance axis, and exactly which of the sixteen dyads are reachable given the categorical bridge.

### E. Paper Organization

Section II reviews the psychological frameworks (categorical, dimensional, Plutchik, PAD) and the machine learning lineage. Section III details the system architecture and fusion methodology. Section IV covers implementation. Section V presents footprints and behavior. Section VI discusses limitations and future work, and Section VII concludes.

---

## II. Psychological Background: The Frameworks Behind the Fusion

### A. The Categorical Tradition: From Darwin to the Six Basic Emotions

The categorical program has three roots. Darwin [1] supplied the evolutionary framing. Tomkins [21] argued that affect is the primary motivational system and the face its dedicated output channel, a thesis Izard developed into differential emotions theory [22]. Ekman and Friesen supplied the cross-cultural evidence: judgments of facial expression are pan-culturally consistent [4], from which the canonical six basic emotions (happiness, sadness, anger, fear, surprise, disgust) were extracted, with neurocultural theory reconciling universality against cultural display rules. The same program produced the two instruments that still define the field's granularity: FACS, an anatomically grounded coding of 44 action units [5], and the observation that faces leak concealed states in micro-expressions [3].

Two later results complicate the neat six-label picture. First, recognition shows reliable in-group advantage: expressions carry cultural dialects over the universal substrate [6]. Second, large-scale self-report [9] recovers an order of 27 categories bridged by gradients, evidence that even rich categorical lists sit on a continuum. The applied vocabulary of FER datasets (FER-2013's seven labels [7]) is therefore a data-driven descendant of the basic-emotions program: useful, universal-ish, and structurally too coarse on its own.

### B. The Dimensional Tradition: From Wundt to the Circumplex

The dimensional program predates the categorical one. Wundt [23] proposed that feelings vary along three bipolar dimensions (pleasure/displeasure, tension/relaxation, excitement/calm). Schlosberg [24], working from facial-expression judgments, reduced this to two dimensions and noted their circular arrangement. Mehrabian and Russell [13] introduced the three-dimensional Pleasure-Arousal-Dominance state space with an explicit measurement battery (Section II-D). Russell [10] then formalized the circumplex: 28 affect words placed at approximately fixed angular positions in the valence-arousal plane. Two structural hypotheses of the circumplex matter for this project:

1. **Circumplex ordering.** Adjacent words in the circular arrangement are more similar in meaning; angular position encodes the qualitative kind of affect.
2. **Radial intensity.** Distance from the origin tracks activation level; a word farther out along the same direction denotes a more intense version of the same affect. This is the hypothesis the engine's intensity tiers (Section III-E5) implement.

The circumplex supplies geometry, not labels. It also, crucially, says nothing about the vertical distance between emotion *words* at similar angles: as Sections II-C and III-E4 show, Fear and Anger live almost on top of each other in the plane.

### C. Plutchik's Wheel: Eight Petals, Three Intensities, a Combinatorial Algebra

Plutchik's psychoevolutionary synthesis [11], [12] begins from the same intuition as the circumplex, that emotions form a structured continuum, but organizes it combinatorially. Eight primary emotions (joy, trust, fear, surprise, sadness, disgust, anger, anticipation) are derived from a similarity scaling of emotion terms and arranged as petals of a flower:

- **Opposites** sit across the wheel (joy/sadness, trust/disgust, fear/anger, surprise/anticipation).
- Each petal carries **three intensity tiers** along a vertical axis: for example annoyance, anger, rage.
- **Composition.** Petal pairs combine into dyads: adjacent petals (one step apart) form **primary dyads** (Love = Joy + Trust, Awe = Fear + Surprise); petals two apart form **secondary dyads** (Pride = Anger + Joy); petals three apart form **tertiary dyads**.

Counting labels: eight basic emotions plus eight primary dyads form sixteen distinct emotion names, with eight secondary dyads on top of that. (Tertiary dyads exist in the full catalogue but are not implemented in this engine; Section III-E6.)

**TABLE I**
**Plutchik Primary and Secondary Dyads** [11], [12]

| Type | Components | Dyad |
|---|---|---|
| Primary | Joy + Trust | Love |
| Primary | Trust + Fear | Submission |
| Primary | Fear + Surprise | Awe |
| Primary | Surprise + Sadness | Disapproval |
| Primary | Sadness + Disgust | Remorse |
| Primary | Disgust + Anger | Contempt |
| Primary | Anger + Anticipation | Aggressiveness |
| Primary | Anticipation + Joy | Optimism |
| Secondary | Joy + Fear | Guilt |
| Secondary | Trust + Surprise | Curiosity |
| Secondary | Fear + Sadness | Despair |
| Secondary | Surprise + Disgust | Unbelief |
| Secondary | Sadness + Anger | Envy |
| Secondary | Disgust + Anticipation | Cynicism |
| Secondary | Anger + Joy | Pride |
| Secondary | Anticipation + Trust | Hope |

One geometric warning is essential, and Semeraro et al. [25] state it plainly: spatial proximity in the wheel is semantic proximity, but the wheel is **not** a valence-arousal plot. Adjacent petals are semantically adjacent, but distinct petals can map to nearly identical valence-arousal coordinates, Fear and Anger being the canonical example. Any system that assigns Plutchik labels from valence-arousal coordinates must therefore handle that degeneracy explicitly (Section III-E4).

### D. Mehrabian's PAD: Pleasure, Arousal, Dominance, and the Empirical Bridge

The PAD model is the load-bearing bridge of this project, so it deserves its own account.

**Origin.** Mehrabian and Russell introduced the three-dimensional framework in their 1974 monograph on environmental psychology [13], as a general state space for affective responses to environments: three bipolar, approximately orthogonal dimensions, Pleasure, Arousal, and Dominance, each measurable by semantic-differential item batteries (happy/unhappy, stimulated/relaxed, controlling/controlled).

**Elaboration.** Mehrabian's 1995 monograph [14] developed the framework into a comprehensive description and measurement system for emotional states: an emotion is a transient point in PAD space, stable individual differences are the corresponding temperament coordinates, and a large vocabulary of emotion words and phrases carries empirically measured (P, A, D) centroids, obtained by averaging rated words. The 1996 paper [15] recast PAD as a general framework for temperament. The practical consequence for engineering is what matters: discrete emotion labels come with **metric** coordinates attached.

**Mapping onto the other two frameworks.** Pleasure is valence and Arousal is arousal; PAD and the circumplex share axes, with PAD adding a third. The repo's bridge table (Table VII) is stated to be the (P, A) projections of Mehrabian's measured word coordinates for the eight Plutchik petals, e.g., Joy at (P, A) = (+0.76, +0.48). Russell's circumplex supplies *ordinal* structure (angles of words around the origin); PAD supplies *metric* centroids (where words actually sit). A nearest-centroid classifier needs the latter.

**TABLE II**
**The PAD Dimensions and Their Role in This Project**

| Dimension | Semantic-differential anchors | Construct | Role here |
|---|---|---|---|
| Pleasure (P) | happy ... unhappy | valence | Axis 1 of the regressor output |
| Arousal (A) | stimulated ... relaxed | activation | Axis 2 of the regressor output |
| Dominance (D) | controlling ... controlled | power, control | **Truncated.** The axis that separates Fear (low D) from Anger (high D) in Mehrabian's measurements [14], [15] |

**Why Dominance matters, and what its absence costs.** In Mehrabian's measured word-space, Fear and Anger have nearly coincident (P, A) coordinates (Table VII: Fear at (-0.64, +0.60), Anger at (-0.51, +0.59), Euclidean distance 0.13) but sit at opposite ends of Dominance. The two-dimensional pipeline here inherits exactly this degeneracy: felt valence-arousal cannot tell terror from rage; only the face (through the categorical branch) or Dominance (absent) can. This is not a bug in the mapping; it is a documented property of the affect space itself [15], [25], and the engine's disambiguation logic (Section III-E4) is designed around it.

### E. What This Project Deliberately Does Not Use

Appraisal theories (the component process model [26]) treat an emotion as the output of a sequence of stimulus-evaluation checks (novelty, goal relevance, coping potential), which is arguably the most causally complete account of affect. It is also unusable here: the checks require contextual and antecedent information that a single face ROI in a webcam frame does not contain. The engine's fusion operates strictly on visible outcomes (labels, probabilities, coordinates), which is a deliberate scope decision.

### F. The Machine Learning Lineage

The engineering substrate is deliberately standard. FER-2013 [7] operationalized categorical FER; the mini-XCEPTION of Arriaga et al. [8], an Xception-style [19] depthwise-separable network with residual blocks, made categorical FER real-time on commodity hardware. AffectNet [16] operationalized the dimensional model "in the wild" with both expression labels and continuous valence-arousal annotations, plus baseline regressors whose errors this project's VA model matches (RMSE ≈ 0.39 / 0.38). MobileNetV2 [22] supplies the regressor backbone with inverted residuals and linear bottlenecks. BlazeFace [17], distributed through MediaPipe [18], supplies a 225 KB SSD-style short-range face detector. Everything executes under TensorFlow Lite [27].

---

## III. System Architecture and Methodology

### A. Pipeline Overview

The end-to-end pipeline (Fig. 2) captures 720×480 mirrored frames from a webcam via OpenCV, converts them to RGB for MediaPipe's BlazeFace detector (minimum detection confidence 0.5), and extracts a bounding-box ROI per detection. Each ROI is preprocessed twice, once for each branch:

- **Categorical branch:** grayscale ROI, resized to 64×64, scaled to [-1, 1] (Eq. 1), batch and channel expanded to (1, 64, 64, 1).
- **Dimensional branch:** color ROI, resized to 224×224, BGR→RGB converted, scaled to [-1, 1], batch expanded to (1, 224, 224, 3).

Both branches are submitted concurrently to a `ThreadPoolExecutor` and their results are joined before fusion:

```
                   ┌───────────────────────────────┐
 Webcam (720×480)  │  Frame capture + RGB conversion │
        ─────────►│  OpenCV / cv2.flip (selfie)     │
                   └───────────────┬───────────────┘
                                   ▼
                   ┌───────────────────────────────┐
                   │  MediaPipe BlazeFace (short)   │
                   │  min_detection_confidence 0.5   │
                   └───────────────┬───────────────┘
                                   ▼  per-detection bounding box
                   ┌───────────────────────────────┐
                   │  ROI split: gray ROI │ color ROI│
                   └──────┬─────────────────┬──────┘
                          ▼                 ▼
        ┌───────────────────────┐ ┌────────────────────────┐
        │  64×64 grayscale      │ │  224×224 RGB           │
        │  Mini Xception (TFLite)│ │  MobileNetV2 (TFLite) │
        │  → 7-class softmax    │ │  → [valence, arousal]  │
        └───────────┬───────────┘ └───────────┬────────────┘
                    └────────────┬──────────────┘
                                 ▼
                   ┌───────────────────────────────┐
                   │  AffectFusionEngine            │
                   │  EMA → Russell check →         │
                   │  PAD-bridged Plutchik map →    │
                   │  intensity → dyads             │
                   └───────────────┬───────────────┘
                                   ▼
                   ┌───────────────────────────────┐
                   │  Overlay + VA HUD + display     │
                   └───────────────────────────────┘
```

**Fig. 2.** Dual-branch parallel inference pipeline with affect fusion.

The normalizing scaling for both branches is:

```
x_norm = (x / 255.0 − 0.5) × 2.0 = 2x/255 − 1                (1)
```

mapping pixel intensities from [0, 255] to [-1, 1].

### B. Face Detection

BlazeFace short-range (Table III) provides axis-aligned bounding boxes. When no detection is present in a frame, the fusion engine's temporal state is explicitly reset (Section III-E2), preventing stale valence-arousal values from bleeding into a subsequent face's smoothing state.

**TABLE III**
**Face Detector Configuration**

| Property | Value |
|---|---|
| Model | BlazeFace short-range [17] |
| Runtime | MediaPipe Tasks API (`mp.tasks.vision.FaceDetector`) |
| Artifact | `models/blaze_face_short_range.tflite` (225 KB) |
| Minimum detection confidence | 0.5 |
| Input format | RGB frame (`mp.Image`, `SRGB`) |
| Output | Bounding boxes (origin clamped to ≥ 0) |

### C. Categorical Branch: Mini Xception

The classifier is a compact Xception variant [8], [19]: two 3×3 convolutions (8 filters each), four residual blocks of doubling width (16→32→64→128), each composed of two depthwise separable convolutions, a 3×3 max-pool, and a 1×1 convolutional skip projection; a final 1×1 convolution to 7 channels, global average pooling, and softmax (Fig. 3).

```
 Input (64×64×1)
    ├─ Conv2D(8, 3×3) → BN → ReLU
    ├─ Conv2D(8, 3×3) → BN → ReLU
    ├─ Residual Block (16) ─┐   SepConv→SepConv→MaxPool
    │                       └── 1×1 Conv skip
    ├─ Residual Block (32)  (same pattern)
    ├─ Residual Block (64)  (same pattern)
    ├─ Residual Block (128) (same pattern)
    ├─ Conv2D(7, 1×1) → GlobalAveragePooling2D
    └─ Softmax → 7 classes
```

**Fig. 3.** Mini Xception categorical branch [8], [19].

**TABLE IV**
**Categorical Branch Specification**

| Property | Value |
|---|---|
| Architecture | Mini Xception [8], [19] |
| Training dataset | FER-2013 (35,887 images, 48×48 grayscale) [7] |
| Reported top accuracy (source) | ~66% [8] |
| Input | (1, 64, 64, 1) float32 in [-1, 1] |
| Output | (1, 7) softmax |
| Classes | 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral |
| Total parameters | 58,423 (1,472 non-trainable BN) |
| TFLite artifact | `models/emotionModel.tflite` (235.6 KB) |

### D. Dimensional Branch: MobileNetV2 Valence-Arousal Regressor

The regressor is a MobileNetV2 backbone [22] (inverted residual blocks with linear bottlenecks, input 224×224×3) followed by GlobalAveragePooling2D, Dense(1024, ReLU), and Dense(2, linear) (Fig. 4). The checkpoint stores weights only (`models/regressor_weights.h5`); the architecture is reconstructed in code before TFLite conversion (Section IV-B). The model was trained on the manually annotated subset of AffectNet [16] with valence and arousal targets in [-1, 1], and its test error matches the AffectNet baselines (RMSE ≈ 0.39 valence / 0.38 arousal) [16].

```
 Input (224×224×3 RGB)
    ├─ Conv2D(32, 3×3, stride=2) → BN → ReLU6
    ├─ Inverted Residual Blocks (t, c, s):
    │     ×1 (1, 16, 1)   ×2 (6, 24, 2)   ×3 (6, 32, 2)   ×4 (6, 64, 2)
    │     ×3 (6, 96, 1)   ×3 (6, 160, 2)  ×1 (6, 320, 1)
    ├─ Conv2D(1280, 1×1) → BN → ReLU6
    ├─ GlobalAveragePooling2D
    ├─ Dense(1024, ReLU)
    └─ Dense(2, linear) → [valence, arousal]
```

**Fig. 4.** MobileNetV2 with regression head [22]. `t` = expansion factor, `c` = output channels, `s` = stride.

**TABLE V**
**Dimensional Branch Specification**

| Property | Value |
|---|---|
| Architecture | MobileNetV2 [22] + Dense(1024) + Dense(2) |
| Training dataset | AffectNet manual-annotation subset [16] |
| Input | (1, 224, 224, 3) float32 RGB in [-1, 1] |
| Output | (1, 2) linear: [valence, arousal], nominal range [-1, 1] |
| Total parameters | 3,571,778 (34,112 non-trainable BN) |
| TFLite artifact | `models/vaModel.tflite` (13.8 MB) |

The final activation is linear, so predictions are unbounded and may slightly exceed [-1, 1]; downstream fusion is safe by construction (all centroids and thresholds are defined on [-1, 1]).

### E. The Affect Fusion Engine

The engine (`emot_recog/affect_fusion.py`) is a stateful per-frame-update object fusing `(emotion_idx, emotion_prob, softmax_probs, valence, arousal)` into a `FusionResult` dataclass. Its stages are as follows.

#### 1) Design position: decision-level fusion

The two branches were trained on different corpora (FER-2013 grayscale vs. AffectNet color), under different label geometries (probability simplex vs. continuous plane), with no shared latent space; feature-level fusion would require retraining. The psychological frameworks operate on observable quantities (a label, a probability distribution, a coordinate), not on internals. And both branches must remain independently runnable: they ship as standalone scripts (`facemot.py`, `faceva.py`). The fusion layer is therefore late (decision-level), additive, and fully separable from both branches.

#### 2) Temporal smoothing (EMA)

Raw per-frame regressor output jitters. Both coordinates are smoothed with an exponential moving average:

```
v_t = α · v_raw + (1 − α) · v_(t−1)
a_t = α · a_raw + (1 − α) · a_(t−1)          with α = 0.3   (2)
```

Step-response analysis: a coordinate reaches 50% of a step change after `n` frames where `(1−α)^n ≤ 0.5`, i.e., `n ≈ ln 0.5 / ln 0.7 ≈ 1.94` frames, and 90% after `0.7^n ≤ 0.1`, i.e., `n ≈ 6.46` frames. At 30 fps that is roughly 65 ms and 215 ms. Two consequences:

- **Micro-expressions are deliberately suppressed.** Sub-second leakage fragments [3] fall inside this smoothing window; the engine tracks macro-expression trajectories, not micro-expression spikes. That is the correct scope for a continuous overlay, and it is a happy coincidence with the motivation: a system fascinated by micro-expressions should at least *know* it is filtering them out.
- **State resets on face loss.** When no detection is present, the EMA state resets, so a new face's first frames are not contaminated by the previous occupant of the frame.

#### 3) Russell consistency check

Each of the 7 categorical classes is associated with an expected centroid `(μ_V, μ_A)` in VA space (Table VI), taken from the empirical AffectNet annotations [16]. The **conflict score** is the Euclidean distance between the smoothed VA position and the centroid of the class predicted by the categorical branch:

```
CS = √[(v_t − μ_V(c))² + (a_t − μ_A(c))²]                    (3)
```

where `c` is the argmax class. A prediction is deemed **reliable** when `CS ≤ τ_CS` with `τ_CS = 0.5`.

**TABLE VI**
**Russell-Check Centroids per Categorical Class** [16]

| Index | Class | μ_Valence | μ_Arousal |
|---|---|---|---|
| 0 | Angry | -0.45 | +0.31 |
| 1 | Disgust | -0.48 | +0.22 |
| 2 | Fear | -0.36 | +0.38 |
| 3 | Happy | +0.55 | +0.28 |
| 4 | Sad | -0.41 | -0.12 |
| 5 | Surprise | +0.13 | +0.45 |
| 6 | Neutral | +0.02 | -0.05 |

Threshold analysis: adjacent-category centroids in Table VI sit 0.1–0.3 apart (e.g., Angry/Disgust 0.095, Angry/Fear 0.114), while gross cross-quadrant errors put the smoothed point ~1.0 away (e.g., Happy/Sad 1.04). `τ_CS = 0.5` therefore sits well above neighborhood uncertainty and well below categorical contradiction: it fires precisely when the two representations *disagree about the kind of affect*, not when they disagree about minor degree. Reliability gates dyad detection (Section III-E6) and dims the overlay when it fails.

#### 4) PAD-bridged Plutchik mapping (nearest centroid, with disambiguation)

The smoothed point `(v_t, a_t)` is assigned to the nearest of the 8 Plutchik petal centroids (Table VII), which the repository documents as the (P, A) projections of Mehrabian's measured PAD word coordinates [14], [15]:

```
petal = argmin_p ∈ P  ‖ (v_t, a_t) − (V_p, A_p) ‖₂          (4)
```

**TABLE VII**
**Plutchik Petal Centroids in VA Space (PAD Projection) [14], [15], with Intensity Labels**

| Petal | V_p | A_p | Mild | Basic | Intense |
|---|---|---|---|---|---|
| Joy | +0.76 | +0.48 | Serenity | Joy | Ecstasy |
| Trust | +0.52 | +0.20 | Acceptance | Trust | Admiration |
| Fear | -0.64 | +0.60 | Apprehension | Fear | Terror |
| Surprise | +0.14 | +0.67 | Distraction | Surprise | Amazement |
| Sadness | -0.63 | -0.27 | Pensiveness | Sadness | Grief |
| Disgust | -0.60 | +0.35 | Boredom | Disgust | Loathing |
| Anger | -0.51 | +0.59 | Annoyance | Anger | Rage |
| Anticipation | +0.22 | +0.62 | Interest | Anticipation | Vigilance |

As established in Section II-D, Eq. 4 is underdetermined exactly where PAD's truncated Dominance axis mattered: Fear and Anger are 0.13 apart in (P, A), closer than any reasonable resolution. The engine handles this with **categorical disambiguation**: if the two smallest centroid distances differ by less than a margin `δ = 0.15`, and the categorical branch's petal (Table VIII) is one of the two tied petals, the tie resolves in its favor. This is psychologically sound: felt valence-arousal does not separate fear from anger (that is Dominance's job [14], [15]), but the *visible facial configuration* does, and that is what the categorical branch reads.

**TABLE VIII**
**Categorical Class → Plutchik Petal Bridge**

| Class index | Class | Petal |
|---|---|---|
| 0 | Angry | Anger |
| 1 | Disgust | Disgust |
| 2 | Fear | Fear |
| 3 | Happy | Joy |
| 4 | Sad | Sadness |
| 5 | Surprise | Surprise |
| 6 | Neutral | none (no Plutchik equivalent) |

The margin `δ = 0.15` is not arbitrary; it is calibrated against the actual degeneracies of Table VII. Computing all pairwise petal-centroid distances:

**TABLE IX**
**Petal Pairs Within the Disambiguation Margin (δ = 0.15)**

| Pair | Distance | Disambiguated by class bridge? |
|---|---|---|
| Surprise / Anticipation | 0.094 | No (neither petal is bridge-reachable) |
| Fear / Anger | 0.130 | Yes (Fear ↔ class 2, Anger ↔ class 0) |
| (next nearest: Fear / Disgust) | 0.253 | outside margin |
| (next nearest: Anger / Disgust) | 0.256 | outside margin |

Only two petal pairs in the PAD projection lie within δ = 0.15. Of these, Fear/Anger is fully handled by the categorical bridge (both classes exist in FER-2013), while Surprise/Anticipation cannot be tiebroken because Anticipation has no FER-2013 class (Table VIII): in that sliver of the plane the engine's assignment is geometry-only. A **neutral override** closes the mapping: if the categorical branch predicts Neutral and the smoothed VA magnitude `r < 0.2`, no petal is assigned (the label is simply "Neutral"), consistent with the circumplex reading of the origin as absence of strong affect [10].

#### 5) Intensity inference

Intensity is a function of polar radius, implementing the circumplex's radial hypothesis [10] directly:

```
r = √(v_t² + a_t²)                                          (5)
θ = atan2(a_t, v_t)   [degrees, HUD visualization only]
intensity = mild    if r < 0.33
          = basic   if 0.33 ≤ r < 0.66
          = intense if r ≥ 0.66                              (6)
```

The resolved label is the petal's intensity-tier word (Table VII), e.g., the Fear petal at `r = 0.81` reads "Terror". Note the tiers are linear partitions of radius, not of each petal's own dynamic range: a petal whose centroid sits near `r ≈ 0.2` (none, in this table) would label most of its territory "intense" under this scheme; with these particular centroids the tiers behave, but the choice is documented rather than derived.

#### 6) Dyad detection

Compound emotions are inferred from the categorical branch's **softmax distribution**, not from VA coordinates, because dyads are petal-adjacency relations [11], [12] and the VA model has no notion of petals. Let `p_1 ≥ p_2` be the top two softmax probabilities and `petal(c_1)`, `petal(c_2)` their petal mappings (Table VIII). A dyad is emitted iff the Russell check passed (Section III-E3) and:

```
p_1 < 0.5   ∧   (p_1 − p_2) < 0.15   ∧   petal(c_1) ≠ petal(c_2) ≠ none   (7)
```

The unordered pair `{petal(c_1), petal(c_2)}` is looked up against the 16-dyad table (TABLE I); the dyad type (primary/secondary) is propagated. The thresholds have clear semantics: `p_1 < 0.5` means no confident single label exists (a genuine mixture, not a confident call), and a gap under 0.15 means the top two are statistically tied. The engine implements all 16 primary and secondary dyads; Plutchik's tertiary dyads are not implemented.

**Reachability.** Because the bridge (Table VIII) can only emit petals from {Anger, Disgust, Fear, Joy, Sadness, Surprise}, exactly 8 of the 16 implemented dyads can ever fire: Contempt (Disgust+Anger), Awe (Fear+Surprise), Remorse (Sadness+Disgust), Guilt (Joy+Fear), Despair (Fear+Sadness), Unbelief (Surprise+Disgust), Envy (Sadness+Anger), and Pride (Anger+Joy). The other 8 (Love, Submission, Aggressiveness, Optimism, Hope, Curiosity, Cynicism, and the Anticipation-bearing half of the table) involve Trust or Anticipation and are unreachable through the categorical bridge. They remain implemented for completeness, and would become reachable if the categorical model were extended to more classes (Section VI).

#### 7) Output vocabulary: sixteen from eight

The label algebra, counted precisely:

- 8 basic petals × 3 intensity words = **24 petal words** (e.g., Annoyance, Anger, Rage);
- + **16 dyad labels** (8 primary, 8 secondary, of which 8 are currently reachable);
- + **Neutral**;
- = **41 distinct output labels** in the engine's vocabulary, of which up to 3 display simultaneously per face per frame (base class + confidence, intensity word, dyad).

The "sixteen from eight" of the title-level framing is the theory's own arithmetic: Plutchik's eight basic emotions plus the eight primary dyads yield sixteen distinct emotion names [11], [12], and it is the same arithmetic the engine implements.

#### 8) Worked examples

Two synthetic traces illustrating the two characteristic regimes (all numbers consistent with Tables IV–IX):

**TABLE X**
**Worked Fusion Traces**

| | Trace A: blend | Trace B: conflict |
|---|---|---|
| Softmax top-2 | Fear 0.34, Surprise 0.28 | Sad 0.42, Neutral 0.21 |
| Eq. 7 conditions | p₁<0.5 ✓, gap 0.06<0.15 ✓ | p₁<0.5 ✓, gap 0.21>0.15 ✗ (no dyad anyway) |
| Smoothed VA (v_t, a_t) | (−0.60, +0.55) | (+0.45, +0.10) |
| r (Eq. 5) | 0.81 → intense | 0.46 → basic |
| Conflict score (Eq. 3) | vs Fear centroid: 0.29 ≤ 0.5 → **reliable** | vs Sad centroid: 0.89 > 0.5 → **unreliable** |
| Nearest petals | Fear 0.064, Anger 0.098 → tie gap 0.03 < δ → categorical bridge selects **Fear** | Trust 0.12, Joy 0.49 → no tie → **Trust** |
| Intensity word | Terror | Trust |
| Dyad | {Fear, Surprise} → **Awe** (primary) | suppressed (unreliable) |
| Overlay | "Fear 34%" / "~ Terror" / "+ Awe", full color | "Sad 42%\*" / "~ Trust", box dimmed 50%, no dyad |

Trace A is the system working as designed: a blended expression surfaces a compound emotion with intensity. Trace B is the system's most valuable failure mode: the representations disagree, and rather than committing to a wrong compound label, it flags the disagreement, suppresses dyad inference, dims the box, and shows what each branch believes.

#### 9) Output contract

The engine returns a `FusionResult` dataclass: raw inputs; smoothed VA; conflict score and reliability flag; polar `(r, θ)`; petal, intensity tier, intensity-specific label, petal BGR color; dyad name and type. All rendering is driven exclusively by this contract, cleanly separating perception (branches), fusion (engine), and presentation (overlay).

### F. Concurrency Model

Each per-face inference pair is submitted as two futures to a `ThreadPoolExecutor(max_workers=2)`:

```python
emot_future = executor.submit(run_emotion, grayFace.copy())
va_future   = executor.submit(run_va, colorFace.copy())
```

The main thread joins both results. Because the two TFLite interpreters execute in C++ and release the Python GIL during `invoke()` [27], the two branches overlap despite CPython's threading constraints, and the smaller emotion model's latency hides under the VA model's. ROI buffers are copied before submission to avoid mutation of numpy views during async resize.

---

## IV. Implementation

### A. Runtime

All three networks run as TensorFlow Lite interpreters (`tf.lite.Interpreter`), each with `allocate_tensors()` called once at startup. Per-frame inference uses direct tensor dispatch (`set_tensor` / `invoke` / `get_tensor`) rather than `model.predict()`, eliminating per-call graph tracing and session overhead [27]. Input shapes are read dynamically from `get_input_details()` so the code cannot drift from the artifacts.

### B. TFLite Conversion

Both production artifacts are generated from the repo's conversion scripts:

- **Emotion** (`convert_to_tflite.py`): the HDF5 file contains the full model (architecture + weights). Conversion is direct: `load_model` → `TFLiteConverter.from_keras_model` → `convert()`.
- **Valence-arousal** (`convert_va_to_tflite.py`): the H5 file stores weights only, so the architecture (MobileNetV2 without top, `input_shape=(224, 224, 3)`, plus `GlobalAveragePooling2D → Dense(1024, relu) → Dense(2, linear)`) is reconstructed in code before `load_weights()` and conversion.

### C. Visualization

The fused result drives a layered overlay per face: a bounding box colored by petal (dimmed 50% when the Russell check fails); a primary label `"{class} {conf%}"` (asterisked when unreliable); a secondary line `~ {intensity word}` when it differs from the base class name; a tertiary line `+ {dyad}` when a dyad is detected; and a 150×150 valence-arousal HUD in the top-right corner plotting the smoothed position as a colored dot against quadrant labels ("Pleasant+", "Unpleasant+", etc.) with a numeric `V:/A:` readout.

### D. Software Stack

**TABLE XI**
**Software Dependencies (reqs.txt)**

| Package | Version | Role |
|---|---|---|
| tensorflow | 2.21.0 | TFLite interpreter, Keras conversion path |
| opencv-python | 4.13.0.92 | Capture, preprocessing, overlay, display |
| mediapipe | ≥ 0.10.14 | BlazeFace detection (Tasks API) |
| numpy | 2.4.4 | ROI math, softmax handling, fusion |
| h5py | 3.14.0 | HDF5/H5 weight file access |

---

## V. Results and Discussion

### A. Model Footprints

**TABLE XII**
**Model Comparison**

| Property | Categorical | Dimensional |
|---|---|---|
| Architecture | Mini Xception [8], [19] | MobileNetV2 + head [22] |
| Task | 7-way classification | 2-D regression |
| Dataset | FER-2013 [7] | AffectNet [16] |
| Input | 64×64 × 1ch | 224×224 × 3ch |
| Parameters | 58,423 | 3,571,778 (~61×) |
| TFLite size | 235.6 KB | 13.8 MB (~57×) |
| Output | Softmax over 7 | (valence, arousal) |
| Output activation | Softmax | Linear |

The two branches are asymptotically mismatched by design: the categorical model is 57–61× smaller than the dimensional one, so per-frame latency is dominated by the VA branch. The ThreadPoolExecutor arrangement (Section III-F) therefore yields roughly the latency of the VA branch alone, rather than the sum of both.

### B. TFLite Conversion Results

**TABLE XIII**
**Conversion Outcomes**

| Model | Source artifact | Source size | TFLite size | Reduction |
|---|---|---|---|---|
| Emotion | `emotionModel.hdf5` (full model) | 852.4 KB | 235.6 KB | 72.4% |
| Valence-arousal | `regressor_weights.h5` (weights only) | 14.0 MB | 13.8 MB | 3.7% |

The emotion model's large reduction reflects HDF5's storage of architecture metadata, optimizer state, and training config, all stripped by the FlatBuffer; the VA weights file was already lean, so its reduction is marginal.

### C. Qualitative Fusion Behavior

Observed behaviors of the fused pipeline, with mechanism attributions:

- **Stable trajectories.** Eq. 2's EMA (α = 0.3) removes most per-frame flicker of the raw regressor; petal colors transition smoothly rather than strobing.
- **Self-flagged disagreement.** When the classifier labels a face "Sad" while the regressor reports positive valence, `CS > τ_CS` in Eq. 3, the overlay dims and appends an asterisk, and dyad detection is suppressed (Eq. 7's reliability gate). The system degrades gracefully by reporting disagreement rather than committing to a wrong compound label.
- **Intensity enrichment.** Sustained high-magnitude joy (r ≥ 0.66 by Eq. 5) upgrades the label from "Happy" to "Ecstasy"; mild expressions (r < 0.33) produce tier-appropriate words ("Serenity", "Pensiveness") from Table VII.
- **Dyads.** Blended expressions whose top-2 softmax probabilities satisfy Eq. 7 surface compound labels, e.g., simultaneous Fear/Surprise mass → "Awe", consistent with Plutchik's compositional semantics [11], [12], [25].

### D. Positioning Relative to Related Systems

Single-representation baselines report FER-2013 accuracy of ~66% for this architecture family [8] and AffectNet valence-arousal RMSE of ~0.39/0.38 [16]; those figures bound the fused system's component accuracy, since fusion is a post-hoc rule layer over both models and neither branch is fine-tuned for fusion. The value added is **representational**: intensity tiers, compound labels, and a reliability signal that neither branch can produce alone, at negligible additional compute (the fusion engine is O(1) numpy per frame). In the author's own extended testing, the system's most frequent verdict on its developer was Joy.

---

## VI. Limitations and Future Work

1. **The truncated Dominance axis.** The single deepest limitation is architectural: PAD's third dimension, the one that separates Fear (low D) from Anger (high D) in Mehrabian's measurements [14], [15], is absent from the pipeline. The engine's Fear/Anger disambiguation (Section III-E4) delegates to the categorical branch, which mitigates but does not resolve the degeneracy. The principled fix is a three-dimensional (P, A, D) regressor; AffectNet does not annotate Dominance, so this requires either a new annotation source or an ordinal proxy (e.g., a gaze/pose-derived control signal).
2. **Unreachable petals and dyads.** Trust and Anticipation have no FER-2013 class (Table VIII), so they are reachable only via pure VA proximity, never via disambiguation, and 8 of the 16 implemented dyads can never fire (Section III-E6). A categorical model trained on a richer label set (e.g., an 8-class Plutchik-aligned corpus, or GoEmotions-style affect labels) would activate the full dyad table without changing the engine.
3. **Population-level centroids.** Tables VI and VII are population averages [14], [15], [16]; individual differences in expressive style [6] are unmodeled. The thresholds (τ_CS = 0.5, δ = 0.15, radii 0.33/0.66, dyad 0.5/0.15) are hand-set and fit to the centroid geometry rather than to held-out data. Person-specific calibration is an obvious extension.
4. **Dataset noise and bias.** FER-2013 carries well-documented label noise and class imbalance (Happy overrepresented, Disgust scarce) [7]; AffectNet annotations are subjective with imperfect inter-rater agreement [16]. Both upstream artifacts are consumed as-is, without demographic audit, so the system inherits whatever biases they encode.
5. **Single-face EMA state.** The engine keeps one EMA pair; with multiple simultaneous faces, smoothing state may blend across identities when detection order changes. Per-track state keyed on face identity (tracking-by-detection, using BlazeFace landmarks for cheap association) is straightforward future work.
6. **GIL-bounded parallelism.** The thread pool overlaps the two interpreters only because TFLite's C++ kernel releases the GIL [27]; scaling to many faces would benefit from process pools or a batched single-interpreter design.
7. **No quantitative fusion evaluation.** The fusion layer is evaluated qualitatively; a held-out protocol (agreement of dyad labels with human annotation on a blend-labeled corpus, and calibration analysis of the conflict score against observed misclassifications) remains open.

---

## VII. Conclusion

This project set out to take faces as seriously as they deserve: not as seven mutually exclusive checkboxes, but as a dense, continuous, combinatorial signal sitting on top of a universal low-dimensional structure. The engineering expression of that stance is a pipeline that runs a categorical classifier (Mini Xception, FER-2013) and a dimensional regressor (MobileNetV2, AffectNet) concurrently per face, and fuses them through classical psychological structure: Russell's circumplex for geometric consistency checking [10], Plutchik's wheel for intensity and compound-emotion semantics [11], [12], and Mehrabian's PAD coordinates as the measured, metric bridge between continuous geometry and discrete labels [13], [14], [15]. The resulting system reports not just *what* emotion a face shows, but *how strongly* (intensity tiers), *how reliably* (conflict score), and, when the evidence supports it, *what blend* (dyads): a 41-label vocabulary produced from 8 petals, sixteen of which (8 basic plus 8 primary dyads) are the theory's own arithmetic. Total model cost is 14.2 MB, and per-frame latency is roughly that of a single MobileNetV2 inference, on commodity CPU hardware. The components are open artifacts in this repository, and the engine's clean dataclass contract makes the natural next steps concrete: 3-D PAD regression to restore the missing Dominance axis, per-face temporal state, and a formal evaluation of fusion quality.

---

## References

[1] C. Darwin, *The Expression of the Emotions in Man and Animals*. London, U.K.: John Murray, 1872.

[2] A. J. Fridlund, *Human Facial Expression: An Evolutionary View*. San Diego, CA, USA: Academic Press, 1994.

[3] P. Ekman and W. V. Friesen, "Nonverbal leakage and clues to deception," *Psychiatry*, vol. 32, no. 1, pp. 88–105, 1969.

[4] P. Ekman, E. R. Sorenson, and W. V. Friesen, "Pan-cultural elements in facial expressions of emotion," *Science*, vol. 164, no. 3878, pp. 86–88, 1969.

[5] P. Ekman and W. V. Friesen, *Facial Action Coding System: A Technique for the Measurement of Facial Action*. Palo Alto, CA, USA: Consulting Psychologists Press, 1978.

[6] H. A. Elfenbein and N. Ambady, "On the universality and cultural specificity of emotion recognition: A meta-analysis," *Psychol. Bull.*, vol. 128, no. 2, pp. 203–235, 2002.

[7] I. J. Goodfellow, D. Erhan, P. L. Carrier, A. Courville, M. Mirza, B. Hamner, W. Cukierski, Y. Tang, D. Thaler, D.-H. Lee, Y. Zhou, C. Ramaiah, F. Feng, R. Li, X. Wang, D. Athanasakis, J. Shawe-Taylor, M. Milakov, J. Park, R. Ionescu, M. Popescu, C. Grozea, J. Bergstra, J. Xie, L. Romaszko, B. Xu, Z. Chuang, and Y. Bengio, "Challenges in representation learning: A report on three machine learning contests," in *Proc. ICML Workshop Challenges Represent. Learn.*, Atlanta, GA, USA, 2013. [Online]. Available: https://arxiv.org/abs/1307.0414

[8] O. Arriaga, M. Valdenegro-Toro, and P. Plöger, "Real-time convolutional neural networks for emotion and gender classification," 2017, arXiv:1710.07557. [Online]. Available: https://arxiv.org/abs/1710.07557

[9] A. S. Cowen and D. Keltner, "Self-report captures 27 distinct categories of emotion bridged by continuous gradients," *Proc. Nat. Acad. Sci. USA*, vol. 114, no. 38, pp. E7900–E7909, 2017, doi: 10.1073/pnas.1702247114.

[10] J. A. Russell, "A circumplex model of affect," *J. Pers. Soc. Psychol.*, vol. 39, no. 6, pp. 1161–1178, 1980, doi: 10.1037/h0077714.

[11] R. Plutchik, *Emotion: A Psychoevolutionary Synthesis*. New York, NY, USA: Harper & Row, 1980.

[12] R. Plutchik, "The nature of emotions: Human emotions have deep evolutionary roots, a fact that may explain their complexity and provide tools for clinical practice," *Amer. Sci.*, vol. 89, no. 4, pp. 344–350, 2001, doi: 10.1511/2001.4.344.

[13] A. Mehrabian and J. A. Russell, *An Approach to Environmental Psychology*. Cambridge, MA, USA: MIT Press, 1974.

[14] A. Mehrabian, "Framework for a comprehensive description and measurement of emotional states," *Genet. Soc. Gen. Psychol. Monogr.*, vol. 121, no. 3, pp. 339–361, 1995.

[15] A. Mehrabian, "Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament," *Current Psych.*, vol. 14, no. 4, pp. 261–292, 1996.

[16] A. Mollahosseini, B. Hasani, and M. H. Mahoor, "AffectNet: A database for facial expression, valence, and arousal computing in the wild," *IEEE Trans. Affect. Comput.*, vol. 10, no. 1, pp. 18–31, 2019, doi: 10.1109/TAFFC.2017.2740923.

[17] V. Bazarevsky, Y. Kartynnik, A. Vakunov, K. Raveendran, and M. Grundmann, "BlazeFace: Sub-millisecond neural face detection on mobile GPUs," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, Long Beach, CA, USA, 2019. [Online]. Available: https://arxiv.org/abs/1907.05047

[18] C. Lugaresi, J. Tang, H. Nash, C. McClanahan, E. Uboweja, M. Hays, F. Zhang, C.-L. Chang, M. G. Yong, J. Lee, W.-T. Chang, W. Hua, M. Georg, and M. Grundmann, "MediaPipe: A framework for building perception pipelines," 2019, arXiv:1906.08172. [Online]. Available: https://arxiv.org/abs/1906.08172

[19] F. Chollet, "Xception: Deep learning with depthwise separable convolutions," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Honolulu, HI, USA, 2017, pp. 1251–1258. [Online]. Available: https://arxiv.org/abs/1610.02357

[20] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Salt Lake City, UT, USA, 2018, pp. 4510–4520. [Online]. Available: https://arxiv.org/abs/1801.04381

[21] S. S. Tomkins, *Affect Imagery Consciousness: Vol. I. The Positive Affects*. New York, NY, USA: Springer, 1962.

[22] C. E. Izard, *The Face of Emotion*. New York, NY, USA: Appleton-Century-Crofts, 1971.

[23] W. Wundt, *Grundriss der Psychologie*. Leipzig, Germany: Engelmann, 1896.

[24] H. Schlosberg, "The description of facial expressions in terms of two dimensions," *J. Exp. Psychol.*, vol. 44, no. 4, pp. 229–237, 1952, doi: 10.1037/h0055778.

[25] A. Semeraro, S. Vilella, and G. Ruffo, "PyPlutchik: Visualising and comparing emotion-annotated corpora," *PLOS ONE*, vol. 16, no. 9, p. e0256503, 2021, doi: 10.1371/journal.pone.0256503.

[26] K. R. Scherer, "On the nature and function of emotion: A component process model approach," in *Approaches to Emotion*, K. R. Scherer and P. Ekman, Eds. Hillsdale, NJ, USA: Erlbaum, 1984.

[27] TensorFlow Authors, "TensorFlow Lite converter and interpreter," *TensorFlow Documentation*. [Online]. Available: https://www.tensorflow.org/lite

---

*All architecture parameters, hyperparameters, artifact sizes, preprocessing details, and fusion logic in this report are transcribed from and verified against the repository source: `emot_recog/facecombined.py`, `emot_recog/affect_fusion.py`, `emot_recog/facemot.py`, `emot_recog/faceva.py`, `emot_recog/convert_to_tflite.py`, `emot_recog/convert_va_to_tflite.py`, and `reqs.txt`. Psychological framework details and all citations were verified against the primary literature during preparation; the historical lineage (Wundt → Schlosberg → Mehrabian/Russell → Russell; Tomkins → Izard → Ekman/Friesen; Plutchik 1980) follows the standard account in the cited sources.*