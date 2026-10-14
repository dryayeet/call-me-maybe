# Real-Time Dual-Model Facial Affect Analysis via Fusion of Categorical and Dimensional Predictions Using Russell's Circumplex, Plutchik's Wheel of Emotions, and Mehrabian's PAD Framework

Manipal Academy of Higher Education

October 14, 2026

---

## Abstract

*Automatic facial expression analysis systems typically commit to a single representation of affect: either discrete emotion categories (e.g., happy, sad, angry) or continuous dimensional scores (valence and arousal). Categorical models offer interpretability but discard intensity and blend information; dimensional models capture graded affect but produce coordinates that are difficult to communicate to end users. This project presents a real-time webcam-based affect analysis system that runs both representations concurrently and fuses them under classical psychological structure. A Mini Xception convolutional network trained on FER-2013 performs 7-class expression classification, while a MobileNetV2 regressor trained on AffectNet predicts continuous valence and arousal. Both models execute in parallel per detected face as TensorFlow Lite interpreters, and MediaPipe BlazeFace supplies face detection. An Affect Fusion Engine cross-validates the two model outputs using Russell's Circumplex Model of Affect, maps fused coordinates onto the eight petals of Plutchik's Wheel of Emotions using Mehrabian's empirically measured Pleasure-Arousal-Dominance (PAD) word coordinates as the numeric bridge, and in doing so expands the petal vocabulary into sixteen compound-emotion labels (dyads) and three intensity tiers per petal. The engine additionally emits a per-frame reliability signal derived from cross-representation agreement. A first quantitative evaluation on FER-2013 with FER+ annotator distributions shows the signal is genuine but weak: the conflict score detects classifier errors (AUROC 0.65) and adds a small, statistically reliable gain when combined with softmax confidence (ΔAUROC +0.011), while softmax confidence alone remains the stronger detector (AUROC 0.78); images that trigger a dyad do carry elevated human annotator disagreement (58% vs. 43%), but the specific blend identity is rarely corroborated. The complete pipeline, including a 58K-parameter emotion classifier and a 3.6M-parameter valence-arousal regressor, runs entirely on CPU at a measured median of 20.5 ms per frame for one face (about 49 FPS before capture and rendering) on a 4-core laptop processor, with total model footprint under 14.2 MB.*

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
- **Mehrabian's PAD model (1974/1995)** [13], [14], [15] is perhaps the most important piece. It supplies empirically measured Pleasure-Arousal-Dominance coordinates for emotion words [28]: the numeric bridge needed to convert continuous geometry back into discrete Plutchik labels without arbitrary hand-tuning.
- **The Affect Fusion Engine** (`emot_recog/affect_fusion.py`) is the software statement of that itinerary. The rest is standard computer vision: MediaPipe BlazeFace [17], [18], Mini Xception [8], [19], and MobileNetV2 regression [20], all under TensorFlow Lite [27], trained on FER-2013 [7] and AffectNet [16].

### D. Contributions

1. A **dual-model inference pipeline** in which a Mini Xception classifier [8], [19] and a MobileNetV2 valence-arousal regressor [20] run per detected face as independent TensorFlow Lite interpreters, with measured per-component latency (Table XIII).
2. An **Affect Fusion Engine** that:
   - smooths valence and arousal with an exponential moving average,
   - computes a **conflict score** between categorical and dimensional predictions against per-category centroids in AffectNet valence-arousal space [16], gating downstream inference on their agreement,
   - maps valence-arousal coordinates onto the eight Plutchik petals via nearest-centroid classification over published PAD word-rating coordinates [28],
   - disambiguates near-ties (notably Fear/Anger, which are nearly coincident in two dimensions) using the categorical prediction as a tiebreaker,
   - infers a three-tier **intensity label** (mild / basic / intense) from the polar magnitude of the valence-arousal vector,
   - detects **compound emotions (dyads)** from the classifier's softmax distribution, following Plutchik's primary and secondary dyad definitions [11], [12].
3. A **quantified system description**: per-layer parameter counts, dataset provenance, preprocessing pipelines, all hyperparameters, TFLite conversion results, and measured per-component latency, verified against the source code.
4. A documented account of the **psychometric limits of two-dimensional fusion**: the Fear/Anger collinearity in valence-arousal space [25], [28], its origin in the truncation of PAD's Dominance axis, and exactly which of the sixteen dyads are reachable given the categorical bridge.
5. A **first quantitative evaluation** of the fusion layer on FER-2013 with FER+ annotator distributions [38], including an out-of-domain EmoNet [34] baseline, with all component accuracy figures re-measured on the exact artifacts in this repository.

### E. Paper Organization

Section II reviews the psychological frameworks (categorical, dimensional, Plutchik, PAD) and the machine learning lineage. Section III details the system architecture and fusion methodology. Section IV covers implementation. Section V presents measured latency, expected behavior, a quantitative evaluation, and positioning. Section VI discusses limitations and future work, and Section VII concludes.

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

| Type      | Components             | Dyad           |
| --------- | ---------------------- | -------------- |
| Primary   | Joy + Trust            | Love           |
| Primary   | Trust + Fear           | Submission     |
| Primary   | Fear + Surprise        | Awe            |
| Primary   | Surprise + Sadness     | Disapproval    |
| Primary   | Sadness + Disgust      | Remorse        |
| Primary   | Disgust + Anger        | Contempt       |
| Primary   | Anger + Anticipation   | Aggressiveness |
| Primary   | Anticipation + Joy     | Optimism       |
| Secondary | Joy + Fear             | Guilt          |
| Secondary | Trust + Surprise       | Curiosity      |
| Secondary | Fear + Sadness         | Despair        |
| Secondary | Surprise + Disgust     | Unbelief       |
| Secondary | Sadness + Anger        | Envy           |
| Secondary | Disgust + Anticipation | Cynicism       |
| Secondary | Anger + Joy            | Pride          |
| Secondary | Anticipation + Trust   | Hope           |

One geometric warning is essential, and Semeraro et al. [25] state it plainly: spatial proximity in the wheel is semantic proximity, but the wheel is **not** a valence-arousal plot. Adjacent petals are semantically adjacent, but distinct petals can map to nearly identical valence-arousal coordinates, Fear and Anger being the canonical example. Any system that assigns Plutchik labels from valence-arousal coordinates must therefore handle that degeneracy explicitly (Section III-E4).

### D. Mehrabian's PAD: Pleasure, Arousal, Dominance, and the Empirical Bridge

The PAD model is the load-bearing bridge of this project, so it deserves its own account.

**Origin.** Mehrabian and Russell introduced the three-dimensional framework in their 1974 monograph on environmental psychology [13], as a general state space for affective responses to environments: three bipolar, approximately orthogonal dimensions, Pleasure, Arousal, and Dominance, each measurable by semantic-differential item batteries (happy/unhappy, stimulated/relaxed, controlling/controlled).

**Elaboration.** Mehrabian's 1995 monograph [14] developed the framework into a comprehensive description and measurement system for emotional states: an emotion is a transient point in PAD space, stable individual differences are the corresponding temperament coordinates, and a large vocabulary of emotion words and phrases carries empirically measured (P, A, D) centroids, obtained by averaging rated words. The 1996 paper [15] recast PAD as a general framework for temperament. The practical consequence for engineering is what matters: discrete emotion labels come with **metric** coordinates attached.

**Mapping onto the other two frameworks.** Pleasure is valence and Arousal is arousal; PAD and the circumplex share axes, with PAD adding a third. Russell's circumplex supplies *ordinal* structure (angles of words around the origin); PAD supplies *metric* centroids (where words actually sit). A nearest-centroid classifier needs the latter.

**Provenance of the engine's coordinates.** For six of the eight petals, the centroids used by the engine (Table VII) trace to published PAD ratings of emotion terms: the Russell and Mehrabian ratings of 151 terms on a [-1, 1] scale [28] (values as reproduced in [29]) and Mehrabian's PAD emotion-scale terms [30]. Several petals are represented by the nearest available *word* rather than the petal name: Joy by *joyful*, Fear by *fearful*, Sadness by *sad*, Disgust by *disgusted*, Anger by *angry*, and Anticipation by *curious*. Two entries are not supported by any located source: the Surprise valence (0.14 in the engine, whereas *surprised* is rated +0.40 in [28]) and the Trust centroid, for which no matching term was found. These are word ratings, i.e., semantic judgments of emotion vocabulary with substantial standard deviations, not measurements of facial expressions. Using them as facial-affect centroids is a modeling assumption and is stated as such.

**TABLE II**
**The PAD Dimensions and Their Role in This Project**

| Dimension     | Semantic-differential anchors | Construct      | Role here                                                                                              |
| ------------- | ----------------------------- | -------------- | ------------------------------------------------------------------------------------------------------ |
| Pleasure (P)  | happy ... unhappy             | valence        | Axis 1 of the regressor output                                                                         |
| Arousal (A)   | stimulated ... relaxed        | activation     | Axis 2 of the regressor output                                                                         |
| Dominance (D) | controlling ... controlled    | power, control | **Truncated.** The axis that separates Fear (low D) from Anger (high D) in PAD word ratings [28] |

**Why Dominance matters, and what its absence costs.** In the published PAD word ratings, *fearful* and *angry* have nearly coincident (P, A) coordinates (Table VII: (-0.64, +0.60) and (-0.51, +0.59), Euclidean distance 0.13) but opposite-signed Dominance (D = -0.43 and +0.25 respectively) [28]. The two-dimensional pipeline here inherits exactly this degeneracy: felt valence-arousal cannot tell terror from rage; only the face (through the categorical branch) or Dominance (absent) can. This is not a bug in the mapping; it is a documented property of the affect space itself [25], [28], and the engine's disambiguation logic (Section III-E4) is designed around it.

### E. What This Project Deliberately Does Not Use

Appraisal theories (the component process model [26]) treat an emotion as the output of a sequence of stimulus-evaluation checks (novelty, goal relevance, coping potential), which is arguably the most causally complete account of affect. It is also unusable here: the checks require contextual and antecedent information that a single face ROI in a webcam frame does not contain. The engine's fusion operates strictly on visible outcomes (labels, probabilities, coordinates), which is a deliberate scope decision.

### F. The Machine Learning Lineage

The engineering substrate is deliberately standard. FER-2013 [7] operationalized categorical FER; the mini-XCEPTION of Arriaga et al. [8], an Xception-style [19] depthwise-separable network with residual blocks, made categorical FER real-time on commodity hardware. AffectNet [16] operationalized the dimensional model "in the wild" with both expression labels and continuous valence-arousal annotations, plus baseline regressors. The upstream source of this project's VA checkpoint reports errors comparable to those baselines (RMSE ≈ 0.39 / 0.38); that figure is inherited, not re-measured here. The classifier's accuracy is re-measured in Section V-C. MobileNetV2 [20] supplies the regressor backbone with inverted residuals and linear bottlenecks. BlazeFace [17], distributed through MediaPipe [18], supplies a 225 KB SSD-style short-range face detector. Everything executes under TensorFlow Lite [27].

### G. Closest Prior Work

Three lines of work overlap directly with this project, and each constrains what it can claim.

**Compound expressions.** Du et al. [31] showed that compound expressions (e.g., happily surprised, fearfully angry) are produced with distinct and consistent facial configurations, and defined 21 categories. RAF-DB [32], [33] brought compound expressions to in-the-wild data with 11 compound classes. These works treat compounds as perceptual categories with their own labels. This engine instead infers Plutchik dyads post hoc from a basic-category classifier, which is a weaker form of evidence (Section III-E6).

**Joint categorical-dimensional models.** EmoNet [34] predicts discrete expression and continuous valence-arousal from a single network, and the Aff-Wild2 dataset and ABAW challenges [35], [36] made multi-task affect recognition (expressions, valence-arousal, action units) a standard benchmark. These models learn both representations jointly. This project fuses two independently trained models through explicit psychological rules. Joint models are likely more accurate per representation. The rule layer's claimed advantages are interpretability, swappable components, and an explicit cross-representation disagreement signal; whether that signal is useful is an empirical question (Section VI).

**Ambiguity and uncertainty in FER.** A substantial literature attributes ambiguity in FER labels and predictions to ambiguous expressions, low image quality, and annotator subjectivity [37], and models it with crowd-sourced label distributions [38] or label-distribution learning [39]. This literature is the main counter-argument to the dyad heuristic and is addressed in Section III-E6.

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

| Property                     | Value                                                  |
| ---------------------------- | ------------------------------------------------------ |
| Model                        | BlazeFace short-range [17]                             |
| Runtime                      | MediaPipe Tasks API (`mp.tasks.vision.FaceDetector`) |
| Artifact                     | `models/blaze_face_short_range.tflite` (225 KB)      |
| Minimum detection confidence | 0.5                                                    |
| Input format                 | RGB frame (`mp.Image`, `SRGB`)                     |
| Output                       | Bounding boxes (origin clamped to ≥ 0)                |

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

| Property                       | Value                                                                    |
| ------------------------------ | ------------------------------------------------------------------------ |
| Architecture                   | Mini Xception [8], [19]                                                  |
| Training dataset               | FER-2013 (35,887 images, 48×48 grayscale) [7]                           |
| Reported top accuracy (source) | ~66% [8]                                                                 |
| Input                          | (1, 64, 64, 1) float32 in [-1, 1]                                        |
| Output                         | (1, 7) softmax                                                           |
| Classes                        | 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral |
| Total parameters               | 58,423 (1,472 non-trainable BN)                                          |
| TFLite artifact                | `models/emotionModel.tflite` (235.6 KB)                                |

### D. Dimensional Branch: MobileNetV2 Valence-Arousal Regressor

The regressor is a MobileNetV2 backbone [20] (inverted residual blocks with linear bottlenecks, input 224×224×3) followed by GlobalAveragePooling2D, Dense(1024, ReLU), and Dense(2, linear) (Fig. 4). The checkpoint stores weights only (`models/regressor_weights.h5`); the architecture is reconstructed in code before TFLite conversion (Section IV-B). The model was trained on the manually annotated subset of AffectNet [16] with valence and arousal targets in [-1, 1], and its upstream source reports test error comparable to the AffectNet baselines [16] (RMSE ≈ 0.39 valence / 0.38 arousal). This figure is inherited, not re-measured here.

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

**Fig. 4.** MobileNetV2 with regression head [20]. `t` = expansion factor, `c` = output channels, `s` = stride.

**TABLE V**
**Dimensional Branch Specification**

| Property         | Value                                                    |
| ---------------- | -------------------------------------------------------- |
| Architecture     | MobileNetV2 [20] + Dense(1024) + Dense(2)                |
| Training dataset | AffectNet manual-annotation subset [16]                  |
| Input            | (1, 224, 224, 3) float32 RGB in [-1, 1]                  |
| Output           | (1, 2) linear: [valence, arousal], nominal range [-1, 1] |
| Total parameters | 3,571,778 (34,112 non-trainable BN)                      |
| TFLite artifact  | `models/vaModel.tflite` (13.8 MB)                      |

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

- **Micro-expressions are deliberately suppressed.** Sub-second leakage fragments [3] fall inside this smoothing window; the engine tracks macro-expression trajectories, not micro-expression spikes. That is the intended scope for a continuous overlay. Capturing micro-expressions would require high-frame-rate capture and no smoothing, which is outside this system's design.
- **State resets on face loss.** When no detection is present, the EMA state resets, so a new face's first frames are not contaminated by the previous occupant of the frame.

#### 3) Russell consistency check

Each of the 7 categorical classes is associated with an expected centroid `(μ_V, μ_A)` in VA space (Table VI). These values are intended to approximate per-category mean valence-arousal in AffectNet [16]. However, AffectNet does not publish per-category means, and no source for the specific numbers was located, so they should be treated as hand-estimated until recomputed from the AffectNet annotation files. The **conflict score** is the Euclidean distance between the smoothed VA position and the centroid of the class predicted by the categorical branch:

```
CS = √[(v_t − μ_V(c))² + (a_t − μ_A(c))²]                    (3)
```

where `c` is the argmax class. A prediction is deemed **reliable** when `CS ≤ τ_CS` with `τ_CS = 0.5`.

**TABLE VI**
**Russell-Check Centroids per Categorical Class (Hand-Estimated; See Text)**

| Index | Class    | μ_Valence | μ_Arousal |
| ----- | -------- | ---------- | ---------- |
| 0     | Angry    | -0.45      | +0.31      |
| 1     | Disgust  | -0.48      | +0.22      |
| 2     | Fear     | -0.36      | +0.38      |
| 3     | Happy    | +0.55      | +0.28      |
| 4     | Sad      | -0.41      | -0.12      |
| 5     | Surprise | +0.13      | +0.45      |
| 6     | Neutral  | +0.02      | -0.05      |

Threshold analysis: adjacent-category centroids in Table VI sit 0.1–0.3 apart (e.g., Angry/Disgust 0.095, Angry/Fear 0.114), while gross cross-quadrant errors put the smoothed point ~1.0 away (e.g., Happy/Sad 1.04). `τ_CS = 0.5` therefore sits well above neighborhood uncertainty and well below categorical contradiction: it fires precisely when the two representations *disagree about the kind of affect*, not when they disagree about minor degree. Reliability gates dyad detection (Section III-E6) and dims the overlay when it fails.

#### 4) PAD-bridged Plutchik mapping (nearest centroid, with disambiguation)

The smoothed point `(v_t, a_t)` is assigned to the nearest of the 8 Plutchik petal centroids (Table VII). Where a source exists, these are the (P, A) projections of published PAD word ratings (Section II-D) [28]–[30]:

```
petal = argmin_p ∈ P  ‖ (v_t, a_t) − (V_p, A_p) ‖₂          (4)
```

**TABLE VII**
**Plutchik Petal Centroids in VA Space, Source Terms, and Intensity Labels**

| Petal        | V_p   | A_p   | Source term (P, A, D) [28]–[30]                 | Mild         | Basic        | Intense    |
| ------------ | ----- | ----- | ------------------------------------------------ | ------------ | ------------ | ---------- |
| Joy          | +0.76 | +0.48 | *joyful* (.76, .48, .35)                       | Serenity     | Joy          | Ecstasy    |
| Trust        | +0.52 | +0.20 | none located                                     | Acceptance   | Trust        | Admiration |
| Fear         | -0.64 | +0.60 | *fearful* (-.64, .60, -.43)                    | Apprehension | Fear         | Terror     |
| Surprise     | +0.14 | +0.67 | *surprised* (.40, .67, -.13): valence mismatch | Distraction  | Surprise     | Amazement  |
| Sadness      | -0.63 | -0.27 | *sad* (-.63, -.27, -.33)                       | Pensiveness  | Sadness      | Grief      |
| Disgust      | -0.60 | +0.35 | *disgusted* (-.60, .35, .11)                   | Boredom      | Disgust      | Loathing   |
| Anger        | -0.51 | +0.59 | *angry* (-.51, .59, .25)                       | Annoyance    | Anger        | Rage       |
| Anticipation | +0.22 | +0.62 | *curious* (.22, .62, -.01)                     | Interest     | Anticipation | Vigilance  |

V_p and A_p are the values used by the engine. The Dominance means for *disgusted* and *surprised* are reported in the source as not significantly different from zero.

As established in Section II-D, Eq. 4 is underdetermined exactly where PAD's truncated Dominance axis mattered: Fear and Anger are 0.13 apart in (P, A), closer than any reasonable resolution. The engine handles this with **categorical disambiguation**: if the two smallest centroid distances differ by less than a margin `δ = 0.15`, and the categorical branch's petal (Table VIII) is one of the two tied petals, the tie resolves in its favor. This is psychologically sound: felt valence-arousal does not separate fear from anger (that is Dominance's job [28]), but the *visible facial configuration* does, and that is what the categorical branch reads.

**TABLE VIII**
**Categorical Class → Plutchik Petal Bridge**

| Class index | Class    | Petal                         |
| ----------- | -------- | ----------------------------- |
| 0           | Angry    | Anger                         |
| 1           | Disgust  | Disgust                       |
| 2           | Fear     | Fear                          |
| 3           | Happy    | Joy                           |
| 4           | Sad      | Sadness                       |
| 5           | Surprise | Surprise                      |
| 6           | Neutral  | none (no Plutchik equivalent) |

The margin `δ = 0.15` can be checked against the actual degeneracies of Table VII. Computing all pairwise petal-centroid distances:

**TABLE IX**
**Petal Pairs Within the Disambiguation Margin (δ = 0.15)**

| Pair                            | Distance | Disambiguated by class bridge?                             |
| ------------------------------- | -------- | ---------------------------------------------------------- |
| Surprise / Anticipation         | 0.094    | Partially (Surprise ↔ class 5; Anticipation has no class) |
| Fear / Anger                    | 0.130    | Yes (Fear ↔ class 2, Anger ↔ class 0)                    |
| (next nearest: Fear / Disgust)  | 0.253    | outside margin                                             |
| (next nearest: Anger / Disgust) | 0.256    | outside margin                                             |

Only two petal pairs lie within δ = 0.15. Fear/Anger is fully handled by the categorical bridge, since both classes exist in FER-2013. Surprise/Anticipation is handled in one direction only: when the classifier predicts Surprise the tie resolves to Surprise, but no class can pull it toward Anticipation, so otherwise the assignment is geometry-only. This second near-tie is also partly an artifact of the unsourced Surprise valence: with the published *surprised* value (V = 0.40) [28], the pair would be 0.19 apart, outside the margin. The margin was therefore fit to the engine's own table rather than derived independently. A **neutral override** closes the mapping: if the categorical branch predicts Neutral and the smoothed VA magnitude `r < 0.2`, no petal is assigned (the label is simply "Neutral"), consistent with the circumplex reading of the origin as absence of strong affect [10].

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

The unordered pair `{petal(c_1), petal(c_2)}` is looked up against the 16-dyad table (TABLE I); the dyad type (primary/secondary) is propagated. The thresholds encode an intent: `p_1 < 0.5` means no confident single label exists, and a gap under 0.15 means the top two are nearly tied. Whether such a tie reflects a blend in the face or uncertainty in the model is not settled by the thresholds (see the caveat below). The engine implements all 16 primary and secondary dyads; Plutchik's tertiary dyads are not implemented.

**Reachability.** Because the bridge (Table VIII) can only emit petals from {Anger, Disgust, Fear, Joy, Sadness, Surprise}, exactly 9 of the 16 implemented dyads can ever fire: the primary dyads Awe (Fear+Surprise), Disapproval (Surprise+Sadness), Remorse (Sadness+Disgust), and Contempt (Disgust+Anger), and the secondary dyads Guilt (Joy+Fear), Despair (Fear+Sadness), Unbelief (Surprise+Disgust), Envy (Sadness+Anger), and Pride (Anger+Joy). The other 7 (Love, Submission, Aggressiveness, Optimism, Curiosity, Cynicism, Hope) involve Trust or Anticipation and are unreachable through the categorical bridge. They remain implemented for completeness, and would become reachable if the categorical model were extended to more classes (Section VI).

**Caveat: softmax ambiguity versus blended emotion.** A near-tie between two classes admits two readings. Under the first, the face displays a genuine compound expression; Du et al. [31] and RAF-DB [32] show that such compounds exist as consistent production patterns. Under the second, the tie reflects uncertainty: low resolution, occlusion, pose, or the annotator disagreement that the model absorbed during training [37]–[39]. A single softmax vector cannot distinguish these readings. The reliability gate removes some uncertainty cases (those where valence-arousal contradicts the class), but not all. Dyad outputs should therefore be read as *hypotheses* of a blend, not detections. The question is testable: if the first reading dominates, the classifier's top-2 pairs should agree with the top-2 pairs of human annotator distributions on the same images, which FER+ [38] provides for FER-2013. This evaluation is carried out in Section V-C, which supports the ambiguity half of the reading and rejects the precise-blend half.

#### 7) Output vocabulary: sixteen from eight

The label algebra, counted precisely:

- 8 basic petals × 3 intensity words = **24 petal words** (e.g., Annoyance, Anger, Rage);
- + **16 dyad labels** (8 primary, 8 secondary, of which 9 are currently reachable);
- + **Neutral**;
- = **41 distinct output labels** in the engine's vocabulary, of which up to 3 display simultaneously per face per frame (base class + confidence, intensity word, dyad).

The "sixteen from eight" of the title-level framing is the theory's own arithmetic: Plutchik's eight basic emotions plus the eight primary dyads yield sixteen distinct emotion names [11], [12], and it is the same arithmetic the engine implements.

#### 8) Worked examples

Two synthetic traces illustrating the two characteristic regimes (all numbers consistent with Tables IV–IX):

**TABLE X**
**Worked Fusion Traces**

|                        | Trace A: blend                                                                           | Trace B: conflict                                  |
| ---------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Softmax top-2          | Fear 0.34, Surprise 0.28                                                                 | Sad 0.42, Neutral 0.21                             |
| Eq. 7 conditions       | p₁<0.5 ✓, gap 0.06<0.15 ✓                                                             | p₁<0.5 ✓, gap 0.21>0.15 ✗ (no dyad anyway)      |
| Smoothed VA (v_t, a_t) | (−0.60, +0.55)                                                                          | (+0.45, +0.10)                                     |
| r (Eq. 5)              | 0.81 → intense                                                                          | 0.46 → basic                                      |
| Conflict score (Eq. 3) | vs Fear centroid: 0.29 ≤ 0.5 →**reliable**                                       | vs Sad centroid: 0.89 > 0.5 →**unreliable** |
| Nearest petals         | Fear 0.064, Anger 0.098 → tie gap 0.03 < δ → categorical bridge selects**Fear** | Trust 0.12, Joy 0.49 → no tie →**Trust**   |
| Intensity word         | Terror                                                                                   | Trust                                              |
| Dyad                   | {Fear, Surprise} →**Awe** (primary)                                               | suppressed (unreliable)                            |
| Overlay                | "Fear 34%" / "~ Terror" / "+ Awe", full color                                            | "Sad 42%\*" / "~ Trust", box dimmed 50%, no dyad   |

Trace A is the system working as designed: a blended expression surfaces a compound emotion with intensity. Trace B is the system's most valuable failure mode: the representations disagree, and rather than committing to a wrong compound label, it flags the disagreement, suppresses dyad inference, dims the box, and shows what each branch believes.

#### 9) Output contract

The engine returns a `FusionResult` dataclass: raw inputs; smoothed VA; conflict score and reliability flag; polar `(r, θ)`; petal, intensity tier, intensity-specific label, petal BGR color; dyad name and type. All rendering is driven exclusively by this contract, cleanly separating perception (branches), fusion (engine), and presentation (overlay).

### F. Concurrency Model

Each per-face inference pair is submitted as two futures to a `ThreadPoolExecutor(max_workers=2)`:

```python
emot_future = executor.submit(run_emotion, grayFace.copy())
va_future   = executor.submit(run_va, colorFace.copy())
```

The main thread joins both results. Because the two TFLite interpreters execute in C++ and release the Python GIL during `invoke()` [27], the two branches can overlap despite CPython's threading constraints. ROI buffers are copied before submission to avoid mutation of numpy views during async resize. In practice the overlap yields no measurable speedup, because the emotion branch is about 12× cheaper than the VA branch (Section V-A, Table XIII).

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

| Package       | Version    | Role                                      |
| ------------- | ---------- | ----------------------------------------- |
| tensorflow    | 2.21.0     | TFLite interpreter, Keras conversion path |
| opencv-python | 4.13.0.92  | Capture, preprocessing, overlay, display  |
| mediapipe     | ≥ 0.10.14 | BlazeFace detection (Tasks API)           |
| numpy         | 2.4.4      | ROI math, softmax handling, fusion        |
| h5py          | 3.14.0     | HDF5/H5 weight file access                |

---

## V. Results and Discussion

### A. Model Footprints

**TABLE XII**
**Model Comparison**

| Property          | Categorical             | Dimensional             |
| ----------------- | ----------------------- | ----------------------- |
| Architecture      | Mini Xception [8], [19] | MobileNetV2 + head [20] |
| Task              | 7-way classification    | 2-D regression          |
| Dataset           | FER-2013 [7]            | AffectNet [16]          |
| Input             | 64×64 × 1ch           | 224×224 × 3ch         |
| Parameters        | 58,423                  | 3,571,778 (~61×)       |
| TFLite size       | 235.6 KB                | 13.8 MB (~57×)         |
| Output            | Softmax over 7          | (valence, arousal)      |
| Output activation | Softmax                 | Linear                  |

The two branches are mismatched by design: the categorical model is 57–61× smaller than the dimensional one, so per-frame latency is dominated by the VA branch (Table XIII).

**TABLE XIII**
**Measured Per-Component Latency (One Face, CPU Only)**

| Stage                                             | Median (ms)     | Mean (ms)       | p95 (ms)        |
| ------------------------------------------------- | --------------- | --------------- | --------------- |
| BlazeFace detection (720×480 frame)              | 4.78            | 5.18            | 6.98            |
| Emotion branch (preprocess + invoke)              | 1.19            | 1.23            | 1.79            |
| VA branch (preprocess + invoke)                   | 14.34           | 15.88           | 21.93           |
| Both branches, sequential                         | 16.28           | 17.16           | 22.67           |
| Both branches, parallel (2 threads)               | 16.40           | 17.03           | 21.80           |
| Fusion engine update                              | 0.04            | 0.04            | 0.07            |
| **Full frame (detect + branches + fusion)** | **20.51** | **21.27** | **28.78** |

Measured with `eval/benchmark_latency.py`: 300 iterations after 30 warm-up iterations, on an Intel Core i5-1035G7 (4 cores, 8 threads, laptop), Windows 11, Python 3.12.10, TensorFlow 2.21.0 (XNNPACK CPU delegate), MediaPipe 1.1.0, OpenCV 5.0.0, using the repository image `carly.png` with one BlazeFace-detected face. Camera capture and overlay rendering are excluded.

Two results follow. First, the full pipeline runs at a median of 20.5 ms per frame for one face, i.e., about 49 FPS before capture and rendering, so the real-time claim holds on a mid-range laptop CPU. Second, **the thread-pool parallelism provides no measurable benefit**: parallel and sequential execution of the two branches are within noise (16.40 vs. 16.28 ms median, a 0.99× ratio). The emotion branch costs only about 1.2 ms, so the most that overlapping it with the VA branch could save is about 1.2 ms, and thread dispatch overhead consumes that. The concurrency design of Section III-F is therefore architecturally clean but not a performance contribution.

### B. TFLite Conversion Results

**TABLE XIV**
**Conversion Outcomes**

| Model           | Source artifact                         | Source size | TFLite size | Reduction |
| --------------- | --------------------------------------- | ----------- | ----------- | --------- |
| Emotion         | `emotionModel.hdf5` (full model)      | 852.4 KB    | 235.6 KB    | 72.4%     |
| Valence-arousal | `regressor_weights.h5` (weights only) | 14.0 MB     | 13.8 MB     | 3.7%      |

The emotion model's large reduction reflects HDF5's storage of architecture metadata, optimizer state, and training config, all stripped by the FlatBuffer; the VA weights file was already lean, so its reduction is marginal.

### C. Quantitative Evaluation on FER-2013

All fusion-layer results in this section are produced by `eval/evaluate_fer.py` on the FER-2013 challenge splits [7] with the FER+ annotator distributions [38] (3,517 of 3,589 PrivateTest images carry valid FER+ votes). Every image is treated as an independent still: the EMA is reset per image, since a still image has no temporal context. Every quantity that is fitted (per-class centroids in the regressor's output space, logistic combination weights) is fitted on PublicTest only, and every number reported below is computed on PrivateTest. Two protocol caveats apply. First, the upstream training script selects checkpoints on PublicTest and PrivateTest jointly [8], so PrivateTest results may be slightly optimistic for the classifier. Second, the VA branch is trained on color AffectNet images while FER-2013 is grayscale, so all VA outputs here come from a domain-shifted input (grayscale replicated to three channels), which affects the conflict score equally.

#### 1) Classifier re-measurement

The shipped emotion model reaches **65.9% accuracy** on PrivateTest (95% CI [64.4%, 67.5%]), with macro-F1 0.643 and a 15-bin expected calibration error of 0.057, on 3,589 images. This matches the ~66% figure reported by the architecture's source [8]. Per-class recall spans 42.0% (Fear) to 87.7% (Happy); Disgust recall (60.0%) is unreliable at its sample size (55 test images). PublicTest accuracy is 64.5%.

#### 2) The conflict score as an error detector

**TABLE XV**
**Error-Detection AUROC and AURC on PrivateTest (higher AUROC = better error ranking)**

| Detector                                     | AUROC | 95% CI         | AURC  |
| -------------------------------------------- | ----- | -------------- | ----- |
| Conflict score (shipped centroids)           | 0.648 | [0.629, 0.667] | 0.259 |
| Conflict score (PublicTest-fitted centroids) | 0.619 | [0.599, 0.638] | 0.282 |
| − max softmax (MSP)                         | 0.782 | [0.766, 0.796] | 0.153 |
| Entropy                                      | 0.776 | [0.761, 0.791] | 0.156 |
| − top-2 margin                              | 0.777 | [0.762, 0.793] | 0.155 |
| Logistic (MSP + fitted CS)                   | 0.789 | [0.776, 0.805] | 0.155 |
| Logistic (MSP + shipped CS)                  | 0.793 | —             | —    |
| Oracle                                       | 1.000 | —             | 0.066 |

Four findings. First, **the reliability signal is genuine but weak standalone**: the conflict score ranks errors at 0.648 AUROC, well above chance, but far below the classifier's own max-softmax confidence (0.782); the same holds for the risk-coverage area (AURC 0.259 vs. 0.153, oracle 0.066). Second, the signal is **complementary to confidence**: the conflict score correlates only weakly with MSP (Spearman ρ = −0.27) and retains AUROC 0.58–0.63 *within* every MSP stratum, including 0.63 among the most confident predictions (MSP > 0.9, error rate 7.4%, n = 1,111). Third, a two-feature logistic combination on PublicTest yields a **small but statistically reliable gain** over MSP alone: ΔAUROC = +0.011 with the shipped centroids (bootstrap CI [0.006, 0.016]) and +0.007 with fitted centroids ([0.001, 0.013]). Fourth, **as a hard gate the score is dominated by confidence**: at τ_CS = 0.5 (86% coverage) the reliable subset is 69.0% accurate vs. 46.8% for flagged images, but simply keeping the most-confident 86% of images by MSP yields 70.7%. As an overlay cue the asterisk is honest; as an abstention policy it should not replace confidence.

An incidental observation: the shipped hand-estimated centroids detect errors better than centroids fitted on PublicTest (0.648 vs. 0.619). The fitted centroids collapse in the grayscale domain (e.g., Fear at (−0.14, +0.26), Sad at (−0.22, +0.06), Neutral at (−0.05, +0.07) become nearly collinear), shrinking the distances the score relies on; the shipped, more spread table separates errors better. This is a property of the domain shift, not evidence for the shipped values' accuracy.

#### 3) Dyads versus annotator ambiguity (FER+)

**TABLE XVI**
**Dyad Flags vs. FER+ Human Labels (Default Thresholds: 187 of 3,517 Images)**

| Quantity                                                | Dyad-fired  | Not fired |
| ------------------------------------------------------- | ----------- | --------- |
| Annotator split rate (runner-up class ≥ 2 of 10 votes) | 57.8%       | 42.6%     |
| Model top-2 classes both receive ≥ 2 annotator votes   | 8.6%        | 17.0%     |
| Pair-support permutation baseline (mean / p95)          | 2.9% / 4.3% | —        |

Model entropy also correlates moderately with annotator entropy over all valid images (Spearman ρ = 0.38), as does MSP with the annotators' majority share (ρ = 0.39). Where the human top-2 is well defined (second-place votes ≥ 2 and strictly above third, 94 images), the model's top-2 pair matches the annotators' in 8.5% of cases.

The evaluation splits the dyad hypothesis cleanly in half:

- **Supported (ambiguity detection).** Images that trigger a dyad are significantly more ambiguous to humans: 57.8% of them have a runner-up class with at least 2 of 10 votes, vs. 42.6% for the rest. The fire condition is a genuine, if coarse, ambiguity detector.
- **Rejected (blend identification).** The *identity* of the inferred blend almost never matches the humans' disagreement. Pair support on fired images (8.6%) is below even the not-fired rate (17.0%) and only moderately above the permutation baseline; exact top-2 agreement is 8.5%. The classifier's near-ties mostly reflect model confusion at FER-2013's resolution and annotation noise [37], [38], not a human-perceivable mixture, and when humans do split, they rarely split the same way the model does.

**TABLE XVII**
**Dyad-Threshold Ablation (Selected Rows)**

| p_max | gap            | Fires | Fire rate | Split fired / not | Pair support fired / perm. |
| ----- | -------------- | ----- | --------- | ----------------- | -------------------------- |
| 0.4   | 0.05           | 41    | 1.2%      | 65.9% / 43.2%     | 9.8% / 3.0%                |
| 0.4   | 0.25           | 105   | 3.0%      | 57.1% / 43.0%     | 4.8% / 2.9%                |
| 0.5   | 0.05           | 73    | 2.1%      | 61.6% / 43.0%     | 12.3% / 3.3%               |
| 0.5   | 0.15 (default) | 187   | 5.3%      | 57.8% / 42.6%     | 8.6% / 2.9%                |
| 0.5   | 0.25           | 248   | 7.1%      | 58.9% / 42.2%     | 8.1% / 3.0%                |
| 0.6   | 0.05           | 75    | 2.1%      | 61.3% / 43.0%     | 12.0% / 3.2%               |
| 0.6   | 0.25           | 310   | 8.8%      | 61.6% / 41.7%     | 10.6% / 3.5%               |

No threshold setting rescues blend identification: stricter settings raise pair support (up to ~12%) only by shrinking the sample to a few dozen images. At the default thresholds the fired set is dominated by Despair (Fear+Sadness, 67) and Envy (Sadness+Anger, 66), i.e., the classifier's confusion structure around Sad, with Awe (Fear+Surprise, 24) a distant third.

#### 4) EmoNet baseline

**TABLE XVIII**
**This System vs. Unmodified EmoNet-8 on FER-2013 PrivateTest**

| Metric                                              | This system          | EmoNet-8 (unmodified) |
| --------------------------------------------------- | -------------------- | --------------------- |
| Accuracy                                            | 65.9%                | 43.7% [42.2, 45.3]    |
| Macro-F1                                            | 0.643                | 0.373                 |
| ECE (15-bin)                                        | 0.057                | 0.414                 |
| Best own-error AUROC                                | 0.782 (−MSP)        | 0.645 (entropy)       |
| AUROC of this system's errors, using the other's VA | 0.535 [0.516, 0.556] | —                    |
| VA Pearson r (valence / arousal)                    | —                   | 0.700 / 0.732         |

EmoNet reports 75% accuracy on its AffectNet test set [34]; here it collapses to 43.7% with severe miscalibration (ECE 0.41). The comparison is deliberately out-of-domain for *both* branches (48-px grayscale upsampled to 256 px), so it is a caution about cross-domain claims rather than a head-to-head. Two usable facts survive: EmoNet's valence and arousal correlate moderately with this system's VA branch (r = 0.70 / 0.73), an independent sanity check on the regressor; and conflict scores computed with EmoNet's VA detect this system's errors at only 0.535, weaker than with its own VA, so the disagreement signal does not transfer across regressors.

### D. Expected Fusion Behavior

The behaviors below are derived from the engine's logic and the worked traces of Table X. They describe what the design produces rather than measurements; the disagreement and dyad mechanisms are evaluated quantitatively in Section V-C.

- **Stable trajectories.** Eq. 2's EMA (α = 0.3) removes most per-frame flicker of the raw regressor; petal colors transition smoothly rather than strobing.
- **Self-flagged disagreement.** When the classifier labels a face "Sad" while the regressor reports positive valence, `CS > τ_CS` in Eq. 3, the overlay dims and appends an asterisk, and dyad detection is suppressed (Eq. 7's reliability gate). The system degrades gracefully by reporting disagreement rather than committing to a wrong compound label.
- **Intensity enrichment.** Sustained high-magnitude joy (r ≥ 0.66 by Eq. 5) upgrades the label from "Happy" to "Ecstasy"; mild expressions (r < 0.33) produce tier-appropriate words ("Serenity", "Pensiveness") from Table VII.
- **Dyads.** Blended expressions whose top-2 softmax probabilities satisfy Eq. 7 surface compound labels, e.g., simultaneous Fear/Surprise mass → "Awe", consistent with Plutchik's compositional semantics [11], [12], [25]; on FER-2013, flagged images do carry elevated human ambiguity, but the specific blend is rarely corroborated (Section V-C).

### E. Positioning Relative to Related Systems

The categorical branch, re-measured here on FER-2013 PrivateTest, reaches 65.9% (Section V-C), matching the ~66% reported by its source [8]. The VA checkpoint's RMSE (~0.39/0.38, comparable to the AffectNet baselines [16]) remains an inherited figure. The fused system's intended value added is **representational**: intensity tiers, compound labels, and a reliability signal, at negligible additional compute (O(1) numpy per frame). The evaluation tempers this: the reliability signal is genuine but weaker than softmax confidence as a standalone detector, adding a small reliable gain only in combination (ΔAUROC +0.011), and the dyad mechanism tracks human ambiguity without identifying the blend (Section V-C). Relative to joint models such as EmoNet [34], the fair comparison remains open: the out-of-domain run in Table XVIII is a caution about cross-domain evaluation, not a verdict.

---

## VI. Limitations and Future Work

1. **The truncated Dominance axis.** The single deepest limitation is architectural: PAD's third dimension, the one that separates Fear (low D) from Anger (high D) in PAD word ratings [28], is absent from the pipeline. The engine's Fear/Anger disambiguation (Section III-E4) delegates to the categorical branch, which mitigates but does not resolve the degeneracy. The principled fix is a three-dimensional (P, A, D) regressor; AffectNet does not annotate Dominance, so this requires either a new annotation source or an ordinal proxy (e.g., a gaze/pose-derived control signal).
2. **Unreachable petals and dyads.** Trust and Anticipation have no FER-2013 class (Table VIII), so they are reachable only via pure VA proximity, never via disambiguation, and 7 of the 16 implemented dyads can never fire (Section III-E6). A categorical model trained on a richer label set (e.g., an 8-class Plutchik-aligned corpus, or GoEmotions-style affect labels) would activate the full dyad table without changing the engine.
3. **Centroid provenance.** Table VII consists of population-average *word* ratings [28], with one entry contradicting its source (Surprise valence) and one unsourced (Trust). Table VI is hand-estimated (Section III-E3) and should be recomputed from AffectNet annotations. Individual differences in expressive style [6] are unmodeled. The thresholds (τ_CS = 0.5, δ = 0.15, radii 0.33/0.66, dyad 0.5/0.15) are hand-set and fit to the centroid geometry rather than to held-out data. Person-specific calibration is an obvious extension.
4. **Dataset noise and bias.** FER-2013 carries well-documented label noise and class imbalance (Happy overrepresented, Disgust scarce) [7]; AffectNet annotations are subjective with imperfect inter-rater agreement [16]. Both upstream artifacts are consumed as-is, without demographic audit, so the system inherits whatever biases they encode.
5. **Single-face EMA state.** The engine keeps one EMA pair; with multiple simultaneous faces, smoothing state may blend across identities when detection order changes. Per-track state keyed on face identity (tracking-by-detection, using BlazeFace landmarks for cheap association) is straightforward future work.
6. **Concurrency gives no speedup.** The thread pool releases the GIL during `invoke()` [27], but with a 1.2 ms emotion branch there is almost nothing to overlap (Table XIII). Gains would only appear with many faces per frame, which would call for batched inference or process pools.
7. **Evaluation scope.** The evaluation of Section V-C uses FER-2013 still images: no video, no active EMA, and a grayscale domain for the color-trained VA branch. It does not evaluate dyads against true compound-expression labels [31], [32], does not touch AffectNet (so VA accuracy and the hand-estimated centroids remain unvalidated), and the EmoNet comparison is outside the baseline's domain. The natural follow-ups, in order of value: compound-labeled corpora (RAF-DB compound [32], [33]), VA evaluation on AffectNet with recomputed per-category centroids, and video evaluation with the EMA active.
8. **Inherited metrics.** The ~0.39/0.38 VA RMSE is reported by the upstream checkpoint source and has not been re-measured; the classifier's accuracy has (Section V-C).

---

## VII. Conclusion

This project set out to take faces as seriously as they deserve: not as seven mutually exclusive checkboxes, but as a dense, continuous, combinatorial signal sitting on top of a universal low-dimensional structure. The engineering expression of that stance is a pipeline that runs a categorical classifier (Mini Xception, FER-2013) and a dimensional regressor (MobileNetV2, AffectNet) concurrently per face, and fuses them through classical psychological structure: Russell's circumplex for geometric consistency checking [10], Plutchik's wheel for intensity and compound-emotion semantics [11], [12], and published PAD word ratings as the metric bridge between continuous geometry and discrete labels [13], [28]. The resulting system reports not just *what* emotion a face shows, but *how strongly* (intensity tiers), *how reliably* (conflict score), and, when the evidence supports it, *what blend* (dyads): a 41-label vocabulary produced from 8 petals, sixteen of which (8 basic plus 8 primary dyads) are the theory's own arithmetic. Total model cost is 14.2 MB, and the full pipeline runs at a median of 20.5 ms per frame for one face on a 4-core laptop CPU (Table XIII). The first quantitative evaluation (Section V-C) tempers the fusion layer's claims: the reliability signal is genuine but dominated by softmax confidence (0.648 vs. 0.782 AUROC, +0.011 in combination), and dyad flags track human ambiguity (58% vs. 43% annotator splits) without identifying the blend. The system's honest contribution is therefore a psychologically structured presentation layer over measured components, plus a concrete template, three-dimensional PAD output, compound-labeled training categories, and temporal evaluation, for making the stronger claims true. The components are open artifacts in this repository, and the engine's clean dataclass contract keeps those next steps cheap: 3-D PAD regression to restore the missing Dominance axis, per-face temporal state, and evaluation on compound-labeled data.

---

## References

[1] C. Darwin, *The Expression of the Emotions in Man and Animals*. London, U.K.: John Murray, 1872.

[2] A. J. Fridlund, *Human Facial Expression: An Evolutionary View*. San Diego, CA, USA: Academic Press, 1994.

[3] P. Ekman and W. V. Friesen, "Nonverbal leakage and clues to deception," *Psychiatry*, vol. 32, no. 1, pp. 88–106, 1969, doi: 10.1080/00332747.1969.11023575.

[4] P. Ekman, E. R. Sorenson, and W. V. Friesen, "Pan-cultural elements in facial displays of emotion," *Science*, vol. 164, no. 3875, pp. 86–88, 1969, doi: 10.1126/science.164.3875.86.

[5] P. Ekman and W. V. Friesen, *Facial Action Coding System: A Technique for the Measurement of Facial Movement*. Palo Alto, CA, USA: Consulting Psychologists Press, 1978.

[6] H. A. Elfenbein and N. Ambady, "On the universality and cultural specificity of emotion recognition: A meta-analysis," *Psychol. Bull.*, vol. 128, no. 2, pp. 203–235, 2002, doi: 10.1037/0033-2909.128.2.203.

[7] I. J. Goodfellow, D. Erhan, P. L. Carrier, A. Courville, M. Mirza, B. Hamner, W. Cukierski, Y. Tang, D. Thaler, D.-H. Lee, Y. Zhou, C. Ramaiah, F. Feng, R. Li, X. Wang, D. Athanasakis, J. Shawe-Taylor, M. Milakov, J. Park, R. Ionescu, M. Popescu, C. Grozea, J. Bergstra, J. Xie, L. Romaszko, B. Xu, Z. Chuang, and Y. Bengio, "Challenges in representation learning: A report on three machine learning contests," in *Proc. ICML Workshop Challenges Represent. Learn.*, Atlanta, GA, USA, 2013. [Online]. Available: https://arxiv.org/abs/1307.0414

[8] O. Arriaga, M. Valdenegro-Toro, and P. Plöger, "Real-time convolutional neural networks for emotion and gender classification," 2017, arXiv:1710.07557. [Online]. Available: https://arxiv.org/abs/1710.07557

[9] A. S. Cowen and D. Keltner, "Self-report captures 27 distinct categories of emotion bridged by continuous gradients," *Proc. Nat. Acad. Sci. USA*, vol. 114, no. 38, pp. E7900–E7909, 2017, doi: 10.1073/pnas.1702247114.

[10] J. A. Russell, "A circumplex model of affect," *J. Pers. Soc. Psychol.*, vol. 39, no. 6, pp. 1161–1178, 1980, doi: 10.1037/h0077714.

[11] R. Plutchik, *Emotion: A Psychoevolutionary Synthesis*. New York, NY, USA: Harper & Row, 1980.

[12] R. Plutchik, "The nature of emotions: Human emotions have deep evolutionary roots, a fact that may explain their complexity and provide tools for clinical practice," *Amer. Sci.*, vol. 89, no. 4, pp. 344–350, 2001, doi: 10.1511/2001.4.344.

[13] A. Mehrabian and J. A. Russell, *An Approach to Environmental Psychology*. Cambridge, MA, USA: MIT Press, 1974.

[14] A. Mehrabian, "Framework for a comprehensive description and measurement of emotional states," *Genet. Soc. Gen. Psychol. Monogr.*, vol. 121, no. 3, pp. 339–361, 1995.

[15] A. Mehrabian, "Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament," *Current Psych.*, vol. 14, no. 4, pp. 261–292, 1996, doi: 10.1007/BF02686918.

[16] A. Mollahosseini, B. Hasani, and M. H. Mahoor, "AffectNet: A database for facial expression, valence, and arousal computing in the wild," *IEEE Trans. Affect. Comput.*, vol. 10, no. 1, pp. 18–31, 2019, doi: 10.1109/TAFFC.2017.2740923.

[17] V. Bazarevsky, Y. Kartynnik, A. Vakunov, K. Raveendran, and M. Grundmann, "BlazeFace: Sub-millisecond neural face detection on mobile GPUs," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, Long Beach, CA, USA, 2019. [Online]. Available: https://arxiv.org/abs/1907.05047

[18] C. Lugaresi, J. Tang, H. Nash, C. McClanahan, E. Uboweja, M. Hays, F. Zhang, C.-L. Chang, M. G. Yong, J. Lee, W.-T. Chang, W. Hua, M. Georg, and M. Grundmann, "MediaPipe: A framework for building perception pipelines," 2019, arXiv:1906.08172. [Online]. Available: https://arxiv.org/abs/1906.08172

[19] F. Chollet, "Xception: Deep learning with depthwise separable convolutions," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Honolulu, HI, USA, 2017, pp. 1251–1258. [Online]. Available: https://arxiv.org/abs/1610.02357

[20] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Salt Lake City, UT, USA, 2018, pp. 4510–4520. [Online]. Available: https://arxiv.org/abs/1801.04381

[21] S. S. Tomkins, *Affect, Imagery, Consciousness: Vol. I. The Positive Affects*. New York, NY, USA: Springer, 1962.

[22] C. E. Izard, *The Face of Emotion*. New York, NY, USA: Appleton-Century-Crofts, 1971.

[23] W. Wundt, *Grundriss der Psychologie*. Leipzig, Germany: Engelmann, 1896.

[24] H. Schlosberg, "The description of facial expressions in terms of two dimensions," *J. Exp. Psychol.*, vol. 44, no. 4, pp. 229–237, 1952, doi: 10.1037/h0055778.

[25] A. Semeraro, S. Vilella, and G. Ruffo, "PyPlutchik: Visualising and comparing emotion-annotated corpora," *PLOS ONE*, vol. 16, no. 9, p. e0256503, 2021, doi: 10.1371/journal.pone.0256503.

[26] K. R. Scherer, "On the nature and function of emotion: A component process approach," in *Approaches to Emotion*, K. R. Scherer and P. Ekman, Eds. Hillsdale, NJ, USA: Erlbaum, 1984, pp. 293–317.

[27] TensorFlow Authors, "TensorFlow Lite converter and interpreter," *TensorFlow Documentation*. [Online]. Available: https://www.tensorflow.org/lite

[28] J. A. Russell and A. Mehrabian, "Evidence for a three-factor theory of emotions," *J. Res. Pers.*, vol. 11, pp. 273–294, 1977, doi: 10.1016/0092-6566(77)90037-X.

[29] C. Becker-Asano, "WASABI: Affect simulation for agents with believable interactivity," Ph.D. dissertation, Univ. Bielefeld, Bielefeld, Germany, 2008, Table 2.4. [Online]. Available: https://gki.informatik.uni-freiburg.de/papers/becker-asano-phdthesis2008.pdf

[30] A. Mehrabian, "The PAD comprehensive emotion tests." [Online]. Available: http://www.kaaj.com/psych/scales/emotion.html

[31] S. Du, Y. Tao, and A. M. Martinez, "Compound facial expressions of emotion," *Proc. Nat. Acad. Sci. USA*, vol. 111, no. 15, pp. E1454–E1462, 2014, doi: 10.1073/pnas.1322355111.

[32] S. Li, W. Deng, and J. Du, "Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 2584–2593, doi: 10.1109/CVPR.2017.277.

[33] S. Li and W. Deng, "Reliable crowdsourcing and deep locality-preserving learning for unconstrained facial expression recognition," *IEEE Trans. Image Process.*, vol. 28, no. 1, pp. 356–370, 2019, doi: 10.1109/TIP.2018.2868382.

[34] A. Toisoul, J. Kossaifi, A. Bulat, G. Tzimiropoulos, and M. Pantic, "Estimation of continuous valence and arousal levels from faces in naturalistic conditions," *Nat. Mach. Intell.*, vol. 3, no. 1, pp. 42–50, 2021, doi: 10.1038/s42256-020-00280-0.

[35] D. Kollias and S. Zafeiriou, "Expression, affect, action unit recognition: Aff-Wild2, multi-task learning and ArcFace," in *Proc. British Mach. Vis. Conf. (BMVC)*, 2019. [Online]. Available: https://arxiv.org/abs/1910.04855

[36] D. Kollias, A. Schulc, E. Hajiyev, and S. Zafeiriou, "Analysing affective behavior in the first ABAW 2020 competition," in *Proc. 15th IEEE Int. Conf. Autom. Face Gesture Recognit. (FG)*, 2020, pp. 637–643, doi: 10.1109/FG47880.2020.00126.

[37] K. Wang, X. Peng, J. Yang, S. Lu, and Y. Qiao, "Suppressing uncertainties for large-scale facial expression recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 6896–6905, doi: 10.1109/CVPR42600.2020.00693.

[38] E. Barsoum, C. Zhang, C. Canton Ferrer, and Z. Zhang, "Training deep networks for facial expression recognition with crowd-sourced label distribution," in *Proc. 18th ACM Int. Conf. Multimodal Interact. (ICMI)*, 2016, pp. 279–283, doi: 10.1145/2993148.2993165.

[39] S. Chen, J. Wang, Y. Chen, Z. Shi, X. Geng, and Y. Rui, "Label distribution learning on auxiliary label space graphs for facial expression recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 13981–13990, doi: 10.1109/CVPR42600.2020.01400.

---

*Architecture parameters, hyperparameters, artifact sizes, preprocessing details, and fusion logic are transcribed from the repository source (`emot_recog/facecombined.py`, `emot_recog/affect_fusion.py`, `emot_recog/facemot.py`, `emot_recog/faceva.py`, `emot_recog/convert_to_tflite.py`, `emot_recog/convert_va_to_tflite.py`, `reqs.txt`). Citations were checked against Crossref, OpenAlex, arXiv, and library catalogs. Remaining uncertainties: the subtitle of [5] and the page range of [26] are confirmed only by secondary sources, and the PAD values of [28] were read from the reproduction in [29] rather than the original table.*
