import numpy as np
from dataclasses import dataclass
from typing import Optional


# --- Plutchik's Wheel: 8 Primary Emotions ---
# Canonical ordering at 45° intervals (Plutchik 1980, PyPlutchik Semeraro et al. 2021)
# Intensity levels: mild (outer ring) → basic (middle) → intense (inner ring)
PLUTCHIK_PETALS = {
    'Joy':          {'mild': 'Serenity',     'basic': 'Joy',          'intense': 'Ecstasy',     'color': (0, 200, 255)},
    'Trust':        {'mild': 'Acceptance',   'basic': 'Trust',        'intense': 'Admiration',  'color': (0, 180, 0)},
    'Fear':         {'mild': 'Apprehension', 'basic': 'Fear',         'intense': 'Terror',      'color': (0, 128, 0)},
    'Surprise':     {'mild': 'Distraction',  'basic': 'Surprise',     'intense': 'Amazement',   'color': (255, 200, 0)},
    'Sadness':      {'mild': 'Pensiveness',  'basic': 'Sadness',      'intense': 'Grief',       'color': (255, 100, 0)},
    'Disgust':      {'mild': 'Boredom',      'basic': 'Disgust',      'intense': 'Loathing',    'color': (80, 0, 128)},
    'Anger':        {'mild': 'Annoyance',    'basic': 'Anger',        'intense': 'Rage',        'color': (0, 0, 255)},
    'Anticipation': {'mild': 'Interest',     'basic': 'Anticipation', 'intense': 'Vigilance',   'color': (0, 165, 255)},
}

# VA centroids from Mehrabian PAD model (1980/1995), [-1, 1] scale
# These are empirically validated, not hand-tuned
PLUTCHIK_VA = {
    'Joy':          ( 0.76,  0.48),
    'Trust':        ( 0.52,  0.20),
    'Fear':         (-0.64,  0.60),
    'Surprise':     ( 0.14,  0.67),
    'Sadness':      (-0.63, -0.27),
    'Disgust':      (-0.60,  0.35),
    'Anger':        (-0.51,  0.59),
    'Anticipation': ( 0.22,  0.62),
}

# Russell sanity check centroids from AffectNet (Mollahosseini et al. 2017)
# Keyed by Xception class index
RUSSELL_CENTROIDS = {
    0: ('Angry',    -0.45,  0.31),
    1: ('Disgust',  -0.48,  0.22),
    2: ('Fear',     -0.36,  0.38),
    3: ('Happy',     0.55,  0.28),
    4: ('Sad',      -0.41, -0.12),
    5: ('Surprise',  0.13,  0.45),
    6: ('Neutral',   0.02, -0.05),
}

# Xception class → Plutchik petal (for categorical disambiguation)
XCEPTION_TO_PLUTCHIK = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Joy',
    4: 'Sadness',
    5: 'Surprise',
    6: None,  # Neutral has no Plutchik equivalent
}

# --- Dyads ---
# Primary (adjacent petals, 1 apart)
_PRIMARY_DYADS = {
    ('Joy', 'Trust'):          'Love',
    ('Trust', 'Fear'):         'Submission',
    ('Fear', 'Surprise'):      'Awe',
    ('Surprise', 'Sadness'):   'Disapproval',
    ('Sadness', 'Disgust'):    'Remorse',
    ('Disgust', 'Anger'):      'Contempt',
    ('Anger', 'Anticipation'): 'Aggressiveness',
    ('Anticipation', 'Joy'):   'Optimism',
}

# Secondary (2 petals apart)
_SECONDARY_DYADS = {
    ('Joy', 'Fear'):            'Guilt',
    ('Trust', 'Surprise'):      'Curiosity',
    ('Fear', 'Sadness'):        'Despair',
    ('Surprise', 'Disgust'):    'Unbelief',
    ('Sadness', 'Anger'):       'Envy',
    ('Disgust', 'Anticipation'):'Cynicism',
    ('Anger', 'Joy'):           'Pride',
    ('Anticipation', 'Trust'):  'Hope',
}

# Build bidirectional lookup with frozenset keys
ALL_DYADS = {}
for (a, b), name in _PRIMARY_DYADS.items():
    ALL_DYADS[frozenset({a, b})] = (name, 'primary')
for (a, b), name in _SECONDARY_DYADS.items():
    ALL_DYADS[frozenset({a, b})] = (name, 'secondary')

# Neutral fallback color (gray)
NEUTRAL_COLOR = (140, 140, 140)


@dataclass
class FusionResult:
    # Raw inputs (passed through)
    emotion_idx: int
    emotion_prob: float
    valence: float
    arousal: float

    # Smoothed VA
    valence_smooth: float
    arousal_smooth: float

    # Russell sanity check
    conflict_score: float
    is_reliable: bool

    # Polar coordinates
    r: float       # VA magnitude, used for intensity
    theta: float   # atan2(A, V) in degrees, for HUD visualization only

    # Plutchik inference
    plutchik_petal: Optional[str]   # None for neutral
    plutchik_intensity: str         # "mild", "basic", "intense"
    plutchik_label: str             # The intensity-specific word (e.g. "Ecstasy")
    plutchik_color: tuple           # BGR

    # Dyad (optional)
    dyad_name: Optional[str]
    dyad_type: Optional[str]        # "primary", "secondary", or None


class AffectFusionEngine:
    def __init__(self, smoothing_alpha=0.3, conflict_threshold=0.5,
                 dyad_confidence_gap=0.15, dyad_max_confidence=0.5,
                 disambiguation_margin=0.15):
        self.alpha = smoothing_alpha
        self.conflict_threshold = conflict_threshold
        self.dyad_gap = dyad_confidence_gap
        self.dyad_max_conf = dyad_max_confidence
        self.disambig_margin = disambiguation_margin

        # EMA state
        self.v_smooth = None
        self.a_smooth = None

        # Pre-compute centroid arrays for vectorized distance
        self._petal_names = list(PLUTCHIK_VA.keys())
        self._petal_coords = np.array([PLUTCHIK_VA[p] for p in self._petal_names])

    def reset(self):
        self.v_smooth = None
        self.a_smooth = None

    def update(self, emotion_idx, emotion_prob, softmax_probs,
               valence, arousal):

        # 1. EMA smoothing
        if self.v_smooth is None:
            self.v_smooth = float(valence)
            self.a_smooth = float(arousal)
        else:
            self.v_smooth = self.alpha * float(valence) + (1 - self.alpha) * self.v_smooth
            self.a_smooth = self.alpha * float(arousal) + (1 - self.alpha) * self.a_smooth

        vs, as_ = self.v_smooth, self.a_smooth

        # 2. Russell conflict score
        _, mu_v, mu_a = RUSSELL_CENTROIDS[emotion_idx]
        conflict_score = np.sqrt((vs - mu_v) ** 2 + (as_ - mu_a) ** 2)
        is_reliable = conflict_score <= self.conflict_threshold

        # 3. Plutchik petal — nearest-centroid with disambiguation
        point = np.array([vs, as_])
        distances = np.sqrt(np.sum((self._petal_coords - point) ** 2, axis=1))
        sorted_idx = np.argsort(distances)
        closest = self._petal_names[sorted_idx[0]]
        second = self._petal_names[sorted_idx[1]]

        # Categorical disambiguation when top two centroids are close
        cat_petal = XCEPTION_TO_PLUTCHIK.get(emotion_idx)
        if distances[sorted_idx[1]] - distances[sorted_idx[0]] < self.disambig_margin:
            if cat_petal in (closest, second):
                closest = cat_petal

        # Neutral override
        r = np.sqrt(vs ** 2 + as_ ** 2)
        if emotion_idx == 6 and r < 0.2:
            petal = None
        else:
            petal = closest

        # 4. Polar coordinates and intensity
        theta = np.degrees(np.arctan2(as_, vs))

        if r < 0.33:
            intensity = 'mild'
        elif r < 0.66:
            intensity = 'basic'
        else:
            intensity = 'intense'

        # Resolve label and color
        if petal is not None:
            petal_info = PLUTCHIK_PETALS[petal]
            label = petal_info[intensity]
            color = petal_info['color']
        else:
            label = 'Neutral'
            color = NEUTRAL_COLOR

        # 5. Dyad detection
        dyad_name = None
        dyad_type = None

        if is_reliable:
            softmax = np.array(softmax_probs)
            top2_idx = np.argsort(softmax)[-2:][::-1]
            top1_prob = softmax[top2_idx[0]]
            top2_prob = softmax[top2_idx[1]]

            if top1_prob < self.dyad_max_conf and (top1_prob - top2_prob) < self.dyad_gap:
                p1 = XCEPTION_TO_PLUTCHIK.get(top2_idx[0])
                p2 = XCEPTION_TO_PLUTCHIK.get(top2_idx[1])
                if p1 is not None and p2 is not None and p1 != p2:
                    key = frozenset({p1, p2})
                    if key in ALL_DYADS:
                        dyad_name, dyad_type = ALL_DYADS[key]

        return FusionResult(
            emotion_idx=emotion_idx,
            emotion_prob=float(emotion_prob),
            valence=float(valence),
            arousal=float(arousal),
            valence_smooth=vs,
            arousal_smooth=as_,
            conflict_score=float(conflict_score),
            is_reliable=is_reliable,
            r=float(r),
            theta=float(theta),
            plutchik_petal=petal,
            plutchik_intensity=intensity,
            plutchik_label=label,
            plutchik_color=color,
            dyad_name=dyad_name,
            dyad_type=dyad_type,
        )
