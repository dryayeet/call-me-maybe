"""Offline evaluation of the AffectFusionEngine on FER-2013 (+ FER+ labels).

Experiments
-----------
E1  Classifier re-measurement: accuracy, macro-F1, ECE on PrivateTest.
E2  Conflict score as an error detector: AUROC for predicting classifier
    misclassification, vs. max-softmax (MSP), entropy, and top-2 margin
    baselines, with bootstrap 95% CIs. Two centroid sets:
      - "hand":   RUSSELL_CENTROIDS as shipped in affect_fusion.py
      - "fitted": per-class mean VA-model output on PublicTest images of
                  that (ground-truth) class, i.e. centroids in the
                  regressor's own output space
    Plus a 2-feature logistic combination (MSP + CS) fit on PublicTest.
E3  Reliability gate: accuracy of reliable vs. unreliable subsets at
    tau_CS = 0.5, and a tau sweep (risk-coverage).
E4  Softmax ambiguity vs. annotator ambiguity (FER+): for images where the
    engine would emit a dyad, does the classifier's top-2 class pair match
    the top-2 pair of the 10-annotator FER+ distribution? Compared with a
    permutation baseline. Also Spearman correlation between model entropy
    and annotator entropy over all test images.
E5  Dyad-threshold ablation (p_max, gap) on firing rate and FER+ agreement.

Protocol: anything fitted or tuned (fitted centroids, logistic weights) uses
PublicTest only; all reported numbers are on PrivateTest.

Single still images have no temporal context, so EMA smoothing is disabled
(engine reset before every image).

Data
----
FER-2013 in the original challenge CSV format (columns: emotion, pixels,
Usage), e.g. `fer2013.csv` from the Kaggle "Challenges in Representation
Learning: Facial Expression Recognition Challenge". FER+ labels
(`fer2013new.csv`, Microsoft FERPlus repo) align row-by-row with it.

Usage (from repo root):
    python -m eval.evaluate_fer --fer data/fer2013/fer2013.csv \
        --ferplus data/ferplus/fer2013new.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EMOT_DIR = REPO / "emot_recog"
MODELS = EMOT_DIR / "models"
RESULTS = REPO / "eval" / "results"
CACHE = REPO / "data" / "cache"

sys.path.insert(0, str(EMOT_DIR))
from affect_fusion import RUSSELL_CENTROIDS, AffectFusionEngine  # noqa: E402

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# FER+ columns -> FER-2013 class index (contempt / unknown / NF have no FER class)
FERPLUS_TO_FER = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3,
                  "sadness": 4, "surprise": 5, "neutral": 6}
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- data ---
def load_fer(path):
    images, labels, usage = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            px = np.asarray(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            images.append(px)
            labels.append(int(row["emotion"]))
            usage.append(row["Usage"])
    return np.stack(images), np.asarray(labels), np.asarray(usage)


def load_ferplus(path, n_rows):
    """Returns (n_rows, 7) vote counts in FER class order, plus a validity mask."""
    votes = np.zeros((n_rows, 7), dtype=np.float64)
    valid = np.zeros(n_rows, dtype=bool)
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n_rows:
                break
            if int(row["NF"]) > 0 or not row["Image name"].strip():
                continue
            for col, idx in FERPLUS_TO_FER.items():
                votes[i, idx] = float(row[col])
            valid[i] = votes[i].sum() > 0
    return votes, valid


# -------------------------------------------------------------- models ---
def run_models(images, cache_name):
    """Runs both TFLite branches on every image (cached to disk)."""
    cache_file = CACHE / f"{cache_name}.npz"
    if cache_file.exists():
        d = np.load(cache_file)
        return d["softmax"], d["va"]

    import tensorflow as tf

    def load(p):
        it = tf.lite.Interpreter(model_path=str(p))
        it.allocate_tensors()
        return it, it.get_input_details(), it.get_output_details()

    e_it, e_in, e_out = load(MODELS / "emotionModel.tflite")
    v_it, v_in, v_out = load(MODELS / "vaModel.tflite")
    e_size = tuple(e_in[0]["shape"][1:3])
    v_size = tuple(v_in[0]["shape"][1:3])

    softmax = np.zeros((len(images), 7), dtype=np.float32)
    va = np.zeros((len(images), 2), dtype=np.float32)
    for i, img in enumerate(images):
        # Emotion branch: identical preprocessing to facecombined.run_emotion
        g = cv2.resize(img, e_size).astype("float32")
        g = ((g / 255.0 - 0.5) * 2.0)[None, ..., None]
        e_it.set_tensor(e_in[0]["index"], g)
        e_it.invoke()
        softmax[i] = e_it.get_tensor(e_out[0]["index"])[0]
        # VA branch: grayscale replicated to 3 channels (FER-2013 has no color)
        c = cv2.cvtColor(cv2.resize(img, v_size), cv2.COLOR_GRAY2RGB).astype("float32")
        c = ((c / 255.0 - 0.5) * 2.0)[None, ...]
        v_it.set_tensor(v_in[0]["index"], c)
        v_it.invoke()
        va[i] = v_it.get_tensor(v_out[0]["index"])[0]
        if (i + 1) % 1000 == 0:
            print(f"  {cache_name}: {i + 1}/{len(images)}", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, softmax=softmax, va=va)
    return softmax, va


# ------------------------------------------------------------- metrics ---
def auroc(scores, positives):
    """Mann-Whitney AUROC; positives = 1 for the event to detect (an error)."""
    positives = np.asarray(positives)
    n_pos = int(positives.sum())
    n_neg = len(positives) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks with ties
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    avg_rank = ends - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    rank_pos = ranks[positives == 1].sum()
    return float((rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_ci(fn, n, iters=1000):
    vals = []
    for _ in range(iters):
        idx = RNG.integers(0, n, n)
        v = fn(idx)
        if not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def ece(probs, labels, bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(total)


def macro_f1(pred, labels, k=7):
    f1s = []
    for c in range(k):
        tp = np.sum((pred == c) & (labels == c))
        fp = np.sum((pred == c) & (labels != c))
        fn = np.sum((pred != c) & (labels == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(1)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def fit_logistic(X, y, iters=50):
    """Newton-Raphson logistic regression with a bias term."""
    X = np.hstack([X, np.ones((len(X), 1))])
    mu, sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-9
    X[:, :-1] = (X[:, :-1] - mu) / sd
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w, mu, sd


def predict_logistic(model, X):
    w, mu, sd = model
    X = (X - mu) / sd
    return 1 / (1 + np.exp(-(X @ w[:-1] + w[-1])))


# --------------------------------------------------------------- fusion ---
def conflict_scores(pred, va, centroids):
    mu = np.array([centroids[c] for c in pred])
    return np.sqrt(((va - mu) ** 2).sum(1))


def engine_dyads(softmax, va, p_max=0.5, gap=0.15, tau=0.5):
    """Runs the real engine per image (EMA reset each time)."""
    eng = AffectFusionEngine(conflict_threshold=tau, dyad_confidence_gap=gap,
                             dyad_max_confidence=p_max)
    out = []
    for s, (v, a) in zip(softmax, va):
        eng.reset()
        idx = int(np.argmax(s))
        r = eng.update(idx, float(s[idx]), s, float(v), float(a))
        out.append(r.dyad_name)
    return out


def top2_pair(row):
    t = np.argsort(row)[-2:]
    return frozenset(int(x) for x in t)


# ----------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer", required=True, help="fer2013.csv (challenge format)")
    ap.add_argument("--ferplus", default=str(REPO / "data" / "ferplus" / "fer2013new.csv"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--tag", default="fer2013", help="cache / output name prefix")
    args = ap.parse_args()

    images, labels, usage = load_fer(args.fer)
    print(f"Loaded {len(images)} images: "
          + ", ".join(f"{u}={np.sum(usage == u)}" for u in np.unique(usage)))
    val = usage == "PublicTest"
    test = usage == "PrivateTest"

    sm_val, va_val = run_models(images[val], f"{args.tag}_publictest")
    sm_te, va_te = run_models(images[test], f"{args.tag}_privatetest")
    y_val, y_te = labels[val], labels[test]
    pred_val, pred_te = sm_val.argmax(1), sm_te.argmax(1)
    err_val = (pred_val != y_val).astype(int)
    err_te = (pred_te != y_te).astype(int)
    n = len(y_te)
    R = {"n_publictest": int(val.sum()), "n_privatetest": int(n)}

    # E1 ---------------------------------------------------------------
    R["E1_classifier"] = {
        "accuracy": float((pred_te == y_te).mean()),
        "accuracy_ci95": bootstrap_ci(lambda i: float((pred_te[i] == y_te[i]).mean()), n, args.boot),
        "macro_f1": macro_f1(pred_te, y_te),
        "ece_15bin": ece(sm_te, y_te),
        "publictest_accuracy": float((pred_val == y_val).mean()),
        "per_class_recall": {CLASSES[c]: float((pred_te[y_te == c] == c).mean()) for c in range(7)},
    }

    # E2 ---------------------------------------------------------------
    hand = {k: (v[1], v[2]) for k, v in RUSSELL_CENTROIDS.items()}
    fitted = {c: tuple(va_val[y_val == c].mean(0)) for c in range(7)}
    R["fitted_centroids_publictest"] = {CLASSES[c]: [round(float(x), 3) for x in fitted[c]] for c in range(7)}
    R["va_output_stats_privatetest"] = {
        "valence_mean": float(va_te[:, 0].mean()), "valence_std": float(va_te[:, 0].std()),
        "arousal_mean": float(va_te[:, 1].mean()), "arousal_std": float(va_te[:, 1].std()),
        "per_true_class_mean": {CLASSES[c]: [round(float(x), 3) for x in va_te[y_te == c].mean(0)] for c in range(7)},
    }

    cs_hand = conflict_scores(pred_te, va_te, hand)
    cs_fit = conflict_scores(pred_te, va_te, fitted)
    msp = sm_te.max(1)
    srt = np.sort(sm_te, 1)
    margin = srt[:, -1] - srt[:, -2]
    ent = entropy(sm_te)

    logit_model = fit_logistic(
        np.column_stack([sm_val.max(1), conflict_scores(pred_val, va_val, fitted)]), err_val)
    combo = predict_logistic(logit_model, np.column_stack([msp, cs_fit]))

    detectors = {  # higher score = more likely an error
        "conflict_score_hand": cs_hand,
        "conflict_score_fitted": cs_fit,
        "neg_max_softmax": -msp,
        "entropy": ent,
        "neg_top2_margin": -margin,
        "logistic_msp_plus_cs_fitted": combo,
    }
    R["E2_error_detection_auroc"] = {
        k: {"auroc": auroc(s, err_te),
            "ci95": bootstrap_ci(lambda i, s=s: auroc(s[i], err_te[i]), n, args.boot)}
        for k, s in detectors.items()
    }
    R["E2_logistic_weights_standardized"] = {
        "msp": float(logit_model[0][0]), "cs_fitted": float(logit_model[0][1]),
        "bias": float(logit_model[0][2])}

    # E3 ---------------------------------------------------------------
    def gate(cs, tau):
        rel = cs <= tau
        return {"tau": tau, "coverage": float(rel.mean()),
                "acc_reliable": float((pred_te[rel] == y_te[rel]).mean()) if rel.any() else None,
                "acc_unreliable": float((pred_te[~rel] == y_te[~rel]).mean()) if (~rel).any() else None}

    R["E3_gate_tau_0.5"] = {"hand": gate(cs_hand, 0.5), "fitted": gate(cs_fit, 0.5)}
    R["E3_tau_sweep_hand"] = [gate(cs_hand, t) for t in np.round(np.arange(0.1, 1.51, 0.1), 2)]
    R["E3_tau_sweep_fitted"] = [gate(cs_fit, t) for t in np.round(np.arange(0.1, 1.51, 0.1), 2)]

    def risk_coverage(score):
        order = np.argsort(score)  # keep lowest-risk first
        errs = err_te[order]
        cov = np.arange(1, n + 1) / n
        risk = np.cumsum(errs) / np.arange(1, n + 1)
        return cov, risk, float(risk.mean())  # mean risk = AURC

    R["E3_aurc"] = {k: risk_coverage(s)[2] for k, s in detectors.items()}
    R["E3_aurc_oracle"] = float(np.mean(np.cumsum(np.sort(err_te)) / np.arange(1, n + 1)))

    # E6: EmoNet baseline (outputs produced by eval/emonet_baseline.py) ------
    em_val_f = CACHE / f"{args.tag}_emonet_publictest.npz"
    em_te_f = CACHE / f"{args.tag}_emonet_privatetest.npz"
    if em_val_f.exists() and em_te_f.exists():
        emo_to_fer = {0: 6, 1: 3, 2: 4, 3: 5, 4: 2, 5: 1, 6: 0}  # 7 = contempt, dropped

        def to_fer7(p8):
            p7 = np.zeros((len(p8), 7), np.float64)
            for e, f in emo_to_fer.items():
                p7[:, f] = p8[:, e]
            return p7 / p7.sum(1, keepdims=True)

        ev, et = np.load(em_val_f), np.load(em_te_f)
        e7_val, e7_te = to_fer7(ev["softmax8"]), to_fer7(et["softmax8"])
        eva_val, eva_te = ev["va"], et["va"]
        epred_val, epred_te = e7_val.argmax(1), e7_te.argmax(1)
        eerr_te = (epred_te != y_te).astype(int)
        efit = {c: tuple(eva_val[y_val == c].mean(0)) for c in range(7)}
        ecs = conflict_scores(epred_te, eva_te, efit)
        # our classifier, but conflict score computed with EmoNet's VA
        efit_ours = {c: tuple(eva_val[y_val == c].mean(0)) for c in range(7)}
        cs_ours_emova = conflict_scores(pred_te, eva_te, efit_ours)
        R["E6_emonet"] = {
            "accuracy": float((epred_te == y_te).mean()),
            "accuracy_ci95": bootstrap_ci(lambda i: float((epred_te[i] == y_te[i]).mean()), n, args.boot),
            "macro_f1": macro_f1(epred_te, y_te),
            "ece_15bin": ece(e7_te, y_te),
            "contempt_argmax_rate": float((et["softmax8"].argmax(1) == 7).mean()),
            "auroc_own_errors": {
                "conflict_score_fitted": auroc(ecs, eerr_te),
                "neg_max_softmax": auroc(-e7_te.max(1), eerr_te),
                "entropy": auroc(entropy(e7_te), eerr_te),
            },
            "auroc_ours_errors_cs_with_emonet_va": {
                "auroc": auroc(cs_ours_emova, err_te),
                "ci95": bootstrap_ci(lambda i: auroc(cs_ours_emova[i], err_te[i]), n, args.boot)},
            "va_pearson_ours_vs_emonet": {
                "valence": float(np.corrcoef(va_te[:, 0], eva_te[:, 0])[0, 1]),
                "arousal": float(np.corrcoef(va_te[:, 1], eva_te[:, 1])[0, 1])},
            "fitted_centroids_publictest": {CLASSES[c]: [round(float(x), 3) for x in efit[c]] for c in range(7)},
        }
    else:
        print("EmoNet caches not found; skipping E6 (run eval/emonet_baseline.py first)")

    # E4 / E5 -----------------------------------------------------------
    if Path(args.ferplus).exists():
        votes, valid_all = load_ferplus(args.ferplus, len(images))
        v_te, ok = votes[test], valid_all[test]
        ann = v_te / np.clip(v_te.sum(1, keepdims=True), 1, None)
        R["E4_ferplus"] = {"n_valid": int(ok.sum())}
        R["E4_ferplus"]["spearman_model_entropy_vs_annotator_entropy"] = spearman(ent[ok], entropy(ann[ok]))
        R["E4_ferplus"]["spearman_model_msp_vs_annotator_majority_share"] = spearman(msp[ok], ann[ok].max(1))

        def ferplus_agreement(p_max, gap, tau=0.5):
            dy = engine_dyads(sm_te, va_te, p_max=p_max, gap=gap, tau=tau)
            fired = np.array([d is not None for d in dy]) & ok
            if fired.sum() == 0:
                return {"p_max": p_max, "gap": gap, "n_fired": 0}
            model_pairs = [top2_pair(sm_te[i]) for i in np.where(fired)[0]]
            ann_pairs = [top2_pair(ann[i]) for i in np.where(fired)[0]]
            # annotators "split" = second class got >= 2 of ~10 votes
            split = np.array([np.sort(v_te[i])[-2] >= 2 for i in np.where(fired)[0]])
            agree = np.array([m == a for m, a in zip(model_pairs, ann_pairs)])
            perm = []
            for _ in range(500):
                sh = RNG.permutation(len(ann_pairs))
                perm.append(np.mean([model_pairs[k] == ann_pairs[sh[k]] for k in range(len(sh))]))
            # also: base rate of annotator splits on non-fired valid images
            nonfired = ok & ~fired
            split_nonfired = np.mean([np.sort(v_te[i])[-2] >= 2 for i in np.where(nonfired)[0]])
            return {"p_max": p_max, "gap": gap, "n_fired": int(fired.sum()),
                    "fire_rate": float(fired.sum() / ok.sum()),
                    "top2_pair_agreement": float(agree.mean()),
                    "permutation_baseline_mean": float(np.mean(perm)),
                    "permutation_baseline_p95": float(np.percentile(perm, 95)),
                    "annotator_split_rate_fired": float(split.mean()),
                    "annotator_split_rate_not_fired": float(split_nonfired),
                    "dyad_counts": {k: int(sum(1 for d, f in zip(dy, fired) if f and d == k))
                                    for k in sorted({d for d, f in zip(dy, fired) if f})}}

        R["E4_default_thresholds"] = ferplus_agreement(0.5, 0.15)
        R["E5_dyad_threshold_ablation"] = [ferplus_agreement(p, g)
                                           for p in (0.4, 0.5, 0.6) for g in (0.05, 0.10, 0.15, 0.25)]
    else:
        print(f"FER+ labels not found at {args.ferplus}; skipping E4/E5")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.tag}_eval.json"
    out.write_text(json.dumps(R, indent=2))
    print(json.dumps({k: R[k] for k in ("E1_classifier", "E2_error_detection_auroc")}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
