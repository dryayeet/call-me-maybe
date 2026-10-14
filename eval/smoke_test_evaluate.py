"""Smoke test for eval.evaluate_fer with synthetic, tie-heavy model outputs.

Exercises the dyad-firing / FER+ agreement branch, which random-noise images
never reach. Run from repo root:  python -m eval.smoke_test_evaluate
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from eval import evaluate_fer as ef

rng = np.random.default_rng(3)
N = 300  # per split


def fake_load_fer(_path):
    usage = np.array(["Training"] * N + ["PublicTest"] * N + ["PrivateTest"] * N)
    labels = rng.integers(0, 7, 3 * N)
    return np.zeros((3 * N, 48, 48), np.uint8), labels, usage


def fake_run_models(images, _name):
    n = len(images)
    sm = rng.dirichlet(np.ones(7) * 0.7, n).astype(np.float32)
    # force Fear/Surprise near-ties on a third of rows -> Awe should fire
    k = n // 3
    sm[:k] = 0.06
    sm[:k, 2], sm[:k, 5] = 0.34, 0.30
    sm[:k] /= sm[:k].sum(1, keepdims=True)
    va = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)]).astype(np.float32)
    va[:k] = (-0.40, 0.40)  # close to the Fear centroid -> reliable
    return sm, va


def main():
    ef.load_fer = fake_load_fer
    ef.run_models = fake_run_models
    tmp = Path(tempfile.mkdtemp())
    ferplus = tmp / "fp.csv"
    cols = "Usage,Image name,neutral,happiness,surprise,sadness,anger,disgust,fear,contempt,unknown,NF"
    lines = [cols]
    for i in range(3 * N):
        v = rng.multinomial(10, np.ones(7) / 7)
        lines.append(f"x,fer{i}.png,{v[6]},{v[3]},{v[5]},{v[4]},{v[0]},{v[1]},{v[2]},0,0,0")
    ferplus.write_text("\n".join(lines))
    ef.RESULTS = tmp
    sys.argv = ["x", "--fer", "unused", "--ferplus", str(ferplus), "--boot", "20", "--tag", "smoke"]
    ef.main()
    r = json.loads((tmp / "smoke_eval.json").read_text())
    d = r["E4_default_thresholds"]
    assert d["n_fired"] > 0, d
    assert "Awe" in d["dyad_counts"], d
    assert 0.0 <= d["pair_support_ge2_fired"] <= 1.0
    assert len(r["E5_dyad_threshold_ablation"]) == 12
    assert "delta_combo_hand_minus_msp" in r["E2_added_value"]
    print("SMOKE OK:", {k: d[k] for k in ("n_fired", "pair_support_ge2_fired",
                                         "pair_support_permutation_mean", "dyad_counts")})


if __name__ == "__main__":
    main()
