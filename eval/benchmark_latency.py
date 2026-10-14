"""Per-component latency benchmark for the facial affect pipeline.

Measures, on the current machine (CPU only):
  - BlazeFace detection on a 720x480 frame
  - Emotion model (Mini Xception, TFLite) invoke, incl. preprocessing
  - VA model (MobileNetV2, TFLite) invoke, incl. preprocessing
  - Both branches sequentially vs. in parallel (ThreadPoolExecutor, 2 workers),
    exactly as facecombined.py does it
  - AffectFusionEngine.update
  - Full per-frame pipeline for one face (detect + parallel branches + fusion)

Overlay drawing and camera capture are excluded (camera-dependent).

Usage (from repo root):
    python -m eval.benchmark_latency [--iters 300] [--warmup 30] [--image carly.png]

Results are printed and written to eval/results/latency.json.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EMOT_DIR = REPO / "emot_recog"
MODELS = EMOT_DIR / "models"
RESULTS = REPO / "eval" / "results"

sys.path.insert(0, str(EMOT_DIR))
from affect_fusion import AffectFusionEngine  # noqa: E402

import mediapipe as mp  # noqa: E402
import tensorflow as tf  # noqa: E402


def load_interpreter(path):
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    return interp, inp, out, tuple(inp[0]["shape"][1:3])


def make_branches():
    e_interp, e_in, e_out, e_size = load_interpreter(MODELS / "emotionModel.tflite")
    v_interp, v_in, v_out, v_size = load_interpreter(MODELS / "vaModel.tflite")

    # Mirrors run_emotion / run_va in facecombined.py
    def run_emotion(gray_face):
        face = cv2.resize(gray_face, e_size).astype("float32")
        face = (face / 255.0 - 0.5) * 2.0
        face = face[None, ..., None]
        e_interp.set_tensor(e_in[0]["index"], face)
        e_interp.invoke()
        pred = e_interp.get_tensor(e_out[0]["index"])
        return int(np.argmax(pred)), float(np.max(pred)), pred[0]

    def run_va(color_face):
        face = cv2.resize(color_face, v_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype("float32")
        face = (face / 255.0 - 0.5) * 2.0
        face = face[None, ...]
        v_interp.set_tensor(v_in[0]["index"], face)
        v_interp.invoke()
        pred = v_interp.get_tensor(v_out[0]["index"])
        return float(pred[0][0]), float(pred[0][1])

    return run_emotion, run_va


def make_detector():
    opts = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODELS / "blaze_face_short_range.tflite")),
        min_detection_confidence=0.5,
    )
    return mp.tasks.vision.FaceDetector.create_from_options(opts)


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * (len(samples) - 1))],
        "min_ms": samples[0],
        "n": len(samples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--image", default=str(REPO / "carly.png"))
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")
    frame = cv2.resize(cv2.flip(frame, 1), (720, 480))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detector = make_detector()
    run_emotion, run_va = make_branches()
    engine = AffectFusionEngine()
    executor = ThreadPoolExecutor(max_workers=2)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    dets = detector.detect(mp_image).detections
    if dets:
        b = dets[0].bounding_box
        x, y, w, h = max(0, b.origin_x), max(0, b.origin_y), b.width, b.height
        face_source = "blazeface"
    else:
        # Fall back to a centered crop so model timings are still meaningful.
        h, w = 240, 240
        x, y = (720 - w) // 2, (480 - h) // 2
        face_source = "center-crop (no face detected)"
    gray_face = gray_frame[y:y + h, x:x + w].copy()
    color_face = frame[y:y + h, x:x + w].copy()

    e_idx, e_prob, e_soft = run_emotion(gray_face)
    v, a = run_va(color_face)

    def detect():
        detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))

    def sequential():
        run_emotion(gray_face)
        run_va(color_face)

    def parallel():
        fe = executor.submit(run_emotion, gray_face.copy())
        fv = executor.submit(run_va, color_face.copy())
        fe.result()
        fv.result()

    def fusion():
        engine.update(e_idx, e_prob, e_soft, v, a)

    def full_frame():
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if res.detections:
            bb = res.detections[0].bounding_box
            xx, yy = max(0, bb.origin_x), max(0, bb.origin_y)
            gf = gray[yy:yy + bb.height, xx:xx + bb.width]
            cf = frame[yy:yy + bb.height, xx:xx + bb.width]
        else:
            gf, cf = gray[y:y + h, x:x + w], frame[y:y + h, x:x + w]
        fe = executor.submit(run_emotion, gf.copy())
        fv = executor.submit(run_va, cf.copy())
        ei, ep, es = fe.result()
        vv, aa = fv.result()
        engine.update(ei, ep, es, vv, aa)

    benches = {
        "blazeface_detect": detect,
        "emotion_branch": lambda: run_emotion(gray_face),
        "va_branch": lambda: run_va(color_face),
        "branches_sequential": sequential,
        "branches_parallel": parallel,
        "fusion_update": fusion,
        "full_frame_one_face": full_frame,
    }

    results = {}
    for name, fn in benches.items():
        results[name] = timeit(fn, args.iters, args.warmup)
        r = results[name]
        print(f"{name:22s} mean {r['mean_ms']:8.3f} ms  median {r['median_ms']:8.3f}  "
              f"p95 {r['p95_ms']:8.3f}")

    full = results["full_frame_one_face"]["median_ms"]
    print(f"\nPipeline throughput (one face, median): {1000.0 / full:.1f} FPS "
          f"(excludes capture and overlay)")
    print(f"Parallel speedup over sequential (median): "
          f"{results['branches_sequential']['median_ms'] / results['branches_parallel']['median_ms']:.2f}x")

    executor.shutdown()
    meta = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "mediapipe": mp.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "image": os.path.basename(args.image),
        "face_source": face_source,
        "face_box": [int(x), int(y), int(w), int(h)],
        "iters": args.iters,
        "warmup": args.warmup,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "latency.json"
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    print(f"\nFace source: {face_source}\nWrote {out}")


if __name__ == "__main__":
    main()
