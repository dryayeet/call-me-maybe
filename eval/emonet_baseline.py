"""EmoNet baseline (Toisoul et al., Nat. Mach. Intell. 2021) on FER-2013.

Runs the official, UNMODIFIED pretrained EmoNet-8 checkpoint (CC BY-NC-ND 4.0,
https://github.com/face-analysis/emonet, cloned into third_party/emonet) on the
FER-2013 PublicTest and PrivateTest splits, and caches its expression softmax
and valence/arousal outputs for eval/evaluate_fer.py to consume.

Must be run with the separate PyTorch environment (emonet_env), not the
TensorFlow one:
    emonet_env\\Scripts\\python.exe -m eval.emonet_baseline --fer data/fer2013/fer2013.csv

Preprocessing follows the official demo.py: RGB, resized to 256x256, scaled
to [0, 1], no further normalization. FER-2013 is 48x48 grayscale, so images
are upsampled and replicated to three channels (a domain shift shared with
this project's own VA model).
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
EMONET_DIR = REPO / "third_party" / "emonet"
CACHE = REPO / "data" / "cache"
sys.path.insert(0, str(EMONET_DIR))
from emonet.models import EmoNet  # noqa: E402


def load_split(path, split):
    imgs, labels = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Usage"] != split:
                continue
            imgs.append(np.asarray(row["pixels"].split(), dtype=np.uint8).reshape(48, 48))
            labels.append(int(row["emotion"]))
    return np.stack(imgs), np.asarray(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer", required=True)
    ap.add_argument("--tag", default="fer2013")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    sd = torch.load(str(EMONET_DIR / "pretrained" / "emonet_8.pth"), map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    net = EmoNet(n_expression=8)
    net.load_state_dict(sd, strict=False)
    net.eval()

    CACHE.mkdir(parents=True, exist_ok=True)
    for split in ("PublicTest", "PrivateTest"):
        out = CACHE / f"{args.tag}_emonet_{split.lower()}.npz"
        if out.exists():
            print(f"{out.name} exists, skipping")
            continue
        imgs, _ = load_split(args.fer, split)
        probs = np.zeros((len(imgs), 8), np.float32)
        va = np.zeros((len(imgs), 2), np.float32)
        with torch.no_grad():
            for s in range(0, len(imgs), args.batch):
                chunk = imgs[s:s + args.batch]
                x = np.stack([cv2.cvtColor(cv2.resize(im, (256, 256)), cv2.COLOR_GRAY2RGB)
                              for im in chunk]).astype(np.float32) / 255.0
                t = torch.from_numpy(x).permute(0, 3, 1, 2)
                o = net(t)
                probs[s:s + len(chunk)] = torch.softmax(o["expression"], 1).numpy()
                va[s:s + len(chunk), 0] = o["valence"].clamp(-1, 1).numpy()
                va[s:s + len(chunk), 1] = o["arousal"].clamp(-1, 1).numpy()
                if (s // args.batch) % 20 == 0:
                    print(f"  {split}: {s + len(chunk)}/{len(imgs)}", flush=True)
        np.savez(out, softmax8=probs, va=va)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
