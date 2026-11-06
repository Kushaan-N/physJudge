"""Decode each sampled clip, keep K evenly spaced frames plus the half-stride offset set,
and compute a DINOv2 ViT-S/14 motion score. Writes data/frames/<clip_id>.npz and
results/motion.csv (or $MOTION)."""
import os

import av
import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModel

from common import CLIPS_CSV, FRAMES, RESULTS, VIDEOS, frame_indices

MAX_SIDE = 640


def decode(path):
    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    return np.stack(frames)


def resize(f):
    h, w = f.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s >= 1:
        return f
    return cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def motion_score(frames, proc, model, device):
    x = proc(images=list(frames), return_tensors="pt").to(device)
    emb = model(**x).pooler_output.float()
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return float((1 - (emb[1:] * emb[:-1]).sum(-1)).mean())


def main():
    FRAMES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    clips = pd.read_csv(CLIPS_CSV)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

    rows = []
    for r in clips.itertuples():
        out = FRAMES / f"{r.clip_id}.npz"
        if out.exists():
            z = np.load(out)
            frames, n = z["frames"], int(z["n_total"])
        else:
            allf = decode(VIDEOS / f"{r.clip_id}.mp4")
            n = len(allf)
            idx, idx_alt = frame_indices(n)
            frames = np.stack([resize(f) for f in allf[idx]])
            frames_alt = np.stack([resize(f) for f in allf[idx_alt]])
            np.savez(out, frames=frames, frames_alt=frames_alt, n_total=n, idx=idx, idx_alt=idx_alt)
        rows.append({"clip_id": r.clip_id, "n_frames": n, "motion": motion_score(frames, proc, model, device)})
        print(r.clip_id, rows[-1], flush=True)

    m = pd.DataFrame(rows)
    m.to_csv(RESULTS / os.environ.get("MOTION", "motion.csv"), index=False)
    print(m.describe())


if __name__ == "__main__":
    main()
