"""Shared paths, constants and helpers for the time-order judge test."""
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
# Large artifacts (videos, frames, weights) live on scratch; point WS at the workspace.
WS = Path(os.environ.get("WS", "/scratch3/workspace/knaskar_umass_edu-judgecheck"))
DATA = WS / "data"
LABELS = DATA / "videophy2_test.parquet"
ZIP = DATA / "mirror" / "videophy2_test .zip"
VIDEOS = DATA / "videos"
FRAMES = DATA / "frames"

CLIPS_CSV = REPO / "clips.csv"
RESULTS = REPO / "results"

K = 16
SEED = 0
CONDITIONS = ["original", "original_alt", "reversed", "shuffled", "frozen"]
ALTERED = ["original_alt", "frozen", "shuffled", "reversed"]

# VideoPhy-2's physical-commonsense instruction (VIDEOPHY2/template.py, PROMPT_PHYSICS).
PC_INSTRUCTION = (
    "Does this video adhere to the physical laws? "
    "Rate the video on a scale from 1 to 5, where 5 means full compliance and 1 means "
    "significant violations."
)


def frame_indices(n, k=K):
    """Evenly spaced indices and the same grid offset by half a stride."""
    i = np.arange(k)
    main = np.floor((i + 0.25) * n / k).astype(int)
    alt = np.floor((i + 0.75) * n / k).astype(int)
    return np.clip(main, 0, n - 1), np.clip(alt, 0, n - 1)


def shuffle_perm(clip_id, k=K):
    from scipy.stats import spearmanr

    rng = np.random.default_rng(int(clip_id))
    while True:
        perm = rng.permutation(k)
        if spearmanr(perm, np.arange(k)).correlation < 0.2:
            return perm


def conditions(frames, frames_alt, clip_id):
    k = len(frames)
    perm = shuffle_perm(clip_id, k)
    return {
        "original": frames,
        "original_alt": frames_alt,
        "reversed": frames[::-1].copy(),
        "shuffled": frames[perm],
        "frozen": np.repeat(frames[k // 2][None], k, axis=0),
    }


def load_frames(clip_id):
    z = np.load(FRAMES / f"{clip_id}.npz")
    return z["frames"], z["frames_alt"]


def done_keys(path):
    keys = set()
    if Path(path).exists():
        for line in open(path):
            r = json.loads(line)
            keys.add((r["clip_id"], r["condition"], r["judge"]))
    return keys


def digit_expectation(probs):
    """probs: dict digit->prob over '1'..'5'. Returns renormalized expected score."""
    tot = sum(probs.values())
    if tot <= 0:
        return None
    return sum(int(d) * p for d, p in probs.items()) / tot
