"""Sample 100 plausible (pc 4-5) and 100 implausible (pc 1-2) clips, stratified across
model_name, seed 0. A clip is eligible only if it is in the HF mirror zip and decodes to
at least 2 s. Eligible clips are extracted to data/videos/<clip_id>.mp4.
Writes clips.csv. Run before any judge output exists."""
import zipfile

import av
import numpy as np
import pandas as pd

from common import CLIPS_CSV, LABELS, SEED, VIDEOS, ZIP

N_PER_GROUP = 100
MIN_SECONDS = 2.0
GROUPS = {"plausible": [4, 5], "implausible": [1, 2]}


def member(url, model):
    parts = url.split("/")
    return f"dream_machine/{parts[-1]}" if model == "ray2" else "/".join(parts[-2:])


def probe(zf, name, clip_id):
    out = VIDEOS / f"{clip_id}.mp4"
    if not out.exists():
        out.write_bytes(zf.read(name))
    try:
        with av.open(str(out)) as c:
            s = c.streams.video[0]
            n = sum(1 for _ in c.decode(video=0))
            fps = float(s.average_rate)
    except Exception as e:  # undecodable -> ineligible
        print("decode fail", clip_id, e)
        out.unlink()
        return None
    dur = n / fps if fps else 0.0
    if dur < MIN_SECONDS:
        out.unlink()
        return None
    return {"n_frames": n, "fps": fps, "duration": dur}


def equal_quota(avail, total):
    """Split total as evenly as possible across models, capped at availability."""
    q = {m: 0 for m in avail}
    left = total
    while left > 0:
        open_ = sorted([m for m in avail if q[m] < avail[m]], key=lambda m: (q[m], m))
        if not open_:
            break
        q[open_[0]] += 1
        left -= 1
    return q


def main():
    VIDEOS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(LABELS).reset_index().rename(columns={"index": "clip_id"})
    zf = zipfile.ZipFile(ZIP)
    names = set(zf.namelist())
    df["member"] = [member(u, m) for u, m in zip(df.video_url, df.model_name)]
    df["in_mirror"] = df.member.isin(names)
    print("in mirror:", df.in_mirror.sum(), "/", len(df))

    rng = np.random.default_rng(SEED)
    picked, report = [], []
    for g, pcs in GROUPS.items():
        cand = df[df.pc.isin(pcs) & df.in_mirror]
        order = {m: list(rng.permutation(sub.clip_id.to_numpy())) for m, sub in cand.groupby("model_name")}
        eligible = {m: [] for m in order}
        quota = equal_quota({m: len(v) for m, v in order.items()}, N_PER_GROUP)
        # Fill quotas; if a model runs out of eligible clips, its share moves to the others.
        while True:
            for m in order:
                while len(eligible[m]) < quota[m] and order[m]:
                    cid = order[m].pop(0)
                    meta = probe(zf, df.loc[cid, "member"], cid)
                    if meta:
                        eligible[m].append({"clip_id": cid, **meta})
            got = {m: len(v) for m, v in eligible.items()}
            if sum(got.values()) >= N_PER_GROUP or not any(order.values()):
                break
            quota = equal_quota({m: got[m] + len(order[m]) for m in order}, N_PER_GROUP)
        for m, v in eligible.items():
            report.append({"group": g, "model_name": m, "candidates": int((cand.model_name == m).sum()), "sampled": len(v)})
            for r in v:
                picked.append({**r, "group": g})

    p = pd.DataFrame(picked).merge(df[["clip_id", "model_name", "pc", "action", "caption", "member"]], on="clip_id")
    p = p[["clip_id", "group", "model_name", "pc", "action", "caption", "member", "n_frames", "fps", "duration"]]
    p.sort_values(["group", "model_name", "clip_id"]).to_csv(CLIPS_CSV, index=False)
    print(pd.DataFrame(report).to_string(index=False))
    print(p.groupby("group").agg(n=("clip_id", "size"), dur=("duration", "median"), frames=("n_frames", "min")))


if __name__ == "__main__":
    main()
