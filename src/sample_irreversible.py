"""Sample the irreversible-action replication set (PREREG_IRREVERSIBLE.md): 100 plausible
clips (pc 4-5) not used in the main study, split evenly across seven action categories,
seed 0. Same eligibility as the main sample (in the HF mirror, >= 2 s). Writes
clips_irreversible.csv."""
import zipfile

import numpy as np
import pandas as pd

from common import CLIPS_CSV, LABELS, REPO, SEED, VIDEOS, ZIP
from sample_clips import equal_quota, member, probe

N = 100
OUT = REPO / "clips_irreversible.csv"
CATEGORIES = {
    "pour": ["pouring something into something until it overflows", "pouring something into something",
             "pouring something onto something", "pouring something out of something", "pouring beer",
             "twisting (wringing) something wet until water comes out", "throwing water balloon"],
    "tear": ["tearing something just a little bit", "tearing something into two pieces", "ripping paper"],
    "break": ["breaking boards", "bending something until it breaks",
              "pulling two ends of something so that it separates into two pieces", "smashing", "popping balloons"],
    "cut": ["cutting", "using circular saw", "chiseling stone", "carving ice", "flint knapping", "sharpening pencil",
            "peeling fruit"],
    "fire": ["wood burning (art)", "blowing out candles", "extinguishing fire"],
    "fall": ["something falling like a rock", "something falling like a feather or paper",
             "poking something so that it falls over", "poking a stack of something so the stack collapses",
             "pushing something so that it falls off the table",
             "putting something onto something else that cannot support it so it falls down",
             "putting something that cannot actually stand upright upright on the table so it falls on its side"],
    "collide": ["something colliding with something and both come to a halt"],
}


def main():
    VIDEOS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(LABELS).reset_index().rename(columns={"index": "clip_id"})
    zf = zipfile.ZipFile(ZIP)
    names = set(zf.namelist())
    df["member"] = [member(u, m) for u, m in zip(df.video_url, df.model_name)]
    action_cat = {a: c for c, acts in CATEGORIES.items() for a in acts}
    df["category"] = df.action.map(action_cat)
    used = set(pd.read_csv(CLIPS_CSV).clip_id)
    cand = df[df.pc.isin([4, 5]) & df.member.isin(names) & df.category.notna() & ~df.clip_id.isin(used)]
    missing = set(action_cat) - set(df.action)
    assert not missing, f"actions not in dataset: {missing}"

    rng = np.random.default_rng(SEED)
    order = {c: list(rng.permutation(sub.clip_id.to_numpy())) for c, sub in sorted(cand.groupby("category"))}
    eligible = {c: [] for c in order}
    quota = equal_quota({c: len(v) for c, v in order.items()}, N)
    while True:
        for c in order:
            while len(eligible[c]) < quota[c] and order[c]:
                cid = order[c].pop(0)
                meta = probe(zf, df.loc[cid, "member"], cid)
                if meta:
                    eligible[c].append({"clip_id": cid, **meta})
        got = {c: len(v) for c, v in eligible.items()}
        if sum(got.values()) >= N or not any(order.values()):
            break
        quota = equal_quota({c: got[c] + len(order[c]) for c in order}, N)

    rows = [{**r, "category": c} for c, v in eligible.items() for r in v]
    p = pd.DataFrame(rows).merge(df[["clip_id", "model_name", "pc", "action", "caption", "member"]], on="clip_id")
    p["group"] = "plausible"
    p = p[["clip_id", "group", "category", "model_name", "pc", "action", "caption", "member", "n_frames", "fps", "duration"]]
    p.sort_values(["category", "clip_id"]).to_csv(OUT, index=False)
    print(cand.groupby("category").size().rename("candidates").to_frame().join(p.groupby("category").size().rename("sampled")))
    print(p.model_name.value_counts().to_dict(), "n =", len(p))


if __name__ == "__main__":
    main()
