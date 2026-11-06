"""Pre-registered analysis of the irreversible-action replication (PREREG_IRREVERSIBLE.md).
Writes results/irreversible/metrics.csv and results/irreversible/summary.txt."""
import json
import warnings

import numpy as np
import pandas as pd

import analyze
from common import REPO, RESULTS

OUT = RESULTS / "irreversible"
RUNS = [("autoeval", "scores_autoeval.jsonl"), ("Qwen2.5-VL-7B-Instruct", "scores_qwen.jsonl"),
        ("autoeval_native32", "scores_autoeval_native32.jsonl")]
MIN_N = 30
LINES = []


def say(*a):
    LINES.append(" ".join(str(x) for x in a))
    print(LINES[-1])


def main():
    # judge_metrics also computes motion terciles; this set has none, so those slices are empty.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    clips = pd.read_csv(REPO / "clips_irreversible.csv")
    labels = pd.read_csv(OUT / "labels.csv")[["clip_id", "verdict"]]
    clips = clips.merge(labels, on="clip_id")
    clips["motion_tercile"] = "n/a"
    s = pd.concat([pd.DataFrame([json.loads(l) for l in open(OUT / f)]) for _, f in RUNS if (OUT / f).exists()])
    say("labels:", clips.verdict.value_counts().to_dict())
    subsets = {"violation (primary)": clips[clips.verdict == "violation"].clip_id,
               "violation + unsure": clips[clips.verdict != "plausible"].clip_id,
               "all 100": clips.clip_id}
    rows = []
    for judge, _ in RUNS:
        if judge not in set(s.judge):
            say(f"\n{judge}: not run")
            continue
        say(f"\n== {judge}")
        for name, keep in subsets.items():
            excl = sorted(set(clips.clip_id) - set(keep))
            for col in ["score", "score_cont"]:
                analyze.RNG = np.random.default_rng(0)
                m, _ = analyze.judge_metrics(s, clips, judge, col, exclude=excl)
                for r in m:
                    rows.append(dict(r, subset=name))
                rev = next(r for r in m if r["condition"] == "reversed")
                flag = " (underpowered: n < 30)" if name.startswith("violation") and rev["n"] < MIN_N else ""
                verdict = "penalizes reversal" if rev["diff_lo"] > 0 else "no reversal penalty"
                say(f"   {name:22s} {col:10s} n={rev['n']:3d}  reversed - alt {rev['diff_vs_alt']:+.2f} "
                    f"[{rev['diff_lo']:+.2f}, {rev['diff_hi']:+.2f}] -> {verdict}{flag}")
        for col in ["score_cont"]:
            analyze.RNG = np.random.default_rng(0)
            m, _ = analyze.judge_metrics(s, clips, judge, col)
            for r in m:
                if r["condition"] in ("shuffled", "frozen"):
                    say(f"   all 100 {col}: {r['condition']:8s} - alt {r['diff_vs_alt']:+.2f} [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}]")
        for cat, sub in clips.groupby("category"):
            analyze.RNG = np.random.default_rng(0)
            m, _ = analyze.judge_metrics(s, clips, judge, "score_cont", exclude=sorted(set(clips.clip_id) - set(sub.clip_id)))
            rev = next(r for r in m if r["condition"] == "reversed")
            rows.append(dict(rev, subset=f"category:{cat}"))
            say(f"   category {cat:8s} n={rev['n']:2d} reversed - alt {rev['diff_vs_alt']:+.2f} [{rev['diff_lo']:+.2f}, {rev['diff_hi']:+.2f}]")
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    (OUT / "summary.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
