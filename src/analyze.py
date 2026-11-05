"""Merge judge outputs into scores.jsonl, compute pre-registered metrics with paired
clip-level bootstrap CIs, and draw the figure."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ALTERED, CLIPS_CSV, REPO, RESULTS  # noqa: E402

N_BOOT = 10_000
RNG = np.random.default_rng(0)
# Exploratory (not pre-registered): plausible clips whose action makes an irreversible
# change, hand-picked from a keyword pass over action+caption after the eyeball check
# (pouring 25/1146, wood burning 651/835, falling 1010/2213, saw cut 1150, wringing
# drips 1315, ripping paper 1764, breaking boards 1966).
IRREVERSIBLE = [25, 651, 835, 1010, 1146, 1150, 1315, 1764, 1966, 2213]
JUDGE_ORDER = ["autoeval", "Qwen2.5-VL-7B-Instruct"]
JUDGE_LABEL = {"autoeval": "VideoPhy-2-AutoEval", "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B"}
COLOR = {"autoeval": "#2a78d6", "Qwen2.5-VL-7B-Instruct": "#eb6834"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load():
    clips = pd.read_csv(CLIPS_CSV)
    motion = pd.read_csv(RESULTS / "motion.csv")
    clips = clips.merge(motion[["clip_id", "motion"]], on="clip_id")
    rows = [json.loads(l) for f in sorted(RESULTS.glob("scores_*.jsonl")) for l in open(f)]
    s = pd.DataFrame(rows).merge(clips[["clip_id", "pc", "motion"]], on="clip_id").rename(columns={"pc": "human_pc"})
    cols = ["clip_id", "group", "condition", "judge", "score", "score_cont", "human_pc", "motion", "raw"]
    with open(REPO / "scores.jsonl", "w") as f:
        for r in s[cols].to_dict("records"):
            f.write(json.dumps({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}) + "\n")
    return clips, s


def boot(fn, n):
    """Percentile CI of fn(idx) over clip resamples."""
    stats = np.array([fn(RNG.integers(0, n, n)) for _ in range(N_BOOT)])
    return np.percentile(stats, [2.5, 97.5], axis=0)


def judge_metrics(s, clips, judge, col, exclude=()):
    p = s[(s.judge == judge) & (s.group == "plausible")].pivot(index="clip_id", columns="condition", values=col)
    p = p.drop(index=[c for c in exclude if c in p.index]).dropna()
    p = p.join(clips.set_index("clip_id")[["motion_tercile"]])
    o = p["original"].to_numpy()
    passes = {c: (p[c].to_numpy() < o).astype(float) for c in ALTERED}
    ties = {c: (p[c].to_numpy() == o).astype(float) for c in ALTERED}
    drops = {c: o - p[c].to_numpy() for c in ALTERED}
    n = len(p)
    out = []
    for c in ALTERED:
        lo, hi = boot(lambda i: passes[c][i].mean(), n)
        dlo, dhi = boot(lambda i: drops[c][i].mean(), n)
        rec = dict(judge=judge, score=col, condition=c, n=n, pass_rate=passes[c].mean(), pass_lo=lo, pass_hi=hi,
                   tie_rate=ties[c].mean(), mean_drop=drops[c].mean(), drop_lo=dlo, drop_hi=dhi)
        if c != "original_alt":
            d = passes[c] - passes["original_alt"]
            rec["diff_vs_alt"] = d.mean()
            rec["diff_lo"], rec["diff_hi"] = boot(lambda i: d[i].mean(), n)
        out.append(rec)
    terc = []
    for t in ["low", "mid", "high"]:
        m = (p.motion_tercile == t).to_numpy()
        for c in ["reversed", "original_alt"]:
            x = passes[c][m]
            lo, hi = boot(lambda i: x[i].mean(), len(x))
            terc.append(dict(judge=judge, score=col, tercile=t, condition=c, n=int(m.sum()), pass_rate=x.mean(),
                             pass_lo=lo, pass_hi=hi))
        d = passes["reversed"][m] - passes["original_alt"][m]
        lo, hi = boot(lambda i: d[i].mean(), len(d))
        terc.append(dict(judge=judge, score=col, tercile=t, condition="reversed-minus-alt", n=int(m.sum()),
                         pass_rate=d.mean(), pass_lo=lo, pass_hi=hi))
    return out, terc


def baseline_spearman(s, judge, col):
    o = s[(s.judge == judge) & (s.condition == "original")].dropna(subset=[col])
    r = spearmanr(o[col], o.human_pc)
    a = o.groupby("group")[col].mean()
    return dict(judge=judge, score=col, n=len(o), spearman=r.correlation, p=r.pvalue,
                mean_plausible=a.get("plausible"), mean_implausible=a.get("implausible"),
                unparsed=int(s[(s.judge == judge)][col].isna().sum()))


def plot(m, t, col, path):
    m, t = m[m.score == col], t[t.score == col]
    judges = [j for j in JUDGE_ORDER if j in set(m.judge)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [3, 3]}, facecolor=SURF)
    off = np.linspace(-0.12, 0.12, len(judges)) if len(judges) > 1 else [0]
    conds = ["frozen", "shuffled", "reversed"]
    for ax in axes:
        ax.set_facecolor(SURF)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1)
    ax = axes[0]
    for j, dx in zip(judges, off):
        mj = m[m.judge == j].set_index("condition")
        base = mj.loc["original_alt", "pass_rate"]
        ax.axhline(base, color=COLOR[j], ls="--", lw=1.5, alpha=0.8,
                   label=f"{JUDGE_LABEL[j]}: original_alt baseline ({base:.2f})")
        x = np.arange(len(conds)) + dx
        y = mj.loc[conds, "pass_rate"].to_numpy()
        err = np.vstack([y - mj.loc[conds, "pass_lo"], mj.loc[conds, "pass_hi"] - y])
        ax.errorbar(x, y, yerr=err, fmt="o", ms=8, color=COLOR[j], ecolor=COLOR[j], elinewidth=2, capsize=0,
                    mec=SURF, mew=2, label=JUDGE_LABEL[j])
    ax.set_xticks(range(len(conds)), conds)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("pass rate (altered scored strictly below original)", color=INK2, fontsize=9)
    ax.set_title("Pass rate by condition, plausible clips", color=INK, fontsize=10.5, loc="left")
    h, l = ax.get_legend_handles_labels()
    order = [i for i, x in enumerate(l) if "baseline" not in x] + [i for i, x in enumerate(l) if "baseline" in x]
    ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, labelcolor=INK)
    ax = axes[1]
    terc = ["low", "mid", "high"]
    for j, dx in zip(judges, off):
        tj = t[(t.judge == j) & (t.condition == "reversed")].set_index("tercile").loc[terc]
        ta = t[(t.judge == j) & (t.condition == "original_alt")].set_index("tercile").loc[terc]
        x = np.arange(3) + dx
        y = tj.pass_rate.to_numpy()
        ax.errorbar(x, y, yerr=np.vstack([y - tj.pass_lo, tj.pass_hi - y]), fmt="o", ms=8, color=COLOR[j],
                    elinewidth=2, capsize=0, mec=SURF, mew=2, label=JUDGE_LABEL[j])
        ax.plot(x, ta.pass_rate, ls="none", marker="_", ms=16, mew=2, color=COLOR[j], alpha=0.8)
    ax.set_xticks(range(3), [f"{x} motion\n(n={int(n)})" for x, n in zip(terc, tj.n)])
    ax.set_title("Reversed, by motion tercile (dash = original_alt)", color=INK, fontsize=10.5, loc="left")
    fig.suptitle(f"Do physics judges penalize impossible time order? ({'integer' if col == 'score' else 'expected'} score)",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def main():
    clips, s = load()
    # Terciles over the plausible clips, since that is where pass rates are computed.
    pl = clips.group == "plausible"
    clips["motion_tercile"] = "n/a"
    clips.loc[pl, "motion_tercile"] = pd.qcut(clips[pl].motion.rank(method="first"), 3,
                                              labels=["low", "mid", "high"]).astype(str)
    eye_path = REPO / "results" / "eyeball_reversed.csv"
    eye = pd.read_csv(eye_path) if eye_path.exists() else None
    excl = list(eye[eye.plausible_backward == 1].clip_id) if eye is not None else []
    irrev = clips[pl & clips.clip_id.isin(IRREVERSIBLE)].clip_id
    not_irrev = list(clips[pl & ~clips.clip_id.isin(irrev)].clip_id)

    M, T, B, E, IR = [], [], [], [], []
    judges = [j for j in JUDGE_ORDER if j in set(s.judge)]
    for j in judges:
        for col in ["score", "score_cont"]:
            m, t = judge_metrics(s, clips, j, col)
            M += m
            T += t
            B.append(baseline_spearman(s, j, col))
            if excl:
                m2, _ = judge_metrics(s, clips, j, col, exclude=excl)
                E += [dict(r, excluded=len(excl)) for r in m2 if r["condition"] == "reversed"]
            m3, _ = judge_metrics(s, clips, j, col, exclude=not_irrev)
            IR += [r for r in m3 if r["condition"] in ("reversed", "original_alt")]
    M, T, B = pd.DataFrame(M), pd.DataFrame(T), pd.DataFrame(B)
    M.to_csv(RESULTS / "metrics.csv", index=False)
    T.to_csv(RESULTS / "metrics_terciles.csv", index=False)
    B.to_csv(RESULTS / "baseline_spearman.csv", index=False)
    pd.DataFrame(IR).to_csv(RESULTS / "metrics_irreversible_subset.csv", index=False)
    if E:
        pd.DataFrame(E).to_csv(RESULTS / "metrics_reversed_eyeball_excluded.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(B.round(3).to_string(index=False))
    print(M.round(3).to_string(index=False))
    print(T.round(3).to_string(index=False))
    if E:
        print(pd.DataFrame(E).round(3).to_string(index=False))
    print(f"irreversible-action subset (exploratory), n={len(irrev)}")
    print(pd.DataFrame(IR).round(3).to_string(index=False))
    plot(M, T, "score", REPO / "results" / "passrates.png")
    plot(M, T, "score_cont", REPO / "results" / "passrates_continuous.png")


if __name__ == "__main__":
    main()
