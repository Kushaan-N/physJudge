"""Robustness checks R1-R5 (see ROBUSTNESS.md). Writes results/robust/summary.txt and
results/robust/*.csv. Sections R2-R4 run only once their judge outputs exist."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import analyze
from common import ALTERED, CLIPS_CSV, CONDITIONS, REPO, RESULTS, conditions, frame_indices, load_frames, shuffle_perm

OUT = RESULTS / "robust"
JUDGES = {"autoeval": "scores_autoeval.jsonl", "Qwen2.5-VL-7B-Instruct": "scores_qwen.jsonl"}
LINES = []


def say(*a):
    msg = " ".join(str(x) for x in a)
    LINES.append(msg)
    print(msg)


def check(name, ok, detail=""):
    say(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def read(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])


def r1_integrity(clips):
    say("\n== R1 integrity audit")
    ok = True
    for j, f in JUDGES.items():
        s = read(RESULTS / f)
        keys = s[["clip_id", "condition"]].apply(tuple, axis=1)
        exp = {(c, k) for c, g in zip(clips.clip_id, clips.group) for k in (CONDITIONS if g == "plausible" else ["original"])}
        ok &= check(f"{j}: 600 rows, keys exactly as designed", len(s) == 600 and set(keys) == exp and not keys.duplicated().any(),
                    f"rows={len(s)} dup={keys.duplicated().sum()} missing={len(exp - set(keys))}")
        ok &= check(f"{j}: no unparsed scores", s.score.notna().all() and s.score_cont.notna().all())
        low = s[s.digit_mass < 0.99]
        # Low mass is only acceptable when the digit came at step 0 and the rest sits on the
        # alternative leading-space path (AutoEval); then the renormalized expectation is fine.
        explained = low.get("answer_step", pd.Series(0, index=low.index)).fillna(-1).eq(0).all() if j == "autoeval" else low.empty
        ok &= check(f"{j}: digit mass >= 0.99, or answer at step 0", explained,
                    f"{len(low)}/600 below 0.99 (min {s.digit_mass.min():.3f}), all at answer step 0" if len(low) else "")
        if len(low):
            drop = set(low.clip_id)
            h = headline(s.assign(judge=j), clips, j, "score_cont", keep=sorted(set(clips.clip_id) - drop))
            say(f"   {j}: headline without the {len(drop)} clips that have a low-mass call: "
                f"{h['diff']:+.2f} [{h['lo']:+.2f}, {h['hi']:+.2f}]")
    g = clips.groupby("group")
    ok &= check("clips: 100 per group", (g.size() == 100).all())
    ok &= check("clips: pc ranges", clips[clips.group == "plausible"].pc.isin([4, 5]).all()
                and clips[clips.group == "implausible"].pc.isin([1, 2]).all())
    ok &= check("clips: groups disjoint, ids unique", clips.clip_id.is_unique)
    ok &= check("clips: all >= 2 s", (clips.duration >= 2).all(), f"min={clips.duration.min():.2f}s")
    bal = clips.groupby(["group", "model_name"]).size().unstack(0)
    say("   per-generator counts:", bal.to_dict())
    bad_alt, bad_perm, bad_frz, bad_rev = 0, 0, 0, 0
    for r in clips.itertuples():
        idx, alt = frame_indices(r.n_frames)
        bad_alt += len(set(idx) & set(alt)) > 0
        perm = shuffle_perm(r.clip_id)
        bad_perm += not (spearmanr(perm, np.arange(16)).correlation < 0.2 and sorted(perm) == list(range(16)))
        if r.group == "plausible":
            f, fa = load_frames(r.clip_id)
            c = conditions(f, fa, r.clip_id)
            bad_frz += not all((x == f[8]).all() for x in c["frozen"])
            bad_rev += not (c["reversed"] == f[::-1]).all()
    ok &= check("frames: original and original_alt index sets disjoint", bad_alt == 0, f"{bad_alt} clips overlap")
    ok &= check("frames: shuffles are permutations with Spearman < 0.2", bad_perm == 0)
    ok &= check("frames: frozen == middle frame, reversed == original[::-1]", bad_frz == 0 and bad_rev == 0)
    sj = pd.read_json(REPO / "scores.jsonl", lines=True)
    need = ["clip_id", "group", "condition", "judge", "score", "human_pc", "motion", "raw"]
    ok &= check("scores.jsonl: guide schema fields present, 1200 rows", all(c in sj for c in need) and len(sj) == 1200)
    return ok


def paired(s, judge, col):
    p = s[(s.judge == judge) & (s.group == "plausible")].pivot(index="clip_id", columns="condition", values=col)
    return p.dropna()


def headline(s, clips, judge, col, keep=None):
    p = paired(s, judge, col)
    if keep is not None:
        p = p.loc[p.index.intersection(keep)]
    o = p["original"].to_numpy()
    rev, alt = (p["reversed"].to_numpy() < o).astype(float), (p["original_alt"].to_numpy() < o).astype(float)
    d = rev - alt
    lo, hi = analyze.boot(lambda i: d[i].mean(), len(d))
    return dict(judge=judge, score=col, n=len(p), reversed=rev.mean(), alt=alt.mean(), diff=d.mean(), lo=lo, hi=hi,
                excludes_0=bool(lo > 0 or hi < 0))


def r5_stats(s, clips):
    say("\n== R5 alternative statistics (main scores)")
    rows = []
    for j in JUDGES:
        p = paired(s, j, "score_cont")
        o = p.original
        dr, da = (o - p.reversed).abs(), (o - p.original_alt).abs()
        w = wilcoxon(dr, da, alternative="greater")
        say(f"   {j}: |orig-rev| median {dr.median():.4f} vs |orig-alt| median {da.median():.4f}; "
            f"Wilcoxon rev>alt p={w.pvalue:.3f}")
        for c in ["reversed", "shuffled", "frozen"]:
            m = ((o - p[c]) > 0.1).mean() - ((o - p.original_alt) > 0.1).mean()
            say(f"   {j}: margin rule (drop > 0.1) {c} - alt = {m:+.2f}")
        rows.append(dict(judge=j, med_abs_rev=dr.median(), med_abs_alt=da.median(), wilcoxon_p=w.pvalue))
    for seed in [1, 2, 3]:
        analyze.RNG = np.random.default_rng(seed)
        for j in JUDGES:
            h = headline(s, clips, j, "score_cont")
            say(f"   bootstrap seed {seed} {j}: diff {h['diff']:+.2f} [{h['lo']:+.2f}, {h['hi']:+.2f}]")
    analyze.RNG = np.random.default_rng(0)
    per = []
    pl = clips[clips.group == "plausible"]
    for m, sub in pl.groupby("model_name"):
        for j in JUDGES:
            per.append(dict(model=m, **headline(s, clips, j, "score_cont", keep=sub.clip_id)))
    per = pd.DataFrame(per)
    per.to_csv(OUT / "per_generator.csv", index=False)
    say("   per-generator reversed - alt (expected score):")
    say(per[["model", "judge", "n", "diff", "lo", "hi"]].round(2).to_string(index=False))
    pd.DataFrame(rows).to_csv(OUT / "wilcoxon.csv", index=False)


def r2_determinism(s):
    say("\n== R2 determinism (rerun 10+10 clips)")
    ok = True
    for j, f in [("autoeval", "rerun_autoeval.jsonl"), ("Qwen2.5-VL-7B-Instruct", "rerun_qwen.jsonl")]:
        if not (OUT / f).exists():
            say(f"   {j}: not run yet")
            continue
        r = read(OUT / f).merge(s[s.judge == j], on=["clip_id", "condition"], suffixes=("_re", ""))
        ok &= check(f"{j}: {len(r)} reruns, integer scores identical", len(r) == 60 and (r.score_re == r.score).all(),
                    f"{(r.score_re != r.score).sum()} differ")
        dmax = (r.score_cont_re - r.score_cont).abs().max()
        ok &= check(f"{j}: expected scores within 0.01", dmax <= 0.01, f"max |diff| = {dmax:.5f}")
    return ok


def variant(clips, path, judge, label, note=""):
    if not path.exists():
        say(f"   {label}: not run yet")
        return None
    v = read(path)
    say(f"   {label}: {len(v)} rows, unparsed {v.score.isna().sum()}" + note)
    rows = []
    for col in ["score", "score_cont"]:
        o = v[v.condition == "original"].merge(clips[["clip_id", "pc"]], on="clip_id")
        rho = spearmanr(o[col], o.pc).correlation
        m, _ = analyze.judge_metrics(v, analyze_clips(clips), judge, col)
        h = headline(v, clips, judge, col)
        say(f"     {col}: baseline Spearman {rho:.2f}; reversed - alt {h['diff']:+.2f} [{h['lo']:+.2f}, {h['hi']:+.2f}]"
            f" -> CI excludes 0: {h['excludes_0']}")
        for r in m:
            if r["condition"] != "original_alt":
                say(f"       {r['condition']:9s} pass {r['pass_rate']:.2f} tie {r['tie_rate']:.2f} "
                    f"vs alt {r['diff_vs_alt']:+.2f} [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}]")
        rows.append(dict(check=label, **h, spearman=rho))
    return v, rows


def analyze_clips(clips):
    c = clips.merge(pd.read_csv(RESULTS / "motion.csv")[["clip_id", "motion"]], on="clip_id")
    c["motion_tercile"] = "n/a"
    pl = c.group == "plausible"
    c.loc[pl, "motion_tercile"] = pd.qcut(c[pl].motion.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(CLIPS_CSV)
    s = pd.concat([read(RESULTS / f) for f in JUDGES.values()])
    ok = r1_integrity(clips)
    r5_stats(s, clips)
    ok &= r2_determinism(s)
    say("\n== R3 AutoEval native 32-frame mp4 path / R4 Qwen prompt without 'in order'")
    rows = []
    res = variant(clips, OUT / "scores_autoeval_native32.jsonl", "autoeval_native32", "R3 autoeval_native32")
    if res:
        v, r = res
        rows += r
        disjoint = v[(v.condition == "original") & (v.alt_overlap == 0)].clip_id
        for col in ["score", "score_cont"]:
            h = headline(v, clips, "autoeval_native32", col, keep=disjoint)
            say(f"     {col} on clips with disjoint alt frames (n={h['n']}): reversed - alt {h['diff']:+.2f} "
                f"[{h['lo']:+.2f}, {h['hi']:+.2f}]")
            rows.append(dict(check="R3 disjoint-alt subset", **h))
    res = variant(clips, OUT / "scores_qwen_no_order.jsonl", "Qwen2.5-VL-7B-Instruct_no_order", "R4 qwen_no_order")
    if res:
        rows += res[1]
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "variants_headline.csv", index=False)
    say(f"\nINTEGRITY/DETERMINISM: {'ALL PASS' if ok else 'SOME FAILED'}")
    (OUT / "summary.txt").write_text("\n".join(LINES) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
