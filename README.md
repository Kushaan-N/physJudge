# Do physics judges notice impossible time order?

**No.** I tested two VLM physics judges on 100 VideoPhy-2 clips that humans rated
plausible. Neither scored a clip lower when it was played backward than when its frames
were merely resampled. VideoPhy-2-AutoEval, the judge used as an RL reward, catches
shuffled frames in its native setup, so it notices jerky cuts, not physics. It also
scores a frozen single frame *higher* than the real clip.

![Pass rates](results/passrates_continuous.png)

*Pass rate is the fraction of plausible clips where the altered version scored strictly
below the original, using the expected score `Σ s·p(s)`. The dashed lines are each
judge's `original_alt` noise floor: the same clip, frames offset by half a stride.
Whiskers are 95% paired bootstrap CIs. The integer-score version is
[here](results/passrates.png); 92–97% of its comparisons are ties.*

## 1. Question

Physics judges are now used as the **reward** for training video generators pitched as
world simulators. HiPhy, for example, trains against the VideoPhy-2 judge. If the judge
cannot tell a possible time order from an impossible one, a generator optimized against
it is never pushed toward correct dynamics.

**Do VLM physics judges score a generated clip lower when its frames are reversed,
shuffled or frozen?**

## 2. Method

**Ground truth by construction.** Take clips humans rated plausible (`pc` 4–5). Reversing
or shuffling them makes them implausible by construction, so no new labels are needed.
This adapts the *Metric validity* starting point: the pass/fail rule comes from
construction instead of a hand-written physics check.

**Data.** VideoPhy-2 test split, using the HF video mirror. I sampled 100 plausible and
100 implausible (`pc` 1–2) clips, stratified across the 7 generators with seed 0
([`clips.csv`](clips.csv)).

**Conditions.** K = 16 evenly spaced frames per clip:

| Condition | Frames |
|---|---|
| `original` | in order |
| `original_alt` | the same grid offset by half a stride (noise floor) |
| `reversed` | backward |
| `shuffled` | seeded permutation, Spearman < 0.2 against true order |
| `frozen` | the middle frame × 16 |

All conditions come from the same decoded frames.

**Judges.**

| Judge | How it was run |
|---|---|
| **VideoPhy-2-AutoEval** | Its own code, prompt and parser |
| **Qwen2.5-VL-7B-Instruct** | Frames sent as 16 images, with VideoPhy-2's physics instruction and a number-only answer |

Both judges are decoded greedily. I log the probability of each answer token 1–5, giving
an integer score and an expected score per call.

**Pre-registered rule** ([`6b147d2`](https://github.com/Kushaan-N/physJudge/commit/6b147d2)
and [`2c9bb9e`](https://github.com/Kushaan-N/physJudge/commit/2c9bb9e), committed 21:20
and 21:26 UTC; first scores committed at 21:58).
- A judge passes a clip if the altered version scores strictly below the original; ties
  fail.
- **Headline:** pass rate(`reversed`) − pass rate(`original_alt`), with a 95% CI from
  10,000 clip-level bootstrap resamples. It counts only if the CI excludes 0.

**Controls.** The `original_alt` noise floor, reversal vs shuffle, motion terciles, a
20-clip eyeball check, baseline Spearman correlation with human `pc`, and tie rates.

## 3. Result

**Baseline signal** (Spearman correlation with human `pc` on 200 originals):

| Judge | Integer score | Expected score |
|---|---|---|
| AutoEval | 0.53 | 0.56 |
| Qwen | 0.13 | 0.33 |

AutoEval carries real signal. Qwen's integer score has almost none: it outputs a 5 on 559
of 600 calls.

**Headline** (reversed − `original_alt` pass rate):

| Judge | Integer score | Expected score |
|---|---|---|
| AutoEval | −0.02 [−0.06, 0.02] | +0.01 [−0.11, 0.13] |
| Qwen | 0.00 [−0.03, 0.03] | −0.11 [−0.22, 0.00] |

**Every CI includes 0.**

**AutoEval.**
- Reversal moves its expected score by −0.002 [−0.013, 0.008]. That is under about 2%
  of its 0.58 gap between plausible and implausible clips.
- Its reversed scores correlate 0.994 with the originals, *more* closely than the
  resampled `original_alt` does (0.987).
- Freezing a clip *raises* its score on 83% of clips.

**Qwen.**
- Its scores drift slightly *up* under reordering.
- Freezing is the only alteration it penalizes (+0.18 [0.05, 0.31]).

Full tables are in [DETAILS.md](DETAILS.md).

**Reading against the pre-registered outcome table.** Both judges land in row 1 ("ignores
time; scores appearance"). In its native setup (R3 below), AutoEval moves to row 2: it
detects shuffling (+0.24 [0.12, 0.36]) but not reversal. A smooth, impossible clip still
passes.

**Controls.**
- **Integer vs expected score:** same conclusion, so ties don't drive the null.
- **Top-motion tercile:**
  - AutoEval: +0.15 [−0.06, 0.36].
  - Qwen: −0.15 [−0.36, 0.06].
- **Eyeball check:** I reviewed 20 reversed clips as flipbooks. 15 still look plausible
  backward, 3 are clear violations, and 2 are unsure. The AI first pass agreed on 17 of
  my 18 decided clips. Excluding the 15 leaves the null unchanged.

**Robustness** ([plan](ROBUSTNESS.md) committed before running; details in
[DETAILS.md](DETAILS.md#robustness-checks-plan-committed-before-running)):

| Check | Outcome |
|---|---|
| R1. Integrity audit (keys, frames, permutations, schema) | **Pass** |
| R2. Rerun 60 calls per judge on an A100 vs the original L40S | **Fail:** 2/60 integers flip per judge; expected scores move by up to 0.04 (AutoEval) and 0.08 (Qwen). Paired comparisons within a run are unaffected, but cross-run comparisons carry this much noise. |
| R3. AutoEval via its native 32-frame mp4 / decord path | **Reversal null holds** (+0.02 [−0.11, 0.15]); shuffling is now detected |
| R4. Qwen prompt without "in order" | **Reversal null holds** (−0.03 [−0.14, 0.08]) |
| R5. Wilcoxon, margin rule, bootstrap seeds, per-generator splits | **Null holds** |

**Scope and time.** These are claims about these two judges on these 100 clips. It took
about 1 h from an empty directory to scored results. The 1,200 calls are inference only
(no training) and took 21.5 min on one L40S. Robustness took another 39 min on an A100.

## 4. What would change the conclusion; next week

**Threats.**
- **Most reversed clips are not clearly impossible.** 15 of 20 look fine backward, and
  only 3 are clear violations. The null holds without the 15 and on 10
  irreversible-action clips, but that subset is tiny and was picked after the fact.
- **A real reversal effect could hide in the noise.** With 100 clips, the expected-score
  CIs are ±0.12. A pre-registered set of about 300 irreversible-action clips (pouring,
  breaking, burning), rated by humans, would settle it.

**Next week.**
1. The irreversible-only replication above.
2. The same paired test on rollouts from an action-conditioned world model, adding a
   wrong-action pairing as a second ground-truth negative.
3. Real video (Something-Something v2), to separate "generated" from "judged".
4. A reward-hacking check. AutoEval rewards stillness, so do the low-motion samples
   rank at the top, and does a generator trained with it drift toward less motion?

## 5. What I'd want to work on with you

**Action sensitivity of action-conditioned world models.** When you change the action,
does the rollout follow it, or mostly the first frame? I'd measure this against a
seed-noise floor, the way `original_alt` works here. It is the dynamics-side twin of
this result: here the judge ignores time; there the model may ignore the action.

---

**Reproduce.**

| Step | Script |
|---|---|
| Sample clips | `src/sample_clips.py` |
| Frames and motion | `src/extract_frames.py` |
| Judges | `src/judge_autoeval.py`, `src/judge_qwen.py` (`jobs/judge_all.sbatch`) |
| Metrics and plots | `src/analyze.py` |
| Robustness | `src/robust_checks.py` (`jobs/robust.sbatch`) |

Per-call scores, with raw outputs and digit probabilities, are in
[`scores.jsonl`](scores.jsonl). Large files live on scratch; set `WS` to point at them.
The Qwen part runs on a free Colab T4 with `--model Qwen/Qwen2.5-VL-3B-Instruct` (fp16
fallback).

**AI tools.** Claude (via Claude Code) helped scope the design, wrote and ran the pipeline
on the Unity cluster, made first-pass eyeball labels that I then reviewed, and drafted
this write-up. I made the research decisions and checked the outputs.
