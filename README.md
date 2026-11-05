# Do physics judges notice impossible time order?

**Short answer: no.** I tested two VLM physics judges on 100 VideoPhy-2 clips that humans
rated physically plausible. Neither scored a clip lower when it was played backward. For
VideoPhy-2-AutoEval, the judge used as an RL reward, reversing a clip moved its score
*less* than resampling the same clip's frames by half a stride.

![Pass rates](results/passrates_continuous.png)

*Pass rate is the fraction of plausible clips where the altered version scored strictly
below the original, using expected scores (`sum s * p(s)`). The dashed lines are each
judge's `original_alt` noise floor: the same clip, frames offset by half a stride. Whiskers
are 95% paired bootstrap CIs over clips. The integer-score version is
[`results/passrates.png`](results/passrates.png); there, every condition is near 0
because 92 to 97% of comparisons are ties.*

## 1. Question and why it matters

Physics judges no longer only grade video generators. They are also used as the
**reward** when training generators pitched as world simulators. HiPhy, for example,
trains against the VideoPhy-2 judge. If a judge cannot tell a physically possible time
order from an impossible one, a generator optimized against it is never pushed toward
correct dynamics.

**Question:** do VLM physics judges score a generated clip lower when its frames are
reversed, shuffled, or frozen?

## 2. Method

**Ground truth by construction.** I start from clips humans rated plausible (`pc` of 4
or 5). Reversing or shuffling such a clip makes it implausible by construction, so no
new human labels are needed. This adapts the *Metric validity* starting point: the
pass/fail check comes from construction rather than from a hand-written physics rule.

**Data.** I used the VideoPhy-2 test split with videos from the HF mirror. I sampled
100 plausible (`pc` 4–5) and 100 implausible (`pc` 1–2) clips, stratified across the 7
generators with seed 0 ([`clips.csv`](clips.csv)). 205 rows missing from the mirror were
dropped before sampling. All clips are 3.2–6.1 s.

**Conditions** (plausible clips only). I took K = 16 evenly spaced frames per clip:
- `original`: the 16 frames in order.
- `original_alt`: the same grid offset by half a stride (noise floor).
- `reversed`: the 16 frames backward.
- `shuffled`: a seeded permutation with Spearman < 0.2 against the true order.
- `frozen`: the middle frame repeated 16 times.

Every condition, `original` included, is built from the same decoded frame arrays.

**Judges.**

| Judge | How it was run |
|---|---|
| **VideoPhy-2-AutoEval** (`videophysics/videophy_2_auto`, mPLUG-Owl-Video) | The repo's own model code, `PROMPT_PHYSICS` template and output parser. The 16 frames go straight into its image processor at 224×224, which is what its `load_video` produces. |
| **Qwen2.5-VL-7B-Instruct** | 16 frames sent as separate images, with VideoPhy-2's instruction plus "Answer with a single number". The prompt is identical for every condition and never mentions reordering. |

Both judges were decoded greedily. For both, I read the probabilities of tokens 1–5 at
the answer position. These sum to more than 0.998 on every call, so the expected score is
well defined. No output failed to parse.

**Pre-registered rule** ([commit `6b147d2`](https://github.com/Kushaan-N/physJudge/commit/6b147d2), 21:20 UTC;
`clips.csv` in [`2c9bb9e`](https://github.com/Kushaan-N/physJudge/commit/2c9bb9e), 21:26; first judge scores committed
at 21:58).
- A judge **passes** a clip if the altered version's score is strictly below the
  original's. A tie is a fail.
- **Headline:** pass rate(`reversed`) − pass rate(`original_alt`), with a 95% bootstrap
  CI (10,000 resamples of clips, paired within each clip). It counts as real only if the
  CI excludes 0.

**Controls.**
1. The `original_alt` noise floor.
2. Reversal vs shuffle, which separates physics from detecting jerky cuts.
3. DINOv2-S motion terciles.
4. A 20-clip eyeball check of reversed clips.
5. Baseline Spearman correlation against human `pc` on the 200 originals.
6. Tie rates.

## 3. Result

**Baseline signal.** On the 200 originals, Spearman correlation with human `pc` is:
- AutoEval: **0.53** (expected score 0.56).
- Qwen: **0.13** (p = 0.07; expected score 0.33).

So AutoEval carries real physics signal. Qwen's integer score has almost none: it gives
a 5 on 559 of 600 calls.

**Headline (reversed − `original_alt` pass rate, 95% CI):**

| Judge | Integer score | Expected score |
|---|---|---|
| AutoEval | −0.02 [−0.06, 0.02] | +0.01 [−0.11, 0.13] |
| Qwen2.5-VL-7B | 0.00 [−0.03, 0.03] | −0.11 [−0.22, 0.00] |

**Every interval includes 0.** Neither judge penalizes reversal beyond its noise floor.

**Full table** (100 plausible clips; pass rate, tie rate and mean drop are vs
`original`):

| Judge | Condition | Pass, integer | Tie, integer | Pass, expected | Mean drop, expected |
|---|---|---|---|---|---|
| AutoEval | original_alt | 0.04 [0.01, 0.08] | 0.92 | 0.50 [0.40, 0.60] | −0.008 [−0.023, 0.007] |
| AutoEval | frozen | 0.02 [0.00, 0.05] | 0.68 | **0.17** [0.10, 0.25] | **−0.252** [−0.315, −0.194] |
| AutoEval | shuffled | 0.05 [0.01, 0.10] | 0.92 | 0.62 [0.52, 0.71] | +0.020 [−0.005, 0.043] |
| AutoEval | reversed | 0.02 [0.00, 0.05] | 0.95 | 0.51 [0.41, 0.61] | −0.002 [−0.013, 0.008] |
| Qwen | original_alt | 0.01 [0.00, 0.03] | 0.97 | 0.47 [0.37, 0.57] | −0.013 [−0.043, 0.019] |
| Qwen | frozen | 0.02 [0.00, 0.05] | 0.92 | 0.65 [0.55, 0.74] | +0.053 [0.007, 0.104] |
| Qwen | shuffled | 0.00 [0.00, 0.00] | 0.96 | 0.33 [0.24, 0.43] | −0.064 [−0.096, −0.034] |
| Qwen | reversed | 0.01 [0.00, 0.03] | 0.95 | 0.36 [0.27, 0.45] | −0.045 [−0.082, −0.006] |

A negative mean drop means the altered clip scored *higher* than the original.

**Reading against the pre-registered outcome table.**

Both judges fall in the first row ("reversed, shuffled, frozen all near baseline: the
judge scores appearance, not time").

AutoEval:
- On reversal, its mean expected-score change is −0.002 (CI ±0.01).
- Its plausible-minus-implausible gap on the same scale is 0.58, so any reversal penalty
  is under about 2% of the gap it shows between plausible and implausible clips.
- Its scores for reversed clips correlate 0.994 with the originals, versus 0.987 for
  `original_alt`.
- Shuffling is borderline (+0.12 [0.00, 0.24]). The CI touches 0, so under the rule it
  does not count.
- **Freezing a clip *raises* its score.** The frozen version scores higher on 83% of
  clips, by +0.25 on average.

Qwen:
- Its scores go slightly *up* under reordering (shuffled −0.14 [−0.26, −0.02] vs the
  noise floor).
- Freezing gets only a small penalty (+0.18 [0.05, 0.31] vs the noise floor).

**Motion terciles.** On the top-motion third (n = 33), which is the fair test,
reversed − `original_alt` is:
- AutoEval: +0.15 [−0.06, 0.36].
- Qwen: −0.15 [−0.36, 0.06].

Both are inconclusive. Even in the clips where reversal changes the most, neither judge
clears its noise floor.

**Integer vs expected scores.** Both give the same conclusion. The integer result is
mostly ties (92–97%); the expected score removes ties and still shows no reversal
penalty. So ties are not what drive the null.

**Eyeball check.** I inspected 20 reversed plausible clips
([`results/eyeball_reversed.csv`](results/eyeball_reversed.csv), sheets in
`results/eyeball/`). 15 of 20 still look physically plausible backward: human motion,
camera moves, near-static scenes. Only 5 show a clear arrow-of-time violation (syrup
flowing up into the bottle, a dust cloud collapsing into the ground, vehicles moving
backward). Excluding the 15, the reversal result is unchanged:
- AutoEval: −0.01 [−0.14, 0.12].
- Qwen: −0.15 [−0.27, −0.04].

On an exploratory set of 10 clips with clearly irreversible actions (pouring, cutting,
breaking, burning, ripping, falling), both judges are also null, with very wide CIs.

**Scope.** These are claims about these two judges, run this way, on these 100 VideoPhy-2
clips. Total time: about 1 h from empty directory to scored results (21:07 → 22:02 UTC;
the 1,200 judge calls took 21.5 min on one L40S), plus the write-up.

## 4. What would change the conclusion, and next week

**Threats.**
- **Most reversed clips are not clearly impossible.** By the eyeball check, 15 of 20
  look fine backward, so the reversal test is weaker than it looks. The null still holds
  on the 85 non-flagged clips and on the 10 irreversible-action clips, but that subset is
  tiny and was picked after the fact. The fix is a pre-registered set of ~100 clips with
  irreversible actions (pouring, breaking, burning), sampled from the full test split.
  It would also need a human rater; the 20 labels here are my first pass from still
  frames and should be re-checked by watching the clips.
- **Input format.** AutoEval was trained with 32 frames decoded from mp4. I fed it 16
  frames directly. Its baseline Spearman (0.53) suggests the format is not crippling,
  but a rerun at 32 frames via its own `load_video` on re-encoded mp4s would rule this
  out.
- **Qwen's prompt.** Qwen's prompt says the frames are "in order". That framing is what
  makes the image sequence a video, but it could also make Qwen trust the order instead
  of checking it.
- **Scores vs tokens.** If expected and integer scores had disagreed, ties would have
  been driving the result. They don't disagree here.

**Next week.**
1. **Irreversible-only replication** (as above), with 2–3 human raters per reversed
   clip.
2. **Action-conditioned world models.** Run the same paired test on rollouts from an
   open action-conditioned world model. Add a *wrong-action* pairing (a rollout scored
   against a different action) as a second ground-truth negative.
3. **Real video.** Repeat on Something-Something v2 to separate "generated" from
   "judged".
4. **Reward-hacking check.** The frozen result already points this way: AutoEval
   rewards a still frame over the real clip. Rank a generator's samples by AutoEval
   score and measure whether low-motion or reversed clips get promoted. Then test
   whether a generator trained with this reward should be expected to drift toward less
   motion.

## 5. What I'd want to work on with you

**Action sensitivity of action-conditioned world models.** When you change the action,
does the rollout follow it, or mostly the first frame? I'd measure this against a
seed-noise floor, the same way `original_alt` works here. It is the dynamics-side twin
of this result: here the judge ignores time; there the model may ignore the action.

---

**Reproduce.** [`PREREGISTRATION.md`](PREREGISTRATION.md) records the rule.

| Step | Script |
|---|---|
| Sample clips | `src/sample_clips.py` |
| Extract frames and motion | `src/extract_frames.py` (`jobs/prep.sbatch`) |
| Run both judges | `src/judge_qwen.py`, `src/judge_autoeval.py` (`jobs/judge_all.sbatch`) |
| Metrics and plots | `src/analyze.py` |
| Eyeball contact sheets | `src/contact_sheets.py` |

Per-call scores, including raw outputs and digit probabilities, are in
[`scores.jsonl`](scores.jsonl) and `results/scores_*.jsonl`.

Environments:
- Main: Python 3.11, torch 2.5.1, transformers 4.51.3, qwen-vl-utils.
- AutoEval: Python 3.10, torch 2.1.2, transformers 4.28.1, plus the
  [VideoPhy repo](https://github.com/Hritikbansal/videophy/tree/main/VIDEOPHY2).

Large files (videos, frames, weights) live on scratch; set `WS` to point at them. The
Qwen part also runs on a free Colab T4: use
`python src/judge_qwen.py --model Qwen/Qwen2.5-VL-3B-Instruct`, which falls back to fp16
when bf16 is unavailable.

**AI tools.** Claude (via Claude Code) helped scope the design, wrote and ran the
pipeline on the Unity cluster, made the first-pass eyeball labels, and drafted this
write-up. I made the research decisions and checked the outputs.
