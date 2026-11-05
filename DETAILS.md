# Details: full tables, controls and robustness

Companion to the [README](README.md). All CIs are 95% percentile bootstraps over clips
(10,000 resamples, paired within each clip).

## Main run: all conditions (100 plausible clips)

| Judge | Condition | Pass, integer | Tie, integer | Pass, expected | Mean drop, expected |
|---|---|---|---|---|---|
| AutoEval | original_alt | 0.04 [0.01, 0.08] | 0.92 | 0.50 [0.40, 0.60] | −0.008 [−0.023, 0.007] |
| AutoEval | frozen | 0.02 [0.00, 0.05] | 0.68 | 0.17 [0.10, 0.25] | −0.252 [−0.315, −0.194] |
| AutoEval | shuffled | 0.05 [0.01, 0.10] | 0.92 | 0.62 [0.52, 0.71] | +0.020 [−0.005, 0.043] |
| AutoEval | reversed | 0.02 [0.00, 0.05] | 0.95 | 0.51 [0.41, 0.61] | −0.002 [−0.013, 0.008] |
| Qwen | original_alt | 0.01 [0.00, 0.03] | 0.97 | 0.47 [0.37, 0.57] | −0.013 [−0.043, 0.019] |
| Qwen | frozen | 0.02 [0.00, 0.05] | 0.92 | 0.65 [0.55, 0.74] | +0.053 [0.007, 0.104] |
| Qwen | shuffled | 0.00 [0.00, 0.00] | 0.96 | 0.33 [0.24, 0.43] | −0.064 [−0.096, −0.034] |
| Qwen | reversed | 0.01 [0.00, 0.03] | 0.95 | 0.36 [0.27, 0.45] | −0.045 [−0.082, −0.006] |

Mean drop is original minus condition. A negative value means the altered clip scored
*higher* than the original.

**Each condition's pass rate minus `original_alt`'s, on the expected score:**

| Judge | Frozen | Shuffled | Reversed |
|---|---|---|---|
| AutoEval | −0.33 [−0.45, −0.21] | +0.12 [0.00, 0.24] | +0.01 [−0.11, 0.13] |
| Qwen | +0.18 [0.05, 0.31] | −0.14 [−0.26, −0.02] | −0.11 [−0.22, 0.00] |

Differences come in steps of 1/100, so a lower bound of 0.00 does not exclude 0.

**Baseline signal.** Spearman correlation with human `pc` on the 200 `original` clips:

| Judge | Integer score | Expected score | Mean score, plausible vs implausible (expected) |
|---|---|---|---|
| AutoEval | 0.53 | 0.56 | 3.73 vs 3.15 |
| Qwen | 0.13 (p = 0.07) | 0.33 | 4.23 vs 3.98 |

Qwen gives an integer 5 on 559 of 600 calls.

**Motion terciles.** Reversed − `original_alt`, expected score, n = 33–34 per tercile:

| Judge | Low motion | Mid motion | High motion |
|---|---|---|---|
| AutoEval | −0.09 [−0.26, 0.09] | −0.03 [−0.24, 0.18] | +0.15 [−0.06, 0.36] |
| Qwen | 0.00 [−0.18, 0.18] | −0.18 [−0.33, −0.03] | −0.15 [−0.36, 0.06] |

**Eyeball check.** 20 reversed plausible clips, labeled from contact sheets of the 16
frames. 15 of 20 still look plausible backward. Excluding those 15, reversed −
`original_alt` on the remaining 85 clips is:
- AutoEval: −0.01 [−0.14, 0.12].
- Qwen: −0.15 [−0.27, −0.04].

The labels are a first pass by the AI assistant from still frames
([`results/eyeball_reversed.csv`](results/eyeball_reversed.csv)), not a human rating.

**Irreversible-action subset** (exploratory, chosen after the eyeball check, n = 10):
both judges are 0.0, with CIs of about ±0.3.

## Robustness checks ([plan](ROBUSTNESS.md), committed before running)

**R1. Integrity audit: all checks pass.**
- 600 rows per judge, with every designed (clip, condition) key present exactly once.
- No unparsed outputs.
- Clip counts, `pc` ranges and durations as designed.
- The `original` and `original_alt` frame sets are disjoint for every clip.
- Every shuffle is a true permutation with Spearman < 0.2.
- `frozen` and `reversed` are built correctly.
- `scores.jsonl` has every schema field from the guide.

One correction: the digit-probability mass at the answer position is ≥ 0.99 on every
Qwen call but only on 588 of 600 AutoEval calls. In the other 12, the digit was emitted
at the first step, and the remaining mass (down to 0.71) sits on the alternative
leading-space token. They are not parse failures. Dropping the 4 clips involved leaves
the AutoEval headline at +0.02 [−0.09, +0.14].

**R2. Determinism: fails the pre-specified bar.**

I reran 10 + 10 clips (60 calls per judge) in bf16 on an A100. The main run used bf16
on an L40S.

| Judge | Integer scores that differ | Max expected-score difference |
|---|---|---|
| AutoEval | 2 of 60 | 0.04 |
| Qwen | 2 of 60 | 0.08 |

The bar was identical integers and expected scores within 0.01. The likely cause is
bf16 numerics differing between GPU models. Same-hardware determinism was not tested.

This does not affect the headline: every paired comparison is between conditions scored
in the same run on the same GPU, so a run-level hardware offset cancels out. It does mean
comparisons *across* runs (for example, the main run vs R3) carry noise of about this
size. That noise is comparable to each judge's own `original_alt` noise (median
|original − alt| of 0.035 for AutoEval and 0.065 for Qwen).

**R3. AutoEval through its native input path: the reversal null holds.**

Native path: 32 frames on the repo's `get_index` grid, written to H.264 mp4 at 8 fps,
then read back by the repo's `processor` and decord `load_video`.

| Condition − `original_alt` | Integer score | Expected score |
|---|---|---|
| reversed | +0.01 [0.00, 0.03] | +0.02 [−0.11, 0.15] |
| shuffled | **+0.06 [0.02, 0.11]** | **+0.24 [0.12, 0.36]** |
| frozen | 0.00 [−0.04, 0.04] | −0.35 [−0.47, −0.23] |

- Baseline Spearman is 0.53 (integer) and 0.57 (expected).
- 57 of the 100 plausible clips have fewer than 64 frames, so their 32-frame
  `original_alt` partly overlaps `original`. On the 43 clips with fully disjoint frames,
  reversed − alt is 0.00 [0.00, 0.00] (integer) and +0.05 [−0.16, +0.26] (expected).
- **So in its native setup, AutoEval detects shuffling but not reversal.** It notices
  discontinuity, not physics, and it still rewards freezing.

**R4. Qwen without "in order" in the prompt: the reversal null holds.**

| Condition − `original_alt` | Integer score | Expected score |
|---|---|---|
| reversed | −0.02 [−0.06, 0.02] | −0.03 [−0.14, 0.08] |
| shuffled | −0.01 [−0.03, 0.00] | −0.09 [−0.21, 0.03] |
| frozen | +0.01 [−0.03, 0.05] | +0.28 [0.16, 0.40] |

Baseline Spearman is 0.14 (integer) and 0.32 (expected). Without the phrase, the upward
shift for shuffled clips is no longer significant. Freezing is still the only alteration
Qwen penalizes.

**R5. Alternative statistics on the main scores.**

- **Wilcoxon signed-rank test**, |original − reversed| vs |original − alt|, paired per clip:
  - AutoEval: reversal moves scores *less* than resampling does (medians 0.026 vs 0.035;
    p = 0.98 for "more").
  - Qwen: reversal moves scores *more* than resampling (0.094 vs 0.065, p = 0.015), but
    upward. Qwen reacts to reversal by rating the clip *more* plausible.
- **Margin rule** (pass only if the expected score drops by more than 0.1), reversed −
  alt: AutoEval −0.01, Qwen −0.01.
- **Bootstrap seeds 1–3:** the headline CIs are identical to two decimals.
- **Per generator** (n = 14–15 each, 7 generators × 2 judges): no CI clearly excludes 0.
  Two touch it (AutoEval on hunyuan +0.29 [0.00, 0.57]; Qwen on videocrafter +0.21
  [0.00, 0.43]), which is what chance predicts across 14 comparisons.

## Deviations from the guide

- AutoEval in the main run got the 16 frames directly rather than re-encoded mp4s. R3
  re-runs it through the native mp4 path.
- The 205 test rows missing from the HF mirror were dropped before sampling.
- The optional Claude API judge (Judge C) was not run.
- The 20-clip eyeball check was done by the AI assistant from still frames, not by a
  human watching the clips.
