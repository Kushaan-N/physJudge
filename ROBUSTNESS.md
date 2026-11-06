# Robustness plan

Written and committed after the main result, but before any of these checks were run.
The main pre-registered result stays primary. These checks only test whether it
survives the threats listed in README section 4.

## R1. Integrity audit (no new judge calls)
- Each judge has exactly 600 rows, with no duplicate (clip, condition) keys: 5 conditions
  × 100 plausible clips plus `original` × 100 implausible clips. No unparsed scores.
- `clips.csv`: 100 per group, the `pc` ranges are right, the groups are disjoint, every
  clip is ≥ 2 s, and generators are balanced.
- Frames: the `original` and `original_alt` index sets are disjoint for every clip. Every
  shuffle permutation has Spearman < 0.2. `frozen` equals the middle frame. `reversed`
  equals `original[::-1]`.
- `scores.jsonl` has every schema field from the guide.

## R2. Determinism
Rescore 10 plausible + 10 implausible clips (60 calls per judge) and require integer
scores to match exactly, with expected scores within 0.01.

## R3. AutoEval through its native input path (answers the 16-frame / no-mp4 deviation)
Take 32 frames per clip (the repo default `num_frames`), using the repo's own
`get_index` spacing. Build the same five conditions from those 32 frames
(`original_alt` = half-stride offset). Write each condition to an H.264 mp4 at one
shared fps (8). Score it with the repo's own `processor(videos=[path], num_frames=32)`,
which goes through decord's `load_video`. That is 600 calls.

**Pass criterion for the main conclusion to stand:** the reversed − `original_alt` pass
rate CI includes 0 for both the integer and the expected score.

## R4. Qwen without "in order" in the prompt
Change the intro to "The following 16 images are frames from a video." and leave
everything else the same. 600 calls. Same criterion as R3.

## R5. Alternative statistics on the main scores (no new calls)
- **Wilcoxon signed-rank test**, paired per clip: |original − reversed| vs
  |original − original_alt|. Does reversal move scores more than resampling does?
- **Margin rule:** pass only if the drop is > 0.1 on the expected score.
- **Per-generator** reversed − `original_alt` on the expected score.
- **Bootstrap seed:** rerun the headline CIs with 3 different seeds.

## R2b. Same-GPU determinism (added after R2 failed; committed before running)
R2 reran on an A100, while the main run used an L40S. To separate hardware from
pipeline nondeterminism, rerun the same 10 + 10 clips (60 calls per judge) pinned to an
L40S in bf16. **Criterion:** integer scores identical, expected scores within 0.01. R2's
failure stands either way. This check only says whether it came from the hardware.
