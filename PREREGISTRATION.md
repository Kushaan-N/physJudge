# Pre-registration: do physics judges notice impossible time order?

Written and committed before any judge has scored any clip. `clips.csv` is added in a
follow-up commit, also before scoring. The commit timestamps are the record.

## Question

Do VLM physics judges (VideoPhy-2-AutoEval; Qwen2.5-VL-7B-Instruct) score a generated
video lower when its frames are reversed, shuffled or frozen?

## Data

VideoPhy-2 test split (`videophysics/videophy2_test`), with videos from the HF mirror.
- 100 **plausible** clips (human `pc` of 4 or 5) and 100 **implausible** clips (`pc` of 1
  or 2), each stratified across `model_name`, seed 0. Clips under 2 s or missing from the
  mirror are excluded before sampling completes, never after scoring.
- If fewer than 100 eligible plausible clips exist, use all of them and report the count.

## Frames and conditions

K = 16 frames at evenly spaced indices `floor((i + 0.25) * N / K)`. `original_alt` uses
`floor((i + 0.75) * N / K)`, the same grid offset by half a stride.

Plausible clips get all five conditions; implausible clips get `original` only.
- `original`: the 16 frames in order.
- `original_alt`: the offset 16 frames in order (noise floor).
- `reversed`: the 16 frames in reverse order.
- `shuffled`: a permutation seeded by `clip_id`, redrawn until Spearman(perm, identity) < 0.2.
- `frozen`: the middle frame (index 8) repeated 16 times.

Every condition, `original` included, is built from the same decoded frame arrays.
Neither judge ever sees the source file.

## Judges

- **AutoEval** (`videophysics/videophy_2_auto`), using its own `PROMPT_PHYSICS` template and
  greedy decoding. The 16 frames go straight into its image processor at 224x224,
  matching its `load_video`, and are not re-encoded to mp4.
- **Qwen2.5-VL-7B-Instruct**, receiving the 16 frames as separate images, with VideoPhy-2's
  physical-commonsense instruction plus a request for the number only. Greedy decoding.
  The prompt is identical across conditions and never mentions that frames may be
  reordered.
- For both judges I also log the probabilities of tokens 1 to 5 at the answer position,
  and use the continuous score `sum_s s * p(s)` (renormalized over 1 to 5).

## Pass/fail rule

For each plausible clip and condition c in {original_alt, reversed, shuffled, frozen}:
**pass iff score(c) < score(original), strictly.** A tie counts as a fail, and I report
the tie rate separately.

## Headline test

For each judge: pass rate(`reversed`) minus pass rate(`original_alt`), with a 95%
percentile bootstrap CI (10,000 resamples of clips, so conditions stay paired within a
clip). The effect counts as real only if the CI excludes 0. I report this for the integer
score (primary) and the continuous score (secondary).

## Controls

1. `original_alt` as the noise floor (above).
2. `reversed` vs `shuffled`: if shuffled drops but reversed does not, the judge is
   detecting discontinuity, not physics.
3. Motion terciles: mean DINOv2 ViT-S/14 cosine distance between consecutive original
   frames. Pass rates per tercile; the top tercile is the fair test.
4. Eyeball check: 20 reversed clips inspected; I report the fraction that look physically
   plausible backward, and rerun the reversal result without them.
5. Baseline signal: Spearman correlation of judge `original` score vs human `pc` on all
   200 clips.

## Scope

All claims are about these two judges on these VideoPhy-2 clips.
