# Pre-registration: replication on irreversible actions

Written and committed before this sample is drawn, scored or reviewed. It follows up on
the main study: there, 15 of 20 reviewed reversed clips still looked plausible backward,
so reversal was often not a real violation.

## Question

On generated clips whose action makes an irreversible change, do VideoPhy-2-AutoEval and
Qwen2.5-VL-7B score the reversed clip lower than the frame-resampling noise floor?

## Sample (`clips_irreversible.csv`)

**Eligibility:**
- Plausible VideoPhy-2 test clips (`pc` 4–5) in the HF mirror, ≥ 2 s long.
- Not used in the main study.
- Action is in one of the categories below.

**Sampling:** 100 clips, quotas split evenly across categories (capped by availability),
random within each category, seed 0.

**Categories,** chosen from action names only, before seeing any of these videos or
scores:

| Category | Actions |
|---|---|
| **pour** | pouring something into something until it overflows; pouring something into something; pouring something onto something; pouring something out of something; pouring beer; twisting (wringing) something wet until water comes out; throwing water balloon |
| **tear** | tearing something just a little bit; tearing something into two pieces; ripping paper |
| **break** | breaking boards; bending something until it breaks; pulling two ends of something so that it separates into two pieces; smashing; popping balloons |
| **cut** | cutting; using circular saw; chiseling stone; carving ice; flint knapping; sharpening pencil; peeling fruit |
| **fire** | wood burning (art); blowing out candles; extinguishing fire |
| **fall** | something falling like a rock; something falling like a feather or paper; poking something so that it falls over; poking a stack of something so the stack collapses; pushing something so that it falls off the table; putting something onto something else that cannot support it so it falls down; putting something that cannot actually stand upright upright on the table so it falls on its side |
| **collide** | something colliding with something and both come to a halt |

## Conditions and judges

Identical to the main study and its code:
- **Conditions:** `original`, `original_alt`, `reversed`, `shuffled`, `frozen`, K = 16.
- **Judges:** AutoEval (16 frames fed directly) and Qwen2.5-VL-7B (default prompt).
- **Secondary judge setting:** AutoEval through its native 32-frame mp4 path (as in R3).

## Human confirmation (blind to scores)

I review every reversed clip as a flipbook of its 16 frames, before looking at any
scores from this sample. Each clip gets one label:
- `violation`: clearly physically impossible backward.
- `plausible`: still looks plausible backward.
- `unsure`.

## Analysis

- **Primary:** pass rate(`reversed`) − pass rate(`original_alt`), on the clips I label
  `violation`. 95% bootstrap CI (10,000 clip resamples, paired, seed 0), for the
  integer and expected scores of each judge. **A judge penalizes reversal on clear
  violations only if this CI excludes 0 and is positive.**
- **Secondary:** the same on all 100 clips; the same per category; `shuffled` and
  `frozen` vs `original_alt`; the native-path AutoEval result.
- **Minimum sample:** if fewer than 30 clips are labeled `violation`, I report the
  primary result as underpowered, with its CI.
