# Do physics judges notice impossible time order?

Short answer: no. I took VideoPhy-2 clips that humans rated physically plausible, played
them backward, and asked two VLM physics judges to score them. Neither judge scored the
reversed clips lower than it scored the same clips with slightly different frames
sampled. The result held across 100 clips, five robustness checks, and a pre-registered
replication on 100 more clips with irreversible actions like pouring and breaking. Two
side findings were also consistent. VideoPhy-2-AutoEval, the judge that has been used as
an RL reward, picks up on shuffled frames but not reversed ones, so it seems to notice
jerky cuts rather than physics. It also gives a frozen clip (one frame repeated) a
higher score than the real video.

![Pass rates](results/passrates_continuous.png)

*Each point is the fraction of plausible clips where the altered version scored lower
than the original. The dashed lines are the noise floor: how often the score drops when
the same clip is just resampled at different frames. Whiskers are 95% bootstrap
intervals. A [version using integer scores](results/passrates.png) looks similar but is
mostly ties.*

## 1. Question

Physics judges aren't only used for evaluation anymore. They are being used as reward
signals to train video generators that get pitched as world simulators (HiPhy trains
against the VideoPhy-2 judge, for example). If a judge can't tell a possible time order
from an impossible one, then a generator trained against it never gets pushed toward
correct dynamics. So the question is simple: do these judges score a generated clip lower
when its frames are reversed, shuffled, or frozen?

## 2. Method

The core idea is that I don't need new labels. If humans rated a clip plausible, then
the same clip played backward is (usually) implausible by construction. This is an
adaptation of the metric-validity direction in the brief: the ground truth comes from
the construction instead of a hand-written physics check.

I sampled 100 plausible clips (human score 4 or 5) and 100 implausible ones (1 or 2) from
the VideoPhy-2 test set, balanced across its 7 video generators ([clips.csv](clips.csv)).
From each clip I took 16 evenly spaced frames and built five versions: the original, a
resampled original shifted by half a frame step (the noise floor), the frames reversed,
the frames shuffled, and the middle frame repeated 16 times.

The two judges were VideoPhy-2-AutoEval, run with its own code and prompt, and
Qwen2.5-VL-7B, given the 16 frames as images along with VideoPhy-2's physics instruction.
Both rate on a 1 to 5 scale. Since integer scores tie a lot, I also recorded each judge's
probability over the five answers and used the probability-weighted average as a
continuous score.

I committed the pass/fail rule before scoring anything
([6b147d2](https://github.com/Kushaan-N/physJudge/commit/6b147d2) and
[2c9bb9e](https://github.com/Kushaan-N/physJudge/commit/2c9bb9e), about half an hour before the
first scores). A judge passes a clip if the altered version scores strictly lower than
the original. The headline number is the reversed pass rate minus the noise-floor pass
rate, with a bootstrap confidence interval over clips. It only counts as an effect if
the interval excludes zero. For controls, I compared reversal against shuffling,
split clips by how much motion they contain, eyeballed 20 reversed clips, and checked how
well each judge agrees with the human scores to begin with.

## 3. Results

First, a sanity check. AutoEval's scores correlate reasonably with the human ratings
(Spearman 0.53). Qwen's barely do (0.13), mostly because it gives a 5 to almost
everything (559 of 600 calls).

The headline, reversed pass rate minus noise-floor pass rate:

| Judge | Integer score | Continuous score |
|---|---|---|
| AutoEval | -0.02 (-0.06 to 0.02) | 0.01 (-0.11 to 0.13) |
| Qwen | 0.00 (-0.03 to 0.03) | -0.11 (-0.22 to 0.00) |

All four intervals include zero, so neither judge penalizes reversal.

For AutoEval the effect is close to nothing. Reversing a clip changes its average score by
0.002 points, which is tiny next to the 0.58-point gap it puts between plausible and
implausible clips. Its scores for reversed clips actually track the originals more
closely than its scores for the resampled ones do. Freezing a clip raised its score on
83% of clips. Qwen's scores went slightly up under reordering, and the only alteration
it penalized was freezing.

In terms of the outcomes I wrote down beforehand, both judges look like they score
appearance and ignore time order. When AutoEval is run through its native video pipeline
(32 frames from an mp4 instead of 16 images), it does start catching shuffled clips, but
still not reversed ones. Reversal keeps the motion smooth, so this points to the judge
reacting to discontinuity, not physics.

The controls didn't change this. The integer and continuous scores agree. The
high-motion third of clips, where reversal should be most obvious, is still null for both
judges. I watched 20 of the reversed clips myself, and 15 of them still look plausible
backward. Dropping those 15 leaves the result the same.

I also ran robustness checks, which I wrote down before running them
([ROBUSTNESS.md](ROBUSTNESS.md)). The data passed an integrity audit. AutoEval through its
native mp4 pipeline and Qwen with a reworded prompt both stay null on reversal, and so do
alternative statistics (Wilcoxon test, a stricter pass margin, per-generator splits). One
check failed: rerunning on a different GPU model flipped 2 of 60 integer scores per judge.
Rerunning on the original GPU model reproduced every score exactly, so the pipeline is
deterministic and the difference comes from the hardware.

Since most reversed clips didn't look impossible, I ran a pre-registered replication on
100 new plausible clips whose prompts describe irreversible actions: pouring, tearing,
breaking, cutting, burning, falling and colliding ([PREREG_IRREVERSIBLE.md](PREREG_IRREVERSIBLE.md)).
The labels for which clips clearly break physics backward were made by Claude rather than
by me, which is recorded as a deviation, and they were committed before any scoring. Only
8 of the 100 clearly do. Most of these generated videos never actually show the
irreversible event: the candle never goes out, the wood never splits. Across all 100
clips, neither judge penalizes reversal (AutoEval -0.03, Qwen 0.01, both intervals
spanning zero). It is also null on the 8 clear violations, though 8 is too few to say
much. AutoEval again rewarded frozen clips and caught shuffling in its native pipeline.

Full tables for all of this are in [DETAILS.md](DETAILS.md). These claims are about these
two judges on these clips. It took about an hour from an empty directory to the first
results. Everything is inference, with no training: the main 1,200 judge calls took about
20 minutes on one GPU, and the robustness checks and replication took about 70 more.

## 4. What would change my mind, and what I'd do next week

The biggest weakness is that generated clips rarely contain an irreversible event. Only
3 of my 20 eyeballed clips and 8 of the 100 replication clips are clearly impossible
backward, so reversal on generated video ends up being a weak test. The null holds
everywhere, but on the handful of clear violations the intervals are wide. A judge that
only catches blatant reversals could still be hiding there. The replication labels also
came from still frames, where you can't always tell which way a pour is going. A human
rating would make them more solid.

With another week, I would:
1. Repeat the reversal test on real video of irreversible actions (pouring, tearing and
   dropping from Something-Something v2), where the event actually happens. That gives
   a well-powered test and separates "generated" from "judged".
2. Run the same paired test on rollouts from an action-conditioned world model, adding
   a wrong-action pairing as a second ground-truth negative.
3. Check for reward hacking. Since AutoEval rewards stillness, I'd test whether ranking by
   its score promotes low-motion clips, and whether a generator trained on it drifts
   toward less motion.

## 5. What I'd want to work on with you

Three directions interest me most.

**World models in medicine.** Surgical and endoscopic video generators are starting to be
pitched as simulators for training and planning, and the stakes for getting the physics
wrong are much higher there than in a stock-footage clip. I'd like to test whether these
models, and the judges used to score them, respect basic anatomy and tissue mechanics,
for example tissue that deforms and doesn't spring back after a cut. The approach would
be the same as here: paired, constructed negatives against a noise floor, instead of
trusting an automatic score.

**Touch in world models.** Almost every world model today is vision-only, but a lot of
physical understanding (grip, slip, how hard something is) only shows up through
contact. I'd want to see whether adding tactile signals to a world model actually
improves its predictions in contact-rich tasks. The control is a vision-only model with
the same capacity, which separates a real gain from simply having more inputs.

**World models as robot policy evaluators.** One of the most practical uses of a world
model is ranking robot policies before running them on real hardware. The question I'd
ask is when that ranking can be trusted: does the world model order policies the same way
real-world success does, and does it fail in predictable places, like contact or long
horizons? It extends this project directly, from "can we trust the judge" to "can we
trust the simulator".

---

**AI tools.** I set the question and design, ran the experiments, and directed the
follow-ups (robustness checks, the replication). I used Claude (via Claude Code) to check
design feasibility, implement the pipeline, label the replication clips (see the
pre-registration), and help draft this write-up.
