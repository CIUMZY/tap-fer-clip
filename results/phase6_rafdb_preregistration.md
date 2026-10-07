# Phase 6 pre-registration - RAF-DB confirmation of H1 (written before any RAF-DB model output exists)

Written: 2026-10-06, before any RAF-DB token was extracted and before any RAF-DB evaluation was run.
This file is frozen once the gate in step 1 is evaluated; the outcome is recorded in
phase6_rafdb_report.md and this file is not edited afterwards.

## Hypothesis

**H1**: on frozen CLIP **ViT-B/16**, TAP (token attention pooling) beats CLIP-Adapter on the
**RAF-DB** target.

## Arms (fixed; trained only on the FER2013 source probe-train split, lr selected on FER2013
probe-val)

1. TAP - attention pooling over 196 patch tokens, architecture identical to phase 2 T1.
2. CLIP-Adapter - bottleneck residual adapter (ratio 4, residual 0.2) on the 512-d global feature,
   classified by the frozen text prototypes.
3. Equal-capacity global head - the phase 3/5 global-head form, parameter-matched to TAP.
4. Training-free reference column - max(PFB, support memory), s=100, affinity=max, beta=256,
   prior_clip=1.0, PFB lambda=0.2, min_prior_samples=32.

## Decision rule (fixed)

H1 holds **if and only if** both of the following are true:

1. the paired 95% CI of TAP - CLIP-Adapter excludes 0 and is positive; and
2. the point estimate of TAP - CLIP-Adapter exceeds the between-seed standard deviation of the
   TAP arm, whose known ViT-B/16 value is 0.0116 (11.6pp-scale, i.e. 1.16pp).

Otherwise H1 does **not** hold and will be reported as such, without re-interpretation.

## Seeds

At least 5: seeds 0, 1, 2, 3, 4. Per-seed values and mean +/- sd are reported.

## Protocol

Identical to phase 3 / phase 4: s=100 (logit scale), affinity=max, beta=256, prior_clip=1.0,
PFB lambda=0.2, min_prior_samples=32, 5-way-irrelevant (7 classes incl. neutral). Targets:
RAF-DB full target plus class-imbalance ratios 0/1/2/5/10, where the ratio construction of
phase 3 is used (base = smallest class count; the majority class contributes base*ratio samples).
If RAF-DB class counts cannot realise a ratio, that ratio is reported as skipped, not approximated.

## Secondary (reported under the same rules, not part of H1)

- The same comparison at frozen CLIP ViT-L/14, if the tokens can be extracted within the memory and
  time budget; otherwise explicitly left blank.
- The training-free reference column on both encoders.

## Discipline

- RAF-DB is a **confirmation set** from now on: no selection, tuning or threshold choice on it.
- Training and hyper-parameter selection use the FER2013 source probe-train / probe-val split only.
- The locked test set, the old 2250 holdout and the phase 3/4 confirmation sets (Set A, Set B) are
  not read.
- Precondition (step 1 of the task, evaluated before this experiment may run): the local RAF-DB copy
  must be verified usable - original images with labels, known class mapping, per-class counts, no
  contamination of the source training set. If that cannot be established, the experiment does not
  run and the reason is recorded.
