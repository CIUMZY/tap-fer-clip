# Phase 20 pre-registration: prospective replication of the training-seed distribution

Date written: 2026-10-09, before any phase-20 seed was trained or evaluated.
Author of record: the manuscript's first author. Prepared with the analysis assistant.

## Why

The venue-calibrated review objected that the ViT-L/14 evidence rests on an outcome-aware extension:
the retraining first used five seeds, the surviving contrast was already positive in those five, and
the count was then raised to fifteen. The disclosure is in the manuscript, but disclosure alone does
not restore a prospective guarantee. The only way to address that objection with evidence is to fix a
seed set and a criterion in advance, run it, and report whatever it gives.

## Question

Does the between-seed distribution of the TAP minus CLIP-Adapter contrast replicate on a fresh batch of
training seeds, fixed before any of them was run?

## Design

- Batch 2 seeds: ViT-L/14 seeds 20-34 (15 seeds); ViT-B/16 seeds 15-24 (10 seeds).
- Everything else is held fixed and identical to the batch-1 recipe: the same three arms (TAP,
  CLIP-Adapter, capacity-matched global head), the same two learning rates {1e-3, 3e-3}, the same
  seed-controlled construction (initialisation, batch order and nested rate selection all follow from
  the seed), the same FER2013 probe-train / probe-validation split used for fitting and rate selection,
  the same published rates per arm and encoder, and the same evaluation code.
- The scripts are phase20_train_prospective.py and phase20_eval_prospective.py, generated from the
  phase-12 scripts by phase20_prep.py with the seed list changed and no other functional change.
- Evaluation axes: the twelve-condition corruption axis, the 50-cell crossed grid (Set A) and the
  15-cell second source (Set B). All three axes are already consumed by the manuscript. This
  pre-registration therefore tests the reproducibility of the training-seed distribution and the
  stability of the surviving contrast. It is not a new confirmation claim, and it is not presented as
  one.

## Primary criterion, fixed now

For the ViT-L/14 corruption axis at the published rate, contrast TAP minus CLIP-Adapter:

- H1: the batch-2 mean is positive.
- H2: the batch-2 seed-level 95 percent t interval excludes zero.
- H3 (agreement, reported but not part of the pass rule): the batch-2 mean lies within one pooled
  between-seed standard deviation of the batch-1 mean.

Verdict: batch 2 REPLICATES if H1 and H2 both hold. Otherwise batch 2 DOES NOT REPLICATE and that
is what will be written.

## Secondary, reported without a pass rule

- The same three quantities for TAP minus the capacity-matched head.
- The same quantities for the other five contrasts of the six-contrast family, where the expectation
  from batch 1 is that none reaches significance.
- The pooled 30-seed (ViT-L/14) and 20-seed (ViT-B/16) estimates and their intervals.
- The ViT-B/16 corruption-axis contrast, for the encoder comparison.

## Discipline

- No seed, arm, rate, axis or threshold is added, removed or re-selected after this file is hashed.
- If batch 2 fails, no further seeds are run to rescue it, and the failure is reported in the
  manuscript with the same prominence as batch 1.
- Training and rate selection use only the FER2013 probe-train / probe-validation split. No target
  label enters training or selection. The locked test and the old 2250 holdout are untouched.
- The evaluation axes are consumed, so nothing in this run can become a new primary claim.

## Recorded hash

The sha256 of this file is recorded in phase20_prereg_freeze.json at the moment of writing and before
the first training run. Both are archived in the release repository.
