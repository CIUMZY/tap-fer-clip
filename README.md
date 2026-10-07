# TAP: training-seed variance in frozen-CLIP adaptation

Release artifacts for the manuscript

> Training-seed variance dominates single-run comparisons of token-level and global-feature
> adaptation in frozen CLIP

## Contents

- `code/` - source-domain training, retraining, extraction, attribution, confirmation, diagnostic
  and calibration scripts.
- `results/` - evidence layer: run manifests, per-cell CSVs, paired-bootstrap JSONs, the phase-8
  multi-seed retraining records, diagnostic summaries and per-phase reports.
- `models/` - trained source-domain arms, including the 15 retrained arm sets (10 seeds at
  ViT-B/16, 5 at ViT-L/14) used for the training-variance analysis.

## Headline numbers

Between-seed mean and standard deviation of the TAP minus CLIP-Adapter contrast, over the same
50 crossed-grid cells and the same 12 corruption conditions used in the single-run experiment:

| axis | encoder | single run | multi-seed mean | SD | seeds matching the single-run sign |
|---|---|---|---|---|---|
| crossed grid | ViT-B/16 | +0.0388 | -0.0328 | 0.0381 | 2/10 |
| crossed grid | ViT-L/14 | -0.0196 | -0.0014 | 0.0188 | 4/5 |
| corruption | ViT-B/16 | +0.0181 | -0.0004 | 0.0145 | 4/10 |
| corruption | ViT-L/14 | +0.0320 | +0.0370 | 0.0098 | 5/5 |

The same evaluator reproduces every single-run number to four decimals from the original
checkpoints (`phase8_calib_seed0.json`, `phase8_calib_corruption_seed0.json`).

## Feature-convention sensitivity and statistical robustness (2026-10-08)

The one contrast that survives retraining (ViT-L/14, corruption axis) was re-run under a symmetric
feature convention: the token arm was retrained on the spatially aligned two-view mean
(original + unflip(mirror)) / 2, with the learning rate selected on the source probe split only.

| corruption axis, TAP minus CLIP-Adapter | single view | two views, spatially aligned |
|---|---|---|
| ViT-B/16 | -0.0004 (SD 0.0145), 4/10 seeds positive | +0.0118 (SD 0.0100), 10/10 |
| ViT-L/14 | +0.0370 (SD 0.0098), 5/5 | +0.0357 (SD 0.0151), 5/5 |

The ViT-L/14 margin therefore remains positive under the convention that the global route already
uses, while the ViT-B/16 comparison is convention-dependent and brackets zero. Holding the
ViT-B/16 learning rate at the value the original protocol selected (3e-3) instead of re-selecting
it gives +0.0118 rather than +0.0124, so the convention effect is not a learning-rate effect.
Sources: results/phase9_aligned_corruption_L14.json, phase9_aligned_corruption_B16_lr3e-3.json,
phase9_aligned_corruption_B16.json; scripts phase9_mirror_train_extract.py,
phase9_build_aligned_cache.py, phase9_aligned_corruption.py; aligned arm checkpoints
models/phase9_aligned_tap_*.pt.

Statistical robustness of the six headline contrasts (exact two-sided sign tests, seed-level
bootstrap, within-seed unit-level bootstrap, Holm-Bonferroni and Benjamini-Hochberg) is in
results/STATS-ROBUSTNESS.md and results/phase10_stats_robustness.json, produced by
code/phase10_stats_robustness.py. Headline finding: at five seeds the exact sign test cannot reach
0.05 (its floor is 0.0625), so no contrast is significant under either correction; the surviving
margin is reported on its interval, on the unanimity of its sign and on its effect size.

## Not included

- The image corpora (FER2013/FER+, CK+, KDEF). Public research benchmarks, not redistributed here.
- The frozen patch-token and global-feature caches (tens of GB of fp16 arrays). These are embeddings
  of face images; available from the authors for research use only, under the terms of the corpus
  each cache derives from.
- The large Tip-Adapter cache-key tensors.

## Licensing

- Code: MIT (see LICENSE).
- Evidence files (results/, models/): research use only. The model weights are trained on
  face-image benchmarks, so downstream use must comply with the licence of the originating corpus.

## Reproducing

The scripts were written for a single workstation and use absolute paths rooted at
D:/ResearchVault/... . To run them elsewhere, edit the D, TC, FEA/FEAT and OUT constants at the top
of each script, and consult MANIFEST.json for the recorded command lines, input hashes and exit
codes for each phase.

Discipline used throughout: the encoder stays frozen; all training and hyper-parameter selection
use only the FER2013 probe-train / probe-validation split; no target label is used at any stage.

