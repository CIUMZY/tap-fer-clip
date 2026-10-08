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
50 crossed-grid cells, three second-source targets and 12 corruption conditions used in the
single-run experiment. Retrained from the seed alone (initial weights, batch order and learning
rate all follow from the seed) with 10 seeds at ViT-B/16 and 15 at ViT-L/14:

| axis | encoder | single run | retrained mean | SD | seeds matching the single-run sign | verdict |
|---|---|---|---|---|---|---|
| crossed grid | ViT-B/16 | +0.0388 | -0.0161 | 0.0475 | 2/10 | not recovered |
| crossed grid | ViT-L/14 | -0.0196 | -0.0019 | 0.0242 | 6/15 | not recovered |
| second source | ViT-B/16 | +0.0347 | +0.0034 | 0.0365 | 3/10 | not recovered |
| second source | ViT-L/14 | -0.0137 | +0.0059 | 0.0151 | 9/15 | not recovered |
| corruption | ViT-B/16 | +0.0181 | +0.0049 | 0.0118 | 6/10 | not recovered |
| corruption | ViT-L/14 | +0.0320 | **+0.0350** | 0.0169 | **15/15** | survives Holm and BH correction |

The same evaluator reproduces every single-run number to four decimals from the original
checkpoints (`phase8_calib_seed0.json`, `phase8_calib_corruption_seed0.json`, `phase8_calib_setB_seed0.json`).

## Phase 12 supersedes phase 8 for the retraining distributions

Phase 8 seeded the batch order but constructed each arm *before* `torch.manual_seed(seed)` was called, so the
initial weights of an arm came from whatever global RNG state the process was in. Re-running the
phase-8 training script therefore does not reproduce its own token-arm checkpoints; the
CLIP-Adapter and global-head arms happen to be unaffected, because their construction follows a
`torch.manual_seed(seed)` call left by the previous arm, which is why phase 11 reproduced those two bit for bit
and the token arm not at all. Phase 12 fixes this by building every arm inside `torch.manual_seed(seed)` and
re-runs everything: 10 seeds x 2 rates at ViT-B/16, 15 seeds x 2 rates at ViT-L/14, 150
checkpoints. A rebuild of the same arm from the same seed is bit-identical (field
determinism_rebuild_identical in phase12_train_B16.json).

Two things follow from the re-run. First, the five unstable contrasts move to values
indistinguishable from zero rather than merely differing in size. Second, with 15 seeds the exact
sign test is no longer floored at 0.0625 and the surviving ViT-L/14 degradation margin passes both
Holm-Bonferroni and Benjamini-Hochberg correction over the family of six (adjusted p = 0.0004).

Phase 12 also adds the **nested condition**: every arm is trained at both grid points and each
seed's own probe split chooses the rate, so the reported distribution covers rate selection as
well as fitting. That moves the ViT-B/16 crossed-grid contrast from -1.61 to +1.66 points (a 3.27
point swing, the size of the effect the single run reported there) while leaving the survivor
unchanged (+3.68 versus +3.50 points).

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

