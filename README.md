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

