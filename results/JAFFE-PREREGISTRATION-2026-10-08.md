
# Pre-specification: the JAFFE fresh confirmation axis (phase 13)

Registered 2026-10-08, **before any JAFFE image was opened, decoded or evaluated**.
This file is hashed and its digest recorded in phase13_prereg_freeze.json at registration
time; the hash is the evidence that the criterion below was not written after seeing a result.

## Data

JAFFE (Japanese Female Facial Expression database), obtained from Zenodo record 14974867,
DOI 10.5281/zenodo.14974867, local copy downloads/jaffe_official_v2/jaffe.zip with
sha256 6da27f5954f969c6f65d782911834dd66827ec3580e79b95073eb6cb93de5c3b and md5
fe13f3302eb9968ef04367456f665436. 213 images, 10 models, 7 expressions. The acquisition record
written on 2026-10-04 labels it "candidate untouched confirmation data; not extracted, trained,
tuned or evaluated". It has not been used for extraction, training, tuning or evaluation since.
The terms require non-commercial scientific use and cite both creator-required articles; the
images are not redistributed with the paper and no more than the permitted number of samples (10)
will ever be shown.

## Hypothesis to be tested

**H_J.** At ViT-L/14, the TAP minus CLIP-Adapter contrast on the JAFFE degradation axis has a
positive between-seed mean whose seed-level 95% t interval excludes zero.

Directional prediction: the same direction as the surviving contrast on the FER2013 degradation
axis (TAP more robust). Secondary comparators, reported but not part of the pass condition:
TAP minus the capacity-matched global head, and the same contrast at ViT-B/16 (a negative control;
the FER2013 study found no surviving effect there).

## Frozen preprocessing

1. Extract the 213 TIFFs from the archive. The label is read from the expression token in the
   file name: AN, DI, FE, HA, SA, SU, NE.
2. Map to the seven-class index used throughout this study and in FER2013:
   AN=0 angry, DI=1 disgust, FE=2 fear, HA=3 happy, SA=4 sad, SU=5 surprise, NE=6 neutral.
   Sanity check before any model runs: filenames starting with one model code appear exactly once
   per expression token, i.e. 10 x 7 + 143 = 213 images with no duplicate (model, expression) pair.
3. Convert to grayscale and resample to 48x48 with LANCZOS, matching the native resolution of
   FER2013, then save as PNG. Everything downstream is therefore pixel-comparable to the original
   axis; severities keep their original meaning and are not re-tuned.
4. Apply degradation with the identical implementation and parameter semantics used for the
   FER2013 axis (scripts/make_corrupted_fer2013.py, function corrupt), with the per-image noise
   seed kept at 1000 + row index.

## Frozen degradation grid

20 conditions = 4 families x 5 severities, extending the original 12-condition grid by one step in
each direction rather than selecting new levels:

| family | parameter | severities |
|---|---|---|
| blur | GaussianBlur radius | 0.8, 1.5, 2.5, 4.0, 6.0 |
| jpeg | JPEG quality | 15, 30, 50, 65, 80 |
| lowlight | brightness factor | 0.3, 0.5, 0.7, 0.85, 0.95 |
| noise | additive Gaussian sigma | 10, 25, 40, 55, 70 |

The original 12 conditions are the subset with severity levels {1st, 2nd, 3rd} of the first three
columns of each family as used before; the two extra levels are included so that the fresh axis has
more scoring units, not because a level was chosen by looking at a result.

## Arms evaluated

The source-domain arms already trained in phase 12 are evaluated as-is; **nothing is retrained,
re-tuned or selected on this axis**.

- ViT-L/14: seeds 5-19, learning rates {1e-3, 3e-3}, arms {TAP, CLIP-Adapter, capacity-matched head}
- ViT-B/16: seeds 5-14, learning rates {1e-3, 3e-3}, same three arms
- primary condition: each arm at the learning rate the original protocol selected
  (ViT-L/14: 1e-3 for all three; ViT-B/16: TAP 3e-3, CLIP-Adapter 1e-3, head 3e-3)
- secondary condition: rate re-selected inside each seed by its own source probe-validation split,
  exactly as in the nested analysis of the main paper

The encoder stays frozen. No target label enters training, tuning or selection at any point.

## Decision rule (fixed now)

- **CONFIRMED**: at ViT-L/14, mean of the per-seed TAP minus CLIP-Adapter contrast > 0 and the
  seed-level 95% t interval over the 15 seeds excludes zero.
- **STRENGTHENED**: CONFIRMED and at least 13 of the 15 seeds are positive.
- **NOT CONFIRMED**: the interval includes zero, or the mean is <= 0.

Either outcome is reported in the manuscript with the same prominence. A not-confirmed outcome is
written into the limitations as a failed replication of the one surviving claim; it is not deleted,
downplayed or re-framed. In particular, no severity level, condition, encoder, arm or metric may be
dropped or added after this point in order to change the verdict.

## Consumption

JAFFE becomes a consumed confirmation axis the moment the first extraction in this phase touches
it. It may not afterwards be used for model selection, hyper-parameter selection, threshold choice
or variant design. Any further variant of the token route would need yet another untouched corpus.

