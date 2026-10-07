# TAP: token-level versus global-feature adaptation of frozen CLIP

Release artifacts for the manuscript

> Token-level versus global-feature adaptation of frozen CLIP: an encoder- and shift-dependent
> comparison for cross-dataset facial expression recognition

## Contents

- `code/` - source-domain training, extraction, attribution, confirmation and diagnostic scripts.
- `results/` - evidence layer: run manifests, per-cell CSVs, paired-bootstrap JSONs, diagnostic
  summaries and per-phase reports. These are the files the manuscript cites.
- `models/` - trained source-domain arms (TAP, CLIP-Adapter, capacity-matched head, linear probe)
  as PyTorch state dicts at the selected learning rates.

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

