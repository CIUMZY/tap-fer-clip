# Phase 7 mirror-view sensitivity - STATUS: INCOMPLETE (blocked by an evaluator bug)

**POST-HOC SENSITIVITY - NOT CONFIRMATORY.** This file records the state because the sensitivity
run could not be completed and finished in this session. No sensitivity verdict is issued.

## What is done and verified

Mirror-view ViT-B/16 patch tokens were extracted (new directory token_cache_vitb16_mirror/, no
existing cache touched), see phase7_mirror_extraction.json:

| dataset | n | shape | GB | seconds | peak VRAM | ids/labels aligned | mirror vs original-view-0 cosine (mean) |
|---|---|---|---|---|---|---|---|
| fer2013_train | 28709 | (28709,196,768) fp16 | 8.58 | 125.0 | 1.62 GB | yes | 0.9918 |
| fer2013_test | 7178 | (7178,196,768) fp16 | 2.14 | 30.1 | 1.62 GB | yes | 0.9916 |
| ckplus_test | 902 | (902,196,768) fp16 | 0.27 | 3.9 | 1.62 GB | yes | 0.9942 |
| kdef_test | 2938 | (2938,196,768) fp16 | 0.88 | 25.8 | 1.62 GB | yes | 0.9922 |

Validation as specified: shape / row count / id-label alignment against the existing caches (NOT
numerical equality with the original view). As a side check that mirrored images were really
processed, the mirror global feature differs from the original view-0 global in 99.9-100% of rows
with mean cosine 0.9916-0.9942.

## What is blocked and why

Training of the two-view-mean TAP completed for lr = 1e-3 (460 s; saved as
phase7_mirror_tap_lr0.001.pt, 202,504 parameters as required), but its **probe-val evaluation is
invalid**: the evaluator reads rows in ascending order for sequential I/O
(two_view(np.sort(va[...])) ) while the labels are taken in the requested (unsorted) order, so
predictions and labels are misaligned. The printed probe-val UAR 0.1402 is therefore ~chance and
must not be used; the learning-rate selection built on it is invalid too. This is the same bug
class that was found and fixed during the earlier source-domain diagnostic.

Fix required (one line): either evaluate without sorting the rows, or scatter the predictions back
into the requested order before scoring. The per-cell evaluation path (Set A / Set B) is not
affected because it uses the requested row order directly, but it cannot be reported until a valid
learning rate has been selected.

## Not produced

- phase7_mirror_sensitivity_report.md (no valid result to report)
- phase7_mirror_cells.csv, bootstrap intervals, MANIFEST entry
- L/14 mirror tokens (not attempted; B/16 was the priority and the blocker appeared first)

## Boundary

Read only the existing FER2013 source caches and the two confirmation-set caches already used in
phases 3-5; the locked test, the old 2250 holdout and the RAF-DB tree were not read; no main*.tex
file was modified; nothing was submitted.
