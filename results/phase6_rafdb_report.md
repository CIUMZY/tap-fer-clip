# Phase 6 report - RAF-DB confirmation of H1: **GATE FAILED, experiment not run**

Executor: DeepSeek (deepseek-flash). Date: 2026-10-06. Interpreter
D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe.

**H1 has no verdict.** The pre-registered confirmation was blocked by its own step-1 gate: the
only RAF-DB copy on this machine cannot support the pre-registered protocol. No RAF-DB token was
extracted, no arm was trained on RAF-DB, no cell and no bootstrap interval exists. RAF-DB remains
an unconsumed confirmation set - nothing in this phase was selected, tuned or thresholded on it.

## 0. Pre-registration (verbatim; written before any RAF-DB output existed)

Frozen file: phase6_rafdb_preregistration.md (embedded here byte-for-byte).

~~~markdown
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
~~~

## 1. Gate verdict

Step 1 of the task requires verifying that the local RAF-DB copy is usable - original images with
labels, a known class mapping, per-class counts, and no contamination of the source training set -
and requires stopping honestly if that cannot be established. Of eight checks, four failed:

| check | result | evidence |
|---|---|---|
| local copy ships a provenance note | pass |  |
| provenance is not a quarantine notice | **GATE FAIL** | PROVENANCE.md status: QUARANTINED |
| an official RAF-DB split/label file exists on disk | **GATE FAIL** | searched D:/ResearchVault for list_patition_label.txt and variants |
| class counts are not an artificial balancing signature | **GATE FAIL** | classes sharing the identical count {5920, 6535, 11398, 8166}: ['angry', 'disgust', 'fear', 'surprise'] |
| no duplicated images inside the candidate set | pass | 0 of 49779 images are duplicates of another image after 48x48 normalisation |
| candidate set does not overlap the FER2013 source images | pass | 0 candidate images are pixel-identical (48x48 normalised) to a FER2013 image |
| all pre-registered ratios 0/1/2/5/10 are realisable | **GATE FAIL** | base=5920 majority=happy(11398); realisable ratios [0, 1]; blocked [2, 5, 10] |
| project label mapping is a clean folder->index bijection | pass | {"angry": [0], "disgust": [1], "fear": [2], "happy": [3], "neutral": [6], "sad": [4], "surprise": [5]} |

Verdict: **GATE FAILED - experiment not run**.

## 2. Evidence

**2.1 The copy declares itself unusable.** downloads/rafdb_processed/PROVENANCE.md records the
source as the Kaggle re-upload fahadullaha/facial-emotion-recognition-dataset, states the content
is "preprocessed FER2013 + RAF-DB JPEG files, 49,779 images", marks the status QUARANTINED, and
says: "Do not use these images for manuscript results until the redistribution rights and original
RAF-DB license are verified. Prefer the official RAF-DB request process for the paper."

**2.2 No official split or label file exists.** A recursive search of D:/ResearchVault for
list_patition_label.txt and variants returns nothing. The RAF-DB train/test split and its labels
cannot be recovered from this copy, and the images are re-indexed (angry_00000.jpg), not the
official train_00001_aligned.jpg / test_00001_aligned.jpg naming, so the RAF-DB subset cannot be
separated from the FER2013 images the same bundle contains.

**2.3 Class counts carry a re-balancing signature.**

| class | images | project index |
|---|---|---|
| angry | 5920 | 0 |
| disgust | 5920 | 1 |
| fear | 5920 | 2 |
| happy | 11398 | 3 |
| neutral | 8166 | 6 |
| sad | 6535 | 4 |
| surprise | 5920 | 5 |
| **total** | **49779** | |

angry, disgust, fear and surprise all hold exactly 5,920 images. A natural facial-expression
distribution does not have four identical class counts; this is the signature of oversampling or
synthesis applied to the minority classes, which would make any accuracy measured on this copy
uninterpretable. All 49779 images are 96x96 RGB, so they are not a
faithful copy of either RAF-DB (100x100 aligned colour) or FER2013 (48x48 grayscale) either.

**2.4 The pre-registered protocol is literally inexecutable on this copy.** The phase 3 ratio
construction takes base = smallest class count and gives the majority class base*ratio samples.
Here base = 5920 and the majority class is happy with 11398 images, so:

| ratio | majority samples needed | available | realisable |
|---|---|---|---|
| 0 | 0 | 11398 | yes |
| 1 | 5920 | 11398 | yes |
| 2 | 11840 | 11398 | **no** |
| 5 | 29600 | 11398 | **no** |
| 10 | 59200 | 11398 | **no** |

Only ratios 0 and 1 can be built; 2, 5 and 10 are blocked. The pre-registration explicitly requires
reporting a ratio as skipped rather than approximating it, but a run over ratios {0,1} only would no
longer be the pre-registered experiment, so it was not run.

**2.5 Two checks did pass, and they are worth recording honestly.**

- No duplicated images inside the candidate set: 0 of 49779 images repeat after 48x48
  normalisation. So the identical class counts are not explained by exact duplicates within this
  copy; they would be explained by augmentation or by images drawn from other sources.
- No detected contamination of the source training set: 0 of 49779 candidate images are
  pixel-identical (48x48 normalised) to any of the 35887 FER2013 images this project trains on.
  This rules out the crudest contamination route only. It does not rule out that the bundle mixes
  FER2013 and RAF-DB content as its own provenance note states, because our FER2013 copy is the
  official 48x48 grayscale release while this tree is a 96x96 colour re-render of unnamed images.
  The decisive point is that the source membership of these 49,779 images is not determinable from
  anything on disk, which is disqualifying for a confirmation set on its own.

**2.6 Class space mapping.** From the existing FER2013 token cache, the project label order is
angry 0, disgust 1, fear 2, happy 3, sad 4, surprise 5, neutral 6. The candidate tree uses the same
seven folder names, so the class space itself is not the blocker.

## 3. What was not run, and what that means for H1

- No ViT-B/16 or ViT-L/14 token extraction on RAF-DB.
- No training, no evaluation, no per-cell results, no bootstrap intervals.
- **H1 (TAP beats CLIP-Adapter at ViT-B/16 on RAF-DB) is therefore neither confirmed nor
  refuted. It stays open.**
- RAF-DB was not used for any selection, tuning or threshold choice, so it is still an unconsumed
  confirmation set in the sense that matters for the pre-registration.

Two deliverables requested for the confirmation run are delivered as explicit empty records rather
than omitted, so their absence cannot be mistaken for a missing file: phase6_rafdb_cells.csv carries
the intended schema and zero rows, and phase6_rafdb_bootstrap.json records status "not_run".

## 4. Conditions under which this exact pre-registered test can still be run

The pre-registration file is unmodified and remains valid. To execute it once and only once:

1. Obtain the official RAF-DB through the official request process (the same route the quarantine
   note recommends), or any copy that ships the official list_patition_label.txt with the
   official train/test split and its official filenames.
2. Re-verify the four gate checks that failed here, plus the class counts, from that copy - in
   particular that the ratio grid is realisable, which must be read off the official labels rather
   than assumed (the official basic-emotion set is far more imbalanced than this copy, so the
   minority-class count matters).
3. Run the frozen pre-registration unchanged: same hypothesis, same four arms, same decision rule
   (paired 95% CI excludes 0 and the margin exceeds the token-arm seed sd of 0.0116), same seeds
   0-4, same protocol constants. The only thing that changes is the data path.

If no official copy is obtained, the honest end state is the current one: H1 open, RAF-DB
unconsumed, and the encoder-dependence question answered only by the source-domain diagnostic of
the previous phase.

## 5. Scope, boundaries, unfinished

- All RAF-DB access in this phase was read-only enumeration and decoding for the gate audit;
  nothing was written inside downloads/rafdb_processed.
- The locked test set, the old 2250 holdout, Set A and Set B were not read. No main*.tex file was
  modified; nothing was submitted, no editor was contacted, nothing was paid, nothing transferred.
- Unfinished: the confirmation itself, for the reason above; the official RAF-DB acquisition is
  the only blocking precondition. The audit also did not attempt to determine which of the 49,779
  images originate from RAF-DB, because no on-disk information supports that determination.
- Runtime: gate audit 37.8 s (18.7 s to decode both image sets).
