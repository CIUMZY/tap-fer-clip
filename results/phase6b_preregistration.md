# Phase 6b pre-registration - corruption-sweep confirmation of H2

Written 2026-10-06, after the step-1 inventory and **before any corrupted-image token or global
feature was extracted and before any arm was evaluated on the corruption set**. Frozen from here:
the outcome is recorded in phase6b_confirm_report.md and this file is not edited afterwards.

## Why this axis (chosen from the inventory; no other axis was run)

The RAF-DB confirmation (phase 6) was blocked by its availability gate. The step-1 inventory of the
remaining local material (phase6b_inventory.json, phase6b_provenance.json) found one axis with a
complete, verifiable provenance chain and a real, never-used target condition:

**the graded synthetic-corruption sweep of the FER2013 test set** - 4 corruption types x 3 severity
levels, 7,178 images per condition, images on disk, per-condition manifests, and existing ViT-B/16
feature caches. Its provenance chain is: make_corrupted_fer2013.py (severity definitions and the
deterministic corruption seed 1000+row_index) -> raw/fer2013_shift_sweep/<condition>/ -> 
manifests/fer2013_<condition>.csv (path,label,domain,split; 7,178 rows each) ->
features/shift_sweep/<condition>.npz (ViT-B/16, [7178,2,512], unit-norm views).

The TAP line has never read any of this material. The only matches for corruption/shift/FER+ inside
the phase 1-6 work are descriptive strings in the diagnostic report's "not run" list and one note in
phase1_validate_tokens.py that FER+ images are fer2013_test images. No arm, no hyper-parameter and no
threshold of this project was ever chosen using it.

## Hypothesis

**H2 (primary, ViT-B/16)**: on frozen CLIP ViT-B/16, TAP beats CLIP-Adapter on the 12-condition
corruption set.

**Secondary (same rules, reported but not part of H2)**: the same comparison at frozen CLIP ViT-L/14.

## Arms (fixed; all reused as already trained on the FER2013 source domain - no re-training, no
re-selection, no tuning on the new conditions)

1. TAP - ViT-B/16: model_T1_lr0.003.pt (phase 2 selected lr 3e-3); ViT-L/14:
   phase5_tap_vitl14_lr0.001.pt.
2. CLIP-Adapter - ViT-B/16: phase4_clipadapter_lr0.001.pt; ViT-L/14:
   phase5_clipadapter_vitl14_lr0.001.pt.
3. Equal-capacity global head - ViT-B/16: model_GHead_lr0.003.pt (198,919 parameters);
   ViT-L/14: phase5_ghead_vitl14_lr0.001.pt (269,627 parameters).
4. Training-free reference column - max(PFB, support memory): PFB uses the source linear probe
   (features/linear_probe_train80_seed0.npz, features_vitl14/linear_probe_train80_seed0.npz) with the
   canonical online prior correction (lambda 0.2, min_prior_samples 32, prior_clip 1.0, s=100);
   support memory uses the frozen text prototypes and the source probe-train globals with
   affinity=max, BETA=256.

## Conditions (all 12 reported individually; no condition may be dropped or selected)

blur_0p8, blur_1p5, blur_2p5, jpeg_15, jpeg_30, jpeg_50, lowlight_0p3, lowlight_0p5, lowlight_0p7,
noise_10, noise_25, noise_40. Each: 7,178 FER2013 test images, unchanged labels.

No class-imbalance ratio is applied on this axis: every condition uses the full target, so the phase-3
ratio grid (0/1/2/5/10) does not apply here and is not reported. This is the axis definition, fixed
before the run.

## Seeds

Seeds 0-4 (five). The four arms are frozen, so TAP, CLIP-Adapter and the global head produce identical
values across the five seeds; the seed dimension varies only the online prior-correction order of the
PFB reference. Per-seed values and mean +/- sd are reported for every arm and condition regardless.

## Decision rule (fixed)

H2 holds **if and only if both**:

1. the paired 95% CI of TAP - CLIP-Adapter excludes 0 and is positive; and
2. the point estimate of TAP - CLIP-Adapter exceeds the between-seed standard deviation of the token
   arm, taken as the known ViT-B/16 value 0.0116 (secondary ViT-L/14: 0.0134).

Otherwise H2 does not hold and is reported as such. Every condition, arm and seed is reported whether
it favours TAP or not.

## Statistics

Paired shared-unit bootstrap, 2000 draws, hierarchical: seeds (0-4) and conditions (12) are both
resampled with replacement; within each sampled cell the 7,178 query indices are resampled with
replacement and shared across arms, so every arm is scored on identical resampled units. The reported
statistic is the mean over sampled cells. Because the trained arms are frozen, this interval covers
query and condition sampling; the training-seed component is the one handled by criterion 2 above.

## New derived data to be created by this phase (declared before the run)

- token_cache_vitb16_confirm/<condition>_tokens_fp16.npy + <condition>_meta.npz (196x768 fp16,
  memory-mapped, no shards, written incrementally)
- token_cache_vitl14_confirm/<condition>_tokens_fp16.npy + <condition>_meta.npz (256x1024 fp16)
- features_confirm_shift/vitb16/<condition>.npz and features_confirm_shift/vitl14/<condition>.npz
  (globals in the existing feature-cache convention: ids, views [n,2,d] unit-norm, labels, meta_json)
- extraction uses open_clip preprocess and visual(x) -> (pooled, tokens), i.e. exactly the convention
  of the existing clean caches: pooled L2-normalised, two views (image, mirror), patch tokens as stored

## Extraction validation gate (must pass before any evaluation is reported)

1. Recompute the ViT-B/16 globals for the clean fer2013 test images and match them against
   features/fer2013_test.npz.
2. Recompute the ViT-B/16 globals for at least one corrupted condition and match them against
   features/shift_sweep/<condition>.npz (cosine >= 0.9999 elementwise after view alignment).
If either check fails, the run stops and no H2 verdict is issued.

## Cost (stated before the run)

- New data: about 26 GB (ViT-B/16 tokens) + about 45 GB (ViT-L/14 tokens) + about 0.9 GB (globals).
- Extraction: about 8 min at ViT-B/16 and about 24 min at ViT-L/14, measured against the recorded
  clean-cache rates (191 and 61 images/s).
- Evaluation plus bootstrap: about 10-15 min. Total about 45-50 min of GPU.

## Discipline

- The corruption conditions become a **confirmation set** with this file: no selection, tuning or
  threshold choice on them, before or after this run.
- Training and hyper-parameter selection remain the FER2013 source probe-train / probe-val choices
  already fixed in phases 2-5; nothing is re-selected.
- Not read: the locked test set, the old 2250 holdout, the phase 3/4 confirmation sets (Set A, Set B),
  and the RAF-DB tree.
- No main*.tex file is modified; nothing is submitted, no editor is contacted, nothing is paid.

---

## Pre-run addendum (approved 2026-10-06, written before any extraction)

Approved by the user before execution. It adds reporting statements only and **does not change the
hypothesis, the axis, the arms, the conditions, the seeds or the decision rule** above.

1. **What this axis measures.** This is a **within-domain** robustness axis: the target is FER2013
   test itself, degraded synthetically, not a cross-database target. If H2 holds, the claim the paper
   can make is "TAP is more robust than CLIP-Adapter under within-domain input degradation". It must
   **not** be presented as a confirmation of cross-database accuracy.
2. **Locked-set boundary.** This axis reads only the FER2013 test images and their 12 degraded
   versions (a target that this line already used as fer2013_test / fer2013_shift in phases 3-5).
   The old 2250 holdout was not read, Set A and Set B were not read, and the RAF-DB tree was not
   read.

## Known limitation (recorded before the run, to be stated in the report)

The four arms are **frozen** models, so seeds 0-4 change only the online prior-correction order of
the PFB reference column; TAP, CLIP-Adapter, the equal-capacity global head and the support-memory
column are identical across the five seeds. The training-seed variance is therefore carried by
criterion 2 (the given between-seed sd 0.0116 at ViT-B/16 and 0.0134 at ViT-L/14), and the
bootstrap covers query and condition sampling only. This must not be described as "five independent
seed repeats".
