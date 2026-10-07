# Phase 6b report - corruption-sweep confirmation of H2

## Two statements that define what this run can claim

1. **What this axis measures.** This is a **within-domain robustness axis**: the target is
   FER2013 test itself, degraded synthetically in 12 conditions - not a cross-database target.
   If H2 holds, the claim the paper can make is "TAP is more robust than CLIP-Adapter under
   within-domain input degradation". It must **not** be presented as a confirmation of
   cross-database accuracy.
2. **Locked-set boundary.** This run reads only the FER2013 test images and their 12 degraded
   versions (a target this line already used as fer2013_test / fer2013_shift in phases 3-5). It
   did not read the old 2250 holdout, did not read Set A or Set B, and did not read the RAF-DB
   tree.

**Known limitation, stated plainly.** The four arms are frozen models, so seeds 0-4 change only
the online prior-correction order of the PFB reference column; TAP, CLIP-Adapter, the
equal-capacity global head and the support-memory column are identical across the five seeds.
This must not be read as "five independent seed repeats". The training-seed variance is carried
by decision criterion 2 (the given between-seed sd 0.0116 at ViT-B/16 and 0.0134 at ViT-L/14),
and the bootstrap covers query and condition sampling only.

## Verdict

**H2 (primary, ViT-B/16): HOLDS.**

- TAP - CLIP-Adapter = +0.0181 (1.81pp), paired 95% CI [+0.0075, +0.0292].
- Criterion 1 (CI excludes 0 and is positive): True. Criterion 2 (|margin| > 0.0116): True.

**ViT-L/14 (secondary, not part of H2):** TAP - CLIP-Adapter = +0.0320 (3.20pp), CI
[+0.0088, +0.0569]; criterion 1 True, criterion 2 True.

## 0. Pre-registration (verbatim, frozen before extraction)

~~~markdown
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
~~~

## 1. Inventory (step 1) and what was chosen

Full audit: phase6b_inventory_report.md, phase6b_inventory.json, phase6b_provenance.json.
The chosen axis is the graded corruption sweep of FER2013 test: 4 corruption types x 3
severities, 7178 images per condition, with a single provenance chain from
make_corrupted_fer2013.py through raw/fer2013_shift_sweep/<condition>/ and
manifests/fer2013_<condition>.csv to the existing features/shift_sweep/<condition>.npz.
The four legacy features/corruptions/*.npz files were excluded (two are exact duplicates of
sweep levels, two come from a different generator script, one of which routes through
grayscale). FER+ 1:1 / 5:1 is available and ratio-feasible but was not chosen; features/
ckplus48_all.npz carries no labels, and CK+/KDEF were already confirmation targets in phases 3-5.

## 2. Extraction and the validation gate

| encoder | conditions | rows per condition | new token bytes | clean-cache cosine | condition cosine | gate |
|---|---|---|---|---|---|---|
| ViT-B/16 | 12 | 7178 | 25.9 GB | 1.000000 | 1.000000 | PASS |
| ViT-L/14 | 12 | 7178 | 45.2 GB | 1.000000 | n/a (no stored L/14 cache) | PASS |

The gate requires elementwise cosine >= 0.9999 against the stored caches. Both encoders passed
at 1.000000, i.e. this extraction reproduces the stored clean caches exactly. The chip
convention is the same one the clean caches and the phase-5 token extraction use: open_clip
preprocess, one original view for the patch tokens (fp16), two views (image + mirror) for the
L2-normalised global features (fp32).

### 2.1 What the ViT-L/14 gate actually verified, and what it could not

The B/16 gate followed the pre-registration literally: clean globals recomputed and matched
against features/fer2013_test.npz, plus one corrupted condition recomputed and matched against
the pre-existing features/shift_sweep/blur_0p8.npz. **There is no equivalent pre-existing
corrupted cache for ViT-L/14** - features/shift_sweep/*.npz are 512-d ViT-B/16 artifacts, and
the L/14 run wrote clean_tok=None / cond=n/a in its gate record for exactly that reason (the
token-cache lookup also used a wrong directory name, token_cache_l14 instead of
token_cache_vitl14). Those two gaps were closed by two follow-up checks, run after the fact and
reported as such rather than folded into the gate:

- phase6b_validate_l14_tokens.py: 512 clean FER2013 test rows recomputed for ViT-L/14, globals
  cosine min 1.000000 against features_vitl14/fer2013_test.npz and patch-token cosine min 1.000000
  against token_cache_vitl14/fer2013_test_tokens_fp16.npy -> PASS.
- phase6b_verify_extraction.py: every one of the 24 condition caches (12 conditions x 2
  encoders) checked for structural completeness (file size equals the .npy header plus nbytes),
  meta id/label alignment against the manifests and the clean test labels, unit-norm views, and
  an independent recomputation of 8 rows per condition. Result: 24/24 verified. Patch tokens
  reproduce to 0.12-1.5 fp16 ULP (per-image cosine 1.000000 at both encoders) and globals to
  cosine 1.000000.

So the honest statement of L/14 evidence strength is: its extraction is verified by exact
agreement with the *clean* L/14 caches plus per-condition recomputation, and its arms are
verified by reproducing the stored phase-5 Set B fer2013_test numbers to four decimals - but it
has **no independent third-party corrupted reference** to agree with, unlike ViT-B/16. L/14 is
the secondary encoder and does not enter the H2 decision; this limitation is recorded rather
than treated as equivalent to the B/16 gate.

The 12 reused L/14 condition files (written before the validation-step CUDA OOM of the first
attempt) were explicitly re-verified before any evaluation: all 12 are byte-complete with
matching ids, labels and unit-norm views, so nothing needed re-extraction.

Cost actually paid: about 12 min for 12 conditions at ViT-B/16 (26 GB) and about 45-50 min at
ViT-L/14 (45 GB). The pre-registration estimated 8 and 24 min by multiplying the recorded
clean-cache rate by the image count; the global-feature convention needs two views per image,
which doubles the forward passes. The estimate was corrected here, after the run, and it
affects no hypothesis, arm, condition or criterion.

## 3. Results - all 12 conditions, both encoders

### ViT-B/16 (primary, H2)

| condition | TAP | GHead | CLIP-Adapter | PFB (mean of 5 seeds) | support memory | max(PFB,memory) | TAP - CLIP-Adapter |
|---|---|---|---|---|---|---|---|
| blur_0p8 | 0.5886 | 0.5511 | 0.5510 | 0.5542 | 0.5005 | 0.5492 | +0.0377 |
| blur_1p5 | 0.3837 | 0.3379 | 0.3275 | 0.3742 | 0.1648 | 0.3253 | +0.0561 |
| blur_2p5 | 0.1981 | 0.2063 | 0.1926 | 0.2033 | 0.1443 | 0.1887 | +0.0055 |
| jpeg_15 | 0.4057 | 0.4195 | 0.4129 | 0.4035 | 0.3028 | 0.3651 | -0.0072 |
| jpeg_30 | 0.4982 | 0.4971 | 0.4996 | 0.4821 | 0.4255 | 0.4645 | -0.0014 |
| jpeg_50 | 0.5482 | 0.5442 | 0.5452 | 0.5241 | 0.5014 | 0.5189 | +0.0030 |
| lowlight_0p3 | 0.6190 | 0.5739 | 0.5832 | 0.5838 | 0.6094 | 0.6158 | +0.0359 |
| lowlight_0p5 | 0.6279 | 0.5903 | 0.5942 | 0.5895 | 0.6328 | 0.6240 | +0.0337 |
| lowlight_0p7 | 0.6323 | 0.5980 | 0.6084 | 0.5938 | 0.6437 | 0.6288 | +0.0239 |
| noise_10 | 0.5289 | 0.4816 | 0.5033 | 0.4863 | 0.4220 | 0.4668 | +0.0257 |
| noise_25 | 0.3793 | 0.3605 | 0.3783 | 0.3902 | 0.2892 | 0.3401 | +0.0010 |
| noise_40 | 0.2784 | 0.2668 | 0.2745 | 0.2914 | 0.2174 | 0.2469 | +0.0040 |
| **mean over conditions** | **0.4740** | **0.4523** | **0.4559** | **0.4564** | **0.4045** | **0.4445** | **+0.0181** |

TAP is above CLIP-Adapter in 10 of 12 conditions. Across conditions the sd of the TAP mean is
0.1440 and of the CLIP-Adapter mean is 0.1370 (condition-level spread, for orientation only).

Per-seed detail: the three trained arms are constant across seeds 0-4 (frozen models); only
the PFB column and therefore max(PFB, memory) move with the seed, through the online
prior-correction order. Per-seed values for every condition and every arm are in
phase6b_cells_B16.csv (columns pfb|0..4, best_family|0..4).

PFB per-seed spread by condition (min/mean/max over seeds 0-4):

| condition | PFB min | PFB mean | PFB max | max(PFB,mem) min | max(PFB,mem) mean | max(PFB,mem) max |
|---|---|---|---|---|---|---|
| blur_0p8 | 0.5537 | 0.5542 | 0.5548 | 0.5487 | 0.5492 | 0.5495 |
| blur_1p5 | 0.3731 | 0.3742 | 0.3754 | 0.3246 | 0.3253 | 0.3262 |
| blur_2p5 | 0.2009 | 0.2033 | 0.2073 | 0.1877 | 0.1887 | 0.1900 |
| jpeg_15 | 0.4008 | 0.4035 | 0.4045 | 0.3649 | 0.3651 | 0.3654 |
| jpeg_30 | 0.4808 | 0.4821 | 0.4836 | 0.4634 | 0.4645 | 0.4658 |
| jpeg_50 | 0.5228 | 0.5241 | 0.5263 | 0.5180 | 0.5189 | 0.5197 |
| lowlight_0p3 | 0.5825 | 0.5838 | 0.5845 | 0.6153 | 0.6158 | 0.6168 |
| lowlight_0p5 | 0.5878 | 0.5895 | 0.5909 | 0.6231 | 0.6240 | 0.6249 |
| lowlight_0p7 | 0.5931 | 0.5938 | 0.5953 | 0.6272 | 0.6288 | 0.6296 |
| noise_10 | 0.4846 | 0.4863 | 0.4886 | 0.4650 | 0.4668 | 0.4679 |
| noise_25 | 0.3885 | 0.3902 | 0.3921 | 0.3390 | 0.3401 | 0.3409 |
| noise_40 | 0.2887 | 0.2914 | 0.2936 | 0.2458 | 0.2469 | 0.2477 |

### ViT-L/14 (secondary)

| condition | TAP | GHead | CLIP-Adapter | PFB (mean of 5 seeds) | support memory | max(PFB,memory) | TAP - CLIP-Adapter |
|---|---|---|---|---|---|---|---|
| blur_0p8 | 0.6307 | 0.5297 | 0.4989 | 0.6254 | 0.4286 | 0.5779 | +0.1319 |
| blur_1p5 | 0.4203 | 0.3636 | 0.3514 | 0.4281 | 0.2542 | 0.4047 | +0.0688 |
| blur_2p5 | 0.2396 | 0.2131 | 0.2357 | 0.2399 | 0.1969 | 0.2369 | +0.0040 |
| jpeg_15 | 0.4783 | 0.4875 | 0.4580 | 0.4748 | 0.3773 | 0.4358 | +0.0204 |
| jpeg_30 | 0.5611 | 0.5748 | 0.5343 | 0.5487 | 0.4889 | 0.5257 | +0.0268 |
| jpeg_50 | 0.5942 | 0.6170 | 0.5780 | 0.5927 | 0.5518 | 0.5774 | +0.0162 |
| lowlight_0p3 | 0.6438 | 0.6494 | 0.6670 | 0.6591 | 0.6473 | 0.6578 | -0.0231 |
| lowlight_0p5 | 0.6740 | 0.6663 | 0.6879 | 0.6620 | 0.6633 | 0.6691 | -0.0139 |
| lowlight_0p7 | 0.6791 | 0.6752 | 0.6887 | 0.6610 | 0.6733 | 0.6757 | -0.0096 |
| noise_10 | 0.6261 | 0.6225 | 0.5868 | 0.6003 | 0.5699 | 0.5811 | +0.0393 |
| noise_25 | 0.4981 | 0.4770 | 0.4238 | 0.4577 | 0.3483 | 0.4102 | +0.0743 |
| noise_40 | 0.3870 | 0.3519 | 0.3381 | 0.3475 | 0.2452 | 0.3057 | +0.0489 |
| **mean over conditions** | **0.5360** | **0.5190** | **0.5040** | **0.5248** | **0.4538** | **0.5048** | **+0.0320** |

TAP is above CLIP-Adapter in 9 of 12 conditions. Across conditions the sd of the TAP mean is
0.1350 and of the CLIP-Adapter mean is 0.1475 (condition-level spread, for orientation only).

Per-seed detail: the three trained arms are constant across seeds 0-4 (frozen models); only
the PFB column and therefore max(PFB, memory) move with the seed, through the online
prior-correction order. Per-seed values for every condition and every arm are in
phase6b_cells_L14.csv (columns pfb|0..4, best_family|0..4).

PFB per-seed spread by condition (min/mean/max over seeds 0-4):

| condition | PFB min | PFB mean | PFB max | max(PFB,mem) min | max(PFB,mem) mean | max(PFB,mem) max |
|---|---|---|---|---|---|---|
| blur_0p8 | 0.6251 | 0.6254 | 0.6256 | 0.5774 | 0.5779 | 0.5782 |
| blur_1p5 | 0.4261 | 0.4281 | 0.4308 | 0.4034 | 0.4047 | 0.4059 |
| blur_2p5 | 0.2388 | 0.2399 | 0.2407 | 0.2366 | 0.2369 | 0.2377 |
| jpeg_15 | 0.4730 | 0.4748 | 0.4756 | 0.4354 | 0.4358 | 0.4362 |
| jpeg_30 | 0.5474 | 0.5487 | 0.5497 | 0.5250 | 0.5257 | 0.5267 |
| jpeg_50 | 0.5921 | 0.5927 | 0.5932 | 0.5772 | 0.5774 | 0.5776 |
| lowlight_0p3 | 0.6584 | 0.6591 | 0.6596 | 0.6575 | 0.6578 | 0.6582 |
| lowlight_0p5 | 0.6610 | 0.6620 | 0.6627 | 0.6682 | 0.6691 | 0.6696 |
| lowlight_0p7 | 0.6602 | 0.6610 | 0.6618 | 0.6747 | 0.6757 | 0.6770 |
| noise_10 | 0.5987 | 0.6003 | 0.6017 | 0.5805 | 0.5811 | 0.5814 |
| noise_25 | 0.4567 | 0.4577 | 0.4595 | 0.4096 | 0.4102 | 0.4112 |
| noise_40 | 0.3444 | 0.3475 | 0.3500 | 0.3049 | 0.3057 | 0.3061 |

## 4. Paired shared-unit bootstrap and the decision rule

2000 draws; seeds (0-4) and conditions (12) resampled with replacement; within each sampled
cell the 7178 query indices are resampled with replacement and shared across arms. Because the
trained arms are frozen, these intervals reflect query and condition sampling only.

### ViT-B/16

| pair | mean | 95% CI |
|---|---|---|
| TAP - CLIPAdapter | +0.0180 | [+0.0075, +0.0292] |
| TAP - GHead | +0.0216 | [+0.0097, +0.0335] |
| TAP - best-of-two-families (seed-0 draw) | +0.0296 | [+0.0178, +0.0408] |
| CLIP-Adapter - best-of-two-families (seed-0 draw) | +0.0116 | [-0.0035, +0.0258] |

Decision on H2: point +0.0181,
criterion 1 True, criterion 2 (|margin| > 0.0116) True -> **HOLDS**.

### ViT-L/14

| pair | mean | 95% CI |
|---|---|---|
| TAP - CLIPAdapter | +0.0317 | [+0.0088, +0.0569] |
| TAP - GHead | +0.0169 | [+0.0003, +0.0384] |
| TAP - best-of-two-families (seed-0 draw) | +0.0311 | [+0.0148, +0.0489] |
| CLIP-Adapter - best-of-two-families (seed-0 draw) | -0.0006 | [-0.0211, +0.0142] |

Decision on the secondary comparison: point +0.0320,
criterion 1 True, criterion 2 (|margin| > 0.0134) True -> **HOLDS**.

## 5. Pipeline cross-check

Before any corrupted condition was scored, the same code path was run on the clean FER2013
test cache and compared with the stored phase 4 / phase 5 Set B values for fer2013_test:

| encoder | arm | computed here | stored | diff |
|---|---|---|---|---|
| ViT-B/16 | TAP | 0.6323 | 0.6323 | +0.0000 |
| ViT-B/16 | GHead | 0.6083 | 0.6083 | +0.0000 |
| ViT-B/16 | CLIPAdapter | 0.6163 | 0.6163 | -0.0000 |
| ViT-L/14 | TAP | 0.6836 | 0.6836 | +0.0000 |
| ViT-L/14 | GHead | 0.6828 | 0.6828 | +0.0000 |
| ViT-L/14 | CLIPAdapter | 0.6916 | 0.6916 | -0.0000 |

The trained arms reproduce the stored numbers to four decimals. The PFB and support-memory
columns here use the **FER2013 source support of the pre-registration** (the Set A convention),
not the KDEF-source support used by the stored Set B reference columns, so those two columns
are reported but not compared - they are a different, pre-registered configuration.

## 6. What this does and does not support

- It supports, or fails to support, the specific claim that TAP is more robust than
  CLIP-Adapter under within-domain input degradation on FER2013 test, under the fixed rule.
- It says nothing about cross-database accuracy. The cross-database evidence remains the
  Phase 3/4 sets (where TAP beat CLIP-Adapter at ViT-B/16 and lost at ViT-L/14) and the
  Phase 5 replication failure.
- The corruption conditions are now a consumed confirmation set. Any further variant designed
  with knowledge of these results needs a fresh untouched set.
- RAF-DB remains unconsumed but locally unavailable (phase 6 gate).

## 7. Scope, boundaries, unfinished

- Read: manifests/fer2013_<condition>.csv (12), raw/fer2013_shift_sweep/<condition>/, the 12
  features/shift_sweep/<condition>.npz (validation only), the frozen source-domain arms of
  phases 2-5 and their source caches, plus the FER2013 test / FER2013 train source caches.
- Not read: the old 2250 holdout, Set A, Set B, the RAF-DB tree, the locked test set.
- No main*.tex file was modified; nothing was submitted, no editor was contacted, nothing was
  paid. No existing file was overwritten: all outputs of this phase are new files.
- Unfinished: the ViT-L/14 half is reported only if its extraction and evaluation completed
  (see section 3); nothing else in the pre-registration was left out.
