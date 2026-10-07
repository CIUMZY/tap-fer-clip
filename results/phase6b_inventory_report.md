# Phase 6b step 1 - inventory of locally available confirmation material

Read-only audit. Machine-readable evidence: phase6b_inventory.json (keys, shapes, dtypes,
labels, ids) and phase6b_provenance.json (per-condition metadata, image folders, identity
relations). Nothing in the data tree was written.

## Available

### A. Graded corruption sweep of the FER2013 test set - 12 conditions, 7178 images each

| condition | images on disk | manifest rows | feature cache | provenance |
|---|---|---|---|---|
| blur_0p8 | 7178 | 7178 | features/shift_sweep/blur_0p8.npz | manifests/fer2013_blur_0p8.csv |
| blur_1p5 | 7178 | 7178 | features/shift_sweep/blur_1p5.npz | manifests/fer2013_blur_1p5.csv |
| blur_2p5 | 7178 | 7178 | features/shift_sweep/blur_2p5.npz | manifests/fer2013_blur_2p5.csv |
| jpeg_15 | 7178 | 7178 | features/shift_sweep/jpeg_15.npz | manifests/fer2013_jpeg_15.csv |
| jpeg_30 | 7178 | 7178 | features/shift_sweep/jpeg_30.npz | manifests/fer2013_jpeg_30.csv |
| jpeg_50 | 7178 | 7178 | features/shift_sweep/jpeg_50.npz | manifests/fer2013_jpeg_50.csv |
| lowlight_0p3 | 7178 | 7178 | features/shift_sweep/lowlight_0p3.npz | manifests/fer2013_lowlight_0p3.csv |
| lowlight_0p5 | 7178 | 7178 | features/shift_sweep/lowlight_0p5.npz | manifests/fer2013_lowlight_0p5.csv |
| lowlight_0p7 | 7178 | 7178 | features/shift_sweep/lowlight_0p7.npz | manifests/fer2013_lowlight_0p7.csv |
| noise_10 | 7178 | 7178 | features/shift_sweep/noise_10.npz | manifests/fer2013_noise_10.csv |
| noise_25 | 7178 | 7178 | features/shift_sweep/noise_25.npz | manifests/fer2013_noise_25.csv |
| noise_40 | 7178 | 7178 | features/shift_sweep/noise_40.npz | manifests/fer2013_noise_40.csv |

Provenance chain: make_corrupted_fer2013.py defines each transform (blur = GaussianBlur radius,
jpeg = JPEG quality, lowlight = Brightness factor, noise = additive Gaussian sigma with the
deterministic seed 1000+row_index) and writes raw/fer2013_shift_sweep/<condition>/, with a
per-condition manifest carrying path, label, domain and split. Every manifest has 7178 rows and
every condition has 7178 PNG files on disk, so the sweep is fully re-extractable. The existing
feature caches are ViT-B/16 [7178,2,512] unit-norm views (ids, labels, meta_json), built with
OpenCLIPEncoder (open_clip preprocess, encode_image, L2-normalise, two views = image + mirror).
Labels are identical to the clean FER2013 test cache in all 12 conditions.

Note on the four legacy files features/corruptions/{blur,jpeg,lowlight,noise}.npz: they carry the
same 7178 labels but a partially ambiguous provenance (two different generator scripts with
different settings, one of which converts through grayscale). blur.npz is array-identical to
shift_sweep/blur_1p5.npz and noise.npz to shift_sweep/noise_25.npz; jpeg.npz and lowlight.npz
match no sweep level. They are therefore excluded from the confirmation axis: the pre-registered
axis uses only the 12 graded conditions with a single unambiguous provenance chain.

### B. FER+ 1:1 / 5:1 (annotation shift) - available, ratio-feasible, not chosen

- features/ferplus_test.npz: 6722 images, labels bincount [626, 53, 165, 1789, 819, 862, 2408].
- ferplus_stats.json: {"total": 7178, "kept": 6722, "excluded_contempt": 50, "excluded_tie": 327, "excluded_invalid": 79, "class_counts": [626, 53, 165, 1789, 819, 862, 2408]}.
- Ratio feasibility with the phase-3 construction (base = smallest class count): base = 53, majority = 2408 -> 1:1 needs 53 (ok), 5:1 needs 265 (ok).
- The images are FER2013 test images, so their tokens and globals already exist in the clean
  caches - this axis would need no new extraction at all. Kept as the fallback axis.

### C. Other material examined

- features/ckplus48_all.npz: 981 CK+ 48x48 frames with labels all -1 in the feature cache (the
  manifest manifests/ckplus48_all.csv would be needed to assign labels), and CK+ was already a
  confirmation target in phases 3-5 (Set A ckplus_prior, Set B ckplus_test). Not chosen.
- features/kdef_test.npz (2938) and features/ckplus_test.npz (902): already used as confirmation
  targets in phases 3-5. Degrading them would create new conditions on already-consumed targets
  and would need full token re-extraction for both encoders. Not chosen.
- RAF-DB: unavailable (phase 6 gate failed; the only local copy is a quarantined re-upload).

## Not available / not usable

- No usable RAF-DB (see phase6_rafdb_report.md).
- features/corruptions/* (4 files): legacy defaults with partially ambiguous provenance; two are
  exact duplicates of sweep levels. Excluded, see above.
- features/ckplus48_all.npz: no labels in the cache.

## Feasibility verdict

The corruption-sweep axis (A) is available, traceable and re-extractable: it is the only candidate
with a complete chain from generator script to on-disk images to manifests to feature caches, and
the only one whose *inputs* were never touched by this project line. It is therefore the axis
pre-registered in phase6b_preregistration.md. Cost: about 26 GB + 45 GB of new token cache plus
0.9 GB of globals, about 8 min extraction at ViT-B/16 and 24 min at ViT-L/14, and about 10-15 min
of evaluation and bootstrap.
