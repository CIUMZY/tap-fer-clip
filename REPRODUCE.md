# Reproducing the analysis

Environment (every number in the paper was produced in it): python 3.12.14, torch 2.8.0+cu128,
numpy 2.5.2, pillow 12.3.0, pandas 3.0.5, scipy 1.18.1, open_clip_torch 3.3.0, one NVIDIA RTX 3080
(10 GB), driver 591.86. See requirements.txt. The scripts use absolute paths rooted at
D:/ResearchVault/...; to relocate, change the D / OUT constants at the top of each script.

## Data (not redistributed)

| corpus | role | note |
|---|---|---|
| FER2013 | source-domain training and the corruption axis | public benchmark |
| FER+ | labels for the same images | public benchmark |
| CK+ | crossed-grid target | public benchmark |
| KDEF | crossed-grid target and the second-source split | public benchmark |
| JAFFE | fresh confirmation axis | Zenodo record 14974867, archive sha256 6da27f5954f969c6f65d782911834dd66827ec3580e79b95073eb6cb93de5c3b |

Corrupted images are generated deterministically by code/make_corrupted_fer2013.py with a per-image
seed of 1000 + row index, so any condition can be rebuilt from its manifest alone.

## Pre-specifications (written before the corresponding run)

- results/phase6b_Pre-specification.md - the 12-condition corruption axis
- results/phase13_jaffe_preregistration.md and ..._amendment1.md - the fresh JAFFE axis, with the
  sha256 of both files recorded in phase13_prereg_freeze.json together with the freeze timestamps

## Order of execution

1. Extraction
   - phase2_extract_tokens.py (ViT-B/16), phase5_extract_tokens_vitl14_v2.py (ViT-L/14)
   - phase6b_extract.py --encoder {B16,L14} - corrupted caches for the 12-condition axis
   - phase9_mirror_train_extract.py --encoder L14 then phase9_build_aligned_cache.py --encoder L14
     (mirror view and the spatially aligned two-view mean)
   - phase13_build_jaffe_axis.py then phase13_extract_jaffe.py --encoder {B16,L14}
2. Single-run comparison (the study whose numbers the paper audits)
   - phase2_train_modules.py, phase4_baselines_train.py, phase5_train_vitl14_v2.py
   - phase3_attribution.py, phase4_baselines_confirm.py, phase5_confirm_vitl14_v3.py,
     phase3_kdef_source_confirm.py
3. Seed-controlled retraining (the paper's main analysis)
   - phase12_train_seeded.py -> phase12_eval_seeded.py -> phase12_report.py -> phase12_write_stats.py
4. Feature-convention sensitivity
   - phase9_aligned_corruption.py --encoder L14, and --encoder B16 --lr 0.003 --suffix _lr3e-3
   - phase14_aligned_l14_15seeds.py (training) then phase14b_eval_aligned.py (evaluation)
5. Fresh confirmation axis
   - phase13_eval_jaffe.py -> phase13_clean_fer_control.py -> phase13_report.py
6. Stability checks
   - phase15_leave_one_seed_out.py
7. Source-domain diagnostic behind Fig. 3
   - diagnostic_token_vs_global_run.py, diagnostic_token_vs_global_mlp.py,
     diagnostic_token_vs_global_report.py

## Determinism

From phase 12 onwards every arm is constructed inside torch.manual_seed(seed), so its initial
weights, its batch order and (in the nested condition) its learning rate all follow from the seed.
Rebuilding an arm from its seed is bit-identical; the flag is recorded in phase12_train_B16.json
under determinism_rebuild_identical.

The files named phase8_* are the superseded first pass of the retraining analysis: it seeded the batch
order but not the initial weights, so it did not reproduce its own token-arm checkpoints. Phase 12
supersedes it for every retraining distribution in the paper; the phase-8 calibration files are still
used, because those reproduce the *original* single-run checkpoints rather than retrained ones.

## Where each number comes from

| paper element | file |
|---|---|
| Table 1 calibration | phase8_calib_seed0.json, phase8_calib_corruption_seed0.json, phase8_calib_setB_seed0.json |
| Tables 2-5, 7 per-seed rows | phase12_train_{B16,L14}.json, phase12_eval.json |
| Table 7 summary, Tables 8-9 statistics | phase12_stats.json |
| Table 6 fresh axis and clean controls | phase13_eval_jaffe.json, phase13_clean_fer_control.json, phase13_summary.json |
| Table 10 convention sensitivity | phase14_aligned_l14_15seeds.json, phase9_aligned_corruption_B16_lr3e-3.json, phase12_stats.json |
| leave-one-seed-out paragraph | phase15_leave_one_out.json |
| Fig. 2 | figures/make_fig2.py reading phase12_eval.json and phase12_stats.json |
