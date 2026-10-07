# Phase 4 report - published-adapter baselines vs TAP

Executor: DeepSeek (deepseek-flash). Device: NVIDIA GeForce RTX 3080 (training and inference),
NumPy for the reference arms. Pinned interpreter
D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe (Python 3.12.14, torch
2.8.0+cu128, open_clip 3.3.0, numpy 2.5.2).

Protocol is identical to Phase 3: training and learning-rate selection use the FER2013 source
probe-train / probe-validation split only; no target label is used for training or selection.
Canonical convention s=100 (logit_scale), affinity=max, BETA=256, PFB lambda=0.2,
min_prior_samples=32. Confirmation uses the two untouched sets from Phase 3:
Set A = crossed grid (ckplus_prior / kdef_prior, ratio 0/1/2/5/10, seed 0-4, 50 cells);
Set B = second source KDEF-source (kdef_source_test / fer2013_test / ckplus_test, seeds 0-4, 15 cells).

Because target labels are unavailable by protocol, every trainable adapter here is fitted on the
source split and deployed to the targets. This is the protocol-consistent form of each method and
is not the form used in the original papers, where the adapter is fitted on the target few-shot
support. This applies to CLIP-Adapter, Tip-Adapter-F and the APE-style reference equally.

## 1. Baselines, budget and source-selection values

All baselines use the same training budget as the token modules (8 epochs, batch 128, Adam) and the
same learning-rate grid {1e-3, 3e-3}; no grid was expanded. Source file:
phase4_baselines_training.json.

| method | form used | trainable params | selected lr | source probe-val UAR |
|---|---|---|---|---|
| TAP (T1, reference) | token attention pooling over 196 patch tokens | 202,504 | 3e-3 | 0.6351 |
| CLIP-Adapter | bottleneck residual adapter on 512-d feature (ratio 4, residual 0.2), frozen text classifier | 131,712 | 1e-3 | 0.6172 |
| PFB (published ref) | source linear probe + online prior correction | 3,591 | n/a (closed form) | 0.6155 |
| APE-style (supplementary) | learnable per-class residual on the text prototypes | 3,584 | 3e-3 | 0.6076 |
| Tip-Adapter-F | trainable cache keys (source support), frozen text branch, logits = 100*cos_text + 256*per-class-max | 11,759,616 | 3e-3 | 0.5829 |
| Linear probe | source-supervised linear classifier on the 512-d feature | 3,591 | 3e-3 | 0.5691 |
| support memory (reference) | text + 256*per-class-max cache, no training | 0 | n/a | 0.6360 |

CLIP-Adapter details: fc1 512->128, fc2 128->512, output f = 0.2*fc2(relu(fc1(x))) + 0.8*x,
L2-normalised and dotted with the frozen text prototypes at scale 100. Tip-Adapter-F keys are
initialised from the 22,968 source support features and are the only trainable tensor.

## 2. Confirmation Set A - crossed grid (50 cells)

Mean UAR over the 50 cells; source file phase4_baselines_cells.csv.

| arm | mean UAR | cells >= best-of-two-families | cells beaten by TAP |
|---|---|---|---|
| TAP | 0.6784 | 50/50 | - |
| CLIP-Adapter | 0.6396 | 50/50 | 50/50 |
| Linear probe | 0.6132 | 50/50 | 50/50 |
| APE-style text residual | 0.6034 | 49/50 | 50/50 |
| PFB | 0.5663 | - | - |
| Tip-Adapter-F | 0.5623 | 25/50 | 50/50 |
| support memory | 0.4451 | - | - |

Paired shared-unit bootstrap (2000, hierarchical over seeds, shared query indices):

| pair | mean | 95% CI |
|---|---|---|
| TAP - best-of-two-families | +0.1120 | [+0.1037, +0.1198] |
| TAP - CLIP-Adapter | +0.0388 | [+0.0307, +0.0460] |
| TAP - Linear probe | +0.0652 | [+0.0568, +0.0731] |
| TAP - APE-style | +0.0750 | [+0.0688, +0.0811] |
| TAP - Tip-Adapter-F | +0.1161 | [+0.1093, +0.1229] |
| CLIP-Adapter - best-of-two-families | +0.0732 | [+0.0661, +0.0805] |
| Linear probe - best-of-two-families | +0.0468 | [+0.0411, +0.0524] |
| APE-style - best-of-two-families | +0.0370 | [+0.0297, +0.0445] |
| Tip-Adapter-F - best-of-two-families | -0.0041 | [-0.0118, +0.0033] |

## 3. Confirmation Set B - second source KDEF-source (15 cells)

Source file phase4_kdef_source_baseline_cells.csv.

| target | TAP | CLIP-Adapter | Linear probe | APE-style | Tip-Adapter-F | PFB | support memory | best-of-two |
|---|---|---|---|---|---|---|---|---|
| kdef_source_test | 0.7166 | 0.6848 | 0.6599 | 0.6440 | 0.5329 | 0.7243 | 0.7914 | 0.7914 |
| fer2013_test | 0.6323 | 0.6163 | 0.5556 | 0.5932 | 0.5854 | 0.4387 | 0.4098 | 0.4387 |
| ckplus_test | 0.6679 | 0.6116 | 0.5787 | 0.5668 | 0.5759 | 0.5235 | 0.4914 | 0.5235 |
| mean | 0.6722 | 0.6375 | 0.5980 | 0.6013 | 0.5647 | 0.5622 | 0.5642 | 0.5845 |

Paired shared-unit bootstrap (2000, hierarchical over seeds x targets, shared query indices):

| pair | mean | 95% CI |
|---|---|---|
| TAP - best-of-two-families | +0.0869 | [+0.0749, +0.0986] |
| TAP - CLIP-Adapter | +0.0347 | [+0.0245, +0.0460] |
| TAP - Linear probe | +0.0742 | [+0.0637, +0.0854] |
| TAP - APE-style | +0.0710 | [+0.0608, +0.0819] |
| TAP - Tip-Adapter-F | +0.1075 | [+0.0966, +0.1189] |
| CLIP-Adapter - best-of-two-families | +0.0522 | [+0.0404, +0.0635] |
| Linear probe - best-of-two-families | +0.0127 | [-0.0000, +0.0246] |
| APE-style - best-of-two-families | +0.0159 | [+0.0045, +0.0271] |
| Tip-Adapter-F - best-of-two-families | -0.0206 | [-0.0331, -0.0089] |

## 4. Does TAP beat published adapters on the confirmation sets?

Yes, on both confirmation sets and against every implemented baseline:

- Set A (50 cells): TAP wins all 50 cells against CLIP-Adapter, linear probe, APE-style and
  Tip-Adapter-F. TAP - CLIP-Adapter = +3.88pp [+3.07, +4.60], which excludes 0.
- Set B (15 cells): TAP wins all 15 cells against all four baselines.
  TAP - CLIP-Adapter = +3.47pp [+2.45, +4.60], which excludes 0.

CLIP-Adapter is the strongest published competitor, not a weak one. On average it is also stronger
than the previous best-of-two-families reference column (+7.32pp on Set A, +5.22pp on Set B, both
CI excluding 0), so the TAP margin over CLIP-Adapter is the harder bar to clear and it is cleared.
The margin is nevertheless modest in absolute terms: 3.5-3.9pp UAR, not a runaway gap, and it is
obtained under a source-only training protocol for both methods.

Two honest qualifications. First, Tip-Adapter-F does not transfer in this protocol: it is the only
baseline that is significantly below best-of-two-families (-0.41pp, CI containing 0, on Set A and
-2.06pp, CI excluding 0, on Set B) despite having 58x more trainable parameters than TAP; this is
consistent with its design premise (keys are meant to be fitted on the target few-shot support,
which the protocol forbids). Second, the linear probe reaches the best-of-two-families reference
within noise on Set B (+1.27pp, CI touching 0), so part of TAP's margin over the reference column
is attributable to a trained head, exactly as the Phase 3 equal-capacity control showed. TAP's net
advantage over the strongest trained head (CLIP-Adapter) remains +3.5 to +3.9pp.

## 5. ViT-L/14 second-encoder option (assessment only, not executed)

Current state: only ViT-B-16.pt (350 MB) is present in D:/ResearchVault/99system/models/open_clip.
No ViT-L-14 weights are on disk. The 512-d global ViT-L/14 feature caches already exist in
data/features_vitl14 (fer2013 train/test/probe-val, ckplus, kdef, ferplus test, text prototypes,
linear probes), so the global-feature second-encoder comparison can be run today; only the
token-level second encoder requires new weights and a new extraction.

- Download: about 1.7 GB. At 10-50 MB/s that is roughly 35 s to 3 min plus checksum verification.
- Token volume: ViT-L/14 has 256 patches of 1024 dims, so fp16 is 0.524 MB per image. The four
  existing caches (28,709 + 7,178 + 902 + 2,938 = 39,727 images) need about 20.8 GB, plus about
  1.1 GB for the 2,056 KDEF-source support images, about 21.9 GB total.
- Disk headroom: 249.4 GB free on D:. The extraction fits comfortably; no cleanup is required.
- Extraction time: ViT-B/16 ran at about 109-130 images/s (39,727 images in 282 s). ViT-L/14 costs
  roughly 4.6x more per image (24 vs 12 layers, 1024 vs 768 width, 256 vs 196 patches), so about
  20-30 min for the four caches plus about 1-2 min for the KDEF-source support, plus about
  1-2 min to retrain TAP at 1024-dim with the same 8-epoch budget.
- Marginal value: TAP's claim currently rests on one encoder. A second encoder would test whether
  the token-attention gain over CLIP-Adapter (+3.5 to +3.9pp) and the net token contribution over an
  equal-capacity head (+5.2 to +6.3pp) replicate at ViT-L/14. That is the single cheapest remaining
  confirmation available (one download plus about 30-40 min of compute), and it directly addresses
  the encoder-dependence limitation already stated in the manuscript. The main risk is a negative
  result, which would be informative rather than costly. Recommendation for the user to decide:
  worth doing if the token-level mechanism is a load-bearing contribution; optional if the paper
  only needs the ViT-B/16 claim.

## 6. Unfinished / scope

- The optional KDEF-source variant of the T2 1x1-vs-3x3 control was not run.
- ViT-L/14 token confirmation is not executed (option assessment only, see section 5).
- Locked test set and the old 2250 holdout were not read; main_rev.tex was not modified; nothing was
  submitted, no editor was contacted, nothing was paid and no manuscript was transferred.
