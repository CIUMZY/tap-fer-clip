# Phase 5 report - ViT-L/14 replication of the token-attention (TAP) result

Executor: DeepSeek (deepseek-flash). Device: NVIDIA GeForce RTX 3080 (10.7 GB) for training,
token inference and the global arms; NumPy for the reference columns and the bootstrap. Pinned
interpreter D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe (torch 2.8.0+cu128,
open_clip 3.3.0). Protocol constants are the canonical ones: logit scale s=100, affinity=max,
BETA=256, PFB lambda=0.2, min_prior_samples=32, prior_clip=1.0.

This phase asks one question: do the two ViT-B/16 margin quantities - TAP minus CLIP-Adapter
(+3.88pp on Set A, +3.47pp on Set B) and TAP minus an equal-capacity global head (+6.25pp /
+5.21pp) - reproduce at ViT-L/14? Training and learning-rate selection use the FER2013 source
probe-train / probe-validation split only; no target label enters training or selection. The two
confirmation sets are the Phase 3/4 sets, unchanged: Set A = crossed grid (ckplus_prior /
kdef_prior x ratio 0/1/2/5/10 x seed 0-4, 50 cells); Set B = second source KDEF-source
(kdef_source_test / fer2013_test / ckplus_test, seed 0-4, 15 cells).

## 1. Provenance and preflight

- Token input: memory-mapped fp16 patch tokens in D:/ResearchVault/99system/data/rb-tta-fer-fg2027/
  token_cache_vitl14 (256x1024 per image, 21.9 GB total, five datasets). No array is concatenated;
  columns are read in row blocks through the .npy memory map, which keeps the 16.2 GB host RAM safe.
- Preflight audit phase5_preflight_vitl14.py: 57 checks, all pass. It verifies token/meta/global id and
  label agreement for all five caches, the probe split size (22968 train / 5741 val), the PFB weight
  shape (7x768), the text-prototype shape, that every KDEF-source support/val/test id resolves
  inside the kdef_test ViT-L cache, and CUDA availability.
- Training command: python phase5_train_vitl14_v2.py (exit code 0, 496.7 s).
- Confirmation command: python phase5_confirm_vitl14_v3.py (exit code 0, 84.3 s).

Two provenance notes. First, phase5_confirm_vitl14_v3.py differs from the v2 script in one
line-group: the KDEF-source held-out diagnostic was read from the 512-d ViT-B views shipped in
data/features_kdef_source, but the probe retrained at ViT-L expects the 768-d rows of kdef_test that
the same ids map to; v2 therefore raised a mat1/mat2 shape error after Set A had been computed.
v3 maps the validation ids into the same 768-d rows the support uses. This affects only the reported
held-out source UAR of that probe (0.7256); no prediction, cell value, reference column or bootstrap
draw depends on it, and the v2 script is left in place unmodified.
Second, at 19:49:41 - before this session started training at 19:51:04 - a second process tree had
already launched the same phase5_train_vitl14_v2.py against the same output paths (observed via the
process command lines). That duplicate was stopped at about 19:57, after it had completed only its
TAP lr 1e-3 checkpoint; the superseded run of 19:38 is preserved as
phase5_tap_vitl14_lr0.001.interrupted_20261006T1938.pt. All artifacts below come from the run
described in this section, whose log files record START/EXIT and whose exit codes are 0.

## 2. Source-only selection at ViT-L/14

Same budget as every earlier arm: 8 epochs, batch 128, Adam, learning-rate grid {1e-3, 3e-3},
selection by probe-validation UAR on the FER2013 source split. Source file
phase5_training_vitl14.json.

| arm | form at ViT-L/14 | trainable params | probe-val lr 1e-3 | probe-val lr 3e-3 | selected lr | selected probe-val UAR | ViT-B/16 selected UAR |
|---|---|---|---|---|---|---|---|
| TAP | attention pooling over 256 patch tokens (dim 1024) | 269,832 | 0.6754 | 0.6510 | 0.001 | 0.6754 | 0.6351 |
| GHead | global-head MLP on the 768-d feature, width 260 (capacity-matched to TAP) | 269,627 | 0.6720 | 0.6543 | 0.001 | 0.6720 | 0.6144 |
| CLIPAdapter | bottleneck residual adapter on the 768-d feature (ratio 4, residual 0.2) | 295,872 | 0.6799 | 0.6578 | 0.001 | 0.6799 | 0.6172 |

Capacity match: TAP has 269,832 trainable parameters; the global head width was solved for that budget
and lands at 269,627 (h=260), an absolute difference of 205 parameters or 0.0760%.
CLIP-Adapter keeps the Phase 4 form (bottleneck ratio 4, residual weight 0.2) scaled to 768 dims,
so it carries 295,872 parameters - slightly more than the other two, exactly as at ViT-B/16.

The source ranking already differs from ViT-B/16. There, TAP (0.6351) > CLIP-Adapter (0.6172) >
global head (0.6144) on probe-val. Here CLIP-Adapter (0.6799) > TAP (0.6754) > global head (0.6720).
Training the same three arms with the same budget on the same source split therefore gives no
source-side reason to prefer TAP at this encoder, and the selection rule picks CLIP-Adapter.

## 3. Confirmation Set A - crossed grid (50 cells)

Mean UAR over the 50 cells. Source file phase5_vitl14_cells.csv; reference columns are the
training-free families (PFB and support memory) plus the two trained comparators.
ViT-B/16 columns are recomputed here from phase4_baselines_cells.csv (TAP, CLIP-Adapter, PFB,
memory, best-of-two) and phase3_attribution_cells.csv (equal-capacity head), not copied by hand.

| arm | ViT-L/14 mean UAR | cells beaten by TAP | ViT-B/16 mean UAR |
|---|---|---|---|
| TAP | 0.7570 | n/a | 0.6784 |
| GHead | 0.7681 | 25/50 | 0.6158 |
| CLIPAdapter | 0.7767 | 25/50 | 0.6396 |
| pfb | 0.6639 | - | 0.5663 |
| memory | 0.6351 | - | 0.4451 |
| best_family | 0.6639 | - | 0.5663 |
| LinearProbe (ViT-B/16 phase 4 only) | - | - | 0.6132 |
| APE-style (ViT-B/16 phase 4 only) | - | - | 0.6034 |
| Tip-Adapter-F (ViT-B/16 phase 4 only) | - | - | 0.5623 |

Paired shared-unit bootstrap (2000 draws, hierarchical over seeds, shared query indices):

| pair | mean | 95% CI |
|---|---|---|
| TAP - CLIPAdapter | -1.96 | [-2.59, -1.31] |
| TAP - GHead | -1.10 | [-1.82, -0.38] |
| TAP - best_family | +9.03 | [+8.38, +9.66] |
| CLIPAdapter - best_family | +10.99 | [+10.45, +11.52] |
| GHead - best_family | +10.13 | [+9.55, +10.73] |

Breakdown by target experiment (25 cells each):

| experiment | TAP | GHead | CLIP-Adapter | best-of-two-families | TAP cells > CLIP-Adapter |
|---|---|---|---|---|---|
| ckplus_prior | 0.7334 | 0.8269 | 0.8070 | 0.7043 | 0/25 |
| kdef_prior | 0.7807 | 0.7092 | 0.7463 | 0.6235 | 25/25 |

Breakdown by majority ratio (10 cells each, both experiments pooled):

| ratio | n | TAP | GHead | CLIP-Adapter | best-of-two-families |
|---|---|---|---|---|---|
| 0 | 10 | 0.7557 | 0.7712 | 0.7801 | 0.6574 |
| 1 | 10 | 0.7589 | 0.7666 | 0.7760 | 0.6679 |
| 2 | 10 | 0.7569 | 0.7681 | 0.7737 | 0.6669 |
| 5 | 10 | 0.7567 | 0.7661 | 0.7763 | 0.6649 |
| 10 | 10 | 0.7569 | 0.7684 | 0.7772 | 0.6624 |

## 4. Confirmation Set B - second source KDEF-source (15 cells)

Support and prior are built from the KDEF-source training split, never from the targets.
Source file phase5_vitl14_kdef_source_cells.csv.
ViT-B/16 columns: phase4_kdef_source_baseline_cells.csv / phase4_baselines.json for the trained
arms and phase3_kdef_source.json per_target for PFB, memory, best-of-two and the equal-capacity head.

| arm | ViT-L/14 mean UAR | cells beaten by TAP | ViT-B/16 mean UAR |
|---|---|---|---|
| TAP | 0.7405 | n/a | 0.6722 |
| GHead | 0.7412 | 10/15 | 0.6202 |
| CLIPAdapter | 0.7543 | 5/15 | 0.6375 |
| pfb | 0.6677 | - | 0.5622 |
| memory | 0.6391 | - | 0.5642 |
| best_family | 0.6777 | - | 0.5845 |

Per target (5 seeds each):

| target | n | TAP | GHead | CLIP-Adapter | best-of-two-families | TAP - best-of-two |
|---|---|---|---|---|---|---|
| ckplus_test | 902 | 0.7305 | 0.8333 | 0.8140 | 0.6454 | +8.51 |
| fer2013_test | 7178 | 0.6836 | 0.6828 | 0.6916 | 0.5191 | +16.45 |
| kdef_source_test | 441 | 0.8073 | 0.7075 | 0.7574 | 0.8685 | -6.12 |

Paired shared-unit bootstrap (2000 draws, hierarchical over seeds x targets):

| pair | mean | 95% CI |
|---|---|---|
| TAP - CLIPAdapter | -1.37 | [-2.38, -0.39] |
| TAP - GHead | -0.07 | [-1.08, +0.92] |
| TAP - best_family | +6.18 | [+5.09, +7.26] |
| CLIPAdapter - best_family | +7.56 | [+6.63, +8.51] |
| GHead - best_family | +6.25 | [+5.25, +7.28] |

## 5. Does the ViT-B/16 result replicate at ViT-L/14? No.

| quantity | ViT-B/16 | ViT-L/14 | replicated? |
|---|---|---|---|
| TAP - CLIP-Adapter, Set A | +3.88pp [+3.07, +4.60] | -1.96pp [-2.59, -1.31] | no, sign reversed |
| TAP - CLIP-Adapter, Set B | +3.47pp [+2.45, +4.60] | -1.37pp [-2.38, -0.39] | no, sign reversed |
| TAP - equal-capacity head, Set A | +6.25pp [+5.61, +6.87] | -1.10pp [-1.82, -0.38] | no, sign reversed |
| TAP - equal-capacity head, Set B | +5.21pp [+4.18, +6.28] | -0.07pp [-1.08, +0.92] | no (wash) |

Both ViT-B/16 quantities fail to replicate. The sign of TAP minus CLIP-Adapter reverses on both
sets and both intervals exclude zero against TAP (-1.96pp on Set A, -1.37pp on Set B). TAP minus
the equal-capacity global head is -1.10pp on Set A with an interval excluding zero, and -0.07pp
on Set B with an interval that straddles zero, i.e. the two trained heads are indistinguishable
there. TAP still beats both training-free families by a wide margin on Set A (+9.03pp over
best-of-two-families, CI excluding zero) and on Set B (+6.18pp), so the token branch is not
broken at ViT-L/14 - it simply stops being better than the strongest alternative trained head.

Cell-level counts agree with the averages. TAP wins 25 of 50 Set A cells against each comparator,
and the split is exactly by experiment: it wins the 25 kdef_prior cells and loses all 25
ckplus_prior cells, against both the global head and CLIP-Adapter. On Set B it wins 10 of 15 against
the global head and only 5 of 15 against CLIP-Adapter.

## 6. Reading of the result

The ViT-L/14 encoder changes which method wins. Every arm improves with the larger encoder, but
the improvements are very uneven: the training-free best-of-two-families reference rises from
0.5663 to 0.6639 on Set A (+9.8pp), TAP from 0.6784 to 0.7570 (+7.9pp), and CLIP-Adapter from
0.6396 to 0.7767 (+13.7pp). The 768-d global representation plus a residual bottleneck adapter
therefore gains more from scale than the token-attention pooling head does. That is the opposite
of the ViT-B/16 ordering, and it is consistent with the source-side selection values: on the
FER2013 probe-val split, CLIP-Adapter already ranked first at ViT-L/14 (0.6799 vs 0.6754).

Three candidate explanations are worth separating, and only the first is currently supported by
this phase. (i) At ViT-B/16 the 512-d global feature is relatively weak, so a token-level branch
supplies information the global branch lacks; at ViT-L/14 the trained global pooling of a 24-layer
model already carries most of the class-relevant signal, and re-learning pooling over 256 tokens
from 22,968 source images with 8 epochs adds little. (ii) The TAP head is a softmax-weighted mean
over patch tokens, so the pooled vector is a convex combination of token features; a residual
adapter keeps the original feature in the path, which is a strong prior for cross-database
transfer. (iii) TAP may be underfitted at 1024 dims: 8 epochs is enough at ViT-B/16 but the
ViT-L/14 token statistics (mean norm about 30.8) may need more steps. This phase cannot separate
(i) from (ii) or (iii); it can only say the ViT-B/16 conclusion is encoder-specific.

Consequence for the manuscript. The claim that token attention pooling beats an equal-capacity
global head by +5 to +6pp currently holds at one encoder and fails at the next one, which is
exactly the check a reviewer would demand before accepting a mechanism-level claim. The ViT-B/16
numbers are not wrong, but they can no longer be presented as a general property of the method.
Recommended handling, in order of preference:

1. Report the encoder dependence as a result rather than hiding it. State the ViT-B/16 gains and
   the ViT-L/14 null side by side, and downgrade the mechanism claim to "at ViT-B/16". The
   source-only selection discipline makes this defensible: selection never saw a target label, and
   the source ranking already anticipated the outcome. This costs the general claim but keeps the
   paper honest and adds a reproducibility result that most submissions lack.
2. Before writing anything else, spend one cheap diagnostic - linear probes on the frozen global
   feature versus frozen mean-pooled and attention-pooled token features, at both encoders, on the
   source split only. If the pooled tokens carry as much class information as the global feature at
   ViT-L/14, explanation (i) is supported and the story becomes "token pooling helps when the
   global feature is weak". If they do not, the TAP pooling itself is the problem and (ii)/(iii)
   move to the front.
3. Only if step 2 points at the TAP head: test a residual TAP (global feature plus a token-attention
   residual) and a longer schedule. These must be treated as exploratory, not confirmatory: the
   two confirmation sets have now been evaluated at ViT-L/14, so any variant designed after
   seeing these numbers needs a fresh untouched set to support a confirmatory claim.

## 7. Scope, boundaries and unfinished items

- Locked test set: not read. Old 2250 holdout: not read. No target label was used for training or
  for learning-rate selection at any point of this phase.
- No main*.tex file was modified; nothing was submitted, no editor was contacted, nothing was paid,
  no manuscript was transferred.
- The Phase 4 published-adapter baselines (APE-style, Tip-Adapter-F, the 512-d linear probe) were
  not retrained at ViT-L/14. The required reference columns for this phase are the four that appear
  in the tables above: best-of-two-families, equal-capacity global head, CLIP-Adapter and TAP.
- The KDEF-source linear probe used for the Set B PFB column was retrained in the 768-d ViT-L
  space (50 epochs, class-weighted cross-entropy, seed 0) and saved as
  phase5_linear_probe_kdef_source_vitl14.npz; its held-out source UAR is 0.7256. The ViT-B/16 Set B
  PFB column used the earlier 512-d probe, so the Set B PFB reference is not identical across
  encoders by construction.
- Set A ratio 0 is repeated across the five seeds by design (the full target is used), so its five
  cells are identical; this mirrors Phase 3/4 exactly and keeps the cell count at 50.
- The confirmation sets have now been evaluated at two encoders. Any further method variant
  designed after this negative result needs a fresh untouched confirmation set.
- A duplicate launch of the training script by another process tree was detected and stopped
  (section 1); the checkpoint of the superseded 19:38 run is retained under an .interrupted_ name.
