# Diagnostic report - token pooling versus the global feature (source domain, both encoders)

Executor: DeepSeek (deepseek-flash). Device: NVIDIA GeForce RTX 3080. Interpreter
D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe (torch 2.8.0+cu128).

This is a **source-domain diagnostic**, not a new method and not a confirmation run. Phase 5
established that TAP's margins over CLIP-Adapter (+3.88pp / +3.47pp) and over an equal-capacity
global head (+6.25pp / +5.21pp) at ViT-B/16 do not replicate at ViT-L/14, where the signs
reverse, while TAP still beats both training-free families at both encoders. This report asks
*why*, using only the FER2013 source probe-train / probe-val split:

- (i) the ViT-L/14 global feature is already strong enough that token pooling adds nothing, or
- (ii) TAP's convex combination of patch tokens loses information because it has no direct path
  from the global feature.

Every arm trains and selects on the source split only (Adam, 8 epochs, batch 128, learning-rate
grid {1e-3, 3e-3}, selection by probe-val UAR, seeds 0/1/2). No target label is used anywhere.
Neither confirmation set (Set A crossed grid, Set B KDEF-source), the locked test set nor the old
2250 holdout was read.

## 1. Protocol, provenance and pipeline validation

Representations, frozen unless noted:

- **R1** global pooled feature, L2-normalised (512-d ViT-B/16, 768-d ViT-L/14) - the existing
  PFB / global-head input.
- **R2** uniform mean pooling of the patch tokens (196x768, 256x1024), raw; plus two robustness
  variants (mean of L2-normalised tokens, L2-normalised mean).
- **R3** the TAP head: softmax attention over tokens -> 1xd pooling -> linear head, architecture
  identical to Phase 2 T1, trained with the same budget. **R3refit** freezes that learned pooling
  and refits a fresh linear head.
- **R4a** frozen concat(R1, R2) + linear probe; **R4b** concat(R1, attention-pooled vector)
  trained jointly - two forms of the direct-path / residual test.
- **MLP extension** (marked, and added because the comparators TAP lost to in Phase 5 are
  nonlinear): the same representations probed with Linear(d,256) -> Tanh -> Linear(256,256) ->
  Tanh -> Linear(256,7). This is exactly the Phase 5 global-head form: 198,919 parameters on the
  512-d global input and 264,455 on the 768-d input.

Inputs are read through the existing caches in 128-256 row blocks: the ViT-L/14 .npy memory map,
and for ViT-B/16 a one-off assembly of the 57 train shards into
token_cache_vitb16/fer2013_train_tokens_fp16.npy (28709x196x768 fp16). The assembly was needed
because a random 128-row batch touches about 44 of the 57 shards, so a shard-granular reader would
copy about 6.8 GB per step. Twelve random rows of the assembled file were verified equal to their
source shard rows, and the frozen R2/R4a probe results computed from the assembled file are
bit-identical to those computed from the shards in an aborted earlier attempt. Nothing is
concatenated in host memory; the largest transient is one 0.3-0.5 GB token block.

Both encoders use the **identical** source split (verified element-wise: 22,968 train / 5,741 val,
same indices), which is what makes the two columns comparable.

Cross-checks against the stored artifacts:

- R1 linear at ViT-B/16, lr 3e-3, seed 0: **0.5691**, matching the stored Phase 4 linear probe to
  four decimals.
- The MLP extension on the frozen global feature reproduces the stored Phase 5 global head:
  ViT-B/16 0.6201 vs stored 0.6144 (lr 3e-3 seed mean), and
  ViT-L/14 0.6714 vs stored 0.6720 (lr 1e-3 seed mean).
- R3 at ViT-L/14 lands at 0.6720 (lr 1e-3, seed 0) and 0.6861 (seed 1), bracketing the stored
  Phase 5 TAP value 0.6754 from a different code path.

Two instrumentation bugs were found and fixed here; they are recorded because they explain the
aborted logs kept alongside the results:

1. The first version built each streaming model *before* seeding it, so its initial weights came
   from a leftover RNG state. The model is now constructed inside the seeded scope.
2. The streaming evaluator read rows in ascending order for sequential I/O and wrote predictions
   back in that order, while labels were taken in the requested order. Because the stored row order
   is class-grouped (only 6 class changes across 28,709 rows), this mispaired predictions with
   labels and gave a chance-level UAR (0.137-0.141) while the training loss fell to 0.88. The
   evaluator now scatters results back into the requested order. Only the streaming arms (R3, R4b,
   R3refit) were affected; the frozen-probe numbers R1/R2/R4a never were, and all numbers in this
   report come from the fixed code.

Checkpoints and pooled matrices from the buggy intermediate runs were moved to
discarded_buggy_run/ rather than deleted. Aborted logs are kept as
diagnostic_token_vs_global_{B16_aborted_shardreader,B16_aborted_unseeded_init,
B16_aborted_eval_order_bug}.log and diagnostic_token_vs_global_mlp_first_attempt_indexerror.log.

## 2. Results - probe-val UAR on the FER2013 source split

"selected" is the Phase 3/4/5 rule (best single run on probe-val); it is biased upward by seed
noise, so the fairer comparator is "seed mean at best lr" with its standard deviation and range.

### ViT-B/16 (ViT-B/16, 22968 train / 5741 val)

| representation | dim | selected UAR | selected +prior | seed mean at best lr | sd | seed range |
|---|---|---|---|---|---|---|
| R1 global (frozen) | 512 | 0.5692 | 0.5781 | 0.5664 | 0.0047 | 0.5610-0.5692 |
| R2 mean tokens (frozen) | 768 | 0.6200 | 0.6286 | 0.6000 | 0.0177 | 0.5860-0.6200 |
| R2 L2-normalised mean | 768 | 0.5461 | 0.5638 | 0.5441 | 0.0018 | 0.5428-0.5461 |
| R2 mean of L2-normalised tokens | 768 | 0.5360 | 0.5518 | 0.5341 | 0.0017 | 0.5328-0.5360 |
| R3 attention pooling (TAP head) | - | 0.6280 | 0.6424 | 0.6200 | 0.0116 | 0.6066-0.6280 |
| R3refit frozen pooling | 768 | 0.6289 | 0.6321 | 0.6137 | 0.0171 | 0.5952-0.6289 |
| R4a concat(global, mean tokens) | 1280 | 0.6283 | 0.6382 | 0.6077 | 0.0181 | 0.5947-0.6283 |
| R4a concat(global, normalised mean) | 1280 | 0.5931 | 0.6096 | 0.5903 | 0.0025 | 0.5885-0.5931 |
| R4a concat(global, mean of normalised) | 1280 | 0.5888 | 0.6086 | 0.5862 | 0.0033 | 0.5825-0.5888 |
| R4b concat(global, attention pool) | - | 0.6217 | 0.6392 | 0.6104 | 0.0132 | 0.5959-0.6217 |
| R1 global + MLP head | 512 | 0.6439 | 0.6443 | 0.6201 | 0.0146 | 0.6080-0.6363 |
| R2 mean tokens + MLP head | 768 | 0.6424 | 0.6531 | 0.6281 | 0.0130 | 0.6169-0.6424 |
| R3 attention pool + MLP head | 768 | 0.6344 | 0.6447 | 0.6315 | 0.0010 | 0.6306-0.6326 |
| R4a concat + MLP head | 1280 | 0.6414 | 0.6494 | 0.6278 | 0.0123 | 0.6174-0.6414 |

Per-learning-rate seed means (every run is listed in
diagnostic_token_vs_global_probes_*.csv and diagnostic_token_vs_global_mlp_probes_*.csv):

| representation | lr 1e-3 | lr 3e-3 | selected lr | selected seed | params |
|---|---|---|---|---|---|
| R1 global (frozen) | 0.5209 | 0.5664 | 0.003 | 2 | 3591 |
| R2 mean tokens (frozen) | 0.5875 | 0.6000 | 0.003 | 1 | 5383 |
| R2 L2-normalised mean | 0.5062 | 0.5441 | 0.003 | 2 | 5383 |
| R2 mean of L2-normalised tokens | 0.4909 | 0.5341 | 0.003 | 2 | 5383 |
| R3 attention pooling (TAP head) | 0.6200 | 0.5989 | 0.001 | 1 | 202504 |
| R3refit frozen pooling | 0.6079 | 0.6137 | 0.003 | 1 | 5383 |
| R4a concat(global, mean tokens) | 0.5966 | 0.6077 | 0.003 | 1 | 8967 |
| R4a concat(global, normalised mean) | 0.5483 | 0.5903 | 0.003 | 0 | 8967 |
| R4a concat(global, mean of normalised) | 0.5433 | 0.5862 | 0.003 | 0 | 8967 |
| R4b concat(global, attention pool) | 0.6104 | 0.5945 | 0.001 | 2 | 206088 |
| R1 global + MLP head | 0.6188 | 0.6201 | 0.001 | 1 | 198919 |
| R2 mean tokens + MLP head | 0.6183 | 0.6281 | 0.003 | 1 | 264455 |
| R3 attention pool + MLP head | 0.6315 | 0.6297 | 0.003 | 2 | 264455 |
| R4a concat + MLP head | 0.6278 | 0.6267 | 0.001 | 1 | 395527 |

### ViT-L/14 (ViT-L/14, 22968 train / 5741 val)

| representation | dim | selected UAR | selected +prior | seed mean at best lr | sd | seed range |
|---|---|---|---|---|---|---|
| R1 global (frozen) | 768 | 0.6227 | 0.6309 | 0.6172 | 0.0076 | 0.6085-0.6227 |
| R2 mean tokens (frozen) | 1024 | 0.6633 | 0.6687 | 0.6513 | 0.0107 | 0.6431-0.6633 |
| R2 L2-normalised mean | 1024 | 0.5884 | 0.6177 | 0.5854 | 0.0029 | 0.5826-0.5884 |
| R2 mean of L2-normalised tokens | 1024 | 0.5732 | 0.5895 | 0.5714 | 0.0019 | 0.5695-0.5732 |
| R3 attention pooling (TAP head) | - | 0.6861 | 0.6899 | 0.6724 | 0.0134 | 0.6592-0.6861 |
| R3refit frozen pooling | 1024 | 0.6719 | 0.6800 | 0.6665 | 0.0049 | 0.6623-0.6719 |
| R4a concat(global, mean tokens) | 1792 | 0.6656 | 0.6736 | 0.6562 | 0.0082 | 0.6507-0.6656 |
| R4a concat(global, normalised mean) | 1792 | 0.6417 | 0.6623 | 0.6380 | 0.0037 | 0.6344-0.6417 |
| R4a concat(global, mean of normalised) | 1792 | 0.6391 | 0.6561 | 0.6376 | 0.0018 | 0.6356-0.6391 |
| R4b concat(global, attention pool) | - | 0.6895 | 0.6910 | 0.6678 | 0.0188 | 0.6556-0.6895 |
| R1 global + MLP head | 768 | 0.6903 | 0.6892 | 0.6714 | 0.0208 | 0.6492-0.6903 |
| R2 mean tokens + MLP head | 1024 | 0.6751 | 0.6791 | 0.6656 | 0.0114 | 0.6530-0.6751 |
| R3 attention pool + MLP head | 1024 | 0.6686 | 0.6725 | 0.6639 | 0.0046 | 0.6595-0.6686 |
| R4a concat + MLP head | 1792 | 0.6872 | 0.6912 | 0.6668 | 0.0126 | 0.6547-0.6799 |

Per-learning-rate seed means (every run is listed in
diagnostic_token_vs_global_probes_*.csv and diagnostic_token_vs_global_mlp_probes_*.csv):

| representation | lr 1e-3 | lr 3e-3 | selected lr | selected seed | params |
|---|---|---|---|---|---|
| R1 global (frozen) | 0.5506 | 0.6172 | 0.003 | 2 | 5383 |
| R2 mean tokens (frozen) | 0.6358 | 0.6513 | 0.003 | 1 | 7175 |
| R2 L2-normalised mean | 0.5480 | 0.5854 | 0.003 | 0 | 7175 |
| R2 mean of L2-normalised tokens | 0.5333 | 0.5714 | 0.003 | 2 | 7175 |
| R3 attention pooling (TAP head) | 0.6724 | 0.6542 | 0.001 | 1 | 269832 |
| R3refit frozen pooling | 0.6665 | 0.6608 | 0.001 | 0 | 7175 |
| R4a concat(global, mean tokens) | 0.6425 | 0.6562 | 0.003 | 1 | 12551 |
| R4a concat(global, normalised mean) | 0.5898 | 0.6380 | 0.003 | 0 | 12551 |
| R4a concat(global, mean of normalised) | 0.5827 | 0.6376 | 0.003 | 0 | 12551 |
| R4b concat(global, attention pool) | 0.6678 | 0.6523 | 0.001 | 1 | 275208 |
| R1 global + MLP head | 0.6714 | 0.6610 | 0.001 | 1 | 264455 |
| R2 mean tokens + MLP head | 0.6656 | 0.6432 | 0.001 | 0 | 329991 |
| R3 attention pool + MLP head | 0.6639 | 0.6570 | 0.001 | 2 | 329991 |
| R4a concat + MLP head | 0.6668 | 0.6617 | 0.003 | 1 | 526599 |

## 3. Answers to the four questions

### Q1. At ViT-L/14, is R2 (mean-pooled tokens) already not worse than R1 (global)?

Yes under a linear probe, by the same margin as at ViT-B/16:

- ViT-L/14: R2 0.6513 vs R1 0.6172 = +3.41pp UAR (prior-corrected: 0.6602 vs 0.6296).
- ViT-B/16: R2 0.6000 vs R1 0.5664 = +3.36pp UAR.

Both frozen representations have small seed spread (sd 0.0076 and 0.0107 for R1 / R2 at ViT-L/14), so this gap is well
outside noise. **The token field is not information-poor at ViT-L/14 and token pooling does add
information a linear readout of the global feature does not recover.** Hypothesis (i) is therefore
not supported in the form "mean-pooled tokens cannot beat the global feature at L/14".

The picture changes once the readout is nonlinear, which is the regime the Phase 5 comparators
live in. With the capacity-matched MLP head at ViT-L/14, R1 is 0.6714 and R2 is 0.6656 - the global feature is now ahead by +0.58pp, whereas at ViT-B/16 the token field stays ahead by +0.80pp. So at ViT-L/14 the global feature carries
information comparable to or better than the token field once the readout can exploit it
nonlinearly, and at ViT-B/16 it does not catch up. That is the specific, much weaker form of (i)
the data support.

### Q2. Ordering at ViT-B/16 versus ViT-L/14 - reversed or not?

| comparison | ViT-B/16 | ViT-L/14 | reversed? |
|---|---|---|---|
| R2 - R1 (linear) | +3.36pp | +3.41pp | no: same sign and same size |
| R3 - R2 (attention vs mean pooling) | +2.00pp | +2.12pp | no |
| R4a - R2 (does the global feature add on top of tokens?) | +0.77pp | +0.49pp | no: both about zero |
| R4b - R3 (does a direct global path help the learned pooling?) | -0.96pp | -0.46pp | no: both at or below zero |
| R2_MLP - R1_MLP (nonlinear head) | +0.80pp | -0.58pp | **yes: the sign flips** |

Only the nonlinear-head comparison reverses. Everything else - tokens beating the linearly-read
global feature, attention pooling beating mean pooling by a small constant, the global feature
adding nothing on top of the tokens, and the direct path adding nothing - behaves the same way at
both encoders.

One caveat on the attention-versus-mean row: that gap is learning-rate dependent, not a constant.
At lr 1e-3 the attention head beats raw mean pooling by +3.25pp at ViT-B/16 and +3.66pp at ViT-L/14, but at lr 3e-3 the two poolings are indistinguishable (-0.11pp and +0.29pp). The headline row above compares each arm at its own selected lr; the same-lr view is the
honest range for the attention claim.

### Q3. What do the residual variants (R4) do?

- **R4a** (frozen concat of the global feature with mean tokens) is above R1 by construction, but
  only +0.77pp above R2 at ViT-B/16 and +0.49pp at ViT-L/14 - inside one seed sd. The global feature
  contributes essentially nothing once the token field is present, at either encoder.
- **R4b** (the global feature given directly to the classifier alongside the learned attention
  pooling) is -0.96pp vs R3 at ViT-B/16 and -0.46pp at ViT-L/14: the direct path does not help, and is if
  anything slightly negative.

So R4 does not beat R1 in the sense that matters (the part of R4 carrying the gain is the token
part, not the global part), and R4 does not beat R3. The direct path is not the missing
ingredient.

### Q4. Which explanation is supported, which is excluded

- **(ii) - "the convex token combination loses information because it has no direct path from the
  global feature" - is excluded.** Two independent forms of the direct path (a frozen concat, and
  a jointly trained concat with the learned pooling) change the result by at most +0.8pp over the
  token-only arm and are negative at ViT-L/14. If the missing residual were the cause, the
  ViT-L/14 gap should close when the path is added; it does not.
- **(i) is supported only in its narrow form**: at ViT-L/14 the global feature is as usable as the
  token field once the readout is nonlinear (R1_MLP 0.6714 vs R2_MLP 0.6656), and the nonlinear readout is exactly what the Phase 5 comparators
  (equal-capacity global head, CLIP-Adapter) use. It is not supported in the form "the token field
  carries no usable information at ViT-L/14": under a linear probe the tokens are still +3.41pp ahead, the same margin as at ViT-B/16.
- **The direct driver of the Phase 5 reversal is the global-feature route under a nonlinear
  readout, not the token route.** Across encoder scale the MLP head on the frozen global feature
  moves 0.6201 -> 0.6714, the token route moves 0.6200 -> 0.6724, and the two routes are essentially tied at
  both encoders once the heads are matched. The stored ViT-B/16 advantage of TAP over the
  equal-capacity global head (+2.1pp in Phases 3/5) is not reproduced by this independent
  implementation (0.6200 vs 0.6201), and the token
  route edge over a nonlinear global head is only -0.01pp at ViT-B/16 and +0.10pp at ViT-L/14.

A third factor is visible and matters for the paper: **seed and initialisation noise in the trained
token arms is of the same order as the effect being claimed.** Across seeds 0/1/2 the R3 arm spans
  0.6592-0.6861 at ViT-L/14 (sd 0.0134) and 0.6066-0.6280 at ViT-B/16 (sd 0.0116), while the frozen representations R1/R2/R4a are
stable to about 0.2-1.8pp. The Phase 5 ViT-B/16 margins (+3.88pp over CLIP-Adapter, +6.25pp over
the global head) are larger than this noise, but the matched-head difference between the token
route and the global route is not.

## 4. What this implies for the paper

1. The load-bearing claim cannot be "token attention pooling beats a global-feature head". Under
   matched heads the two routes are essentially tied at both encoders in this diagnostic, and the
   attention mechanism itself contributes only a small, roughly constant +2.00pp / +2.12pp over plain uniform mean pooling.
2. The defensible claims are narrower and better supported: (a) the token field is a better
   linear read-out target than the global feature at both encoders (+3.36pp / +3.41pp); (b) a direct global path does not repair or improve
   token pooling; (c) attention pooling is a small, consistent gain over mean pooling.
3. If the encoder dependence itself is to be reported as a result, it needs the same treatment as
   any other claim: a pre-specified comparison, several seeds, and a stated noise floor. This
   diagnostic supplies the noise floor.

## 5. Discipline for any follow-up

This diagnostic consumed no confirmation data: it ran entirely on the source probe-train /
probe-val split. But it was designed *after* seeing the Phase 5 result, so **any new method
variant derived from it must be confirmed on a fresh, untouched set** - the two existing
confirmation sets (Set A crossed grid, Set B KDEF-source) have now been evaluated at two encoders
and cannot support a confirmatory claim about a variant chosen with this information. Candidate
fresh sets, in the order I would rank them:

1. **RAF-DB** as a third cross-database target (different collection and label noise than
   CK+ / KDEF / FER2013), with the same prior/support protocol. Strongest generalisation test.
2. **Corruption sweep** on the existing source-trained arms (blur, noise, JPEG levels): tests
   whether the token route is more robust than the global route rather than only more accurate -
   a different axis from the one that failed, and cheap to run.
3. **FER+ 1:1 and 5:1 imbalance** as a prior-shift test, the regime where prior correction and
   support memory matter most and where the two routes have not been compared.

Design requirements for whichever is used: fixed before running; all arms (token route, global
route with matched heads, CLIP-Adapter, training-free families) evaluated on it; at least three
seeds with the spread reported; and no arm or hyper-parameter chosen on the new set.

## 6. Scope, boundaries, unfinished

- Source domain only: the locked test set, the old 2250 holdout, Set A and Set B were not read.
- No main*.tex file was modified; nothing was submitted, no editor was contacted, nothing was paid.
- Delivered: this report; diagnostic_token_vs_global_{B16,L14}.json;
  diagnostic_token_vs_global_mlp_{B16,L14}.json; diagnostic_token_vs_global_probes_{B16,L14}.csv
  (every run); diagnostic_token_vs_global_mlp_probes_{B16,L14}.csv;
  diagnostic_token_vs_global_recall_{B16,L14}.csv (per-class recall, raw and prior-corrected, for
  the selected configuration of each representation); diagnostic_token_vs_global_summary.{json,csv}
  (merged cross-encoder table); the run scripts, the report generator and logs with START/EXIT.
- Derived artifacts created: token_cache_vitb16/fer2013_train_tokens_fp16.npy (the assembled
  ViT-B/16 train-token memory map, verified against the shards) and the pooled-token val matrices
  saved by the run script. The original ViT-B/16 token cache was not modified.
- The MLP extension is an addition to the requested R1-R4 linear-probe design, marked as such,
  included because the task logic for Q3 cannot be evaluated without knowing how the same
  representations behave under the nonlinear head class the Phase 5 comparators use.
- Not run: an R4 variant combining the direct global path with a *nonlinear* head (a
  CLIP-Adapter-style route with token pooling added). Both R4 forms tested here put a linear head
  on a concatenation, so they bound the linear-head version of hypothesis (ii), not every
  nonlinear variant of it; the MLP rows partially cover this for the frozen representations.
- Not run: corruption sweep, RAF-DB, FER+ imbalance variants, and any hyper-parameter search
  beyond the fixed {1e-3, 3e-3} grid.
