# Statistical robustness of the six training-seed contrasts

Artefact for the statistical-robustness subsection of the TVC submission (PAPER-TVC-TAP-2026-10-06). Every number below is read from the stored Phase-8 JSON records listed under each table; no model was retrained and no evaluation was re-run.

## What is estimated, and in what units

The statistical unit is the **training seed**. Each of the six core contrasts is a paired difference between two source-domain adaptation arms fitted on the same source split (TAP minus CLIP-Adapter), and the per-seed value is that difference averaged over the axis's own evaluation units: the 50 crossed class-prior cells of Set A, the three second-source targets of Set B, and the 12 pre-specified corruption conditions. Two intervals are reported for every contrast: an ordinary 95% Student-t interval on the n per-seed differences, and a seed-level percentile bootstrap (B = 10,000, fixed <code>numpy.random.default_rng(0)</code>) that resamples the seeds with replacement. Both intervals describe between-training-seed variability only, conditional on the frozen evaluation units and the frozen query sets, so they exclude uncertainty about which cells, targets or conditions were chosen and about query sampling inside them. Exact two-sided sign tests (zeros dropped, p capped at 1.0) are reported alongside; their six p-values form the family corrected in the next section.

## Table 1. Six core contrasts, seed-level statistics (TAP minus CLIP-Adapter)

| # | Axis | Encoder | n seeds | Mean (pp) | SD (pp) | 95% t-CI (pp) | Seeds positive | Sign-test p | Seed bootstrap 95% CI (pp) |
|---|------|---------|---------|-----------|---------|---------------|----------------|-------------|------------------------------|
| 1 | Set A (50 crossed class-prior cells) | ViT-B/16 | 10 | -3.28 | 3.81 | [-6.01, -0.56] | 2/10 | 0.1094 | [-5.33, -0.88] |
| 2 | Set A (50 crossed class-prior cells) | ViT-L/14 | 5 | -0.14 | 1.88 | [-2.47, 2.20] | 1/5 | 0.3750 | [-1.20, 1.52] |
| 3 | Set B (second source, 3 targets) | ViT-B/16 | 10 | -0.77 | 3.62 | [-3.37, 1.82] | 3/10 | 0.3438 | [-2.75, 1.46] |
| 4 | Set B (second source, 3 targets) | ViT-L/14 | 5 | 0.34 | 0.85 | [-0.71, 1.39] | 4/5 | 0.3750 | [-0.37, 0.95] |
| 5 | Corruption axis (12 pre-specified conditions) | ViT-B/16 | 10 | -0.04 | 1.45 | [-1.07, 1.00] | 4/10 | 0.7539 | [-0.83, 0.88] |
| 6 | Corruption axis (12 pre-specified conditions) | ViT-L/14 | 5 | 3.70 | 0.98 | [2.48, 4.91] | 5/5 | 0.0625 | [2.94, 4.46] |

Standard deviations are sample SDs (ddof = 1) and the t-interval uses df = n - 1. The seed bootstrap resamples the n per-seed differences with replacement (B = 10,000) and reports the 2.5 and 97.5 percentiles. pp = percentage points of unweighted average recall (UAR); the stored values are fractions.

Source files and fields (all under <code>results/token_module_tailor_20261006/</code>):

| # | Source file | Per-seed difference field | Seed field |
|---|-------------|---------------------------|------------|
| 1 | <code>phase8_setA_trainvar.json</code> | <code>B16.rows[*].TAP_minus_CA</code> | <code>B16.rows[*].train_seed</code> |
| 2 | <code>phase8_setA_trainvar.json</code> | <code>L14.rows[*].TAP_minus_CA</code> | <code>L14.rows[*].train_seed</code> |
| 3 | <code>phase8_setB_trainvar.json</code> | <code>B16.rows[*].TAP_minus_CA</code> | <code>B16.rows[*].train_seed</code> |
| 4 | <code>phase8_setB_trainvar.json</code> | <code>L14.rows[*].TAP_minus_CA</code> | <code>L14.rows[*].train_seed</code> |
| 5 | <code>phase8_corruption_trainvar.json</code> | <code>B16.rows[*].TAP_minus_CA</code> | <code>B16.rows[*].train_seed</code> |
| 6 | <code>phase8_corruption_trainvar.json</code> | <code>L14.rows[*].TAP_minus_CA</code> | <code>L14.rows[*].train_seed</code> |

Reproduce: <code>D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/phase10_stats_robustness.py</code>

## Table 2. Family-wise and false-discovery control over the six sign tests

The family is the six core contrasts defined above (three confirmation axes x two encoders), tested with the exact two-sided sign test at alpha = 0.05. Sorting the six p-values puts <code>corruption_L14</code> (3.70 pp, 5 of 5 seeds positive) at rank 1, and no p-value in the family reaches the rank-1 Holm threshold of alpha/6 = 0.0083 or the rank-1 Benjamini-Hochberg threshold of (1/6) x alpha = 0.0083.

| Rank | Axis | Encoder | Sign-test p | Holm-Bonferroni adjusted p | BH adjusted p | Rejected by Holm | Rejected by BH |
|------|------|---------|-------------|----------------------------|---------------|------------------|----------------|
| 1 | Corruption axis (12 pre-specified conditions) | ViT-L/14 | 0.0625 | 0.3750 | 0.3281 | no | no |
| 2 | Set A (50 crossed class-prior cells) | ViT-B/16 | 0.1094 | 0.5469 | 0.3281 | no | no |
| 3 | Set B (second source, 3 targets) | ViT-B/16 | 0.3438 | 1.0000 | 0.4500 | no | no |
| 4 | Set A (50 crossed class-prior cells) | ViT-L/14 | 0.3750 | 1.0000 | 0.4500 | no | no |
| 5 | Set B (second source, 3 targets) | ViT-L/14 | 0.3750 | 1.0000 | 0.4500 | no | no |
| 6 | Corruption axis (12 pre-specified conditions) | ViT-B/16 | 0.7539 | 1.0000 | 0.7539 | no | no |

**No contrast survives either correction.** Under Holm-Bonferroni the step-down procedure stops at the first rank, because the smallest p-value (0.0625) already exceeds alpha/6 = 0.0083; under Benjamini-Hochberg no rank k satisfies p_(k) <= (k/m) x alpha either. The rejection set is therefore empty for both procedures, and the adjusted p-value of the strongest contrast is 0.3750 under Holm-Bonferroni and 0.3281 under Benjamini-Hochberg.

The reason is the seed budget rather than the size of the effect. The strongest contrast - ViT-L/14 on the corruption axis, mean 3.70 pp - is positive in 5 of 5 seeds, and its exact two-sided sign-test p is exactly 0.0625, which is the floor of the test at n = 5 (2 / 2^5 = 0.0625): with five seeds no outcome, not even a unanimous one, can produce p < 0.05. Since the sign test cannot reach 0.05 at n = 5, no multiplicity correction can make any ViT-L/14 contrast significant, and the same floor applies to the Set B contrast at ViT-L/14 (p = 0.3750). The evidence separating the corruption result from the other five therefore rests on the intervals of Table 1 (its t-interval [2.48, 4.91] pp and its seed bootstrap interval [2.94, 4.46] pp both exclude zero, unlike every other contrast) and on the unanimity of its sign, not on a family-wise-significant p-value.

Source files and fields: the six p-values are computed from the <code>p</code> entries derived from <code>TAP_minus_CA</code> in <code>phase8_setA_trainvar.json</code>, <code>phase8_setB_trainvar.json</code> and <code>phase8_corruption_trainvar.json</code> (exact field paths in Table 1). Correction settings: alpha = 0.05, m = 6.

Reproduce: <code>D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/phase10_stats_robustness.py</code>

## What the intervals cover, and what they do not

The t-intervals and the seed-level bootstrap intervals quantify exactly one source of randomness: which training run produced the two arms. They answer the question a reader of a single-run comparison should ask - would the sign and the size of this margin hold if the arms had been fitted from a different initialisation - and they hold everything else fixed. They do not cover uncertainty in the evaluation units: the 50 crossed cells, the three second-source targets and the 12 corruption conditions are treated as the population, and no resampling of cells, targets or conditions enters the two intervals of Table 1. They do not cover query sampling within a cell or a condition, and they do not cover any part of the source-domain training pipeline that was held fixed across seeds, such as data order, learning-rate selection or augmentation. The seed bootstrap of Table 1 varies the training run at fixed units, whereas the within-seed unit bootstrap of Table 3 varies the units at fixed training runs; the two must not be read as a single interval.

One consequence is worth stating plainly. Because these intervals condition on the confirmation units they are narrower than an interval that would also propagate unit-level sampling, and they are wider than the single-run numbers they replace. The ViT-L/14 corruption margin is the case in point: the single-run estimate was +3.20 pp, the retrained mean is 3.70 pp, the interval over seeds is [2.48, 4.91] pp, and the sign is consistent in all five seeds. The same logic cuts the other way for the ViT-B/16 crossed-grid margin: the single run reported +3.88 pp, the retrained mean is -3.28 pp, and the interval over seeds is [-6.01, -0.56] pp, which excludes zero on the negative side - so the reversal is itself supported by the interval, and what the retraining overturns is the sign of that effect, not its existence.

Source files and fields: <code>phase8_setA_trainvar.json</code>, <code>phase8_setB_trainvar.json</code>, <code>phase8_corruption_trainvar.json</code>, field <code>&lt;encoder&gt;.rows[*].TAP_minus_CA</code>. The single-run reference values (+3.20 pp and +3.88 pp) are the original-run numbers quoted in the manuscript, not values stored in these files.

Reproduce: <code>D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/phase10_stats_robustness.py</code>

## Table 3. Within-seed (unit-level) paired bootstrap, where it exists

| Axis | Encoder | Unit | Units per seed | B | 95% CI (pp) | Available |
|------|---------|------|----------------|---|-------------|-----------|
| Set A (50 crossed class-prior cells) | ViT-B/16 | cells | 50 | 2000 | [-3.82, -2.78] | yes |
| Set A (50 crossed class-prior cells) | ViT-L/14 | cells | 50 | 2000 | [-0.44, 0.16] | yes |
| Set B (second source, 3 targets) | ViT-B/16 | targets | 0 | n/a | n/a | **no** |
| Set B (second source, 3 targets) | ViT-L/14 | targets | 0 | n/a | n/a | **no** |
| Corruption axis (12 pre-specified conditions) | ViT-B/16 | conditions | 12 | 2000 | [-0.53, 0.45] | yes |
| Corruption axis (12 pre-specified conditions) | ViT-L/14 | conditions | 12 | 2000 | [2.75, 4.60] | yes |

Where per-unit values are stored, the units are resampled with replacement inside each training seed and the same resample is shared by both arms, so the pairing is preserved; the resulting per-seed means are averaged over seeds (B = 2,000). The seed dimension is held fixed here, so these intervals say how much of each margin is attributable to which cells or conditions were evaluated, not how much is attributable to retraining.

**Set B has no per-cell values.** <code>phase8_setB_pertarget.json</code> stores only three per-target aggregate UAR values per seed (<code>&lt;encoder&gt;[*].seed&lt;k&gt;.{TAP, CA, GH}</code>, one entry per target, with n = 441, 7178 and 902), so the 15 crossed cells of that axis are not recoverable from the stored files and the cell-level within-seed paired bootstrap cannot be computed for Set B. We did not substitute the three target aggregates for cells: a three-unit resample is degenerate, and it is recorded in the JSON only under <code>target_level_fallback_not_interpretable</code> with <code>interpretable = false</code>.

Source files and fields:

| Axis | Unit file | Per-unit fields |
|------|-----------|-----------------|
| Set A | <code>phase8_setA_percell.json</code> | <code>&lt;encoder&gt;.&lt;seed&gt;.{TAP, CA}</code>, 50 per-cell UAR values per seed |
| Corruption | <code>phase8_corruption_percond.json</code> | <code>&lt;encoder&gt;[*].seed&lt;k&gt;.{TAP, CA}</code>, 12 per-condition UAR values per seed |
| Set B | <code>phase8_setB_pertarget.json</code> | <code>&lt;encoder&gt;[*].seed&lt;k&gt;.{TAP, CA}</code>, 3 per-target aggregates per seed, no per-cell values |

Reproduce: <code>D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/phase10_stats_robustness.py</code>

## Table 4. Alignment of the stored aggregates with the recomputed per-seed statistics

| Axis | Encoder | Aggregate | n | Stored mean | Recomputed mean | Match (4 dp) | Stored SD | Recomputed SD | Match (4 dp) |
|------|---------|-----------|----|-------------|-----------------|--------------|-----------|---------------|--------------|
| Set A (50 crossed class-prior cells) | ViT-B/16 | <code>TAP_minus_CA</code> | 10 | -0.032824 | -0.032824 | yes | 0.038117 | 0.038117 | yes |
| Set A (50 crossed class-prior cells) | ViT-B/16 | <code>TAP_minus_GH</code> | 10 | -0.032349 | -0.032349 | yes | 0.043931 | 0.043931 | yes |
| Set A (50 crossed class-prior cells) | ViT-L/14 | <code>TAP_minus_CA</code> | 5 | -0.001362 | -0.001362 | yes | 0.018775 | 0.018775 | yes |
| Set A (50 crossed class-prior cells) | ViT-L/14 | <code>TAP_minus_GH</code> | 5 | 0.025220 | 0.025220 | yes | 0.022268 | 0.022268 | yes |
| Set B (second source, 3 targets) | ViT-B/16 | <code>TAP_minus_CA</code> | 10 | -0.007732 | -0.007732 | yes | 0.036242 | 0.036242 | yes |
| Set B (second source, 3 targets) | ViT-B/16 | <code>TAP_minus_GH</code> | 10 | -0.007425 | -0.007425 | yes | 0.038209 | 0.038209 | yes |
| Set B (second source, 3 targets) | ViT-L/14 | <code>TAP_minus_CA</code> | 5 | 0.003393 | 0.003393 | yes | 0.008476 | 0.008476 | yes |
| Set B (second source, 3 targets) | ViT-L/14 | <code>TAP_minus_GH</code> | 5 | 0.025805 | 0.025805 | yes | 0.007610 | 0.007610 | yes |
| Corruption axis (12 pre-specified conditions) | ViT-B/16 | <code>TAP_minus_CA</code> | 10 | -0.000355 | -0.000355 | yes | 0.014498 | 0.014498 | yes |
| Corruption axis (12 pre-specified conditions) | ViT-B/16 | <code>TAP_minus_GH</code> | 10 | 0.001058 | 0.001058 | yes | 0.017372 | 0.017372 | yes |
| Corruption axis (12 pre-specified conditions) | ViT-L/14 | <code>TAP_minus_CA</code> | 5 | 0.036975 | 0.036975 | yes | 0.009794 | 0.009794 | yes |
| Corruption axis (12 pre-specified conditions) | ViT-L/14 | <code>TAP_minus_GH</code> | 5 | 0.034328 | 0.034328 | yes | 0.008133 | 0.008133 | yes |
| Source probe-validation (source-domain diagnostic) | ViT-B/16 | <code>TAP_minus_CLIPAdapter</code> | 10 | -0.009394 | -0.009394 | yes | 0.021415 | 0.021415 | yes |
| Source probe-validation (source-domain diagnostic) | ViT-B/16 | <code>TAP_minus_GHead</code> | 10 | -0.001797 | -0.001797 | yes | 0.025877 | 0.025877 | yes |
| Source probe-validation (source-domain diagnostic) | ViT-L/14 | <code>TAP_minus_CLIPAdapter</code> | 5 | 0.003229 | 0.003229 | yes | 0.004894 | 0.004894 | yes |
| Source probe-validation (source-domain diagnostic) | ViT-L/14 | <code>TAP_minus_GHead</code> | 5 | 0.014754 | 0.014754 | yes | 0.011039 | 0.011039 | yes |

16 of 16 stored aggregate entries match the recomputed per-seed statistics at four decimals, for both the mean and the SD; there are 0 mismatches. The stored SDs are sample SDs (ddof = 1): the population SDs (ddof = 0) are also recorded in the JSON and do not match, which fixes the convention used by the Phase-8 aggregation. No stored value was adjusted.

The per-unit files reproduce the per-seed arm means as well: recomputing <code>TAP_mean</code>, <code>CA_mean</code> and <code>TAP_minus_CA</code> from the stored per-cell and per-condition values agrees with the trainvar records to within <code>max_abs_difference = 0.0</code> for all six axis x encoder combinations (see <code>unit_table_consistency</code> in the JSON).

Source files and fields: <code>phase8_trainvar.json</code> (<code>{B16,L14}.TAP_minus_{CLIPAdapter,GHead}.{mean,sd}</code>), <code>phase8_setA_trainvar.json</code>, <code>phase8_setB_trainvar.json</code> and <code>phase8_corruption_trainvar.json</code> (<code>{B16,L14}.TAP_minus_{CA,GH}.{mean,sd}</code>), cross-checked against the per-seed fields of the same files.

Reproduce: <code>D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/phase10_stats_robustness.py</code>

## Appendix A. Supplementary contrasts

### A.1 Parameter-matched global head as comparator (TAP minus GHead)

| Axis | Encoder | n seeds | Mean (pp) | SD (pp) | 95% t-CI (pp) | Seeds positive | Sign-test p | Seed bootstrap 95% CI (pp) |
|------|---------|---------|-----------|---------|---------------|----------------|-------------|------------------------------|
| Set A (50 crossed class-prior cells) | ViT-B/16 | 10 | -3.23 | 4.39 | [-6.38, -0.09] | 2/10 | 0.1094 | [-5.58, -0.45] |
| Set A (50 crossed class-prior cells) | ViT-L/14 | 5 | 2.52 | 2.23 | [-0.24, 5.29] | 5/5 | 0.0625 | [1.29, 4.56] |
| Set B (second source, 3 targets) | ViT-B/16 | 10 | -0.74 | 3.82 | [-3.48, 1.99] | 3/10 | 0.3438 | [-2.87, 1.60] |
| Set B (second source, 3 targets) | ViT-L/14 | 5 | 2.58 | 0.76 | [1.64, 3.53] | 5/5 | 0.0625 | [2.01, 3.20] |
| Corruption axis (12 pre-specified conditions) | ViT-B/16 | 10 | 0.11 | 1.74 | [-1.14, 1.35] | 4/10 | 0.7539 | [-0.94, 1.11] |
| Corruption axis (12 pre-specified conditions) | ViT-L/14 | 5 | 3.43 | 0.81 | [2.42, 4.44] | 5/5 | 0.0625 | [2.86, 4.11] |

These six contrasts are outside the corrected family above; they are reported because the manuscript also quotes the parameter-matched head as a secondary comparator. Fields: <code>&lt;encoder&gt;.rows[*].TAP_minus_GH</code> in the three <code>phase8_*_trainvar.json</code> files.

Reproduce: <code>D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/phase10_stats_robustness.py</code>

### A.2 Source probe-validation axis (diagnostic, not part of the family)

| Comparator | Encoder | n seeds | Mean (pp) | SD (pp) | 95% t-CI (pp) | Seeds positive | Sign-test p | Seed bootstrap 95% CI (pp) |
|------------|---------|---------|-----------|---------|---------------|----------------|-------------|------------------------------|
| CLIP-Adapter | ViT-B/16 | 10 | -0.94 | 2.14 | [-2.47, 0.59] | 4/10 | 0.7539 | [-2.24, 0.30] |
| parameter-matched global head (GHead) | ViT-B/16 | 10 | -0.18 | 2.59 | [-2.03, 1.67] | 5/10 | 1.0000 | [-1.75, 1.31] |
| CLIP-Adapter | ViT-L/14 | 5 | 0.32 | 0.49 | [-0.28, 0.93] | 3/5 | 1.0000 | [-0.08, 0.68] |
| parameter-matched global head (GHead) | ViT-L/14 | 5 | 1.48 | 1.10 | [0.10, 2.85] | 5/5 | 0.0625 | [0.60, 2.35] |

This is the source-domain probe-validation diagnostic recorded in <code>phase8_trainvar.json</code>. It is one held-out split per seed, so no per-unit file exists and no unit-level bootstrap is possible, and it is not one of the manuscript's six contrasts. Fields: <code>&lt;encoder&gt;.rows[*].TAP_minus_{CLIPAdapter,GHead}</code>.

Reproduce: <code>D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/phase10_stats_robustness.py</code>

## Provenance

Input files with SHA-256, all under <code>D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/</code>:

- <code>phase8_trainvar.json</code> - <code>2f09414b3b63ffa11f1c9bc8c2ad4b1763888abb92a410966585a5730c2d62b2</code>
- <code>phase8_setA_trainvar.json</code> - <code>e3db05d17310d8266cb921f78068981ab722833aa75353016590a24b74d81ab6</code>
- <code>phase8_setB_trainvar.json</code> - <code>b33bef5f70cbb04db2e97b5d99132236328ae837c6ce36b75d53d6fb2af69421</code>
- <code>phase8_corruption_trainvar.json</code> - <code>e4f1deadb6458de8162f56bbee7a3e487a2e4de3448941b515248625da9473de</code>
- <code>phase8_setA_percell.json</code> - <code>83aaf0a1fe439be491baa4edade44c9e4a377e0d2988bb9edd40d7c4cded66f8</code>
- <code>phase8_setB_pertarget.json</code> - <code>4661a8291b1c4b2427c84bef0fac7cacfbe574b78a42fb23d744a7898237fafa</code>
- <code>phase8_corruption_percond.json</code> - <code>1a1962c53b8b6ecfe385ba4be6c105981921b26093cc239c779884205842bd24</code>

Machine-readable companion: <code>D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/phase10_stats_robustness.json</code>.

Reproduce both artefacts with:

    D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/phase10_stats_robustness.py

