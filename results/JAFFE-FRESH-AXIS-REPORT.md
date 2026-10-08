# Phase 13: the JAFFE fresh confirmation axis

Pre-specification: phase13_jaffe_preregistration.md (sha256 a70aaa6255dc73a1...) plus amendment 1 (sha256 cde86a04442917aa...), both written and hashed before any JAFFE image was opened. JAFFE archive sha256 6da27f5954f969c6...

## Pre-specified verdict

**CONFIRMED AND STRENGTHENED** - at ViT-L/14 the mean of the per-seed TAP minus CLIP-Adapter contrast on the corrupted JAFFE axis is +2.7493 points with a seed-level 95% interval of [+1.7766, +3.7220] and 15 of 15 seeds positive.

## Degraded JAFFE axis (pre-specified grid, 20 conditions)

| encoder | condition | contrast | mean (pp) | SD | 95% interval | seeds positive | sign-test p |
|---|---|---|---|---|---|---|---|
| ViT-L/14 | published | TAP - CA | +2.75 | 1.76 | [+1.78, +3.72] | 15/15 | 0.0001 |
| ViT-L/14 | published | TAP - GH | +3.75 | 1.86 | [+2.72, +4.78] | 14/15 | 0.0010 |
| ViT-L/14 | nested | TAP - CA | +3.20 | 1.63 | [+2.30, +4.10] | 14/15 | 0.0010 |
| ViT-L/14 | nested | TAP - GH | +3.59 | 1.78 | [+2.60, +4.57] | 15/15 | 0.0001 |
| ViT-B/16 | published | TAP - CA | -1.72 | 1.70 | [-2.93, -0.50] | 3/10 | 0.3438 |
| ViT-B/16 | published | TAP - GH | -1.54 | 1.26 | [-2.44, -0.64] | 1/10 | 0.0215 |
| ViT-B/16 | nested | TAP - CA | -0.61 | 2.50 | [-2.39, +1.18] | 3/10 | 0.3438 |
| ViT-B/16 | nested | TAP - GH | -0.98 | 1.33 | [-1.93, -0.03] | 2/10 | 0.1094 |

## Clean controls (not part of the pre-specified grid; reported because they decide how the result should be read)

| data | encoder | contrast | mean (pp) | SD | 95% interval | seeds positive |
|---|---|---|---|---|---|---|
| JAFFE, clean | ViT-L/14 | TAP - CA | +2.04 | 3.30 | [+0.22, +3.87] | 11/15 |
| JAFFE, clean | ViT-L/14 | TAP - GH | +3.58 | 4.47 | [+1.11, +6.06] | 13/15 |
| JAFFE, clean | ViT-B/16 | TAP - CA | -4.61 | 4.63 | [-7.92, -1.29] | 1/10 |
| JAFFE, clean | ViT-B/16 | TAP - GH | -5.80 | 3.06 | [-7.99, -3.61] | 0/10 |
| FER2013 test, clean | ViT-B/16 | TAP - CA | +0.40 | 1.35 | [-0.57, +1.37] | 6/10 |
| FER2013 test, clean | ViT-B/16 | TAP - GH | +1.05 | 2.09 | [-0.44, +2.54] | 6/10 |
| FER2013 test, clean | ViT-L/14 | TAP - CA | -0.17 | 1.16 | [-0.81, +0.47] | 7/15 |
| FER2013 test, clean | ViT-L/14 | TAP - GH | +1.00 | 1.36 | [+0.25, +1.75] | 11/15 |
| FER2013 test, corrupted (pre-specified axis) | ViT-L/14 | TAP - CA | +3.50 | 1.69 | [+2.56, +4.43] | 15/15 |

## What the numbers say

1. The pre-specified criterion passes on a corpus that had never been touched: at ViT-L/14 the token route beats CLIP-Adapter on the fresh degraded axis by +2.75 points, positive in all fifteen seeds. The direction of the surviving FER2013 claim replicates out of sample.
2. The clean control changes what the fresh-axis result means. On clean JAFFE the same contrast is already +2.04 points, so most of the fresh-axis margin is a cross-domain accuracy advantage rather than a robustness advantage; corruption adds about +0.70 points.
3. On the original within-domain axis the opposite holds: on clean FER2013 test the contrast is -0.17 points (interval covering zero) while under the pre-specified corruption it is +3.50 points, so there the margin is degradation-specific.
4. The encoder dependence replicates in sign, and more strongly than on FER2013: at ViT-B/16 the token route is worse than CLIP-Adapter on the fresh axis (-1.72 points, interval excluding zero) and worse still on clean JAFFE (-4.61 points).

Honest reading: the fresh axis confirms the direction of the L/14 advantage and its encoder dependence, and it does not contradict the within-domain robustness finding, but it shows that out of domain the advantage is not degradation-specific. The manuscript reports both rows.

## Consumption

JAFFE is now a consumed confirmation axis. Nothing else may be selected on it.
