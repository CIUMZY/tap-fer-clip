
# Amendment 1 to the JAFFE pre-specification (phase 13)

Recorded 2026-10-08, minutes after the original registration and **before any model was run on
JAFFE, before any feature was extracted from it, and before any JAFFE accuracy or contrast was
computed**. At the moment of writing, the only JAFFE-derived facts that exist are (a) the archive
sha256 matches the recorded value, and (b) the archive's file listing.

## What was wrong

The original registration asserted the structural sanity check "filenames starting with one model
code appear exactly once per expression token, i.e. 10 x 7 + 143 = 213 images with no duplicate
(model, expression) pair". That claim about the corpus is incorrect. JAFFE contains 213 images from
10 models in which each model poses each of the 7 expressions **two to four times**; the arithmetic
"10 x 7 + 143" was wrong.

## Corrected structural check (replaces the check in section "Frozen preprocessing", item 2)

The file listing gives: 213 images; 10 model codes; 7 expression tokens; all 10 x 7 = 70
(model, expression) pairs present; between 2 and 4 images per pair; between 20 and 23 images per
model; between 29 and 32 images per expression. The check that will be enforced before any feature
extraction is: exactly 213 images, exactly 70 distinct (model, expression) pairs, every pair with at
least 2 images, and every expression token in the fixed map {AN 0, DI 1, FE 2, HA 3, SA 4, SU 5,
NE 6}.

## What is unaffected

Nothing that the experiment tests changes. The hypothesis H_J, the directional prediction, the
grayscale 48x48 LANCZOS preprocessing, the 20-condition degradation grid (4 families x 5
severities), the arms evaluated, the primary and secondary learning-rate conditions, the decision
rule (CONFIRMED / STRENGTHENED / NOT CONFIRMED) and the consumption rule are exactly as registered
and are **not** amended. In particular the pass condition - at ViT-L/14, positive between-seed mean
of TAP minus CLIP-Adapter with a seed-level 95% interval excluding zero - is untouched.

The reason for amending at all is that the wrong sentence was a factual error about a corpus, not a
choice about the analysis; leaving it in place would misdescribe the data. The original file and its
sha256 remain in the record unchanged, and this amendment is hashed separately.

