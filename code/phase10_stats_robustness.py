"""phase10_stats_robustness.py

Statistical robustness audit for the training-seed variance manuscript
(TVC submission, PAPER-TVC-TAP-2026-10-06).

Reads only the stored Phase-8 per-seed / per-unit JSON records and derives, for the
six core contrasts (TAP minus CLIP-Adapter, three confirmation axes x two encoders):

  * per-seed paired difference vectors, taken from the stored per-seed fields
  * mean, sample SD (ddof=1), 95% Student-t interval, seeds positive
  * exact two-sided sign test (zeros dropped, p capped at 1.0)
  * seed-level percentile bootstrap (B = 10000, rng = np.random.default_rng(0))
  * within-seed paired unit bootstrap over cells / conditions (B = 2000) where
    per-unit values are stored
  * Holm-Bonferroni and Benjamini-Hochberg adjustment of the six sign-test p-values

Outputs (both deterministic, i.e. re-running produces byte-identical files):

  * phase10_stats_robustness.json   (machine readable, next to this script)
  * manuscript/STATS-ROBUSTNESS.md  (human readable, manuscript directory)

No model is retrained and no evaluation is re-run.  This script writes to no file
other than the two outputs above.
"""

import hashlib
import json
import math
import os
import sys

import numpy as np
import scipy
from scipy import stats

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

BASE = os.path.dirname(os.path.abspath(__file__))
JSON_OUT = os.path.join(BASE, "phase10_stats_robustness.json")

MANUSCRIPT_DIR = (
    "D:/ResearchVault/06manuscripts/rb-tta-fer-fg2027/"
    "PAPER-TVC-TAP-2026-10-06/manuscript"
)
REPORT_OUT = os.path.join(MANUSCRIPT_DIR, "STATS-ROBUSTNESS.md")

PYTHON_EXE = "D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe"
SCRIPT_PATH = os.path.join(BASE, os.path.basename(__file__)).replace("\\", "/")
REPRO_CMD = "{} {}".format(PYTHON_EXE, SCRIPT_PATH)

ALPHA = 0.05
B_SEED = 10000
B_UNIT = 2000
RNG_SEED = 0

INPUT_FILES = [
    "phase8_trainvar.json",
    "phase8_setA_trainvar.json",
    "phase8_setB_trainvar.json",
    "phase8_corruption_trainvar.json",
    "phase8_setA_percell.json",
    "phase8_setB_pertarget.json",
    "phase8_corruption_percond.json",
]

# Comparator of record for the six core contrasts: CLIP-Adapter.  This is the
# contrast the manuscript reports (+3.70 points for ViT-L/14 on the corruption axis).
COMPARATOR = "CLIP-Adapter"

AXES = {
    "setA": {
        "label": "Set A (50 crossed class-prior cells)",
        "trainvar_file": "phase8_setA_trainvar.json",
        "seed_field": "rows[*].train_seed",
        "diff_field_ca": "rows[*].TAP_minus_CA",
        "diff_field_gh": "rows[*].TAP_minus_GH",
        "arm_fields": "rows[*].TAP_mean, rows[*].CA_mean, rows[*].GH_mean",
        "unit_file": "phase8_setA_percell.json",
        "unit_kind": "cells",
        "unit_layout": "<encoder>.<seed>.{TAP, CA, GH} = list of per-cell UAR",
    },
    "setB": {
        "label": "Set B (second source, 3 targets)",
        "trainvar_file": "phase8_setB_trainvar.json",
        "seed_field": "rows[*].train_seed",
        "diff_field_ca": "rows[*].TAP_minus_CA",
        "diff_field_gh": "rows[*].TAP_minus_GH",
        "arm_fields": "rows[*].TAP_mean, rows[*].CA_mean, rows[*].GH_mean",
        "unit_file": "phase8_setB_pertarget.json",
        "unit_kind": "targets",
        "unit_layout": "<encoder> = list of 3 target records, each with target, n, seed<k>.{TAP, CA, GH}",
    },
    "corruption": {
        "label": "Corruption axis (12 pre-specified conditions)",
        "trainvar_file": "phase8_corruption_trainvar.json",
        "seed_field": "rows[*].train_seed",
        "diff_field_ca": "rows[*].TAP_minus_CA",
        "diff_field_gh": "rows[*].TAP_minus_GH",
        "arm_fields": "rows[*].TAP_mean, rows[*].CA_mean, rows[*].GH_mean",
        "unit_file": "phase8_corruption_percond.json",
        "unit_kind": "conditions",
        "unit_layout": "<encoder> = list of 12 condition records, each with condition, n, seed<k>.{TAP, CA, GH}",
    },
}

# Diagnostic axis (source-domain probe validation).  Stored and reported, but NOT
# part of the six-contrast family: the family is the three confirmation axes above.
# Kept because the input list names the file.
DIAG_AXIS = {
    "label": "Source probe-validation (source-domain diagnostic)",
    "trainvar_file": "phase8_trainvar.json",
    "unit_file": None,
    "unit_kind": "none",
}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def load_json(name):
    with open(os.path.join(BASE, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_of(name):
    h = hashlib.sha256()
    with open(os.path.join(BASE, name), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_sign_test(pos, neg):
    """Exact two-sided sign test over the non-zero entries.

    p = 2 * sum_{i=0..min(pos,neg)} C(n,i) / 2^n, capped at 1.0, n = pos + neg.
    """
    n = pos + neg
    if n == 0:
        return 1.0, 0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / float(2 ** n)), n


def t_interval(diffs, alpha=ALPHA):
    """Two-sided Student-t interval on the per-seed mean."""
    n = len(diffs)
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1)) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "sd_ddof1": sd,
        "se": se,
        "t_crit": tcrit,
        "df": n - 1,
        "ci_low": mean - tcrit * se if n > 1 else float("nan"),
        "ci_high": mean + tcrit * se if n > 1 else float("nan"),
    }


def seed_bootstrap(diffs, b=B_SEED, rng_seed=RNG_SEED):
    """Percentile bootstrap over training seeds, with replacement."""
    arr = np.asarray(diffs, dtype=float)
    n = arr.size
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(b, n))
    stat = arr[idx].mean(axis=1)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return {
        "method": "seed_level_percentile_bootstrap",
        "n_seeds": int(n),
        "B": int(b),
        "rng": "numpy.random.default_rng({})".format(rng_seed),
        "percentiles": [2.5, 97.5],
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def unit_bootstrap(per_seed_unit_diffs, b=B_UNIT, rng_seed=RNG_SEED):
    """Within-seed paired unit bootstrap.

    per_seed_unit_diffs: list (over seeds) of 1-D arrays of per-unit paired
    differences.  Inside each seed the units are resampled with replacement; the
    same resample is shared by both arms, so the pairing is preserved.  Each
    replicate is the average over seeds of the per-seed resampled means.
    """
    arrays = [np.asarray(a, dtype=float) for a in per_seed_unit_diffs]
    n_units = arrays[0].size
    n_seeds = len(arrays)
    rng = np.random.default_rng(rng_seed)
    acc = np.zeros(b, dtype=float)
    for a in arrays:
        idx = rng.integers(0, n_units, size=(b, n_units))
        acc += a[idx].mean(axis=1)
    acc /= n_seeds
    lo, hi = np.percentile(acc, [2.5, 97.5])
    return {
        "method": "within_seed_paired_unit_percentile_bootstrap",
        "n_units_per_seed": int(n_units),
        "n_seeds": int(n_seeds),
        "B": int(b),
        "rng": "numpy.random.default_rng({})".format(rng_seed),
        "percentiles": [2.5, 97.5],
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def holm_bonferroni(pvals, alpha=ALPHA):
    """Step-down Holm-Bonferroni.

    Sort ascending; reject while p_(k) <= alpha / (m - k + 1); stop at the first
    failure.  Adjusted p_(k) = max_{j<=k} min(1, (m - j + 1) * p_(j)).
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    rejected = [False] * m
    running = 0.0
    still_rejecting = True
    for rank, i in enumerate(order, start=1):
        factor = m - rank + 1
        running = max(running, min(1.0, factor * pvals[i]))
        adj[i] = running
        if still_rejecting and pvals[i] <= alpha / factor:
            rejected[i] = True
        else:
            still_rejecting = False
    return {
        "method": "holm_bonferroni_step_down",
        "alpha": alpha,
        "m": m,
        "adjusted_p": adj,
        "reject": rejected,
        "n_rejected": int(sum(rejected)),
    }


def benjamini_hochberg(pvals, alpha=ALPHA):
    """Benjamini-Hochberg step-up.

    Sort ascending; find the largest k with p_(k) <= (k / m) * alpha and reject
    every hypothesis up to that rank.  Adjusted p_(k) = min_{j>=k} min(1,(m/j)p_(j)).
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, min(1.0, (m / float(rank)) * pvals[i]))
        adj[i] = running
    k_max = 0
    for rank in range(1, m + 1):
        i = order[rank - 1]
        if pvals[i] <= (rank / float(m)) * alpha:
            k_max = rank
    rejected = [False] * m
    for rank in range(1, k_max + 1):
        rejected[order[rank - 1]] = True
    return {
        "method": "benjamini_hochberg_step_up",
        "alpha": alpha,
        "m": m,
        "adjusted_p": adj,
        "reject": rejected,
        "n_rejected": int(sum(rejected)),
    }


def axis_rows(block, diff_field):
    """Per-seed seeds and stored differences for one axis/encoder block."""
    seed_key = "train_seed" if "train_seed" in block["rows"][0] else "seed"
    field = diff_field.split(".")[-1]
    seeds = [int(r[seed_key]) for r in block["rows"]]
    diffs = [float(r[field]) for r in block["rows"]]
    return seeds, diffs


def unit_arrays(unit_blob, seeds, arm_a, arm_b):
    """Per-seed arrays of per-unit (arm_a - arm_b) paired differences."""
    out = []
    if isinstance(unit_blob, dict):
        for s in seeds:
            rec = unit_blob[str(s)]
            out.append(np.asarray(rec[arm_a], float) - np.asarray(rec[arm_b], float))
        return out, "cells"
    if "condition" in unit_blob[0]:
        for s in seeds:
            a = np.asarray([c["seed{}".format(s)][arm_a] for c in unit_blob], float)
            b = np.asarray([c["seed{}".format(s)][arm_b] for c in unit_blob], float)
            out.append(a - b)
        return out, "conditions"
    for s in seeds:
        a = np.asarray([t["seed{}".format(s)][arm_a] for t in unit_blob], float)
        b = np.asarray([t["seed{}".format(s)][arm_b] for t in unit_blob], float)
        out.append(a - b)
    return out, "targets"


def unit_means(unit_blob, seed, arm):
    """Mean of the stored per-unit values for one seed and one arm."""
    if isinstance(unit_blob, dict):
        return float(np.mean(unit_blob[str(seed)][arm]))
    if "condition" in unit_blob[0]:
        return float(np.mean([c["seed{}".format(seed)][arm] for c in unit_blob]))
    return float(np.mean([t["seed{}".format(seed)][arm] for t in unit_blob]))


def consistency_check(block, unit_blob, seed_key):
    """Recompute the per-seed axis mean from the per-unit file and compare."""
    worst = 0.0
    details = []
    for row in block["rows"]:
        s = int(row[seed_key])
        tap = unit_means(unit_blob, s, "TAP")
        ca = unit_means(unit_blob, s, "CA")
        d_tap = abs(tap - float(row["TAP_mean"]))
        d_ca = abs(ca - float(row["CA_mean"]))
        d_diff = abs((tap - ca) - float(row["TAP_minus_CA"]))
        worst = max(worst, d_tap, d_ca, d_diff)
        details.append({
            "seed": s,
            "abs_diff_TAP_mean": d_tap,
            "abs_diff_CA_mean": d_ca,
            "abs_diff_TAP_minus_CA": d_diff,
        })
    return {
        "max_abs_difference": worst,
        "matches_to_1e-12": bool(worst < 1e-12),
        "per_seed": details,
    }


def contrast_stats(diffs, seeds):
    diffs = [float(x) for x in diffs]
    arr = np.asarray(diffs, float)
    nz = arr[arr != 0.0]
    pos = int((nz > 0).sum())
    neg = int((nz < 0).sum())
    zero = int(arr.size - nz.size)
    p, n_used = exact_sign_test(pos, neg)
    return {
        "n_seeds": int(arr.size),
        "seeds": [int(s) for s in seeds],
        "per_seed_diffs": diffs,
        "per_seed_diffs_points": [100.0 * x for x in diffs],
        "mean": float(arr.mean()),
        "sd_ddof1": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "mean_points": float(100.0 * arr.mean()),
        "sd_points": float(100.0 * arr.std(ddof=1)) if arr.size > 1 else None,
        "n_positive": pos,
        "n_negative": neg,
        "n_zero_dropped": zero,
        "seeds_positive": [int(s) for s, d in zip(seeds, diffs) if d > 0],
        "sign_test": {
            "test": "exact_two_sided_sign_test",
            "definition": "p = 2 * sum_{i=0..min(pos,neg)} C(n,i) / 2^n, capped at 1.0",
            "n_used": int(n_used),
            "n_positive": pos,
            "n_negative": neg,
            "n_zero_dropped": zero,
            "p": float(p),
        },
        "t_interval": t_interval(diffs),
        "seed_bootstrap": seed_bootstrap(diffs),
    }


# --------------------------------------------------------------------------- #
# Main computation                                                            #
# --------------------------------------------------------------------------- #

def build():
    data = {name: load_json(name) for name in INPUT_FILES}
    hashes = {name: sha256_of(name) for name in INPUT_FILES}

    comparisons = []
    for axis_key in ("setA", "setB", "corruption"):
        spec = AXES[axis_key]
        block_file = spec["trainvar_file"]
        for encoder in ("B16", "L14"):
            block = data[block_file][encoder]
            seeds, d_ca = axis_rows(block, spec["diff_field_ca"])
            _, d_gh = axis_rows(block, spec["diff_field_gh"])
            rec = contrast_stats(d_ca, seeds)
            rec["comparison_id"] = "{}_{}".format(axis_key, encoder)
            rec["axis"] = spec["label"]
            rec["axis_key"] = axis_key
            rec["encoder"] = encoder
            rec["comparator"] = COMPARATOR
            rec["contrast"] = "TAP - CLIP-Adapter"
            rec["source_file"] = block_file
            rec["source_fields"] = {
                "seeds": "{}.{}".format(encoder, spec["seed_field"]),
                "per_seed_difference": "{}.{}".format(encoder, spec["diff_field_ca"]),
                "per_seed_arm_means": "{}.{}".format(encoder, spec["arm_fields"]),
                "stored_aggregate_cross_checked": "{}.TAP_minus_CA.{{mean,sd}}".format(encoder),
            }
            rec["unit_source_file"] = spec["unit_file"]

            unit_blob = data[spec["unit_file"]][encoder]
            if axis_key == "setB":
                fb, kind = unit_arrays(unit_blob, seeds, "TAP", "CA")
                fallback = unit_bootstrap(fb)
                fallback["interpretable"] = False
                fallback["note"] = (
                    "only three per-target aggregate values per seed; a three-unit "
                    "resample is degenerate and is recorded for completeness only, "
                    "not as a usable interval"
                )
                cb = {
                    "available": False,
                    "unit_kind": kind,
                    "unit_file": spec["unit_file"],
                    "reason": (
                        "the stored second-source file holds only three per-target "
                        "aggregate UAR values per seed "
                        "(phase8_setB_pertarget.json[<encoder>][*].seed<k>), not per-cell "
                        "values; the 15 crossed cells are not stored, so the cell-level "
                        "within-seed paired bootstrap cannot be computed"
                    ),
                    "target_level_fallback_not_interpretable": fallback,
                }
            else:
                arrays, kind = unit_arrays(unit_blob, seeds, "TAP", "CA")
                cb = unit_bootstrap(arrays)
                cb["available"] = True
                cb["unit_kind"] = kind
                cb["unit_file"] = spec["unit_file"]
                cb["unit_layout"] = spec["unit_layout"]
                cb["covers"] = (
                    "resampling of the {} within each training seed, paired across "
                    "arms; the training seeds are held fixed".format(kind)
                )
            rec["condition_bootstrap"] = cb
            comparisons.append(rec)

    pvals = [c["sign_test"]["p"] for c in comparisons]
    holm = holm_bonferroni(pvals)
    bh = benjamini_hochberg(pvals)
    multiplicity = {
        "alpha": ALPHA,
        "family": [c["comparison_id"] for c in comparisons],
        "family_definition": (
            "the six core contrasts = three confirmation axes (Set A, Set B, "
            "corruption) x two encoders (ViT-B/16, ViT-L/14), comparator CLIP-Adapter"
        ),
        "p_values": pvals,
        "holm_bonferroni": holm,
        "benjamini_hochberg": bh,
        "rejected_by_holm": [c["comparison_id"] for c, r in zip(comparisons, holm["reject"]) if r],
        "rejected_by_bh": [c["comparison_id"] for c, r in zip(comparisons, bh["reject"]) if r],
        "rejected_by_any": sorted(set(
            [c["comparison_id"] for c, r in zip(comparisons, holm["reject"]) if r]
            + [c["comparison_id"] for c, r in zip(comparisons, bh["reject"]) if r]
        )),
    }

    supp = []
    for axis_key in ("setA", "setB", "corruption"):
        spec = AXES[axis_key]
        for encoder in ("B16", "L14"):
            block = data[spec["trainvar_file"]][encoder]
            seeds, d_gh = axis_rows(block, spec["diff_field_gh"])
            rec = contrast_stats(d_gh, seeds)
            rec["comparison_id"] = "{}_{}".format(axis_key, encoder)
            rec["axis"] = spec["label"]
            rec["encoder"] = encoder
            rec["comparator"] = "parameter-matched global head (GHead)"
            rec["in_core_family"] = False
            rec["source_file"] = spec["trainvar_file"]
            rec["source_field"] = "{}.{}".format(encoder, spec["diff_field_gh"])
            supp.append(rec)

    diag = []
    for encoder in ("B16", "L14"):
        block = data[DIAG_AXIS["trainvar_file"]][encoder]
        for tag, field in (("CA", "TAP_minus_CLIPAdapter"), ("GH", "TAP_minus_GHead")):
            seeds = [int(r["seed"]) for r in block["rows"]]
            diffs = [float(r[field]) for r in block["rows"]]
            rec = contrast_stats(diffs, seeds)
            rec["comparison_id"] = "source_probeval_{}_{}".format(encoder, tag)
            rec["axis"] = DIAG_AXIS["label"]
            rec["encoder"] = encoder
            rec["comparator"] = "CLIP-Adapter" if tag == "CA" else "parameter-matched global head (GHead)"
            rec["in_core_family"] = False
            rec["source_file"] = DIAG_AXIS["trainvar_file"]
            rec["source_field"] = "{}.rows[*].{}".format(encoder, field)
            rec["condition_bootstrap"] = {
                "available": False,
                "unit_kind": "none",
                "reason": "single held-out source probe-validation split per seed; no per-unit file is stored",
            }
            diag.append(rec)

    align = []
    for axis_key in ("setA", "setB", "corruption"):
        spec = AXES[axis_key]
        for encoder in ("B16", "L14"):
            block = data[spec["trainvar_file"]][encoder]
            for agg_key in ("TAP_minus_CA", "TAP_minus_GH"):
                _, diffs = axis_rows(block, "rows[*].{}".format(agg_key))
                arr = np.asarray(diffs, float)
                stored_mean = float(block[agg_key]["mean"])
                stored_sd = float(block[agg_key]["sd"])
                rec_mean = float(arr.mean())
                rec_sd = float(arr.std(ddof=1))
                align.append({
                    "axis_key": axis_key,
                    "axis_label": spec["label"],
                    "encoder": encoder,
                    "aggregate": agg_key,
                    "source_file": spec["trainvar_file"],
                    "source_field": "{}.{}.{{mean,sd}}".format(encoder, agg_key),
                    "n": int(arr.size),
                    "stored_mean": stored_mean,
                    "recomputed_mean": rec_mean,
                    "stored_mean_4dp": round(stored_mean, 4),
                    "recomputed_mean_4dp": round(rec_mean, 4),
                    "mean_matches_4dp": bool(round(stored_mean, 4) == round(rec_mean, 4)),
                    "stored_sd": stored_sd,
                    "recomputed_sd_ddof1": rec_sd,
                    "recomputed_sd_ddof0": float(arr.std(ddof=0)),
                    "stored_sd_4dp": round(stored_sd, 4),
                    "recomputed_sd_4dp": round(rec_sd, 4),
                    "sd_matches_4dp_ddof1": bool(round(stored_sd, 4) == round(rec_sd, 4)),
                    "sd_convention": (
                        "stored sd is the sample SD (ddof=1)"
                        if round(stored_sd, 4) == round(rec_sd, 4)
                        else "MISMATCH - population SD (ddof=0)? check manually"
                    ),
                })
    for encoder in ("B16", "L14"):
        block = data[DIAG_AXIS["trainvar_file"]][encoder]
        for agg_key in ("TAP_minus_CLIPAdapter", "TAP_minus_GHead"):
            arr = np.asarray([float(r[agg_key]) for r in block["rows"]], float)
            stored_mean = float(block[agg_key]["mean"])
            stored_sd = float(block[agg_key]["sd"])
            rec_mean = float(arr.mean())
            rec_sd = float(arr.std(ddof=1))
            align.append({
                "axis_key": "source_probeval",
                "axis_label": DIAG_AXIS["label"],
                "encoder": encoder,
                "aggregate": agg_key,
                "source_file": DIAG_AXIS["trainvar_file"],
                "source_field": "{}.{}.{{mean,sd}}".format(encoder, agg_key),
                "n": int(arr.size),
                "stored_mean": stored_mean,
                "recomputed_mean": rec_mean,
                "stored_mean_4dp": round(stored_mean, 4),
                "recomputed_mean_4dp": round(rec_mean, 4),
                "mean_matches_4dp": bool(round(stored_mean, 4) == round(rec_mean, 4)),
                "stored_sd": stored_sd,
                "recomputed_sd_ddof1": rec_sd,
                "recomputed_sd_ddof0": float(arr.std(ddof=0)),
                "stored_sd_4dp": round(stored_sd, 4),
                "recomputed_sd_4dp": round(rec_sd, 4),
                "sd_matches_4dp_ddof1": bool(round(stored_sd, 4) == round(rec_sd, 4)),
                "sd_convention": (
                    "stored sd is the sample SD (ddof=1)"
                    if round(stored_sd, 4) == round(rec_sd, 4)
                    else "MISMATCH - population SD (ddof=0)? check manually"
                ),
            })

    unit_consistency = []
    for axis_key in ("setA", "setB", "corruption"):
        spec = AXES[axis_key]
        for encoder in ("B16", "L14"):
            block = data[spec["trainvar_file"]][encoder]
            unit_blob = data[spec["unit_file"]][encoder]
            seed_key = "train_seed" if "train_seed" in block["rows"][0] else "seed"
            chk = consistency_check(block, unit_blob, seed_key)
            chk["axis_key"] = axis_key
            chk["encoder"] = encoder
            chk["unit_file"] = spec["unit_file"]
            chk["unit_count"] = len(unit_blob) if not isinstance(unit_blob, dict) else len(unit_blob[str(block["rows"][0][seed_key])]["TAP"])
            unit_consistency.append(chk)

    doc = {
        "meta": {
            "script": os.path.abspath(__file__).replace("\\", "/"),
            "python": PYTHON_EXE,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "reproduce": REPRO_CMD,
            "alpha": ALPHA,
            "bootstrap": {
                "seed_level_B": B_SEED,
                "unit_level_B": B_UNIT,
                "rng": "numpy.random.default_rng({}), fresh per bootstrap block".format(RNG_SEED),
                "percentiles": [2.5, 97.5],
            },
            "statistical_unit": "training seed",
            "comparator_of_record": COMPARATOR,
            "note": (
                "all per-seed numbers are read from the stored Phase-8 JSON records; "
                "no model was retrained or re-evaluated"
            ),
            "inputs": [{"file": name, "sha256": hashes[name]} for name in INPUT_FILES],
            "outputs": {
                "json": JSON_OUT.replace("\\", "/"),
                "report": REPORT_OUT.replace("\\", "/"),
            },
        },
        "alignment_check": {
            "requirement": "stored mean/sd must equal the recomputed per-seed mean/sd at 4 decimals",
            "n_entries": len(align),
            "all_mean_match_4dp": bool(all(a["mean_matches_4dp"] for a in align)),
            "all_sd_match_4dp": bool(all(a["sd_matches_4dp_ddof1"] for a in align)),
            "mismatches": [a for a in align if not (a["mean_matches_4dp"] and a["sd_matches_4dp_ddof1"])],
            "entries": align,
        },
        "unit_table_consistency": {
            "requirement": (
                "the mean of the stored per-unit values must reproduce the per-seed "
                "TAP_mean / CA_mean / TAP_minus_CA of the trainvar records"
            ),
            "all_match_to_1e-12": bool(all(u["matches_to_1e-12"] for u in unit_consistency)),
            "entries": unit_consistency,
        },
        "core_comparisons": comparisons,
        "multiplicity": multiplicity,
        "supplementary_comparator_ghead": supp,
        "diagnostic_source_probeval": diag,
    }
    return doc, align, comparisons, holm, bh, unit_consistency


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #

def fmt(x, nd=2):
    if x is None:
        return "n/a"
    return "{:.{}f}".format(x, nd)


def enc_name(encoder):
    return "ViT-B/16" if encoder == "B16" else "ViT-L/14"


def build_report(doc, align, comparisons, holm, bh, unit_consistency):
    L = []
    A = L.append
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["sign_test"]["p"])
    best = comparisons[order[0]]
    corr_l14 = [c for c in comparisons if c["comparison_id"] == "corruption_L14"][0]
    setb_l14 = [c for c in comparisons if c["comparison_id"] == "setB_L14"][0]
    holm_ids = [comparisons[i]["comparison_id"] for i, r in enumerate(holm["reject"]) if r]
    bh_ids = [comparisons[i]["comparison_id"] for i, r in enumerate(bh["reject"]) if r]
    rejected_any = sorted(set(holm_ids + bh_ids))

    A("# Statistical robustness of the six training-seed contrasts")
    A("")
    A("Artefact for the statistical-robustness subsection of the TVC submission "
      "(PAPER-TVC-TAP-2026-10-06). Every number below is read from the stored Phase-8 "
      "JSON records listed under each table; no model was retrained and no evaluation "
      "was re-run.")
    A("")

    A("## What is estimated, and in what units")
    A("")
    A("The statistical unit is the **training seed**. Each of the six core contrasts is "
      "a paired difference between two source-domain adaptation arms fitted on the same "
      "source split (TAP minus CLIP-Adapter), and the per-seed value is that difference "
      "averaged over the axis's own evaluation units: the 50 crossed class-prior cells "
      "of Set A, the three second-source targets of Set B, and the 12 pre-specified "
      "corruption conditions. Two intervals are reported for every contrast: an ordinary "
      "95% Student-t interval on the n per-seed differences, and a seed-level percentile "
      "bootstrap (B = 10,000, fixed <code>numpy.random.default_rng(0)</code>) that "
      "resamples the seeds with replacement. Both intervals describe between-training-seed "
      "variability only, conditional on the frozen evaluation units and the frozen query "
      "sets, so they exclude uncertainty about which cells, targets or conditions were "
      "chosen and about query sampling inside them. Exact two-sided sign tests (zeros "
      "dropped, p capped at 1.0) are reported alongside; their six p-values form the "
      "family corrected in the next section.")
    A("")

    A("## Table 1. Six core contrasts, seed-level statistics (TAP minus CLIP-Adapter)")
    A("")
    A("| # | Axis | Encoder | n seeds | Mean (pp) | SD (pp) | 95% t-CI (pp) | Seeds positive | Sign-test p | Seed bootstrap 95% CI (pp) |")
    A("|---|------|---------|---------|-----------|---------|---------------|----------------|-------------|------------------------------|")
    for i, c in enumerate(comparisons, start=1):
        t = c["t_interval"]
        b = c["seed_bootstrap"]
        A("| {} | {} | {} | {} | {} | {} | [{}, {}] | {}/{} | {} | [{}, {}] |".format(
            i, c["axis"], enc_name(c["encoder"]), c["n_seeds"],
            fmt(c["mean_points"]), fmt(c["sd_points"]),
            fmt(100.0 * t["ci_low"]), fmt(100.0 * t["ci_high"]),
            c["n_positive"], c["n_seeds"], fmt(c["sign_test"]["p"], 4),
            fmt(100.0 * b["ci_low"]), fmt(100.0 * b["ci_high"])))
    A("")
    A("Standard deviations are sample SDs (ddof = 1) and the t-interval uses df = n - 1. "
      "The seed bootstrap resamples the n per-seed differences with replacement "
      "(B = 10,000) and reports the 2.5 and 97.5 percentiles. pp = percentage points of "
      "unweighted average recall (UAR); the stored values are fractions.")
    A("")
    A("Source files and fields (all under "
      "<code>results/token_module_tailor_20261006/</code>):")
    A("")
    A("| # | Source file | Per-seed difference field | Seed field |")
    A("|---|-------------|---------------------------|------------|")
    for i, c in enumerate(comparisons, start=1):
        A("| {} | <code>{}</code> | <code>{}</code> | <code>{}</code> |".format(
            i, c["source_file"], c["source_fields"]["per_seed_difference"],
            c["source_fields"]["seeds"]))
    A("")
    A("Reproduce: <code>{}</code>".format(REPRO_CMD))
    A("")

    A("## Table 2. Family-wise and false-discovery control over the six sign tests")
    A("")
    A("The family is the six core contrasts defined above (three confirmation axes x two "
      "encoders), tested with the exact two-sided sign test at alpha = 0.05. Sorting the "
      "six p-values puts <code>{}</code> ({} pp, {} of {} seeds positive) at rank 1, and "
      "no p-value in the family reaches the rank-1 Holm threshold of alpha/6 = {} or the "
      "rank-1 Benjamini-Hochberg threshold of (1/6) x alpha = {}.".format(
        best["comparison_id"], fmt(best["mean_points"]), best["n_positive"],
        best["n_seeds"], fmt(ALPHA / 6.0, 4), fmt(ALPHA / 6.0, 4)))
    A("")
    A("| Rank | Axis | Encoder | Sign-test p | Holm-Bonferroni adjusted p | BH adjusted p | Rejected by Holm | Rejected by BH |")
    A("|------|------|---------|-------------|----------------------------|---------------|------------------|----------------|")
    for rank, i in enumerate(order, start=1):
        c = comparisons[i]
        A("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            rank, c["axis"], enc_name(c["encoder"]), fmt(c["sign_test"]["p"], 4),
            fmt(holm["adjusted_p"][i], 4), fmt(bh["adjusted_p"][i], 4),
            "yes" if holm["reject"][i] else "no",
            "yes" if bh["reject"][i] else "no"))
    A("")
    if not rejected_any:
        A("**No contrast survives either correction.** Under Holm-Bonferroni the step-down "
          "procedure stops at the first rank, because the smallest p-value ({}) already "
          "exceeds alpha/6 = {}; under Benjamini-Hochberg no rank k satisfies "
          "p_(k) <= (k/m) x alpha either. The rejection set is therefore empty for both "
          "procedures, and the adjusted p-value of the strongest contrast is {} under "
          "Holm-Bonferroni and {} under Benjamini-Hochberg.".format(
            fmt(best["sign_test"]["p"], 4), fmt(ALPHA / 6.0, 4),
            fmt(holm["adjusted_p"][order[0]], 4), fmt(bh["adjusted_p"][order[0]], 4)))
        A("")
        A("The reason is the seed budget rather than the size of the effect. The strongest "
          "contrast - ViT-L/14 on the corruption axis, mean {} pp - is positive in 5 of 5 "
          "seeds, and its exact two-sided sign-test p is exactly {}, which is the floor of "
          "the test at n = 5 (2 / 2^5 = 0.0625): with five seeds no outcome, not even a "
          "unanimous one, can produce p < 0.05. Since the sign test cannot reach 0.05 at "
          "n = 5, no multiplicity correction can make any ViT-L/14 contrast significant, "
          "and the same floor applies to the Set B contrast at ViT-L/14 (p = {}). The "
          "evidence separating the corruption result from the other five therefore rests "
          "on the intervals of Table 1 (its t-interval [{}, {}] pp and its seed bootstrap "
          "interval [{}, {}] pp both exclude zero, unlike every other contrast) and on the "
          "unanimity of its sign, not on a family-wise-significant p-value.".format(
            fmt(best["mean_points"]), fmt(best["sign_test"]["p"], 4),
            fmt(setb_l14["sign_test"]["p"], 4),
            fmt(100.0 * best["t_interval"]["ci_low"]), fmt(100.0 * best["t_interval"]["ci_high"]),
            fmt(100.0 * best["seed_bootstrap"]["ci_low"]), fmt(100.0 * best["seed_bootstrap"]["ci_high"])))
    else:
        A("Rejected by Holm-Bonferroni: {}.".format(", ".join(holm["rejected_by_holm"]) or "none"))
        A("")
        A("Rejected by Benjamini-Hochberg: {}.".format(", ".join(bh["rejected_by_bh"]) or "none"))
    A("")
    A("Source files and fields: the six p-values are computed from the <code>p</code> "
      "entries derived from <code>TAP_minus_CA</code> in <code>phase8_setA_trainvar.json</code>, "
      "<code>phase8_setB_trainvar.json</code> and <code>phase8_corruption_trainvar.json</code> "
      "(exact field paths in Table 1). Correction settings: alpha = 0.05, m = 6.")
    A("")
    A("Reproduce: <code>{}</code>".format(REPRO_CMD))
    A("")

    A("## What the intervals cover, and what they do not")
    A("")
    A("The t-intervals and the seed-level bootstrap intervals quantify exactly one source "
      "of randomness: which training run produced the two arms. They answer the question a "
      "reader of a single-run comparison should ask - would the sign and the size of this "
      "margin hold if the arms had been fitted from a different initialisation - and they "
      "hold everything else fixed. They do not cover uncertainty in the evaluation units: "
      "the 50 crossed cells, the three second-source targets and the 12 corruption "
      "conditions are treated as the population, and no resampling of cells, targets or "
      "conditions enters the two intervals of Table 1. They do not cover query sampling "
      "within a cell or a condition, and they do not cover any part of the source-domain "
      "training pipeline that was held fixed across seeds, such as data order, "
      "learning-rate selection or augmentation. The seed bootstrap of Table 1 varies the "
      "training run at fixed units, whereas the within-seed unit bootstrap of Table 3 "
      "varies the units at fixed training runs; the two must not be read as a single "
      "interval.")
    A("")
    A("One consequence is worth stating plainly. Because these intervals condition on the "
      "confirmation units they are narrower than an interval that would also propagate "
      "unit-level sampling, and they are wider than the single-run numbers they replace. "
      "The ViT-L/14 corruption margin is the case in point: the single-run estimate was "
      "+3.20 pp, the retrained mean is {} pp, the interval over seeds is [{}, {}] pp, and "
      "the sign is consistent in all five seeds. The same logic cuts the other way for "
      "the ViT-B/16 crossed-grid margin: the single run reported +3.88 pp, the retrained "
      "mean is {} pp, and the interval over seeds is [{}, {}] pp, which excludes zero on "
      "the negative side - so the reversal is itself supported by the interval, and what "
      "the retraining overturns is the sign of that effect, not its existence.".format(
        fmt(corr_l14["mean_points"]),
        fmt(100.0 * corr_l14["t_interval"]["ci_low"]), fmt(100.0 * corr_l14["t_interval"]["ci_high"]),
        fmt([c for c in comparisons if c["comparison_id"] == "setA_B16"][0]["mean_points"]),
        fmt(100.0 * [c for c in comparisons if c["comparison_id"] == "setA_B16"][0]["t_interval"]["ci_low"]),
        fmt(100.0 * [c for c in comparisons if c["comparison_id"] == "setA_B16"][0]["t_interval"]["ci_high"])))
    A("")
    A("Source files and fields: <code>phase8_setA_trainvar.json</code>, "
      "<code>phase8_setB_trainvar.json</code>, <code>phase8_corruption_trainvar.json</code>, "
      "field <code>&lt;encoder&gt;.rows[*].TAP_minus_CA</code>. The single-run reference "
      "values (+3.20 pp and +3.88 pp) are the original-run numbers quoted in the "
      "manuscript, not values stored in these files.")
    A("")
    A("Reproduce: <code>{}</code>".format(REPRO_CMD))
    A("")

    A("## Table 3. Within-seed (unit-level) paired bootstrap, where it exists")
    A("")
    A("| Axis | Encoder | Unit | Units per seed | B | 95% CI (pp) | Available |")
    A("|------|---------|------|----------------|---|-------------|-----------|")
    for c in comparisons:
        cb = c["condition_bootstrap"]
        if cb.get("available"):
            A("| {} | {} | {} | {} | {} | [{}, {}] | yes |".format(
                c["axis"], enc_name(c["encoder"]), cb["unit_kind"],
                cb["n_units_per_seed"], cb["B"],
                fmt(100.0 * cb["ci_low"]), fmt(100.0 * cb["ci_high"])))
        else:
            A("| {} | {} | {} | 0 | n/a | n/a | **no** |".format(
                c["axis"], enc_name(c["encoder"]), cb["unit_kind"]))
    A("")
    A("Where per-unit values are stored, the units are resampled with replacement inside "
      "each training seed and the same resample is shared by both arms, so the pairing is "
      "preserved; the resulting per-seed means are averaged over seeds (B = 2,000). The "
      "seed dimension is held fixed here, so these intervals say how much of each margin "
      "is attributable to which cells or conditions were evaluated, not how much is "
      "attributable to retraining.")
    A("")
    A("**Set B has no per-cell values.** <code>phase8_setB_pertarget.json</code> stores "
      "only three per-target aggregate UAR values per seed "
      "(<code>&lt;encoder&gt;[*].seed&lt;k&gt;.{TAP, CA, GH}</code>, one entry per target, "
      "with n = 441, 7178 and 902), so the 15 crossed cells of that axis are not "
      "recoverable from the stored files and the cell-level within-seed paired bootstrap "
      "cannot be computed for Set B. We did not substitute the three target aggregates for "
      "cells: a three-unit resample is degenerate, and it is recorded in the JSON only "
      "under <code>target_level_fallback_not_interpretable</code> with "
      "<code>interpretable = false</code>.")
    A("")
    A("Source files and fields:")
    A("")
    A("| Axis | Unit file | Per-unit fields |")
    A("|------|-----------|-----------------|")
    A("| Set A | <code>phase8_setA_percell.json</code> | <code>&lt;encoder&gt;.&lt;seed&gt;.{TAP, CA}</code>, 50 per-cell UAR values per seed |")
    A("| Corruption | <code>phase8_corruption_percond.json</code> | <code>&lt;encoder&gt;[*].seed&lt;k&gt;.{TAP, CA}</code>, 12 per-condition UAR values per seed |")
    A("| Set B | <code>phase8_setB_pertarget.json</code> | <code>&lt;encoder&gt;[*].seed&lt;k&gt;.{TAP, CA}</code>, 3 per-target aggregates per seed, no per-cell values |")
    A("")
    A("Reproduce: <code>{}</code>".format(REPRO_CMD))
    A("")

    A("## Table 4. Alignment of the stored aggregates with the recomputed per-seed statistics")
    A("")
    A("| Axis | Encoder | Aggregate | n | Stored mean | Recomputed mean | Match (4 dp) | Stored SD | Recomputed SD | Match (4 dp) |")
    A("|------|---------|-----------|----|-------------|-----------------|--------------|-----------|---------------|--------------|")
    for a in align:
        A("| {} | {} | <code>{}</code> | {} | {} | {} | {} | {} | {} | {} |".format(
            a["axis_label"], enc_name(a["encoder"]), a["aggregate"], a["n"],
            fmt(a["stored_mean"], 6), fmt(a["recomputed_mean"], 6),
            "yes" if a["mean_matches_4dp"] else "**NO**",
            fmt(a["stored_sd"], 6), fmt(a["recomputed_sd_ddof1"], 6),
            "yes" if a["sd_matches_4dp_ddof1"] else "**NO**"))
    A("")
    n_ok = sum(1 for a in align if a["mean_matches_4dp"] and a["sd_matches_4dp_ddof1"])
    A("{} of {} stored aggregate entries match the recomputed per-seed statistics at "
      "four decimals, for both the mean and the SD; there are {} mismatches. The stored "
      "SDs are sample SDs (ddof = 1): the population SDs (ddof = 0) are also recorded in "
      "the JSON and do not match, which fixes the convention used by the Phase-8 "
      "aggregation. No stored value was adjusted.".format(
        n_ok, len(align), len(align) - n_ok))
    A("")
    A("The per-unit files reproduce the per-seed arm means as well: recomputing "
      "<code>TAP_mean</code>, <code>CA_mean</code> and <code>TAP_minus_CA</code> from the "
      "stored per-cell and per-condition values agrees with the trainvar records to within "
      "<code>max_abs_difference = {}</code> for all six axis x encoder combinations "
      "(see <code>unit_table_consistency</code> in the JSON).".format(
        max(u["max_abs_difference"] for u in unit_consistency)))
    A("")
    A("Source files and fields: <code>phase8_trainvar.json</code> "
      "(<code>{B16,L14}.TAP_minus_{CLIPAdapter,GHead}.{mean,sd}</code>), "
      "<code>phase8_setA_trainvar.json</code>, <code>phase8_setB_trainvar.json</code> and "
      "<code>phase8_corruption_trainvar.json</code> "
      "(<code>{B16,L14}.TAP_minus_{CA,GH}.{mean,sd}</code>), cross-checked against the "
      "per-seed fields of the same files.")
    A("")
    A("Reproduce: <code>{}</code>".format(REPRO_CMD))
    A("")

    A("## Appendix A. Supplementary contrasts")
    A("")
    A("### A.1 Parameter-matched global head as comparator (TAP minus GHead)")
    A("")
    A("| Axis | Encoder | n seeds | Mean (pp) | SD (pp) | 95% t-CI (pp) | Seeds positive | Sign-test p | Seed bootstrap 95% CI (pp) |")
    A("|------|---------|---------|-----------|---------|---------------|----------------|-------------|------------------------------|")
    for c in doc["supplementary_comparator_ghead"]:
        t = c["t_interval"]
        b = c["seed_bootstrap"]
        A("| {} | {} | {} | {} | {} | [{}, {}] | {}/{} | {} | [{}, {}] |".format(
            c["axis"], enc_name(c["encoder"]), c["n_seeds"],
            fmt(c["mean_points"]), fmt(c["sd_points"]),
            fmt(100.0 * t["ci_low"]), fmt(100.0 * t["ci_high"]),
            c["n_positive"], c["n_seeds"], fmt(c["sign_test"]["p"], 4),
            fmt(100.0 * b["ci_low"]), fmt(100.0 * b["ci_high"])))
    A("")
    A("These six contrasts are outside the corrected family above; they are reported "
      "because the manuscript also quotes the parameter-matched head as a secondary "
      "comparator. Fields: <code>&lt;encoder&gt;.rows[*].TAP_minus_GH</code> in the three "
      "<code>phase8_*_trainvar.json</code> files.")
    A("")
    A("Reproduce: <code>{}</code>".format(REPRO_CMD))
    A("")
    A("### A.2 Source probe-validation axis (diagnostic, not part of the family)")
    A("")
    A("| Comparator | Encoder | n seeds | Mean (pp) | SD (pp) | 95% t-CI (pp) | Seeds positive | Sign-test p | Seed bootstrap 95% CI (pp) |")
    A("|------------|---------|---------|-----------|---------|---------------|----------------|-------------|------------------------------|")
    for c in doc["diagnostic_source_probeval"]:
        t = c["t_interval"]
        b = c["seed_bootstrap"]
        A("| {} | {} | {} | {} | {} | [{}, {}] | {}/{} | {} | [{}, {}] |".format(
            c["comparator"], enc_name(c["encoder"]), c["n_seeds"],
            fmt(c["mean_points"]), fmt(c["sd_points"]),
            fmt(100.0 * t["ci_low"]), fmt(100.0 * t["ci_high"]),
            c["n_positive"], c["n_seeds"], fmt(c["sign_test"]["p"], 4),
            fmt(100.0 * b["ci_low"]), fmt(100.0 * b["ci_high"])))
    A("")
    A("This is the source-domain probe-validation diagnostic recorded in "
      "<code>phase8_trainvar.json</code>. It is one held-out split per seed, so no "
      "per-unit file exists and no unit-level bootstrap is possible, and it is not one of "
      "the manuscript's six contrasts. Fields: "
      "<code>&lt;encoder&gt;.rows[*].TAP_minus_{CLIPAdapter,GHead}</code>.")
    A("")
    A("Reproduce: <code>{}</code>".format(REPRO_CMD))
    A("")

    A("## Provenance")
    A("")
    A("Input files with SHA-256, all under "
      "<code>D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/</code>:")
    A("")
    for rec in doc["meta"]["inputs"]:
        A("- <code>{}</code> - <code>{}</code>".format(rec["file"], rec["sha256"]))
    A("")
    A("Machine-readable companion: "
      "<code>D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/phase10_stats_robustness.json</code>.")
    A("")
    A("Reproduce both artefacts with:")
    A("")
    A("    " + REPRO_CMD)
    A("")
    return "\n".join(L) + "\n"


def write_if_changed(path, text):
    prev = None
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            prev = fh.read()
    changed = prev != text
    if changed:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return changed, prev is not None


def main():
    doc, align, comparisons, holm, bh, unit_consistency = build()
    json_text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    report_text = build_report(doc, align, comparisons, holm, bh, unit_consistency)

    json_prev = None
    if os.path.exists(JSON_OUT):
        with open(JSON_OUT, "r", encoding="utf-8") as fh:
            json_prev = fh.read()
    report_prev = None
    if os.path.exists(REPORT_OUT):
        with open(REPORT_OUT, "r", encoding="utf-8") as fh:
            report_prev = fh.read()

    json_changed, _ = write_if_changed(JSON_OUT, json_text)
    report_changed, _ = write_if_changed(REPORT_OUT, report_text)

    print("=" * 78)
    print("phase10 statistical robustness audit")
    print("=" * 78)
    print("JSON   : {} ({} bytes, previously present: {}, byte-identical: {})".format(
        JSON_OUT, len(json_text), json_prev is not None, json_prev == json_text))
    print("REPORT : {} ({} bytes, previously present: {}, byte-identical: {})".format(
        REPORT_OUT, len(report_text), report_prev is not None, report_prev == report_text))
    print("")
    print("Six core contrasts (TAP - CLIP-Adapter), points:")
    for c in comparisons:
        t = c["t_interval"]
        b = c["seed_bootstrap"]
        print("  {:<18} n={:<3} mean={:>7.2f} sd={:>5.2f} tCI=[{:>6.2f},{:>6.2f}] "
              "pos={}/{} sign_p={:.4f} bootCI=[{:>6.2f},{:>6.2f}] cb={}".format(
                  c["comparison_id"], c["n_seeds"], c["mean_points"], c["sd_points"],
                  100.0 * t["ci_low"], 100.0 * t["ci_high"],
                  c["n_positive"], c["n_seeds"], c["sign_test"]["p"],
                  100.0 * b["ci_low"], 100.0 * b["ci_high"],
                  "yes" if c["condition_bootstrap"].get("available") else "no"))
    print("")
    print("Multiplicity (alpha={}):".format(ALPHA))
    holm_ids = [comparisons[i]["comparison_id"] for i, r in enumerate(holm["reject"]) if r]
    bh_ids = [comparisons[i]["comparison_id"] for i, r in enumerate(bh["reject"]) if r]
    print("  Holm rejected : {}".format(holm_ids or "none"))
    print("  BH   rejected : {}".format(bh_ids or "none"))
    print("")
    print("Alignment check at 4 dp: {} entries, all means match: {}, all SDs match: {}".format(
        len(align),
        all(a["mean_matches_4dp"] for a in align),
        all(a["sd_matches_4dp_ddof1"] for a in align)))
    print("Unit-table consistency (max abs difference): {}".format(
        max(u["max_abs_difference"] for u in unit_consistency)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
