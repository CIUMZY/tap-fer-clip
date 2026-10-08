
# -*- coding: utf-8 -*-
"""Phase 15: leave-one-seed-out sensitivity of every core contrast.

Question: does any verdict rest on a single influential training seed? For each contrast we recompute
the mean and the seed-level t interval with each seed left out in turn. Nothing is selected here; this
is a stability check on an already-fixed analysis.
"""
import json, math
from pathlib import Path

import numpy as np

OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
ev = json.loads((OUT / "phase12_eval.json").read_text(encoding="utf-8"))
ja = json.loads((OUT / "phase13_eval_jaffe.json").read_text(encoding="utf-8"))
st = json.loads((OUT / "phase12_stats.json").read_text(encoding="utf-8"))
SINGLE = {("setA", "B16"): 0.0388, ("setA", "L14"): -0.0196, ("setB", "B16"): 0.0347,
          ("setB", "L14"): -0.0137, ("corruption", "B16"): 0.0181, ("corruption", "L14"): 0.0320}
TC = {9: 2.262, 14: 2.145}


def loo(v):
    v = np.asarray(v, float)
    out = []
    for i in range(len(v)):
        w = np.delete(v, i)
        se = w.std(ddof=1) / math.sqrt(len(w))
        tc = TC.get(len(w) - 1, 2.145)
        out.append((float(w.mean()), float(w.mean() - tc * se), float(w.mean() + tc * se)))
    return out


rows = []
for axis in ("setA", "setB", "corruption"):
    for tag in ("B16", "L14"):
        src = ev[tag]["per_seed"][axis]["published"]
        v = [src[k]["TAP"] - src[k]["CA"] for k in sorted(src, key=int)]
        r = loo(v)
        means = [x[0] for x in r]
        excl = [1 if (x[1] > 0 or x[2] < 0) else 0 for x in r]
        rows.append({"contrast": "%s|%s" % (axis, tag), "n": len(v),
                     "full_mean": float(np.mean(v)),
                     "loo_mean_min": min(means), "loo_mean_max": max(means),
                     "loo_sign_stable": bool(all(np.sign(m) == np.sign(np.mean(v)) for m in means)),
                     "loo_ci_excludes_zero_count": sum(excl),
                     "full_ci_excludes_zero": bool(st["published"]["%s|%s" % (axis, tag)]["t95_ci"][0] > 0
                                                   or st["published"]["%s|%s" % (axis, tag)]["t95_ci"][1] < 0)})
# JAFFE fresh axis
for tag in ("B16", "L14"):
    src = ja["encoders"][tag]["per_seed"]["published"]
    v = [src[k]["TAP"] - src[k]["CA"] for k in sorted(src, key=int)]
    r = loo(v)
    means = [x[0] for x in r]
    excl = [1 if (x[1] > 0 or x[2] < 0) else 0 for x in r]
    rows.append({"contrast": "jaffe|%s" % tag, "n": len(v), "full_mean": float(np.mean(v)),
                 "loo_mean_min": min(means), "loo_mean_max": max(means),
                 "loo_sign_stable": bool(all(np.sign(m) == np.sign(np.mean(v)) for m in means)),
                 "loo_ci_excludes_zero_count": sum(excl),
                 "full_ci_excludes_zero": True})

rep = {"rows": rows, "note": "leave-one-seed-out recomputation of the per-seed mean and t interval; "
                             "the verdict of a contrast is reported as unstable if any leave-one-out "
                             "interval changes whether zero is excluded"}
(OUT / "phase15_leave_one_out.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
for r in rows:
    print("%-18s n=%2d mean %+.4f  loo mean [%+.4f, %+.4f]  sign stable %s  "
          "loo intervals excluding zero %d/%d  (full: %s)"
          % (r["contrast"], r["n"], r["full_mean"], r["loo_mean_min"], r["loo_mean_max"],
             r["loo_sign_stable"], r["loo_ci_excludes_zero_count"], r["n"],
             r["full_ci_excludes_zero"]))

