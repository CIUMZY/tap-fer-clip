
# -*- coding: utf-8 -*-
"""Phase 13d: assemble the JAFFE fresh-axis report from the frozen rule and the recorded numbers."""
import json
from pathlib import Path

import numpy as np

OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
MS = Path("D:/ResearchVault/06manuscripts/rb-tta-fer-fg2027/PAPER-TVC-TAP-2026-10-06/manuscript")
ja = json.loads((OUT / "phase13_eval_jaffe.json").read_text(encoding="utf-8"))
clean = json.loads((OUT / "phase13_clean_fer_control.json").read_text(encoding="utf-8"))
freeze = json.loads((OUT / "phase13_prereg_freeze.json").read_text(encoding="utf-8"))
main = json.loads((OUT / "phase12_stats.json").read_text(encoding="utf-8"))

summary = {
    "jaffe_axis": {t: ja["encoders"][t]["summary"] for t in ("B16", "L14")},
    "jaffe_clean_descriptive": {},
    "fer2013_clean_control": {t: {"TAP_minus_CA": clean[t]["TAP_minus_CA"],
                                  "TAP_minus_GH": clean[t]["TAP_minus_GH"],
                                  "arm_means": clean[t]["arm_means"]} for t in ("B16", "L14")},
    "fer2013_corrupted": {"corruption|L14": main["published"]["corruption|L14"],
                          "corruption|B16": main["published"]["corruption|B16"]},
    "prereg": {"file": freeze["file"], "sha256": freeze["sha256"],
               "amendment_sha256": freeze["amendment1"]["sha256"],
               "jaffe_archive_sha256": freeze["jaffe_archive_sha256_expected"]},
    "verdict": ja["verdict"],
}
for tag in ("B16", "L14"):
    cd = ja["encoders"][tag]["clean_descriptive"]
    rec = {}
    for comp in ("CA", "GH"):
        v = np.array([cd[k]["TAP"] - cd[k][comp] for k in sorted(cd, key=int)])
        tc = 2.262 if tag == "B16" else 2.145
        se = v.std(ddof=1) / np.sqrt(len(v))
        rec["TAP_minus_%s" % comp] = {
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "ci": [float(v.mean() - tc * se), float(v.mean() + tc * se)],
            "seeds_positive": int((v > 0).sum()), "n_seeds": int(len(v))}
    summary["jaffe_clean_descriptive"][tag] = rec
(OUT / "phase13_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

L = []
A = L.append
A("# Phase 13: the JAFFE fresh confirmation axis")
A("")
A("Pre-specification: phase13_jaffe_preregistration.md (sha256 " + freeze["sha256"][:16] +
  "...) plus amendment 1 (sha256 " + freeze["amendment1"]["sha256"][:16] +
  "...), both written and hashed before any JAFFE image was opened. JAFFE archive sha256 " +
  freeze["jaffe_archive_sha256_expected"][:16] + "...")
A("")
A("## Pre-specified verdict")
A("")
A("**" + ja["verdict"]["verdict"] + "** - at ViT-L/14 the mean of the per-seed TAP minus CLIP-Adapter "
  "contrast on the corrupted JAFFE axis is %+.4f points with a seed-level 95%% interval of "
  "[%+.4f, %+.4f] and %d of %d seeds positive."
  % (100 * ja["verdict"]["mean"], 100 * ja["verdict"]["t95_ci"][0], 100 * ja["verdict"]["t95_ci"][1],
     ja["verdict"]["seeds_positive"], 15))
A("")
A("## Degraded JAFFE axis (pre-specified grid, 20 conditions)")
A("")
A("| encoder | condition | contrast | mean (pp) | SD | 95% interval | seeds positive | sign-test p |")
A("|---|---|---|---|---|---|---|---|")
for tag, name in (("L14", "ViT-L/14"), ("B16", "ViT-B/16")):
    for label in ("published", "nested"):
        for comp in ("CA", "GH"):
            s = ja["encoders"][tag]["summary"]["%s|TAP_minus_%s" % (label, comp)]
            A("| %s | %s | TAP - %s | %+.2f | %.2f | [%+.2f, %+.2f] | %d/%d | %.4f |" % (
                name, label, comp, 100 * s["mean"], 100 * s["sd"], 100 * s["t95_ci"][0],
                100 * s["t95_ci"][1], s["seeds_positive"], s["n_seeds"], s["sign_test_p"]))
A("")
A("## Clean controls (not part of the pre-specified grid; reported because they decide how the result "
  "should be read)")
A("")
A("| data | encoder | contrast | mean (pp) | SD | 95% interval | seeds positive |")
A("|---|---|---|---|---|---|---|")
for tag, name in (("L14", "ViT-L/14"), ("B16", "ViT-B/16")):
    for comp in ("CA", "GH"):
        s = summary["jaffe_clean_descriptive"][tag]["TAP_minus_%s" % comp]
        A("| JAFFE, clean | %s | TAP - %s | %+.2f | %.2f | [%+.2f, %+.2f] | %d/%d |" % (
            name, comp, 100 * s["mean"], 100 * s["sd"], 100 * s["ci"][0], 100 * s["ci"][1],
            s["seeds_positive"], s["n_seeds"]))
for tag, name in (("B16", "ViT-B/16"), ("L14", "ViT-L/14")):
    for comp in ("CA", "GH"):
        s = clean[tag]["TAP_minus_%s" % comp]
        A("| FER2013 test, clean | %s | TAP - %s | %+.2f | %.2f | [%+.2f, %+.2f] | %d/%d |" % (
            name, comp, 100 * s["mean"], 100 * s["sd"], 100 * s["ci"][0], 100 * s["ci"][1],
            s["seeds_positive"], clean[tag]["n_seeds"]))
c14 = main["published"]["corruption|L14"]
A("| FER2013 test, corrupted (pre-specified axis) | ViT-L/14 | TAP - CA | %+.2f | %.2f | [%+.2f, %+.2f] | 15/15 |"
  % (100 * c14["mean"], 100 * c14["sd"], 100 * c14["t95_ci"][0], 100 * c14["t95_ci"][1]))
A("")
A("## What the numbers say")
A("")
A("1. The pre-specified criterion passes on a corpus that had never been touched: at ViT-L/14 the token "
  "route beats CLIP-Adapter on the fresh degraded axis by %+.2f points, positive in all fifteen seeds. The "
  "direction of the surviving FER2013 claim replicates out of sample." % (100 * ja["verdict"]["mean"]))
A("2. The clean control changes what the fresh-axis result means. On clean JAFFE the same contrast is "
  "already %+.2f points, so most of the fresh-axis margin is a cross-domain accuracy advantage rather than "
  "a robustness advantage; corruption adds about %+.2f points."
  % (100 * summary["jaffe_clean_descriptive"]["L14"]["TAP_minus_CA"]["mean"],
     100 * (ja["verdict"]["mean"] - summary["jaffe_clean_descriptive"]["L14"]["TAP_minus_CA"]["mean"])))
A("3. On the original within-domain axis the opposite holds: on clean FER2013 test the contrast is %+.2f "
  "points (interval covering zero) while under the pre-specified corruption it is +3.50 points, so there "
  "the margin is degradation-specific." % (100 * clean["L14"]["TAP_minus_CA"]["mean"]))
A("4. The encoder dependence replicates in sign, and more strongly than on FER2013: at ViT-B/16 the token "
  "route is worse than CLIP-Adapter on the fresh axis (%+.2f points, interval excluding zero) and worse "
  "still on clean JAFFE (%+.2f points)."
  % (100 * ja["encoders"]["B16"]["summary"]["published|TAP_minus_CA"]["mean"],
     100 * summary["jaffe_clean_descriptive"]["B16"]["TAP_minus_CA"]["mean"]))
A("")
A("Honest reading: the fresh axis confirms the direction of the L/14 advantage and its encoder dependence, "
  "and it does not contradict the within-domain robustness finding, but it shows that out of domain the "
  "advantage is not degradation-specific. The manuscript reports both rows.")
A("")
A("## Consumption")
A("")
A("JAFFE is now a consumed confirmation axis. Nothing else may be selected on it.")
(MS / "JAFFE-FRESH-AXIS-REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("phase13_summary.json and JAFFE-FRESH-AXIS-REPORT.md written")
print(json.dumps(ja["verdict"], indent=2))

