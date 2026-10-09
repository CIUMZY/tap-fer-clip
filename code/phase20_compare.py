"""Phase 20: compare the prospective batch-2 seed distribution with batch 1.

Implements the criterion frozen in
manuscript/PHASE20-PREREGISTRATION-2026-10-09.md (sha256 1f0554921efe8c3d...):
for the ViT-L/14 corruption axis at the published rate, contrast TAP - CLIP-Adapter,
  H1 the batch-2 mean is positive;
  H2 the batch-2 seed-level 95% t interval excludes zero;
  H3 (agreement only) the batch-2 mean is within one pooled SD of the batch-1 mean.
Batch 2 replicates if and only if H1 and H2 hold.
"""
import json
import math
import statistics
from pathlib import Path

OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
T95 = {9: 2.262157, 14: 2.144787, 29: 2.045230, 19: 2.093024}
AXES = ["corruption", "setA", "setB"]
PAIRS = ["CA", "GH"]


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def margins(rec, axis, label, comp):
    src = rec["per_seed"][axis][label]
    seeds = sorted(int(s) for s in src)
    return seeds, [float(src[str(s)]["TAP"]) - float(src[str(s)][comp]) for s in seeds]


def stats(v, tcrit):
    n = len(v)
    mean = statistics.fmean(v)
    sd = statistics.stdev(v) if n > 1 else 0.0
    half = tcrit * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean_pp": round(mean * 100, 3),
        "sd_pp": round(sd * 100, 3),
        "ci95_pp": [round((mean - half) * 100, 3), round((mean + half) * 100, 3)],
        "positive": sum(1 for x in v if x > 0),
        "excludes_zero": (mean - half) > 0 or (mean + half) < 0,
    }


b1 = load("phase12_eval.json")
b2 = load("phase20_eval.json")
report = {"criterion": "H1 mean>0 and H2 interval excludes zero", "encoder": {}}

for tag in ("L14", "B16"):
    report["encoder"][tag] = {}
    for axis in AXES:
        for label in ("published", "nested"):
            for comp in PAIRS:
                key = axis + "|" + label + "|TAP_minus_" + comp
                s1, v1 = margins(b1[tag], axis, label, comp)
                s2, v2 = margins(b2[tag], axis, label, comp)
                pooled = v1 + v2
                a = stats(v1, T95[len(v1) - 1])
                b = stats(v2, T95[len(v2) - 1])
                p = stats(pooled, T95[len(pooled) - 1])
                agree = abs(b["mean_pp"] - a["mean_pp"]) <= (a["sd_pp"] + b["sd_pp"]) / 2
                report["encoder"][tag][key] = {
                    "batch1": dict(a, seeds=s1),
                    "batch2": dict(b, seeds=s2),
                    "pooled": dict(p, seeds=s1 + s2),
                    "H1_mean_positive": b["mean_pp"] > 0,
                    "H2_interval_excludes_zero": b["excludes_zero"],
                    "H3_within_one_pooled_sd": bool(agree),
                    "batch2_replicates": bool(b["mean_pp"] > 0 and b["excludes_zero"]),
                }
                print("%-4s %-11s %-9s TAP-%-3s  b1 %+6.2f [%+6.2f,%+6.2f] n=%-3d | b2 %+6.2f [%+6.2f,%+6.2f] n=%-3d | pooled %+6.2f [%+6.2f,%+6.2f]  %s"
                      % (tag, axis, label, comp,
                         a["mean_pp"], a["ci95_pp"][0], a["ci95_pp"][1], a["n"],
                         b["mean_pp"], b["ci95_pp"][0], b["ci95_pp"][1], b["n"],
                         p["mean_pp"], p["ci95_pp"][0], p["ci95_pp"][1],
                         "REPLICATES" if report["encoder"][tag][key]["batch2_replicates"] else "-"))

primary = report["encoder"]["L14"]["corruption|published|TAP_minus_CA"]
report["primary_verdict"] = {
    "contrast": "L14 corruption published TAP_minus_CA",
    "H1": primary["H1_mean_positive"],
    "H2": primary["H2_interval_excludes_zero"],
    "H3": primary["H3_within_one_pooled_sd"],
    "batch2_replicates": primary["batch2_replicates"],
}
(OUT / "phase20_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\nPRIMARY:", json.dumps(report["primary_verdict"]))
print("written", OUT / "phase20_comparison.json")
