
"""Phase 12d: emit the manuscript tables from the seed-controlled retraining records."""
import json
from pathlib import Path

OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
ev = json.loads((OUT / "phase12_eval.json").read_text(encoding="utf-8"))
st = json.loads((OUT / "phase12_stats.json").read_text(encoding="utf-8"))
tr = {t: json.loads((OUT / ("phase12_train_%s.json" % t)).read_text(encoding="utf-8")) for t in ("B16", "L14")}

AX = [("setA", "Set A (crossed grid)"), ("setB", "Set B (second source)"), ("corruption", "corruption axis")]
KEYS = ["%s|%s" % (a, e) for a, _ in AX for e in ("B16", "L14")]

print("### T2 probe-val")
print("| encoder | seed | TAP | CLIP-Adapter | matched head | nested TAP | nested CA | nested GH |")
print("|---|---|---|---|---|---|---|---|")
for tag in ("B16", "L14"):
    pub = tr[tag]["published"]
    for r in tr[tag]["rows"]:
        s = r["seed"]
        nc = ev[tag]["nested_lr_choice"]["corruption"][str(s)]
        print("| ViT-%s | %d | %.4f | %.4f | %.4f | %g | %g | %g |" % (
            "B/16" if tag == "B16" else "L/14", s,
            r["TAP_lr%g" % pub["TAP"]], r["CA_lr%g" % pub["CA"]], r["GH_lr%g" % pub["GH"]],
            nc["TAP"], nc["CA"], nc["GH"]))

for axis, name in AX:
    for label in ("published", "nested"):
        print("\n### T per-seed %s %s" % (axis, label))
        print("| encoder | seed | TAP | CLIP-Adapter | matched head | TAP - CLIP-Adapter | TAP - matched |")
        print("|---|---|---|---|---|---|---|")
        for tag in ("B16", "L14"):
            src = ev[tag]["per_seed"][axis][label]
            for k in sorted(src, key=int):
                v = src[k]
                print("| ViT-%s | %s | %.4f | %.4f | %.4f | %+.4f | %+.4f |" % (
                    "B/16" if tag == "B16" else "L/14", k, v["TAP"], v["CA"], v["GH"],
                    v["TAP"] - v["CA"], v["TAP"] - v["GH"]))

print("\n### T6 summary (published rate)")
print("| axis | encoder | seeds | single run | mean | SD | 95% CI | sign-test p | verdict |")
print("|---|---|---|---|---|---|---|---|---|")
for k in KEYS:
    s = st["published"][k]
    axis, tag = k.split("|")
    print("| %s | ViT-%s | %d | %+.4f | %+.4f | %.4f | [%+.4f, %+.4f] | %.4f | %s |" % (
        dict(AX)[axis], "B/16" if tag == "B16" else "L/14", s["n_seeds"], s["single_run"], s["mean"],
        s["sd"], s["t95_ci"][0], s["t95_ci"][1], s["sign_test_p"], s["verdict"]))

print("\n### T7 nested vs published")
print("| axis | encoder | published mean (95% CI) | nested mean (95% CI) | change |")
print("|---|---|---|---|---|")
for k in KEYS:
    a, b = st["published"][k], st["nested"][k]
    axis, tag = k.split("|")
    print("| %s | ViT-%s | %+.4f [%+.4f, %+.4f] | %+.4f [%+.4f, %+.4f] | %+.4f |" % (
        dict(AX)[axis], "B/16" if tag == "B16" else "L/14",
        a["mean"], a["t95_ci"][0], a["t95_ci"][1],
        b["mean"], b["t95_ci"][0], b["t95_ci"][1], b["mean"] - a["mean"]))

print("\n### corrected p-values")
for k in KEYS:
    print(k, "sign p=%.4f" % st["published"][k]["sign_test_p"])
print("holm adj", ["%.4f" % x for x in st["corrections"]["holm"]["adjusted"]])
print("bh adj  ", ["%.4f" % x for x in st["corrections"]["bh"]["adjusted"]])
print("nested holm adj", ["%.4f" % x for x in st["corrections_nested"]["holm"]["adjusted"]])

print("\n### TAP - matched head, published")
for k in KEYS:
    s = st["published"][k]
    # recompute the matched-head contrast series from per-seed data
    axis, tag = k.split("|")
    src = ev[tag]["per_seed"][axis]["published"]
    vals = [src[x]["TAP"] - src[x]["GH"] for x in sorted(src, key=int)]
    import numpy as np
    v = np.array(vals)
    print("%-20s mean %+.4f sd %.4f pos %d/%d" % (k, v.mean(), v.std(ddof=1), int((v > 0).sum()), len(v)))

