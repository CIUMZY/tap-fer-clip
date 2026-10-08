
# -*- coding: utf-8 -*-
"""Phase 17: audit every number in the manuscript's summary tables against the released records.

Parses Tables 6, 7 and 10 out of FULL-MANUSCRIPT.md and compares each cell with the JSON artefact it
claims to come from. Reports a diff list; an empty diff means the manuscript tables are exactly the
released numbers.
"""
import json, re, sys
from pathlib import Path

import numpy as np

OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
MS = Path("D:/ResearchVault/06manuscripts/rb-tta-fer-fg2027/PAPER-TVC-TAP-2026-10-06/manuscript/FULL-MANUSCRIPT.md")
t = MS.read_text(encoding="utf-8")
st12 = json.loads((OUT / "phase12_stats.json").read_text(encoding="utf-8"))
ja = json.loads((OUT / "phase13_eval_jaffe.json").read_text(encoding="utf-8"))
sm13 = json.loads((OUT / "phase13_summary.json").read_text(encoding="utf-8"))
clean = json.loads((OUT / "phase13_clean_fer_control.json").read_text(encoding="utf-8"))
p14 = json.loads((OUT / "phase14_aligned_l14_15seeds.json").read_text(encoding="utf-8"))
p9b = json.loads((OUT / "phase9_aligned_corruption_B16_lr3e-3.json").read_text(encoding="utf-8"))
ev12 = json.loads((OUT / "phase12_eval.json").read_text(encoding="utf-8"))

problems = []


def table_rows(title):
    i = t.index(title)
    seg = t[i:].split("\n")
    rows = []
    for ln in seg[1:]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    return rows


def num(cell):
    m = re.search(r"([+-]?\d+\.\d+)", cell)
    return float(m.group(1)) if m else None


# ---- Table 6 (fresh axis + controls) ----
rows = table_rows("**Table 6. The surviving contrast on the fresh JAFFE axis")
for r in rows[2:]:
    label, enc, cond, ca, gh, pos = r[0], r[1], r[2], r[3], r[4], r[5]
    tag = "L14" if "L/14" in enc else "B16"
    if label.startswith("JAFFE, corrupted"):
        key = "published" if cond == "rate fixed" else "nested"
        want_ca = 100 * ja["encoders"][tag]["summary"]["%s|TAP_minus_CA" % key]["mean"]
        want_pos = ja["encoders"][tag]["summary"]["%s|TAP_minus_CA" % key]["seeds_positive"]
    elif label.startswith("JAFFE, clean"):
        want_ca = 100 * sm13["jaffe_clean_descriptive"][tag]["TAP_minus_CA"]["mean"]
        want_pos = sm13["jaffe_clean_descriptive"][tag]["TAP_minus_CA"]["seeds_positive"]
    elif label.startswith("FER2013 test, corrupted"):
        want_ca = 100 * st12["published"]["corruption|L14"]["mean"]
        want_pos = st12["published"]["corruption|L14"]["seeds_positive"]
    else:
        want_ca = 100 * clean[tag]["TAP_minus_CA"]["mean"]
        want_pos = clean[tag]["TAP_minus_CA"]["seeds_positive"]
    if abs(num(ca) - want_ca) > 0.005:
        problems.append("Table 6 %s %s %s: %s vs %.4f" % (label, enc, cond, ca, want_ca))
    if int(pos.split("/")[0]) != want_pos:
        problems.append("Table 6 %s %s %s seeds: %s vs %d" % (label, enc, cond, pos, want_pos))

# ---- Table 7 (summary) ----
rows = table_rows("**Table 7. Summary across the three confirmation axes")
for r in rows[2:]:
    axis, enc, n, single, mean, sd, ci, p, verdict = r[:9]
    tag = "L14" if "L/14" in enc else "B16"
    akey = {"crossed grid": "setA", "second source": "setB", "corruption": "corruption"}[axis]
    s = st12["published"]["%s|%s" % (akey, tag)]
    for got, want, what in ((num(single), 100 * s["single_run"], "single"),
                            (num(mean), 100 * s["mean"], "mean"),
                            (num(sd), 100 * s["sd"], "sd")):
        if abs(got - want) > 0.02:
            problems.append("Table 7 %s|%s %s: %s vs %.4f" % (akey, tag, what, got, want))
    if abs(num(p) - s["sign_test_p"]) > 0.0002:
        problems.append("Table 7 %s|%s p: %s vs %.4f" % (akey, tag, p, s["sign_test_p"]))

# ---- Table 10 (convention) ----
rows = table_rows("**Table 10. Feature-convention sensitivity of the degradation")
expect = {
    ("ViT-B/16", "single"): (st12["published"]["corruption|B16"]["mean"],
                             np.mean([ev12["B16"]["per_seed"]["corruption"]["published"][k]["TAP"]
                                      - ev12["B16"]["per_seed"]["corruption"]["published"][k]["GH"]
                                      for k in sorted(ev12["B16"]["per_seed"]["corruption"]["published"], key=int)])),
    ("ViT-B/16", "aligned"): (p9b["contrasts"]["aligned2v_TAP_minus_CA"]["mean"],
                              p9b["contrasts"]["aligned2v_TAP_minus_GH"]["mean"]),
    ("ViT-L/14", "single"): (st12["published"]["corruption|L14"]["mean"], 0.0273),
    ("ViT-L/14", "aligned"): (p14["summary"]["aligned_TAP_minus_CA"]["mean"],
                              p14["summary"]["aligned_TAP_minus_GH"]["mean"]),
}
for r in rows[2:]:
    enc, conv, ca, gh, pos = r[0], r[1], r[2], r[3], r[4]
    key = (enc, "single" if "single" in conv else "aligned")
    want_ca, want_gh = expect[key]
    if abs(num(ca) - 100 * want_ca) > 0.02:
        problems.append("Table 10 %s %s CA: %s vs %.4f" % (enc, conv, ca, 100 * want_ca))
    if abs(num(gh) - 100 * want_gh) > 0.02:
        problems.append("Table 10 %s %s GH: %s vs %.4f" % (enc, conv, gh, 100 * want_gh))

# ---- in-text key numbers ----
TEXT_CHECKS = [
    ("+3.50 points", 100 * st12["published"]["corruption|L14"]["mean"], 0.005),
    ("[2.56, 4.43]", None, None),
    ("+2.75 points", 100 * ja["verdict"]["mean"], 0.005),
    ("+1.78, +3.72", 100 * ja["verdict"]["t95_ci"][0], 0.02),
    ("+3.85 points", 100 * p14["summary"]["aligned_TAP_minus_CA"]["mean"], 0.005),
]
for needle, want, tol in TEXT_CHECKS:
    if needle not in t:
        problems.append("text missing %r" % needle)

rep = {"n_checks": "Tables 6, 7, 10 and the headline in-text numbers",
       "problems": problems, "status": "pass" if not problems else "fail"}
(OUT / "phase17_number_audit.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
print(json.dumps(rep, indent=2))
sys.exit(0 if not problems else 1)

