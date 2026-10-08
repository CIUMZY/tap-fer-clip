
"""Phase 12c: statistics and manuscript tables for the seed-controlled retraining set.

Reads phase12_eval.json / phase12_train_*.json and writes
  phase12_stats.json          machine-readable summary
  STATS-ROBUSTNESS.md         human-readable section (overwrites the phase-8 version)
and prints the markdown tables used in the manuscript.
"""
import json, math, sys
from pathlib import Path

import numpy as np

OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
MS = Path("D:/ResearchVault/06manuscripts/rb-tta-fer-fg2027/PAPER-TVC-TAP-2026-10-06/manuscript")

SINGLE = {("setA", "B16"): 0.0388, ("setA", "L14"): -0.0196,
          ("setB", "B16"): 0.0347, ("setB", "L14"): -0.0137,
          ("corruption", "B16"): 0.0181, ("corruption", "L14"): 0.0320}
LABEL = {"setA": "Set A (crossed grid)", "setB": "Set B (second source)",
         "corruption": "corruption axis"}
UNIT = {"setA": "50 crossed cells", "setB": "3 targets", "corruption": "12 conditions"}


def sign_p(a):
    a = np.asarray(a, float)
    pos = int((a > 0).sum())
    n = int((a != 0).sum())
    k = min(pos, n - pos)
    return float(min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)), pos, n


def t_ci(a, df):
    a = np.asarray(a, float)
    tc = {9: 2.262, 14: 2.145}.get(df, 2.262)
    se = a.std(ddof=1) / math.sqrt(len(a))
    return float(a.mean() - tc * se), float(a.mean() + tc * se), tc


def holm(p, alpha=0.05):
    m = len(p)
    order = np.argsort(p)
    rej = [False] * m
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, p[i] * (m - rank)))
        adj[i] = running
    return rej, adj


def bh(p, alpha=0.05):
    m = len(p)
    order = np.argsort(p)
    rej = [False] * m
    kmax = 0
    for rank, i in enumerate(order):
        if p[i] <= (rank + 1) / m * alpha:
            kmax = rank + 1
    for rank, i in enumerate(order):
        if rank + 1 <= kmax:
            rej[i] = True
    adj = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, min(1.0, p[i] * m / (rank + 1)))
        adj[i] = running
    return rej, adj


def unit_bootstrap(series, lr_of, arm_a, arm_b, raw_axis, B=2000):
    """Resample evaluation units with replacement inside each fixed training seed."""
    rng = np.random.default_rng(0)
    seeds = sorted(int(s) for s in series)
    la0, lb0 = lr_of(arm_a, seeds[0]), lr_of(arm_b, seeds[0])
    units = sorted(raw_axis["%s|%d|%g" % (arm_a, seeds[0], la0)].keys())
    draws = []
    for _ in range(B):
        pick = rng.integers(0, len(units), len(units))
        vals = []
        for s in seeds:
            la, lb = lr_of(arm_a, s), lr_of(arm_b, s)
            a = np.array([raw_axis["%s|%d|%g" % (arm_a, s, la)][units[j]] for j in pick])
            b = np.array([raw_axis["%s|%d|%g" % (arm_b, s, lb)][units[j]] for j in pick])
            vals.append(float((a - b).mean()))
        draws.append(float(np.mean(vals)))
    d = np.array(draws)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def verdict(single, mean, lo, hi):
    if lo > 0 or hi < 0:
        if np.sign(mean) == np.sign(single):
            return "replicates"
        return "reversed"
    return "attenuated" if np.sign(mean) == np.sign(single) else "attenuated (sign differs)"


def main():
    ev = json.loads((OUT / "phase12_eval.json").read_text(encoding="utf-8"))
    stats = {"published": {}, "nested": {}, "corrections": {}}
    rows = {"published": [], "nested": []}

    for tag in ("B16", "L14"):
        rec = ev[tag]
        for axis in ("setA", "setB", "corruption"):
            for label in ("published", "nested"):
                src = rec["per_seed"][axis][label]
                seeds = sorted(int(s) for s in src)
                v = np.array([src[str(s) if str(s) in src else s]["TAP"]
                              - src[str(s) if str(s) in src else s]["CA"] for s in seeds])
                mean = float(v.mean())
                lo, hi, tc = t_ci(v, len(seeds) - 1)
                p, pos, n = sign_p(v)
                rng = np.random.default_rng(0)
                boot = np.array([np.mean([v[i] for i in rng.integers(0, len(v), len(v))]) for _ in range(10000)])
                stats[label]["%s|%s" % (axis, tag)] = {
                    "n_seeds": int(len(v)), "mean": mean, "sd": float(v.std(ddof=1)),
                    "min": float(v.min()), "max": float(v.max()), "t95_ci": [lo, hi], "tcrit": tc,
                    "sign_test_p": p, "seeds_positive": pos, "seed_bootstrap_ci95":
                        [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "single_run": SINGLE[(axis, tag)],
                    "verdict": verdict(SINGLE[(axis, tag)], mean, lo, hi)}
                rows[label].append((axis, tag))
        for axis in ("setA", "setB", "corruption"):
            raw = rec["raw"][axis]
            for label in ("published", "nested"):
                src = rec["per_seed"][axis][label]
                if label == "published":
                    lr_of = lambda arm, s, _p=rec["published"]: _p[arm]
                else:
                    lr_of = lambda arm, s, _c=rec["nested_lr_choice"][axis]: _c[str(s)][arm]
                m, lo, hi = unit_bootstrap(src, lr_of, "TAP", "CA", raw)
                stats[label]["%s|%s|unit_bootstrap_CA" % (axis, tag)] = {
                    "mean": m, "ci95": [lo, hi], "unit": UNIT[axis]}

    keys = ["%s|%s" % (a, e) for a in ("setA", "setB", "corruption") for e in ("B16", "L14")]
    ps = [stats["published"][k]["sign_test_p"] for k in keys]
    rj, aj = holm(ps)
    rb, ab = bh(ps)
    stats["corrections"] = {
        "family": keys, "p_values": ps,
        "holm": {"reject": rj, "adjusted": aj, "n_rejected": int(sum(rj))},
        "bh": {"reject": rb, "adjusted": ab, "n_rejected": int(sum(rb))},
        "sign_test_floor_n5": 0.0625, "sign_test_floor_n10": 2 / 2 ** 10}
    pn = [stats["nested"][k]["sign_test_p"] for k in keys]
    rjn, ajn = holm(pn)
    rbn, abn = bh(pn)
    stats["corrections_nested"] = {"p_values": pn,
                                   "holm": {"reject": rjn, "adjusted": ajn, "n_rejected": int(sum(rjn))},
                                   "bh": {"reject": rbn, "adjusted": abn, "n_rejected": int(sum(rbn))}}
    (OUT / "phase12_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ---- console output ----
    print("=== published-rate distribution (TAP - CLIP-Adapter) ===")
    for k in keys:
        s = stats["published"][k]
        print("%-22s n=%2d mean %+.4f sd %.4f ci [%+.4f,%+.4f] pos %2d/%2d p %.4f  %s"
              % (k, s["n_seeds"], s["mean"], s["sd"], s["t95_ci"][0], s["t95_ci"][1],
                 s["seeds_positive"], s["n_seeds"], s["sign_test_p"], s["verdict"]))
    print("=== nested distribution ===")
    for k in keys:
        s = stats["nested"][k]
        print("%-22s n=%2d mean %+.4f sd %.4f ci [%+.4f,%+.4f] pos %2d/%2d p %.4f  %s"
              % (k, s["n_seeds"], s["mean"], s["sd"], s["t95_ci"][0], s["t95_ci"][1],
                 s["seeds_positive"], s["n_seeds"], s["sign_test_p"], s["verdict"]))
    print("=== TAP - capacity-matched head (published) ===")
    for tag in ("B16", "L14"):
        for axis in ("setA", "setB", "corruption"):
            rec = ev[tag]
            src = rec["per_seed"][axis]["published"]
            seeds = sorted(int(s) for s in src)
            v = np.array([src[str(s) if str(s) in src else s]["TAP"] - src[str(s) if str(s) in src else s]["GH"]
                          for s in seeds])
            print("%s %-11s mean %+.4f sd %.4f pos %d/%d" % (tag, axis, v.mean(), v.std(ddof=1),
                                                            int((v > 0).sum()), len(v)))
    print("=== unit-level bootstrap (published, TAP - CA) ===")
    for k in keys:
        s = stats["published"][k + "|unit_bootstrap_CA"]
        print("%-22s mean %+.4f ci [%+.4f,%+.4f] unit=%s" % (k, s["mean"], s["ci95"][0], s["ci95"][1], s["unit"]))
    print("=== multiplicity (published sign tests) ===")
    print("p:", ["%.4f" % x for x in ps])
    print("holm rejects:", sum(rj), "adjusted:", ["%.4f" % x for x in aj])
    print("bh   rejects:", sum(rb), "adjusted:", ["%.4f" % x for x in ab])
    print("=== multiplicity (nested sign tests) ===")
    print("p:", ["%.4f" % x for x in pn], "holm:", sum(rjn), "bh:", sum(rbn))
    print("=== nested rate choices (L14) ===")
    for seed, ch in sorted(ev["L14"]["nested_lr_choice"]["corruption"].items(), key=lambda x: int(x[0])):
        print("seed", seed, ch)
    print("=== probe-val per seed (published rate) ===")
    for tag in ("B16", "L14"):
        tr = json.loads((OUT / ("phase12_train_%s.json" % tag)).read_text(encoding="utf-8"))
        pub = tr["published"]
        for r in tr["rows"]:
            print(tag, r["seed"], "TAP %.4f CA %.4f GH %.4f"
                  % (r["TAP_lr%g" % pub["TAP"]], r["CA_lr%g" % pub["CA"]], r["GH_lr%g" % pub["GH"]]))


if __name__ == "__main__":
    main()

