
# -*- coding: utf-8 -*-
"""Phase 13c: evaluate the phase-12 source-domain arms on the JAFFE fresh confirmation axis.

Nothing is trained, tuned or selected here. The arm checkpoints come from phase 12; the axis and
the decision rule come from the frozen pre-specification (phase13_jaffe_preregistration.md plus
amendment 1). The verdict printed at the end is the pre-specified one, not a judgement formed
while looking at the numbers.
"""
import importlib.util, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = Path("D:/ResearchVault/99system/data/rb-tta-fer-fg2027")
OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
DEV = "cuda"
CONDS = ["blur_0p8", "blur_1p5", "blur_2p5", "blur_4p0", "blur_6p0",
         "jpeg_15", "jpeg_30", "jpeg_50", "jpeg_65", "jpeg_80",
         "lowlight_0p3", "lowlight_0p5", "lowlight_0p7", "lowlight_0p85", "lowlight_0p95",
         "noise_10", "noise_25", "noise_40", "noise_55", "noise_70"]
LRS = [1e-3, 3e-3]

spec = importlib.util.spec_from_file_location("p8", OUT / "phase8_trainvar.py")
p8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p8)

CFG = {
    "B16": dict(dim=768, gd=512, hid=256, gh=256, ca_hid=128,
                tok=D / "token_cache_b16_jaffe", feat=D / "features_jaffe" / "b16",
                seeds=list(range(5, 15)), published=dict(TAP=3e-3, CA=1e-3, GH=3e-3)),
    "L14": dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192,
                tok=D / "token_cache_l14_jaffe", feat=D / "features_jaffe" / "l14",
                seeds=list(range(5, 20)), published=dict(TAP=1e-3, CA=1e-3, GH=1e-3)),
}


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def t_ci(a, df):
    a = np.asarray(a, float)
    tc = {9: 2.262, 14: 2.145}.get(df, 2.145)
    se = a.std(ddof=1) / math.sqrt(len(a))
    return float(a.mean()), float(a.mean() - tc * se), float(a.mean() + tc * se)


def sign_p(a):
    a = np.asarray(a, float)
    pos = int((a > 0).sum())
    n = int((a != 0).sum())
    k = min(pos, n - pos)
    return float(min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)), pos, n


def load_models(tag):
    c = CFG[tag]
    M = {}
    for arm in ("TAP", "CA", "GH"):
        for s in c["seeds"]:
            for lr in LRS:
                if arm == "TAP":
                    m = p8.T1(c["dim"], c["hid"])
                elif arm == "CA":
                    m = p8.CLIPAdapter(np.zeros((7, c["gd"]), np.float32), c["gd"], c["ca_hid"])
                else:
                    m = p8.GHead(c["gd"], c["gh"])
                m.load_state_dict(torch.load(OUT / ("phase12_%s_%s_seed%d_lr%g.pt" % (arm.lower(), tag, s, lr)),
                                             map_location=DEV))
                m.to(DEV).eval()
                M[(arm, s, lr)] = m
    return M


def main():
    t0 = time.perf_counter()
    probes = {t: json.loads((OUT / ("phase12_train_%s.json" % t)).read_text(encoding="utf-8"))
              for t in ("B16", "L14")}
    out = {"conditions": CONDS, "encoders": {}}

    for tag in ("B16", "L14"):
        c = CFG[tag]
        M = load_models(tag)
        res = {}
        clean = {}
        for cond in ["original"] + CONDS:
            gz = np.load(c["feat"] / ("%s.npz" % cond), allow_pickle=False)
            y = gz["labels"].astype(np.int64)
            g = torch.from_numpy(norm(gz["views"].mean(axis=1).astype(np.float32))).to(DEV)
            TOK = np.load(c["tok"] / ("%s_tokens_fp16.npy" % cond), mmap_mode="r")
            preds = {k: [] for k in M}
            with torch.inference_mode():
                for i in range(0, len(y), 256):
                    tok = torch.from_numpy(np.asarray(TOK[i:i + 256])).to(DEV).float()
                    gp = g[i:i + 256]
                    for k, m in M.items():
                        preds[k].append((m(tok) if k[0] == "TAP" else m(gp)).argmax(1).cpu().numpy())
                    del tok
            store = res if cond in CONDS else clean
            for k in M:
                store.setdefault(k, {})[cond] = uar(y, np.concatenate(preds[k]))
            del TOK, g, preds
            torch.cuda.empty_cache()
        print("%s evaluated on %d conditions (%.0fs)" % (tag, len(CONDS), time.perf_counter() - t0), flush=True)

        probe = {(r["seed"], arm, lr): r["%s_lr%g" % (arm, lr)]
                 for r in probes[tag]["rows"] for arm in ("TAP", "CA", "GH") for lr in LRS}

        def arm_mean(arm, s, lr):
            return float(np.mean(list(res[(arm, s, lr)].values())))

        def clean_mean(arm, s, lr):
            return float(np.mean(list(clean[(arm, s, lr)].values())))

        def nested_lr(arm, s):
            best = None
            for lr in LRS:
                v = probe[(s, arm, lr)]
                if best is None or v > best[1] + 1e-12:
                    best = (lr, v)
            return best[0]

        rec = {"n_seeds": len(c["seeds"]), "seeds": c["seeds"], "published": c["published"],
               "per_seed": {}, "summary": {}, "clean_descriptive": {}}
        pub, nest = {}, {}
        for s in c["seeds"]:
            pub[s] = {a: arm_mean(a, s, c["published"][a]) for a in ("TAP", "CA", "GH")}
            ns = {a: nested_lr(a, s) for a in ("TAP", "CA", "GH")}
            nest[s] = {"lr": ns, **{a: arm_mean(a, s, ns[a]) for a in ("TAP", "CA", "GH")}}
            rec["clean_descriptive"][str(s)] = {
                a: clean_mean(a, s, c["published"][a]) for a in ("TAP", "CA", "GH")}
        rec["per_seed"] = {"published": pub, "nested": nest}
        for label, src in (("published", pub), ("nested", nest)):
            for comp in ("CA", "GH"):
                v = np.array([src[s]["TAP"] - src[s][comp] for s in c["seeds"]])
                mean, lo, hi = t_ci(v, len(c["seeds"]) - 1)
                p, pos, n = sign_p(v)
                rng = np.random.default_rng(0)
                boot = np.array([np.mean([v[i] for i in rng.integers(0, len(v), len(v))])
                                 for _ in range(10000)])
                rec["summary"]["%s|TAP_minus_%s" % (label, comp)] = {
                    "n_seeds": int(len(v)), "mean": mean, "sd": float(v.std(ddof=1)),
                    "min": float(v.min()), "max": float(v.max()), "t95_ci": [lo, hi],
                    "sign_test_p": p, "seeds_positive": pos,
                    "seed_bootstrap_ci95": [float(np.percentile(boot, 2.5)),
                                            float(np.percentile(boot, 97.5))]}
        rec["raw_uar"] = {("%s|%d|%g|%s" % (a, s, lr, cond)): res[(a, s, lr)][cond]
                          for (a, s, lr) in res for cond in CONDS}
        out["encoders"][tag] = rec
        (OUT / "phase13_eval_jaffe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---------------- pre-specified decision rule ----------------
    s = out["encoders"]["L14"]["summary"]["published|TAP_minus_CA"]
    confirmed = (s["mean"] > 0) and (s["t95_ci"][0] > 0)
    strengthened = confirmed and s["seeds_positive"] >= 13
    verdict = "NOT CONFIRMED"
    if confirmed:
        verdict = "CONFIRMED"
    if strengthened:
        verdict = "CONFIRMED AND STRENGTHENED"
    out["verdict"] = {"rule": "L14 published-rate TAP - CLIP-Adapter mean > 0 and 95% CI excludes 0",
                      "strengthened_addendum": "and >= 13 of 15 seeds positive",
                      "mean": s["mean"], "t95_ci": s["t95_ci"], "seeds_positive": s["seeds_positive"],
                      "verdict": verdict}
    (OUT / "phase13_eval_jaffe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for tag in ("B16", "L14"):
        print("== %s (published rate)" % tag)
        for k, v in out["encoders"][tag]["summary"].items():
            print("  %-24s mean %+.4f sd %.4f ci [%+.4f,%+.4f] pos %2d/%2d p %.4f"
                  % (k, v["mean"], v["sd"], v["t95_ci"][0], v["t95_ci"][1], v["seeds_positive"],
                     v["n_seeds"], v["sign_test_p"]))
    print("== VERDICT:", verdict, "| mean %+.4f ci [%+.4f,%+.4f] %d/15"
          % (s["mean"], s["t95_ci"][0], s["t95_ci"][1], s["seeds_positive"]))
    print("done %.0fs" % (time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()

