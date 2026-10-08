
# -*- coding: utf-8 -*-
"""Phase 14b: evaluate the 15 aligned two-view ViT-L/14 arms on the pre-specified corruption axis.

The training half (phase14_aligned_l14_15seeds.py) saved phase14_aligned_tap_L14_seed{5..19}.pt;
its evaluation stage used the phase-9 checkpoint names for the comparators and failed. This re-runs
only the evaluation, with the phase-12 checkpoint names, and writes the summary.
"""
import importlib.util, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = Path("D:/ResearchVault/99system/data/rb-tta-fer-fg2027")
OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
DEV = "cuda"
SEEDS = list(range(5, 20))
CONDS = ["blur_0p8", "blur_1p5", "blur_2p5", "jpeg_15", "jpeg_30", "jpeg_50",
         "lowlight_0p3", "lowlight_0p5", "lowlight_0p7", "noise_10", "noise_25", "noise_40"]
CFG = dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192)

spec = importlib.util.spec_from_file_location("p8", OUT / "phase8_trainvar.py")
p8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p8)


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def t_ci(a, tc=2.145):
    a = np.asarray(a, float)
    se = a.std(ddof=1) / math.sqrt(len(a))
    return float(a.mean()), float(a.mean() - tc * se), float(a.mean() + tc * se)


def main():
    t0 = time.perf_counter()
    aligned, single, ca, gh = {}, {}, {}, {}
    for s in SEEDS:
        m = p8.T1(CFG["dim"], CFG["hid"]).to(DEV)
        m.load_state_dict(torch.load(OUT / ("phase14_aligned_tap_L14_seed%d.pt" % s), map_location=DEV))
        aligned[s] = m.eval()
        m = p8.T1(CFG["dim"], CFG["hid"]).to(DEV)
        m.load_state_dict(torch.load(OUT / ("phase12_tap_L14_seed%d_lr0.001.pt" % s), map_location=DEV))
        single[s] = m.eval()
        m = p8.CLIPAdapter(np.zeros((7, CFG["gd"]), np.float32), CFG["gd"], CFG["ca_hid"]).to(DEV)
        m.load_state_dict(torch.load(OUT / ("phase12_ca_L14_seed%d_lr0.001.pt" % s), map_location=DEV))
        ca[s] = m.eval()
        m = p8.GHead(CFG["gd"], CFG["gh"]).to(DEV)
        m.load_state_dict(torch.load(OUT / ("phase12_gh_L14_seed%d_lr0.001.pt" % s), map_location=DEV))
        gh[s] = m.eval()
    print("models loaded %.0fs" % (time.perf_counter() - t0), flush=True)

    per = {s: {"aligned": {}, "single": {}, "ca": {}, "gh": {}} for s in SEEDS}
    for cond in CONDS:
        gz = np.load(D / "features_confirm_shift" / "l14" / ("%s.npz" % cond), allow_pickle=False)
        y = gz["labels"].astype(np.int64)
        g = torch.from_numpy(norm(gz["views"].mean(axis=1).astype(np.float32))).to(DEV)
        TOK = np.load(D / "token_cache_l14_confirm" / ("%s_tokens_fp16.npy" % cond), mmap_mode="r")
        pa = {s: [] for s in SEEDS}
        ps = {s: [] for s in SEEDS}
        with torch.inference_mode():
            for i in range(0, len(y), 256):
                tok = torch.from_numpy(np.asarray(TOK[i:i + 256])).to(DEV).float()
                for s in SEEDS:
                    pa[s].append(aligned[s](tok).argmax(1).cpu().numpy())
                    ps[s].append(single[s](tok).argmax(1).cpu().numpy())
                del tok
        for s in SEEDS:
            cp = ca[s](g).argmax(1).cpu().numpy()
            gp = gh[s](g).argmax(1).cpu().numpy()
            per[s]["aligned"][cond] = uar(y, np.concatenate(pa[s]))
            per[s]["single"][cond] = uar(y, np.concatenate(ps[s]))
            per[s]["ca"][cond] = uar(y, cp)
            per[s]["gh"][cond] = uar(y, gp)
        del TOK, g, pa, ps
        torch.cuda.empty_cache()
        print("  %s %.0fs" % (cond, time.perf_counter() - t0), flush=True)

    def vec(arm):
        return np.array([float(np.mean([per[s][arm][k] for k in CONDS])) for s in SEEDS])

    A, S, C, G = vec("aligned"), vec("single"), vec("ca"), vec("gh")
    rep = {"encoder": "L14", "n_seeds": len(SEEDS), "seeds": SEEDS, "lr": 1e-3, "conditions": CONDS,
           "means": {"aligned": A.tolist(), "single": S.tolist(), "ca": C.tolist(), "gh": G.tolist()},
           "summary": {}, "wall_seconds": round(time.perf_counter() - t0, 1),
           "scope": "extension of the phase-9 aligned two-view sensitivity from 5 to 15 seeds; the recipe, "
                    "learning rate and evaluation axis are unchanged and nothing is selected on the axis"}
    rng = np.random.default_rng(0)
    for key, v in (("aligned_TAP_minus_CA", A - C), ("aligned_TAP_minus_GH", A - G),
                   ("single_TAP_minus_CA", S - C), ("single_TAP_minus_GH", S - G),
                   ("aligned_minus_single", A - S)):
        mean, lo, hi = t_ci(v)
        boot = np.array([np.mean([v[i] for i in rng.integers(0, len(v), len(v))]) for _ in range(10000)])
        rep["summary"][key] = {"mean": mean, "sd": float(v.std(ddof=1)), "t95_ci": [lo, hi],
                               "seeds_positive": int((v > 0).sum()), "n": int(len(v)),
                               "seed_bootstrap_ci95": [float(np.percentile(boot, 2.5)),
                                                       float(np.percentile(boot, 97.5))]}
        print("%-24s mean %+.4f sd %.4f ci [%+.4f,%+.4f] pos %d/%d"
              % (key, mean, v.std(ddof=1), lo, hi, int((v > 0).sum()), len(v)), flush=True)
    (OUT / "phase14_aligned_l14_15seeds.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("done %.0fs" % (time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()

