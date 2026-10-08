
# -*- coding: utf-8 -*-
"""Phase 14: extend the aligned two-view token arm at ViT-L/14 from 5 seeds to 15.

The feature-convention sensitivity for the degradation axis currently rests on five seeds (phase 9).
This re-runs the same construction, unchanged, for seeds 5-19 at the learning rate phase 9 selected on
the source probe split (1e-3), and evaluates it on the same 12 pre-specified corruption conditions
against the same frozen comparator arms of phase 12.

Nothing is selected here: the recipe, the rate and the evaluation axis are those already used.
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
LR = 1e-3
GRID = 16
CONDS = ["blur_0p8", "blur_1p5", "blur_2p5", "jpeg_15", "jpeg_30", "jpeg_50",
         "lowlight_0p3", "lowlight_0p5", "lowlight_0p7", "noise_10", "noise_25", "noise_40"]

spec = importlib.util.spec_from_file_location("p8", OUT / "phase8_trainvar.py")
p8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p8)
CFG = dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192,
           feat=D / "features_vitl14", tokdir=D / "token_cache_vitl14",
           aligned=D / "token_cache_vitl14_aligned2v")


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
    split = np.load(CFG["feat"] / "linear_probe_split_seed0.npz", allow_pickle=False)
    tr = np.sort(np.asarray(split["train_indices"], dtype=np.int64))
    va = np.asarray(split["validation_indices"], dtype=np.int64)
    AL = np.load(CFG["aligned"] / "fer2013_train_tokens_fp16.npy", mmap_mode="r")
    LAB = np.load(CFG["tokdir"] / "fer2013_train_meta.npz", allow_pickle=False)["labels"].astype(np.int64)

    def batches(sel):
        sel = np.asarray(sel, np.int64)
        return (torch.from_numpy(np.asarray(AL[sel])).to(DEV).float(),
                torch.from_numpy(LAB[sel]).to(DEV))

    yva = LAB[va]
    aligned = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = p8.T1(CFG["dim"], CFG["hid"]).to(DEV)
        opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=LR)
        gen = torch.Generator().manual_seed(seed)
        for _ in range(p8.EPOCHS):
            perm = torch.randperm(len(tr), generator=gen)
            for i in range(0, len(tr), p8.BATCH):
                sel = tr[perm[i:i + p8.BATCH].numpy()]
                xb, yb = batches(sel)
                loss = torch.nn.functional.cross_entropy(m(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        m.eval()
        torch.save(m.state_dict(), OUT / ("phase14_aligned_tap_L14_seed%d.pt" % seed))
        preds = []
        with torch.inference_mode():
            for i in range(0, len(va), 256):
                xb, _ = batches(va[i:i + 256])
                preds.append(m(xb).argmax(1).cpu().numpy())
        aligned[seed] = m
        print("  aligned seed %d probe-val %.4f (%.0fs)"
              % (seed, uar(yva, np.concatenate(preds)), time.perf_counter() - t0), flush=True)

    single, ca, gh = {}, {}, {}
    for seed in SEEDS:
        m = p8.T1(CFG["dim"], CFG["hid"]).to(DEV)
        m.load_state_dict(torch.load(OUT / ("phase12_tap_L14_seed%d_lr0.001.pt" % seed), map_location=DEV))
        m.eval()
        single[seed] = m
        m = p8.CLIPAdapter(np.zeros((7, CFG["gd"]), np.float32), CFG["gd"], CFG["ca_hid"]).to(DEV)
        m.load_state_dict(torch.load(OUT / ("phase12_clipadapter_L14_seed%d_lr0.001.pt" % seed), map_location=DEV))
        m.eval()
        ca[seed] = m
        m = p8.GHead(CFG["gd"], CFG["gh"]).to(DEV)
        m.load_state_dict(torch.load(OUT / ("phase12_ghead_L14_seed%d_lr0.001.pt" % seed), map_location=DEV))
        m.eval()
        gh[seed] = m

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
        print("  %s done %.0fs" % (cond, time.perf_counter() - t0), flush=True)

    def vec(arm):
        return np.array([float(np.mean([per[s][arm][k] for k in CONDS])) for s in SEEDS])

    A, S, C, G = vec("aligned"), vec("single"), vec("ca"), vec("gh")
    rep = {"encoder": "L14", "n_seeds": len(SEEDS), "seeds": SEEDS, "lr": LR,
           "conditions": CONDS, "means": {"aligned": A.tolist(), "single": S.tolist(),
                                          "ca": C.tolist(), "gh": G.tolist()},
           "summary": {}, "wall_seconds": round(time.perf_counter() - t0, 1),
           "scope": "extension of the phase-9 aligned two-view sensitivity to 15 seeds; nothing selected "
                    "on the axis"}
    for key, v in (("aligned_TAP_minus_CA", A - C), ("aligned_TAP_minus_GH", A - G),
                   ("single_TAP_minus_CA", S - C), ("aligned_minus_single", A - S)):
        mean, lo, hi = t_ci(v)
        rep["summary"][key] = {"mean": mean, "sd": float(v.std(ddof=1)), "t95_ci": [lo, hi],
                               "seeds_positive": int((v > 0).sum()), "n": int(len(v))}
        print("%-24s mean %+.4f sd %.4f ci [%+.4f,%+.4f] pos %d/%d"
              % (key, mean, v.std(ddof=1), lo, hi, int((v > 0).sum()), len(v)), flush=True)
    (OUT / "phase14_aligned_l14_15seeds.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("done %.0fs" % (time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()

