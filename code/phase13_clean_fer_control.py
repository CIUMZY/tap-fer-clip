
# -*- coding: utf-8 -*-
"""Clean-domain control for the phase-12 arms: UAR on the uncorrupted FER2013 test split.

Needed to separate "the token route is better on this domain" from "the token route is better under
degradation". Same checkpoints, same protocol, no training or selection.
"""
import importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = Path("D:/ResearchVault/99system/data/rb-tta-fer-fg2027")
OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
DEV = "cuda"
LRS = [1e-3, 3e-3]
spec = importlib.util.spec_from_file_location("p8", OUT / "phase8_trainvar.py")
p8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p8)

CFG = {
    "B16": dict(dim=768, gd=512, hid=256, gh=256, ca_hid=128,
                tok=D / "token_cache_vitb16" / "fer2013_test_tokens_fp16.npz",
                feat=D / "features", seeds=list(range(5, 15)),
                published=dict(TAP=3e-3, CA=1e-3, GH=3e-3)),
    "L14": dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192,
                tok=D / "token_cache_vitl14" / "fer2013_test_tokens_fp16.npy",
                feat=D / "features_vitl14", seeds=list(range(5, 20)),
                published=dict(TAP=1e-3, CA=1e-3, GH=1e-3)),
}


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main():
    t0 = time.perf_counter()
    out = {}
    for tag in ("B16", "L14"):
        c = CFG[tag]
        gz = np.load(c["feat"] / "fer2013_test.npz", allow_pickle=False)
        y = gz["labels"].astype(np.int64)
        g = torch.from_numpy(norm(gz["views"].mean(axis=1).astype(np.float32))).to(DEV)
        TOK = np.load(c["tok"], mmap_mode="r") if c["tok"].suffix == ".npy" else None
        if TOK is None:
            z = np.load(c["tok"], allow_pickle=False)
            TOK, y = z["tokens"], z["labels"].astype(np.int64)
        models = {}
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
                    models[(arm, s, lr)] = m
        preds = {k: [] for k in models}
        with torch.inference_mode():
            for i in range(0, len(y), 256):
                tok = torch.from_numpy(np.asarray(TOK[i:i + 256])).to(DEV).float()
                gp = g[i:i + 256]
                for k, m in models.items():
                    preds[k].append((m(tok) if k[0] == "TAP" else m(gp)).argmax(1).cpu().numpy())
                del tok
        res = {("%s|%d|%g" % k): uar(y, np.concatenate(v)) for k, v in preds.items()}
        mean_by = {}
        for arm in ("TAP", "CA", "GH"):
            for lr in LRS:
                v = np.array([res["%s|%d|%g" % (arm, s, lr)] for s in c["seeds"]])
                mean_by["%s_lr%g" % (arm, lr)] = float(v.mean())
        pub = c["published"]
        d_ca = np.array([res["TAP|%d|%g" % (s, pub["TAP"])] - res["CA|%d|%g" % (s, pub["CA"])]
                         for s in c["seeds"]])
        d_gh = np.array([res["TAP|%d|%g" % (s, pub["TAP"])] - res["GH|%d|%g" % (s, pub["GH"])]
                         for s in c["seeds"]])
        tc = 2.262 if tag == "B16" else 2.145
        out[tag] = {"n_seeds": len(c["seeds"]), "n_images": int(len(y)),
                    "arm_means": mean_by, "per_seed_raw": res,
                    "TAP_minus_CA": {"mean": float(d_ca.mean()), "sd": float(d_ca.std(ddof=1)),
                                     "ci": [float(d_ca.mean() - tc * d_ca.std(ddof=1) / np.sqrt(len(d_ca))),
                                            float(d_ca.mean() + tc * d_ca.std(ddof=1) / np.sqrt(len(d_ca)))],
                                     "seeds_positive": int((d_ca > 0).sum())},
                    "TAP_minus_GH": {"mean": float(d_gh.mean()), "sd": float(d_gh.std(ddof=1)),
                                     "ci": [float(d_gh.mean() - tc * d_gh.std(ddof=1) / np.sqrt(len(d_gh))),
                                            float(d_gh.mean() + tc * d_gh.std(ddof=1) / np.sqrt(len(d_gh)))],
                                     "seeds_positive": int((d_gh > 0).sum())}}
        print(tag, json.dumps({k: v for k, v in out[tag].items()
                               if k in ("TAP_minus_CA", "TAP_minus_GH", "arm_means")}, indent=1), flush=True)
        del models, preds, TOK, g
        torch.cuda.empty_cache()
    (OUT / "phase13_clean_fer_control.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("done %.0fs" % (time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()

