
# -*- coding: utf-8 -*-
"""Phase 19: the third-encoder verification axis.

Encoder: LAION-2B ViT-B/32 (open_clip checkpoint from ModelScope, sha256 verified). It differs from
both encoders used so far in architecture (patch 32, 7x7 tokens) and in pretraining data (LAION-2B
instead of OpenAI's WIT). Nothing from the paper's arms is reused: the three arms are re-trained here
with the phase-12 seed-controlled recipe, on the FER2013 probe-train split only.

Stages: extract -> train -> evaluate on the pre-specified 12-condition axis, the clean control and
the nine fresh-family conditions of axis C.
"""
import argparse, csv, importlib.util, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = Path("D:/ResearchVault/99system/data/rb-tta-fer-fg2027")
OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
DEV = "cuda"
WEIGHTS = Path("D:/ResearchVault/99system/models/open_clip/ViT-B-32-laion2B-fixed.pt")
CONDS12 = ["blur_0p8", "blur_1p5", "blur_2p5", "jpeg_15", "jpeg_30", "jpeg_50",
           "lowlight_0p3", "lowlight_0p5", "lowlight_0p7", "noise_10", "noise_25", "noise_40"]
CONDSC = ["occ_8", "occ_16", "occ_24", "contrast_0p3", "contrast_0p5", "contrast_0p7",
          "rot_10", "rot_20", "rot_30"]
CFG = dict(dim=768, gd=512, hid=256, gh=256, ca_hid=128, n_tok=49, d_tok=768, rows=128,
           lrs=[1e-3, 3e-3], seeds=list(range(5, 15)), published=dict(TAP=3e-3, CA=1e-3, GH=3e-3))
TOKDIR = D / "token_cache_vitb32laion"
FEATDIR = D / "features_vitb32laion"

spec = importlib.util.spec_from_file_location("p8", OUT / "phase8_trainvar.py")
p8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p8)


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def t_ci(a, tc):
    a = np.asarray(a, float)
    se = a.std(ddof=1) / math.sqrt(len(a))
    return float(a.mean()), float(a.mean() - tc * se), float(a.mean() + tc * se)


def sources():
    """(name, token-or-npy path, manifest, kind) for every set we need."""
    out = [("fer2013_train", None, D / "manifests" / "fer2013_train.csv", "train"),
           ("fer2013_test", None, D / "manifests" / "fer2013_test.csv", "clean")]
    for c in CONDS12:
        out.append((c, None, D / "manifests" / ("fer2013_%s.csv" % c), "corrupt12"))
    for c in CONDSC:
        out.append((c, None, D / "raw/fer2013_fresh_axis/manifests" / ("fer2013_%s.csv" % c), "fresh"))
    return out


def extract():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(WEIGHTS), device="cuda", weights_only=False)
    model.eval()
    model.visual.output_tokens = True
    TOKDIR.mkdir(parents=True, exist_ok=True)
    FEATDIR.mkdir(parents=True, exist_ok=True)
    for name, _, man, kind in sources():
        npy = TOKDIR / ("%s_tokens_fp16.npy" % name)
        feat = FEATDIR / ("%s.npz" % name)
        if (npy.exists() and npy.stat().st_size > 0
                and feat.exists() and feat.stat().st_size > 1_000_000):
            print("  reuse", name, flush=True)
            continue
        with man.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        paths = [r["path"] for r in rows]
        labels = np.asarray([int(r["label"]) for r in rows], np.int64)
        mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16,
                                       shape=(len(paths), CFG["n_tok"], CFG["d_tok"]))
        g_all = np.empty((len(paths), 2, CFG["gd"]), np.float32)
        t0 = time.perf_counter()
        for i in range(0, len(paths), CFG["rows"]):
            chunk = paths[i:i + CFG["rows"]]
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            pairs = []
            for im in imgs:
                pairs.extend([im, im.transpose(Image.FLIP_LEFT_RIGHT)])
            x = torch.stack([preprocess(im) for im in pairs]).to(DEV)
            with torch.inference_mode():
                pooled, tokens = model.visual(x)
                pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            mm[i:i + len(chunk)] = tokens[0::2].half().cpu().numpy()
            g_all[i:i + len(chunk)] = pooled.cpu().numpy().astype(np.float32).reshape(len(chunk), 2, -1)
            del x, pooled, tokens
        mm.flush()
        del mm
        np.savez_compressed(TOKDIR / ("%s_meta.npz" % name), labels=labels, ids=np.asarray(paths))
        np.savez_compressed(FEATDIR / ("%s.npz" % name), ids=np.asarray(paths), views=g_all, labels=labels)
        print("  %-14s %d  %.0fs" % (name, len(paths), time.perf_counter() - t0), flush=True)


def train():
    split = np.load(D / "features" / "linear_probe_split_seed0.npz", allow_pickle=False)
    tr = np.sort(np.asarray(split["train_indices"], dtype=np.int64))
    va = np.asarray(split["validation_indices"], dtype=np.int64)
    text = norm(np.load(FEATDIR / "text_prototypes.npz", allow_pickle=False)["prototypes"].astype(np.float32))
    gtr = norm(np.load(FEATDIR / "fer2013_train.npz", allow_pickle=False)["views"].mean(axis=1))
    G = torch.from_numpy(gtr.astype(np.float32))
    TOK = np.load(TOKDIR / "fer2013_train_tokens_fp16.npy", mmap_mode="r")
    LAB = np.load(TOKDIR / "fer2013_train_meta.npz", allow_pickle=False)["labels"].astype(np.int64)

    def tok_b(sel):
        return (torch.from_numpy(np.asarray(TOK[sel])).to(DEV).float(), torch.from_numpy(LAB[sel]).to(DEV))

    def glob_b(sel):
        return (G[sel].to(DEV), torch.from_numpy(LAB[sel]).to(DEV))

    def tok_t(sel):
        return tok_b(tr[sel])

    def glob_t(sel):
        return glob_b(tr[sel])

    yva = LAB[va]
    rows = []
    t0 = time.perf_counter()
    for seed in CFG["seeds"]:
        rec = {"seed": seed}
        for arm in ("TAP", "CA", "GH"):
            for lr in CFG["lrs"]:
                torch.manual_seed(seed)
                if arm == "TAP":
                    m = p8.T1(CFG["dim"], CFG["hid"])
                elif arm == "CA":
                    m = p8.CLIPAdapter(text, CFG["gd"], CFG["ca_hid"])
                else:
                    m = p8.GHead(CFG["gd"], CFG["gh"])
                bt = tok_t if arm == "TAP" else glob_t
                sc = tok_b if arm == "TAP" else glob_b
                m2 = p8.train_arm(m, bt, len(tr), lr, seed)
                rec["%s_lr%g" % (arm, lr)] = p8.uar(yva, p8.eval_preds(m2, sc, va))
                torch.save(m2.state_dict(), OUT / ("phase19_%s_B32LAION_seed%d_lr%g.pt" % (arm.lower(), seed, lr)))
                del m2
                torch.cuda.empty_cache()
        rows.append(rec)
        print("  seed %d TAP %.4f CA %.4f GH %.4f (%.0fs)"
              % (seed, rec["TAP_lr0.003"], rec["CA_lr0.001"], rec["GH_lr0.003"], time.perf_counter() - t0), flush=True)
    (OUT / "phase19_train_B32LAION.json").write_text(json.dumps({"rows": rows, "config": CFG, "weights": str(WEIGHTS)}, indent=2), encoding="utf-8")
    return rows


def evaluate():
    models = {}
    for seed in CFG["seeds"]:
        for arm in ("TAP", "CA", "GH"):
            for lr in CFG["lrs"]:
                if arm == "TAP":
                    m = p8.T1(CFG["dim"], CFG["hid"])
                elif arm == "CA":
                    m = p8.CLIPAdapter(np.zeros((7, CFG["gd"]), np.float32), CFG["gd"], CFG["ca_hid"])
                else:
                    m = p8.GHead(CFG["gd"], CFG["gh"])
                m.load_state_dict(torch.load(OUT / ("phase19_%s_B32LAION_seed%d_lr%g.pt" % (arm.lower(), seed, lr)),
                                             map_location=DEV))
                models[(arm, seed, lr)] = m.to(DEV).eval()
    res = {}
    for name, _, man, kind in sources():
        if kind == "train":
            continue
        gz = np.load(FEATDIR / ("%s.npz" % name), allow_pickle=False)
        y = gz["labels"].astype(np.int64)
        g = torch.from_numpy(norm(gz["views"].mean(axis=1).astype(np.float32))).to(DEV)
        TOK = np.load(TOKDIR / ("%s_tokens_fp16.npy" % name), mmap_mode="r")
        preds = {k: [] for k in models}
        with torch.inference_mode():
            for i in range(0, len(y), 256):
                tok = torch.from_numpy(np.asarray(TOK[i:i + 256])).to(DEV).float()
                gp = g[i:i + 256]
                for k, m in models.items():
                    preds[k].append((m(tok) if k[0] == "TAP" else m(gp)).argmax(1).cpu().numpy())
                del tok
        for k in models:
            res.setdefault(k, {})[name] = uar(y, np.concatenate(preds[k]))
        del TOK, g, preds
        torch.cuda.empty_cache()
        print("  scored", name, flush=True)

    pub = CFG["published"]
    rep = {"encoder": "LAION-2B ViT-B/32", "weights": str(WEIGHTS), "conditions12": CONDS12,
           "conditions_fresh": CONDSC, "seeds": CFG["seeds"], "summary": {}, "per_seed": {}}
    for label, conds in (("axis12", CONDS12), ("axisC", CONDSC)):
        for comp in ("CA", "GH"):
            v = np.array([np.mean([res[(("TAP"), s, pub["TAP"])][k] for k in conds])
                          - np.mean([res[(comp, s, pub[comp])][k] for k in conds]) for s in CFG["seeds"]])
            mean, lo, hi = t_ci(v, 2.262)
            rep["summary"]["%s|TAP_minus_%s" % (label, comp)] = {
                "mean": mean, "sd": float(v.std(ddof=1)), "t95_ci": [lo, hi],
                "seeds_positive": int((v > 0).sum()), "n": int(len(v)), "per_seed": [float(x) for x in v]}
            print("%-26s mean %+.4f sd %.4f ci [%+.4f,%+.4f] pos %d/%d"
                  % (label + "|TAP-" + comp, mean, v.std(ddof=1), lo, hi, int((v > 0).sum()), len(v)), flush=True)
    v = np.array([res[("TAP", s, pub["TAP"])]["fer2013_test"] - res[("CA", s, pub["CA"])]["fer2013_test"]
                  for s in CFG["seeds"]])
    mean, lo, hi = t_ci(v, 2.262)
    rep["summary"]["clean|TAP_minus_CA"] = {"mean": mean, "sd": float(v.std(ddof=1)),
                                            "t95_ci": [lo, hi], "seeds_positive": int((v > 0).sum()),
                                            "n": int(len(v)), "per_seed": [float(x) for x in v]}
    print("%-26s mean %+.4f sd %.4f ci [%+.4f,%+.4f] pos %d/%d"
          % ("clean|TAP-CA", mean, v.std(ddof=1), lo, hi, int((v > 0).sum()), len(v)), flush=True)
    s12 = rep["summary"]["axis12|TAP_minus_CA"]
    sC = rep["summary"]["axisC|TAP_minus_CA"]
    rep["verdict"] = {
        "axis12": "CONFIRMED" if (s12["mean"] > 0 and s12["t95_ci"][0] > 0) else "NOT CONFIRMED",
        "axisC": "CONFIRMED" if (sC["mean"] > 0 and sC["t95_ci"][0] > 0) else "NOT CONFIRMED"}
    (OUT / "phase19_eval_B32LAION.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("VERDICT axis12:", rep["verdict"]["axis12"], " axisC:", rep["verdict"]["axisC"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["extract", "train", "eval", "all"])
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.stage in ("all", "extract"):
        print("extracting", flush=True)
        extract()
    if a.stage in ("all", "train"):
        print("training", flush=True)
        train()
    if a.stage in ("all", "eval"):
        print("evaluating", flush=True)
        evaluate()
    print("done %.0fs" % (time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()

