
"""Phase 9b: does the surviving ViT-L/14 degradation-robustness margin survive a symmetric
feature convention?

The main comparison measures each route in its native configuration: the global route consumes
the L2-normalised mean of the image and its mirror, the token route consumes the original view
only.  The manuscript reports a post-hoc sensitivity showing that on the cross-database sets the
aligned two-view token convention reverses the token-route margin over CLIP-Adapter (phase7c).
The same sensitivity was never run on the degradation axis, so whether the one surviving
contrast (ViT-L/14, +3.70 points) is equally convention-dependent is untested.

This script closes that gap.  It

  1. trains the token arm on the spatially aligned two-view mean
     (original + unflip(mirror)) / 2, selecting the learning rate on the source probe-validation
     split only, exactly as the original protocol did;
  2. streams the 12 pre-specified corruption conditions, forwards the mirror view of every
     corrupted image, unflips it into the token grid and forms the aligned two-view mean
     against the existing original-view cache;
  3. scores the aligned two-view arm, the original single-view arm and the frozen per-seed
     comparator arms (CLIP-Adapter, capacity-matched global head) on exactly the same units.

Nothing is selected on the corruption axis: the learning rate is chosen on the source split, the
comparators are the existing per-seed checkpoints, and the axis is reported as a whole.
"""

import argparse, csv, importlib.util, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = Path("D:/ResearchVault/99system/data/rb-tta-fer-fg2027")
OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
DEV = "cuda"
CONDS = ["blur_0p8", "blur_1p5", "blur_2p5", "jpeg_15", "jpeg_30", "jpeg_50",
         "lowlight_0p3", "lowlight_0p5", "lowlight_0p7", "noise_10", "noise_25", "noise_40"]

spec = importlib.util.spec_from_file_location("p8", OUT / "phase8_trainvar.py")
p8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p8)

CFG = {
    "B16": dict(model="ViT-B-16",
                weights=Path("D:/ResearchVault/99system/models/open_clip/ViT-B-16.pt"),
                dim=768, gd=512, hid=256, gh=256, ca_hid=128, grid=14, rows=128,
                clean_tokdir=D / "token_cache_vitb16",
                mirror_tokdir=D / "token_cache_vitb16_mirror",
                label_meta=D / "token_cache_vitb16" / "fer2013_train_tokens_fp16.npz",
                feat=D / "features",
                corr_tokdir=D / "token_cache_b16_confirm",
                corr_feat=D / "features_confirm_shift" / "b16",
                seeds=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    "L14": dict(model="ViT-L-14",
                weights=Path("D:/ResearchVault/99system/models/open_clip/ViT-L-14.pt"),
                dim=1024, gd=768, hid=256, gh=260, ca_hid=192, grid=16, rows=64,
                clean_tokdir=D / "token_cache_vitl14",
                mirror_tokdir=D / "token_cache_vitl14_mirror",
                label_meta=D / "token_cache_vitl14" / "fer2013_train_meta.npz",
                feat=D / "features_vitl14",
                corr_tokdir=D / "token_cache_l14_confirm",
                corr_feat=D / "features_confirm_shift" / "l14",
                seeds=[5, 6, 7, 8, 9]),
}


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def unflip(b, grid):
    n = b.shape[0]
    g = b.reshape(n, grid, grid, -1)
    g = g[:, :, ::-1, :]
    return np.ascontiguousarray(g.reshape(n, grid * grid, -1))


def t_ci(a):
    a = np.asarray(a, float)
    n = len(a)
    if n < 2:
        return (float(a.mean()), float("nan"), float("nan"))
    se = a.std(ddof=1) / math.sqrt(n)
    df = n - 1
    tcrit = {4: 2.776, 9: 2.262, 5: 2.571, 14: 2.145, 19: 2.093}.get(df, 2.042)
    return float(a.mean()), float(a.mean() - tcrit * se), float(a.mean() + tcrit * se)


def sign_test_p(a):
    a = np.asarray(a, float)
    pos = int((a > 0).sum())
    neg = int((a < 0).sum())
    n = pos + neg
    if n == 0:
        return 1.0, pos, n
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2.0
    return float(min(1.0, p)), pos, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="L14", choices=["B16", "L14"])
    ap.add_argument("--lr", type=float, default=None,
                    help="force the token-arm learning rate instead of re-selecting it on the source "
                         "probe split; use this to hold the learning rate at the value the original "
                         "protocol selected so that only the feature convention changes")
    ap.add_argument("--suffix", default="", help="suffix for the output file names")
    args = ap.parse_args()
    tag = args.encoder
    c = CFG[tag]
    grid = c["grid"]
    SUF = args.suffix
    t0all = time.perf_counter()

    split = np.load(c["feat"] / "linear_probe_split_seed0.npz", allow_pickle=False)
    tr = np.asarray(split["train_indices"], dtype=np.int64)
    va = np.asarray(split["validation_indices"], dtype=np.int64)
    TOK = np.load(c["clean_tokdir"] / "fer2013_train_tokens_fp16.npy", mmap_mode="r")
    LAB = np.load(c["label_meta"], allow_pickle=False)["labels"].astype(np.int64)
    MM = np.load(c["mirror_tokdir"] / "fer2013_train_tokens_fp16.npy", mmap_mode="r")
    assert MM.shape[0] == len(LAB), (MM.shape, len(LAB))
    print("%s train tokens %s mirror %s" % (tag, TOK.shape, MM.shape), flush=True)

    def two_view(sel):
        sel = np.asarray(sel, np.int64)
        a = torch.from_numpy(np.asarray(TOK[sel])).to(DEV).float()
        b = torch.from_numpy(np.asarray(MM[sel])).to(DEV).float()
        n = b.shape[0]
        b = b.reshape(n, grid, grid, -1).flip(2).reshape(n, grid * grid, -1)
        return (a + b) / 2.0

    aligned_path = D / ("token_cache_vit%s_aligned2v" % ("l14" if tag == "L14" else "b16")) \
        / "fer2013_train_tokens_fp16.npy"
    ALIGNED = np.load(aligned_path, mmap_mode="r") if aligned_path.exists() else None
    print("%s aligned cache: %s" % (tag, "precomputed" if ALIGNED is not None else "on the fly"), flush=True)

    def train_input(sel):
        sel = np.asarray(sel, np.int64)
        if ALIGNED is not None:
            return torch.from_numpy(np.asarray(ALIGNED[sel])).to(DEV).float()
        return two_view(sel)

    text = norm(np.load(c["feat"] / "text_prototypes.npz", allow_pickle=False)["prototypes"].astype(np.float32))
    gtr = norm(np.load(c["feat"] / "fer2013_train.npz", allow_pickle=False)["views"].mean(axis=1))
    G = torch.from_numpy(gtr.astype(np.float32))

    yva = LAB[va]
    sel_rows = []
    trs = np.sort(tr)
    if args.lr is not None:
        best_lr = float(args.lr)
        sel_rows = [{"lr": best_lr, "forced": True,
                     "note": "learning rate held at the value the original protocol selected"}]
        print("%s forced lr %s (no re-selection)" % (tag, best_lr), flush=True)
    else:
        best_lr = None
    for lr in (() if best_lr is not None else (1e-3, 3e-3)):
        torch.manual_seed(0)
        m = p8.T1(c["dim"], c["hid"]).to(DEV)
        opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=lr)
        g = torch.Generator().manual_seed(0)
        for _ in range(p8.EPOCHS):
            perm = torch.randperm(len(trs), generator=g)
            for i in range(0, len(trs), p8.BATCH):
                sel = trs[perm[i:i + p8.BATCH].numpy()]
                xb = train_input(sel)
                yb = torch.from_numpy(LAB[sel]).to(DEV)
                loss = torch.nn.functional.cross_entropy(m(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        m.eval()
        preds = []
        with torch.inference_mode():
            for i in range(0, len(va), 256):
                s = va[i:i + 256]
                preds.append(m(train_input(s)).argmax(1).cpu().numpy())
        sel_rows.append({"lr": lr, "probe_val_uar": uar(yva, np.concatenate(preds))})
        print("  lr %s probe-val %.4f" % (lr, sel_rows[-1]["probe_val_uar"]), flush=True)
    if best_lr is None:
        best_lr = max(sel_rows, key=lambda r: r["probe_val_uar"])["lr"]
        print("%s selected lr %s" % (tag, best_lr), flush=True)

    aligned = {}
    for seed in c["seeds"]:
        torch.manual_seed(seed)
        m = p8.T1(c["dim"], c["hid"]).to(DEV)
        opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=best_lr)
        gen = torch.Generator().manual_seed(seed)
        for _ in range(p8.EPOCHS):
            perm = torch.randperm(len(trs), generator=gen)
            for i in range(0, len(trs), p8.BATCH):
                sel = trs[perm[i:i + p8.BATCH].numpy()]
                xb = train_input(sel)
                yb = torch.from_numpy(LAB[sel]).to(DEV)
                loss = torch.nn.functional.cross_entropy(m(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        m.eval()
        torch.save(m.state_dict(), OUT / ("phase9_aligned_tap_%s%s_seed%d.pt" % (tag, SUF, seed)))
        preds = []
        with torch.inference_mode():
            for i in range(0, len(va), 256):
                s = va[i:i + 256]
                preds.append(m(train_input(s)).argmax(1).cpu().numpy())
        pv = uar(yva, np.concatenate(preds))
        aligned[seed] = m
        print("  aligned seed %d probe-val %.4f (%.0fs)" % (seed, pv, time.perf_counter() - t0all), flush=True)

    single, ca, gh = {}, {}, {}
    for seed in c["seeds"]:
        m = p8.T1(c["dim"], c["hid"]).to(DEV)
        m.load_state_dict(torch.load(OUT / ("phase8_tap_%s_seed%d.pt" % (tag, seed)), map_location=DEV))
        m.eval()
        single[seed] = m
        m = p8.CLIPAdapter(np.zeros((7, c["gd"]), np.float32), c["gd"], c["ca_hid"]).to(DEV)
        m.load_state_dict(torch.load(OUT / ("phase8_clipadapter_%s_seed%d.pt" % (tag, seed)), map_location=DEV))
        m.eval()
        ca[seed] = m
        m = p8.GHead(c["gd"], c["gh"]).to(DEV)
        m.load_state_dict(torch.load(OUT / ("phase8_ghead_%s_seed%d.pt" % (tag, seed)), map_location=DEV))
        m.eval()
        gh[seed] = m

    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        c["model"], pretrained=str(c["weights"]), device="cuda", weights_only=False)
    model.eval()
    model.visual.output_tokens = True

    per = {s: {"aligned": {}, "single": {}, "ca": {}, "gh": {}} for s in c["seeds"]}
    cond_rows = []
    for cond in CONDS:
        t0 = time.perf_counter()
        gz = np.load(c["corr_feat"] / ("%s.npz" % cond), allow_pickle=False)
        y = gz["labels"].astype(np.int64)
        g = norm(gz["views"].mean(axis=1).astype(np.float32))
        paths = np.load(c["corr_tokdir"] / ("%s_meta.npz" % cond), allow_pickle=False)["ids"].astype(str)
        ORIG = np.load(c["corr_tokdir"] / ("%s_tokens_fp16.npy" % cond), mmap_mode="r")
        assert len(ORIG) == len(y) == len(paths), (len(ORIG), len(y), len(paths))
        gt = torch.from_numpy(g).to(DEV)
        pa = {s: [] for s in c["seeds"]}
        ps = {s: [] for s in c["seeds"]}
        with torch.inference_mode():
            for i in range(0, len(y), c["rows"]):
                chunk = paths[i:i + c["rows"]]
                imgs = [ImageOps.mirror(Image.open(p).convert("RGB")) for p in chunk]
                x = torch.stack([preprocess(im) for im in imgs]).to("cuda")
                _, tokens = model.visual(x)
                mraw = tokens.half().cpu()
                del x, tokens
                o = torch.from_numpy(np.asarray(ORIG[i:i + c["rows"]])).to(DEV).float()
                n = mraw.shape[0]
                mg = mraw.to(DEV).float().reshape(n, grid, grid, -1).flip(2).reshape(n, grid * grid, -1)
                tw = (o + mg) / 2.0
                ov = o
                for s in c["seeds"]:
                    pa[s].append(aligned[s](tw).argmax(1).cpu().numpy())
                    ps[s].append(single[s](ov).argmax(1).cpu().numpy())
                del tw, ov, o, mraw
        for s in c["seeds"]:
            cp = ca[s](gt).argmax(1).cpu().numpy()
            gp = gh[s](gt).argmax(1).cpu().numpy()
            per[s]["aligned"][cond] = uar(y, np.concatenate(pa[s]))
            per[s]["single"][cond] = uar(y, np.concatenate(ps[s]))
            per[s]["ca"][cond] = uar(y, cp)
            per[s]["gh"][cond] = uar(y, gp)
        cond_rows.append({"condition": cond, "n": int(len(y)),
                          **{"seed%d_%s" % (s, a): per[s][a][cond]
                             for s in c["seeds"] for a in ("aligned", "single", "ca", "gh")}})
        print("  %s %s done %.0fs" % (tag, cond, time.perf_counter() - t0), flush=True)
        del ORIG, gt
        torch.cuda.empty_cache()

    def seed_vec(arm):
        return np.array([float(np.mean([per[s][arm][k] for k in CONDS])) for s in c["seeds"]])

    A, Sg, CAv, GHv = seed_vec("aligned"), seed_vec("single"), seed_vec("ca"), seed_vec("gh")
    contrasts = {
        "single_TAP_minus_CA": Sg - CAv,
        "single_TAP_minus_GH": Sg - GHv,
        "aligned2v_TAP_minus_CA": A - CAv,
        "aligned2v_TAP_minus_GH": A - GHv,
        "aligned2v_minus_single_TAP": A - Sg,
    }
    summary = {}
    for k, v in contrasts.items():
        mean, lo, hi = t_ci(v)
        p, pos, n = sign_test_p(v)
        summary[k] = {"mean": mean, "sd": float(v.std(ddof=1)), "min": float(v.min()),
                      "max": float(v.max()), "t95_ci": [lo, hi],
                      "ci_excludes_zero": bool(lo > 0 or hi < 0),
                      "sign_test_p": p, "seeds_positive": pos, "seeds_nondegenerate": n}

    rng = np.random.default_rng(0)
    boot = {}
    for k, pair in (("aligned2v_TAP_minus_CA", ("aligned", "ca")),
                    ("aligned2v_TAP_minus_GH", ("aligned", "gh")),
                    ("single_TAP_minus_CA", ("single", "ca"))):
        draws = []
        for _ in range(2000):
            idx = rng.integers(0, len(CONDS), len(CONDS))
            vals = []
            for s in c["seeds"]:
                a = np.array([per[s][pair[0]][CONDS[j]] for j in idx])
                b = np.array([per[s][pair[1]][CONDS[j]] for j in idx])
                vals.append(float((a - b).mean()))
            draws.append(float(np.mean(vals)))
        draws = np.array(draws)
        boot[k] = {"mean": float(draws.mean()),
                   "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}

    rep = {"encoder": tag, "suffix": SUF, "model": c["model"], "conditions": CONDS,
           "n_seeds": len(c["seeds"]),
           "seeds": c["seeds"], "lr_selection": sel_rows, "selected_lr": best_lr,
           "per_condition": cond_rows,
           "per_seed_means": {"aligned2v_TAP": A.tolist(), "single_TAP": Sg.tolist(),
                              "CLIPAdapter": CAv.tolist(), "GHead": GHv.tolist()},
           "contrasts": summary, "bootstrap_conditions": boot,
           "wall_seconds": round(time.perf_counter() - t0all, 1),
           "scope": "post-hoc sensitivity on a consumed confirmation axis; learning rate selected "
                    "on the source probe-validation split only; no target label used for training "
                    "or selection"}
    (OUT / ("phase9_aligned_corruption_%s%s.json" % (tag, SUF))).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    with (OUT / ("phase9_aligned_corruption_%s%s.csv" % (tag, SUF))).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["encoder", "seed", "aligned2v_TAP", "single_TAP", "CLIPAdapter", "GHead",
                    "aligned_minus_CA", "aligned_minus_GH", "single_minus_CA"])
        for i, s in enumerate(c["seeds"]):
            w.writerow([tag, s, round(A[i], 6), round(Sg[i], 6), round(CAv[i], 6), round(GHv[i], 6),
                        round(A[i] - CAv[i], 6), round(A[i] - GHv[i], 6), round(Sg[i] - CAv[i], 6)])
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(boot, indent=2), flush=True)
    print("done in %.0fs" % (time.perf_counter() - t0all), flush=True)


if __name__ == "__main__":
    main()

