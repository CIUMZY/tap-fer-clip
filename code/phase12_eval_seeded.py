
"""Phase 12b: evaluate the seed-controlled retraining set on all three confirmation axes.

For each encoder and axis two distributions are reported:
  * published rate - each arm at the rate the original protocol selected (ViT-B/16: TAP 3e-3,
    CLIP-Adapter 1e-3, global head 3e-3; ViT-L/14: 1e-3 for all three);
  * nested - for each seed, each arm's own probe-validation split picks its rate from the same
    grid used originally; the contrast is then formed from that seed's selected arms.

Neither axis is used for selection; the rate choice is made on the source probe split only.
"""
import importlib.util, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = Path("D:/ResearchVault/99system/data/rb-tta-fer-fg2027")
OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
DEV = "cuda"
KS = D / "features_kdef_source"
LRS = [1e-3, 3e-3]
CONDS = ["blur_0p8", "blur_1p5", "blur_2p5", "jpeg_15", "jpeg_30", "jpeg_50",
         "lowlight_0p3", "lowlight_0p5", "lowlight_0p7", "noise_10", "noise_25", "noise_40"]

spec = importlib.util.spec_from_file_location("p8", OUT / "phase8_trainvar.py")
p8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p8)

CFG = {
    "B16": dict(dim=768, gd=512, hid=256, gh=256, ca_hid=128, grid=14,
                tokdir=D / "token_cache_vitb16", feat=D / "features",
                corr_tok=D / "token_cache_b16_confirm", corr_feat=D / "features_confirm_shift" / "b16",
                seeds=list(range(5, 15)), published=dict(TAP=3e-3, CA=1e-3, GH=3e-3)),
    "L14": dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192, grid=16,
                tokdir=D / "token_cache_vitl14", feat=D / "features_vitl14",
                corr_tok=D / "token_cache_l14_confirm", corr_feat=D / "features_confirm_shift" / "l14",
                seeds=list(range(5, 20)), published=dict(TAP=1e-3, CA=1e-3, GH=1e-3)),
}


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def subset(lab, ratio, maj, seed):
    if ratio == 0:
        return np.arange(len(lab))
    rng = np.random.default_rng(seed)
    cnt = np.bincount(lab, minlength=7)
    base = int(cnt.min())
    return np.concatenate([rng.choice(np.flatnonzero(lab == c),
                                      size=min(int(cnt[c]), base * (ratio if c == maj else 1)),
                                      replace=False) for c in range(7)])


def load_tok(tokdir, name):
    """ViT-B/16 stores its per-set caches as .npz, ViT-L/14 as .npy; accept both."""
    p = tokdir / (name + "_tokens_fp16.npy")
    if p.exists():
        return np.load(p, mmap_mode="r")
    return np.load(tokdir / (name + "_tokens_fp16.npz"), allow_pickle=False)["tokens"]


def t_ci(a, tcrit):
    a = np.asarray(a, float)
    se = a.std(ddof=1) / math.sqrt(len(a))
    return float(a.mean()), float(a.mean() - tcrit * se), float(a.mean() + tcrit * se)


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


def eval_corruption(tag, M):
    c = CFG[tag]
    res = {}
    for cond in CONDS:
        gz = np.load(c["corr_feat"] / ("%s.npz" % cond), allow_pickle=False)
        y = gz["labels"].astype(np.int64)
        g = torch.from_numpy(norm(gz["views"].mean(axis=1).astype(np.float32))).to(DEV)
        TOK = np.load(c["corr_tok"] / ("%s_tokens_fp16.npy" % cond), mmap_mode="r")
        preds = {k: [] for k in M}
        preds_g = {k: [] for k in M if k[0] != "TAP"}
        with torch.inference_mode():
            for i in range(0, len(y), 256):
                tok = torch.from_numpy(np.asarray(TOK[i:i + 256])).to(DEV).float()
                gp = g[i:i + 256]
                for k, m in M.items():
                    if k[0] == "TAP":
                        preds[k].append(m(tok).argmax(1).cpu().numpy())
                    else:
                        preds_g[k].append(m(gp).argmax(1).cpu().numpy())
                del tok
        for k in M:
            src = preds[k] if k[0] == "TAP" else preds_g[k]
            res.setdefault(k, {})[cond] = uar(y, np.concatenate(src))
        del TOK, g, preds, preds_g
        torch.cuda.empty_cache()
        print("  %s corruption %-12s %.0fs" % (tag, cond, time.perf_counter()), flush=True)
    return res


def eval_setA(tag, M):
    c = CFG[tag]
    res = {}
    for tname, maj in (("ckplus_test", 6), ("kdef_test", 6)):
        zf = np.load(c["feat"] / ("%s.npz" % tname), allow_pickle=False)
        gv = norm(zf["views"].mean(axis=1))
        lab = zf["labels"].astype(np.int64)
        TOKs = load_tok(c["tokdir"], tname)
        for ratio in (0, 1, 2, 5, 10):
            for cseed in range(5):
                idx = subset(lab, ratio, maj, cseed)
                y = lab[idx]
                g = torch.from_numpy(gv[idx].astype(np.float32)).to(DEV)
                tok = torch.from_numpy(np.asarray(TOKs[idx])).to(DEV).float()
                key = "%s|%d|%d" % (tname, ratio, cseed)
                with torch.inference_mode():
                    for k, m in M.items():
                        p = m(tok).argmax(1).cpu().numpy() if k[0] == "TAP" else m(g).argmax(1).cpu().numpy()
                        res.setdefault(k, {})[key] = uar(y, p)
                del tok, g
                torch.cuda.empty_cache()
        print("  %s setA %s %.0fs" % (tag, tname, time.perf_counter()), flush=True)
    return res


def build_targets(tag):
    c = CFG[tag]
    ks = np.load(KS / "kdef_source_test.npz", allow_pickle=False)
    ks_ids = ks["ids"].astype(str)
    ks_y = ks["labels"].astype(np.int64)
    out = []
    if tag == "B16":
        z = np.load(c["tokdir"] / "kdef_test_tokens_fp16.npz", allow_pickle=False)
        pos = {x: i for i, x in enumerate(z["ids"].astype(str))}
        sel = np.asarray([pos[x] for x in ks_ids], dtype=np.int64)
        out.append(("kdef_source_test", z["tokens"][sel], ks["views"].mean(axis=1).astype(np.float32), ks_y))
        for t in ("fer2013_test", "ckplus_test"):
            zt = np.load(c["tokdir"] / ("%s_tokens_fp16.npz" % t), allow_pickle=False)
            gt = np.load(c["feat"] / ("%s.npz" % t), allow_pickle=False)
            out.append((t, zt["tokens"], norm(gt["views"].mean(axis=1).astype(np.float32)),
                        gt["labels"].astype(np.int64)))
    else:
        gz = np.load(c["feat"] / "kdef_test.npz", allow_pickle=False)
        kpos = {x: i for i, x in enumerate(gz["ids"].astype(str))}
        idx = np.asarray([kpos[x] for x in ks_ids], dtype=np.int64)
        out.append(("kdef_source_test", np.load(c["tokdir"] / "kdef_test_tokens_fp16.npy", mmap_mode="r")[idx],
                    norm(gz["views"].mean(axis=1).astype(np.float32))[idx], ks_y))
        for t in ("fer2013_test", "ckplus_test"):
            gt = np.load(c["feat"] / ("%s.npz" % t), allow_pickle=False)
            out.append((t, np.load(c["tokdir"] / ("%s_tokens_fp16.npy" % t), mmap_mode="r"),
                        norm(gt["views"].mean(axis=1).astype(np.float32)), gt["labels"].astype(np.int64)))
    return out


def eval_setB(tag, M):
    res = {}
    for tname, TOK, gv, y in build_targets(tag):
        gt = torch.from_numpy(np.ascontiguousarray(norm(gv))).to(DEV)
        n = len(y)
        with torch.inference_mode():
            for k, m in M.items():
                if k[0] == "TAP":
                    outs = []
                    for i in range(0, n, 256):
                        xb = torch.from_numpy(np.asarray(TOK[i:i + 256])).to(DEV).float()
                        outs.append(m(xb).argmax(1).cpu().numpy())
                    p = np.concatenate(outs)
                else:
                    p = m(gt).argmax(1).cpu().numpy()
                res.setdefault(k, {})[tname] = uar(y, p)
        del gt
        torch.cuda.empty_cache()
        print("  %s setB %s %.0fs" % (tag, tname, time.perf_counter()), flush=True)
    return res


def main():
    t0 = time.perf_counter()
    report = {}
    for tag in ("B16", "L14"):
        c = CFG[tag]
        probes = json.loads((OUT / ("phase12_train_%s.json" % tag)).read_text(encoding="utf-8"))["rows"]
        probe = {(r["seed"], arm, lr): r["%s_lr%g" % (arm, lr)]
                 for r in probes for arm in ("TAP", "CA", "GH") for lr in LRS}
        M = load_models(tag)
        print("%s models loaded %.0fs" % (tag, time.perf_counter() - t0), flush=True)
        axes = {"corruption": eval_corruption(tag, M), "setA": eval_setA(tag, M), "setB": eval_setB(tag, M)}
        tcrit = 2.262 if len(c["seeds"]) == 10 else 2.145

        def arm_mean(axis, arm, s, lr):
            return float(np.mean(list(axes[axis][(arm, s, lr)].values())))

        def nested_lr(arm, s):
            best = None
            for lr in LRS:                     # ties resolve to the smaller rate
                v = probe[(s, arm, lr)]
                if best is None or v > best[1] + 1e-12:
                    best = (lr, v)
            return best[0]

        rec = {"encoder": tag, "seeds": c["seeds"], "lrs": LRS, "published": c["published"],
               "nested_lr_choice": {}, "summary": {}}
        series = {}
        for axis in axes:
            pub, nest = {}, {}
            for s in c["seeds"]:
                pub[s] = {a: arm_mean(axis, a, s, c["published"][a]) for a in ("TAP", "CA", "GH")}
                ns = {a: nested_lr(a, s) for a in ("TAP", "CA", "GH")}
                rec["nested_lr_choice"].setdefault(axis, {})[str(s)] = ns
                nest[s] = {a: arm_mean(axis, a, s, ns[a]) for a in ("TAP", "CA", "GH")}
            series[axis] = {"published": pub, "nested": nest}
            for label, src in (("published", pub), ("nested", nest)):
                for comp in ("CA", "GH"):
                    v = np.array([src[s]["TAP"] - src[s][comp] for s in c["seeds"]])
                    mean, lo, hi = t_ci(v, tcrit)
                    p, pos, n = sign_p(v)
                    rec["summary"]["%s|%s|TAP_minus_%s" % (axis, label, comp)] = {
                        "mean": mean, "sd": float(v.std(ddof=1)), "min": float(v.min()),
                        "max": float(v.max()), "t95_ci": [lo, hi], "sign_test_p": p,
                        "seeds_positive": pos, "n": n, "includes_zero": bool(lo <= 0 <= hi)}
                    print("%s %s %s|TAP-%s mean %+.4f sd %.4f ci [%+.4f,%+.4f] pos %d/%d p %.4f"
                          % (tag, axis, label, comp, mean, v.std(ddof=1), lo, hi, pos, n, p), flush=True)
        rng = np.random.default_rng(0)
        for axis in axes:
            for label in ("published", "nested"):
                src = series[axis][label]
                draws = np.array([np.mean([src[c["seeds"][i]]["TAP"] - src[c["seeds"][i]]["CA"]
                                           for i in rng.integers(0, len(c["seeds"]), len(c["seeds"]))])
                                  for _ in range(10000)])
                rec["summary"]["%s|%s|seed_bootstrap_CA" % (axis, label)] = {
                    "mean": float(draws.mean()),
                    "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
        rec["raw"] = {axis: {("%s|%d|%g" % (a, s, lr)): axes[axis][(a, s, lr)]
                             for (a, s, lr) in axes[axis]} for axis in axes}
        rec["per_seed"] = series
        report[tag] = rec
        (OUT / "phase12_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("done %.0fs" % (time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()

