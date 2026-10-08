
# -*- coding: utf-8 -*-
"""Phase 13b: patch tokens and two-view global features for the JAFFE fresh axis.

Convention copied verbatim from phase6b_extract.py so that the fresh axis is pixel- and
feature-comparable with the original one:
  tokens  = model.visual(x)[1][0::2]      (original view only, fp16)
  globals = normalize(model.visual(x)[0]) (two views: image, mirror; fp32, unit norm)

Conditions are the pre-specified 20 (4 families x 5 severities). The clean images are extracted too,
but only as a descriptive reference; they are not part of the pre-specified grid.
"""
import argparse, csv, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = Path("D:/ResearchVault/99system/data/rb-tta-fer-fg2027")
ROOT = D / "raw/jaffe_confirm"
OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
CONDS = ["blur_0p8", "blur_1p5", "blur_2p5", "blur_4p0", "blur_6p0",
         "jpeg_15", "jpeg_30", "jpeg_50", "jpeg_65", "jpeg_80",
         "lowlight_0p3", "lowlight_0p5", "lowlight_0p7", "lowlight_0p85", "lowlight_0p95",
         "noise_10", "noise_25", "noise_40", "noise_55", "noise_70"]
ENC = {
    "B16": dict(model="ViT-B-16", weights=Path("D:/ResearchVault/99system/models/open_clip/ViT-B-16.pt"),
                d_glob=512, n_tok=196, d_tok=768, rows=128),
    "L14": dict(model="ViT-L-14", weights=Path("D:/ResearchVault/99system/models/open_clip/ViT-L-14.pt"),
                d_glob=768, n_tok=256, d_tok=1024, rows=64),
}


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_manifest(cond):
    path = ROOT / "jaffe_index.csv" if cond == "original" else ROOT / "manifests" / ("jaffe_%s.csv" % cond)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r["path"] for r in rows], np.asarray([int(r["label"]) for r in rows], np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["B16", "L14"])
    args = ap.parse_args()
    tag = args.encoder
    c = ENC[tag]
    low = tag.lower()
    tokdir = D / ("token_cache_%s_jaffe" % low)
    featdir = D / "features_jaffe" / low
    tokdir.mkdir(parents=True, exist_ok=True)
    featdir.mkdir(parents=True, exist_ok=True)
    rep = {"encoder": tag, "model": c["model"], "weights": str(c["weights"]),
           "weights_sha256": sha(c["weights"]), "conditions": {},
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        c["model"], pretrained=str(c["weights"]), device="cuda", weights_only=False)
    model.eval()
    model.visual.output_tokens = True
    print(tag, "loaded", flush=True)

    def forward(paths):
        imgs = [Image.open(p).convert("RGB") for p in paths]
        pairs = []
        for im in imgs:
            pairs.extend([im, ImageOps.mirror(im)])
        x = torch.stack([preprocess(im) for im in pairs]).to("cuda")
        with torch.inference_mode():
            pooled, tokens = model.visual(x)
            pooled = F.normalize(pooled.float(), dim=-1)
        g = pooled.cpu().numpy().astype(np.float32).reshape(len(paths), 2, -1)
        t = tokens[0::2].half().cpu().numpy()
        del x, pooled, tokens
        return t, g

    t0 = time.perf_counter()
    for cond in ["original"] + CONDS:
        npy = tokdir / ("%s_tokens_fp16.npy" % cond)
        meta = tokdir / ("%s_meta.npz" % cond)
        feat = featdir / ("%s.npz" % cond)
        paths, labels = read_manifest(cond)
        assert len(paths) == 213, (cond, len(paths))
        mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16,
                                       shape=(len(paths), c["n_tok"], c["d_tok"]))
        g_all = np.empty((len(paths), 2, c["d_glob"]), np.float32)
        for i in range(0, len(paths), c["rows"]):
            t, g = forward(paths[i:i + c["rows"]])
            mm[i:i + t.shape[0]] = t
            g_all[i:i + g.shape[0]] = g
        mm.flush()
        del mm
        np.savez_compressed(meta, labels=labels, ids=np.asarray(paths))
        np.savez_compressed(feat, ids=np.asarray(paths), views=g_all, labels=labels)
        rep["conditions"][cond] = {"n": len(paths), "npy_GB": round(npy.stat().st_size / 1e9, 3),
                                   "unit_norm_views": bool(np.allclose(np.linalg.norm(g_all, axis=-1), 1.0, atol=1e-4)),
                                   "field_mean": float(g_all[:, 0].mean()),
                                   "npy_sha256": sha(npy)}
        print("  %s %-14s n=%d  %.0fs" % (tag, cond, len(paths), time.perf_counter() - t0), flush=True)

    rep["seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / ("phase13_extract_jaffe_%s.json" % tag)).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("done %.0fs" % (time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()

