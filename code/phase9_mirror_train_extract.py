
"""Phase 9a: mirror-view patch tokens for the FER2013 training split (ViT-L/14, ViT-B/16).

The clean token caches hold the original view only.  The feature-convention sensitivity needs a
spatially aligned two-view mean, so this script forwards the horizontal mirror of every training
image and stores the mirror-view patch tokens as a memory-mapped fp16 array in the SAME row order
as the clean cache.  Convention copied verbatim from phase6b_extract.py:

    tokens = model.visual(x)[1]        # patch tokens, fp16
    x      = open_clip preprocess applied to the mirror image only

so that unflip(mirror[k]) aligns position-by-position with original[k].
"""

import argparse, hashlib, json, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = Path("D:/ResearchVault/99system/data/rb-tta-fer-fg2027")
OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")

ENC = {
    "L14": dict(model="ViT-L-14",
                weights=Path("D:/ResearchVault/99system/models/open_clip/ViT-L-14.pt"),
                meta=D / "token_cache_vitl14" / "fer2013_train_meta.npz",
                cache_dir="token_cache_vitl14_mirror", rows=64),
    "B16": dict(model="ViT-B-16",
                weights=Path("D:/ResearchVault/99system/models/open_clip/ViT-B-16.pt"),
                meta=D / "token_cache_vitb16" / "fer2013_train_tokens_fp16.npz",
                cache_dir="token_cache_vitb16_mirror", rows=128),
}


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="L14", choices=["L14", "B16"])
    args = ap.parse_args()
    tag = args.encoder
    enc = ENC[tag]
    dst = D / enc["cache_dir"]
    dst.mkdir(parents=True, exist_ok=True)
    npy = dst / "fer2013_train_tokens_fp16.npy"
    meta_out = dst / "fer2013_train_meta.npz"

    z = np.load(enc["meta"], allow_pickle=False)
    ids = z["ids"].astype(str)
    labels = z["labels"].astype(np.int64)
    del z
    print("%s rows=%d first=%s" % (tag, len(ids), ids[0]), flush=True)
    assert not [p for p in ids[:64] if not Path(p).exists()]

    rep = {"encoder": tag, "model": enc["model"], "weights": str(enc["weights"]),
           "weights_sha256": sha(enc["weights"]), "n_rows": int(len(ids)),
           "view": "horizontal mirror; spatially unflipped when the two-view mean is formed",
           "source_meta": str(enc["meta"]), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "free_GB_on_D": round(shutil.disk_usage("D:/").free / 1e9, 1)}

    if npy.exists() and meta_out.exists():
        got = np.load(npy, mmap_mode="r")
        rep["reused"] = True
        rep["tokens_shape"] = list(got.shape)
        rep["npy_sha256"] = sha(npy)
        print(tag, "reused", got.shape, flush=True)
        (OUT / ("phase9_extract_%s.json" % tag)).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        return

    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        enc["model"], pretrained=str(enc["weights"]), device="cuda", weights_only=False)
    model.eval()
    model.visual.output_tokens = True
    print(enc["model"], "loaded", flush=True)

    mm = None
    t0 = time.perf_counter()
    for start in range(0, len(ids), enc["rows"]):
        chunk = ids[start:start + enc["rows"]]
        imgs = [ImageOps.mirror(Image.open(p).convert("RGB")) for p in chunk]
        x = torch.stack([preprocess(im) for im in imgs]).to("cuda")
        with torch.inference_mode():
            _, tokens = model.visual(x)
        t = tokens.half().cpu().numpy()
        del x, tokens
        if mm is None:
            mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16,
                                           shape=(len(ids), t.shape[1], t.shape[2]))
        mm[start:start + t.shape[0]] = t
        if (start // enc["rows"]) % 25 == 0:
            print("  %s %d/%d %.0fs" % (tag, start + len(chunk), len(ids),
                                        time.perf_counter() - t0), flush=True)
    mm.flush()
    del mm
    np.savez_compressed(meta_out, ids=ids, labels=labels)
    got = np.load(npy, mmap_mode="r")
    rep.update({"tokens_shape": list(got.shape),
                "npy_GB": round(npy.stat().st_size / 1e9, 3),
                "seconds": round(time.perf_counter() - t0, 1),
                "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "npy_sha256": sha(npy)})
    (OUT / ("phase9_extract_%s.json" % tag)).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()

