
"""Phase 9c: materialise the spatially aligned two-view token cache.

(original[k] + unflip(mirror[k])) / 2, stored as fp16 in the same row order as the clean cache.

Motivation: training an aligned two-view arm by interleaving two 15 GB memory maps on a 15 GB
host thrashes the page cache.  Materialising the mean once turns the training read pattern back
into a single sequential-ish memmap read per epoch, which is what the original protocol used.

Precision note: every token cache in this project is stored as fp16, including the two inputs to
this average.  The mean is computed in float32 from the two fp16 caches and stored back as fp16,
so the aligned cache has exactly the same precision class as the caches it is derived from.
"""

import argparse, hashlib, json, sys, time
from pathlib import Path
import shutil

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = Path("D:/ResearchVault/99system/data/rb-tta-fer-fg2027")
OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")

ENC = {
    "L14": dict(src=D / "token_cache_vitl14", mir=D / "token_cache_vitl14_mirror",
                out="token_cache_vitl14_aligned2v", grid=16),
    "B16": dict(src=D / "token_cache_vitb16", mir=D / "token_cache_vitb16_mirror",
                out="token_cache_vitb16_aligned2v", grid=14),
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
    ap.add_argument("--block", type=int, default=1024)
    args = ap.parse_args()
    tag = args.encoder
    c = ENC[tag]
    grid = c["grid"]
    src = np.load(c["src"] / "fer2013_train_tokens_fp16.npy", mmap_mode="r")
    mir = np.load(c["mir"] / "fer2013_train_tokens_fp16.npy", mmap_mode="r")
    assert src.shape == mir.shape, (src.shape, mir.shape)
    dst_dir = D / c["out"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    npy = dst_dir / "fer2013_train_tokens_fp16.npy"
    if not (c["src"] / "fer2013_train_meta.npz").exists():
        meta_src = c["src"] / "fer2013_train_tokens_fp16.npz"
    else:
        meta_src = c["src"] / "fer2013_train_meta.npz"
    z = np.load(meta_src, allow_pickle=False)
    labels = z["labels"].astype(np.int64)
    ids = z["ids"].astype(str)
    del z
    np.savez_compressed(dst_dir / "fer2013_train_meta.npz", ids=ids, labels=labels)

    n, nt, dt = src.shape
    mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16, shape=(n, nt, dt))
    t0 = time.perf_counter()
    for s in range(0, n, args.block):
        e = min(s + args.block, n)
        a = np.asarray(src[s:e]).astype(np.float32)
        b = np.asarray(mir[s:e]).astype(np.float32)
        b = b.reshape(b.shape[0], grid, grid, -1)[:, :, ::-1, :].reshape(b.shape[0], nt, dt)
        mm[s:e] = ((a + b) / 2.0).astype(np.float16)
        if (s // args.block) % 4 == 0:
            print("  %s %d/%d %.0fs" % (tag, e, n, time.perf_counter() - t0), flush=True)
    mm.flush()
    del mm
    rep = {"encoder": tag, "out": str(npy), "shape": [n, nt, dt],
           "npy_GB": round(npy.stat().st_size / 1e9, 3),
           "formula": "fp16( float32(original) + float32(unflip(mirror)) ) / 2",
           "seconds": round(time.perf_counter() - t0, 1),
           "free_GB_on_D": round(shutil.disk_usage("D:/").free / 1e9, 1),
           "npy_sha256": sha(npy)}
    (OUT / ("phase9_aligned_cache_%s.json" % tag)).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()

