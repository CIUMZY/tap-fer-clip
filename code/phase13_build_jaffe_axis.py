
# -*- coding: utf-8 -*-
"""Phase 13a: build the JAFFE fresh confirmation axis exactly as pre-specified.

Steps: verify the archive hash, extract, map expression tokens to the seven-class index used in
this study, resample to 48x48 grayscale (FER2013 native resolution), then apply the same corruption
implementation with the pre-specified 20-condition grid.

Nothing here inspects a model. The pre-specification
(phase13_jaffe_preregistration.md, sha256 recorded in phase13_prereg_freeze.json) fixes every
choice made below.
"""
import hashlib, io, json, sys, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/scripts")))
from make_corrupted_fer2013 import corrupt  # noqa: E402  (identical implementation, reused)

D = Path("D:/ResearchVault/99system/data/rb-tta-fer-fg2027")
ARCHIVE = D / "downloads/jaffe_official_v2/jaffe.zip"
EXPECT_SHA256 = "6da27f5954f969c6f65d782911834dd66827ec3580e79b95073eb6cb93de5c3b"
OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
ROOT = D / "raw/jaffe_confirm"
LABEL = {"AN": 0, "DI": 1, "FE": 2, "HA": 3, "SA": 4, "SU": 5, "NE": 6}
GRID = [("blur", "0.8"), ("blur", "1.5"), ("blur", "2.5"), ("blur", "4.0"), ("blur", "6.0"),
        ("jpeg", "15"), ("jpeg", "30"), ("jpeg", "50"), ("jpeg", "65"), ("jpeg", "80"),
        ("lowlight", "0.3"), ("lowlight", "0.5"), ("lowlight", "0.7"), ("lowlight", "0.85"),
        ("lowlight", "0.95"),
        ("noise", "10"), ("noise", "25"), ("noise", "40"), ("noise", "55"), ("noise", "70")]


def sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.perf_counter()
    got = sha(ARCHIVE)
    assert got == EXPECT_SHA256, "archive mismatch: %s" % got
    print("archive sha256 verified", flush=True)

    rep = {"archive": str(ARCHIVE), "archive_sha256": got, "conditions": [], "labels": {},
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    with zipfile.ZipFile(ARCHIVE) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith((".tif", ".tiff")))
        assert len(names) == 213, "expected 213 images, found %d" % len(names)
        rows = []
        for idx, n in enumerate(names):
            stem = Path(n).stem                      # e.g. MK.AN3.127
            parts = stem.split(".")
            model, expr = parts[0], parts[1][:2]
            if expr not in LABEL:
                raise SystemExit("unknown expression token %r in %s" % (expr, n))
            rows.append({"index": idx, "name": stem, "member": n, "model": model,
                         "expression": expr, "label": LABEL[expr]})

    # pre-specified sanity check: every model appears exactly once per expression
    key = {}
    for r in rows:
        key.setdefault(r["model"], {}).setdefault(r["expression"], 0)
        key[r["model"]][r["expression"]] += 1
    counts = {m: {e: c for e, c in v.items()} for m, v in key.items()}
    # amended check (phase13_jaffe_preregistration_amendment1.md): all 70 (model, expression)
    # pairs present, at least two images each, exactly seven tokens per model
    dup = [m for m, v in counts.items() if len(v) != 7 or any(c < 2 for c in v.values())]
    pairs = sum(len(v) for v in counts.values())
    assert pairs == 70, pairs
    rep["models"] = {m: counts[m] for m in sorted(counts)}
    rep["n_model_expression_pairs"] = pairs
    rep["models_with_duplicate_or_missing_expression"] = dup
    print("models", len(counts), "duplicate/missing:", dup, flush=True)
    assert not dup, dup

    src_dir = ROOT / "original"
    src_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        for r in rows:
            with Image.open(io.BytesIO(z.read(r["member"]))) as im:
                im.convert("L").resize((48, 48), Image.LANCZOS).save(
                    src_dir / ("%s.png" % r["name"]))
            r["path"] = str(src_dir / ("%s.png" % r["name"]))
    print("resampled 213 originals to 48x48 grayscale", flush=True)

    frame = pd.DataFrame(rows)
    for name, sev in GRID:
        var = "%s_%s" % (name, sev.replace(".", "p"))
        vroot = ROOT / var
        vroot.mkdir(parents=True, exist_ok=True)
        out_rows = []
        for _, r in frame.iterrows():
            dst = vroot / ("%s.png" % r["name"])
            with Image.open(r["path"]) as im:
                corrupt(im, name, seed=1000 + int(r["index"]), severity=float(sev)).save(dst)
            out_rows.append({"path": str(dst), "label": int(r["label"]),
                             "domain": "jaffe_%s" % var, "split": "test"})
        man = ROOT / "manifests" / ("jaffe_%s.csv" % var)
        man.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_csv(man, index=False)
        rep["conditions"].append({"condition": var, "family": name, "severity": float(sev),
                                  "n": len(out_rows), "manifest": str(man)})
        print("  %-16s %d images" % (var, len(out_rows)), flush=True)

    frame[["index", "name", "model", "expression", "label", "path"]].to_csv(
        ROOT / "jaffe_index.csv", index=False)
    rep["n_images"] = len(frame)
    rep["label_counts"] = {k: int(v) for k, v in frame["label"].value_counts().items()}
    rep["seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / "phase13_jaffe_axis.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "conditions"}, indent=2)[:1200])
    print("done %.0fs" % (time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()

