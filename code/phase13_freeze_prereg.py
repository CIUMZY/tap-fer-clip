
# -*- coding: utf-8 -*-
"""Freeze the phase-13 pre-specification: record its hash and byte size before any JAFFE work."""
import hashlib, json, time
from pathlib import Path

OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
MS = Path("D:/ResearchVault/06manuscripts/rb-tta-fer-fg2027/PAPER-TVC-TAP-2026-10-06/manuscript")
src = OUT / "phase13_jaffe_preregistration.md"
b = src.read_bytes()
rec = {
    "file": str(src),
    "bytes": len(b),
    "sha256": hashlib.sha256(b).hexdigest(),
    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "jaffe_archive": "downloads/jaffe_official_v2/jaffe.zip",
    "jaffe_archive_sha256_expected": "6da27f5954f969c6f65d782911834dd66827ec3580e79b95073eb6cb93de5c3b",
    "statement": "written and hashed before any JAFFE image was opened, decoded or evaluated",
    "copy_in_manuscript_dir": str(MS / "JAFFE-PREREGISTRATION-2026-10-08.md"),
}
import shutil
shutil.copyfile(src, MS / "JAFFE-PREREGISTRATION-2026-10-08.md")
(OUT / "phase13_prereg_freeze.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(rec, indent=2))

