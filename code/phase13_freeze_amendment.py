
# -*- coding: utf-8 -*-
import hashlib, json, shutil, time
from pathlib import Path

OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
MS = Path("D:/ResearchVault/06manuscripts/rb-tta-fer-fg2027/PAPER-TVC-TAP-2026-10-06/manuscript")
src = OUT / "phase13_jaffe_preregistration_amendment1.md"
b = src.read_bytes()
rec = json.loads((OUT / "phase13_prereg_freeze.json").read_text(encoding="utf-8"))
rec["amendment1"] = {
    "file": str(src), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(),
    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "reason": "factual correction of the corpus structure check; hypothesis, grid, arms and decision "
              "rule unchanged",
    "state_at_freezing": "archive hash verified and file listing inspected only; no model run and no "
                         "feature extracted",
}
rec["copy_in_manuscript_dir_amendment1"] = str(MS / "JAFFE-PREREGISTRATION-AMENDMENT1-2026-10-08.md")
shutil.copyfile(src, MS / "JAFFE-PREREGISTRATION-AMENDMENT1-2026-10-08.md")
(OUT / "phase13_prereg_freeze.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(rec, indent=2))

