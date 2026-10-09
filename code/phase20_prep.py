"""Build the phase-20 prospective-replication scripts from the phase-12 originals."""
from pathlib import Path

OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")


def build(src_name, dst_name, extra):
    text = (OUT / src_name).read_text(encoding="utf-8")
    subs = [
        ("phase12_%s_%s_seed%d_lr%g.pt", "phase20_%s_%s_seed%d_lr%g.pt"),
        ('("phase12_train_%s.json" % tag)', '("phase20_train_%s.json" % tag)'),
        ('OUT / "phase12_eval.json"', 'OUT / "phase20_eval.json"'),
        ("seeds=list(range(5, 15))", "seeds=list(range(15, 25))"),
        ("seeds=list(range(5, 20))", "seeds=list(range(20, 35))"),
        ('if tag == "B16" and seed == 5 and arm == "TAP"', 'if tag == "B16" and seed == 15 and arm == "TAP"'),
    ]
    for a, b in subs:
        n = text.count(a)
        text = text.replace(a, b)
        print("  %-45s x%d" % (a[:45], n))
    text = text.replace('"""Phase 12a', '"""Phase 20').replace('"""Phase 12b', '"""Phase 20')
    text = extra + text
    (OUT / dst_name).write_text(text, encoding="utf-8")
    print("wrote", dst_name, len(text), "bytes")


HEAD = (
    "# PROSPECTIVE BATCH 2 (phase 20). Built from the phase-12 script by phase20_prep.py.\n"
    "# Seeds are fixed and hashed in manuscript/PHASE20-PREREGISTRATION-2026-10-09.md BEFORE this ran.\n"
    "# Evaluation axes are the already-consumed confirmation axes: this tests the reproducibility of\n"
    "# the training-seed distribution, not a new confirmation claim.\n"
)

print("train script:")
build("phase12_train_seeded.py", "phase20_train_prospective.py", HEAD)
print("eval script:")
build("phase12_eval_seeded.py", "phase20_eval_prospective.py", HEAD)
