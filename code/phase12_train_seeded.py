
"""Phase 12a: retrain all arms with an explicitly seed-controlled initialisation.

Why this rerun exists
---------------------
Inspecting phase 8 showed that its arms were constructed *before* train_arm() called
torch.manual_seed(seed), so the initial weights of an arm were drawn from whatever global RNG
state the process happened to be in. Re-running phase8_trainvar.py therefore does not reproduce
its own checkpoints: phase 11 reproduced the CLIP-Adapter and global-head arms bit for bit but not
the token arm. That is a real reproducibility defect in the retraining analysis, and it is the
defect the reviewers asked us to close.

Here every arm is built inside torch.manual_seed(seed), so both the initialisation and the batch
order follow from the seed alone, for every arm and every encoder, and the whole retraining run is
reproducible from the seed list. Each arm is trained at both grid points so that the retraining
distribution can be reported both at the historically selected rate and under nested per-seed
rate selection.

Discipline: encoder frozen; training and rate selection use the FER2013 probe-train /
probe-validation split only; no target label enters training or selection.
"""
import importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = Path("D:/ResearchVault/99system/data/rb-tta-fer-fg2027")
OUT = Path("D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006")
DEV = "cuda"
LRS = [1e-3, 3e-3]

spec = importlib.util.spec_from_file_location("p8", OUT / "phase8_trainvar.py")
p8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p8)

CFG = {
    "B16": dict(dim=768, gd=512, hid=256, gh=256, ca_hid=128,
                tokdir=D / "token_cache_vitb16", feat=D / "features",
                seeds=list(range(5, 15)), published=dict(TAP=3e-3, CA=1e-3, GH=3e-3)),
    "L14": dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192,
                tokdir=D / "token_cache_vitl14", feat=D / "features_vitl14",
                seeds=list(range(5, 20)), published=dict(TAP=1e-3, CA=1e-3, GH=1e-3)),
}


def run_encoder(tag):
    c = CFG[tag]
    t0 = time.perf_counter()
    Fe = c["feat"]
    split = np.load(Fe / "linear_probe_split_seed0.npz", allow_pickle=False)
    tr = np.asarray(split["train_indices"], dtype=np.int64)
    va = np.asarray(split["validation_indices"], dtype=np.int64)
    text = p8.norm(np.load(Fe / "text_prototypes.npz", allow_pickle=False)["prototypes"].astype(np.float32))
    gtr = p8.norm(np.load(Fe / "fer2013_train.npz", allow_pickle=False)["views"].mean(axis=1))
    G = torch.from_numpy(gtr.astype(np.float32))
    if (c["tokdir"] / "fer2013_train_tokens_fp16.npy").exists():
        TOK = np.load(c["tokdir"] / "fer2013_train_tokens_fp16.npy", mmap_mode="r")
    else:
        TOK = np.load(c["tokdir"] / "fer2013_train_tokens_fp16.npz", allow_pickle=False)["tokens"]
    LAB = np.load(c["tokdir"] / "fer2013_train_meta.npz", allow_pickle=False)["labels"].astype(np.int64) \
        if (c["tokdir"] / "fer2013_train_meta.npz").exists() else \
        np.load(c["tokdir"] / "fer2013_train_tokens_fp16.npz", allow_pickle=False)["labels"].astype(np.int64)
    trs = np.sort(tr)
    print("%s tokens %s probe-train %d probe-val %d" % (tag, TOK.shape, len(trs), len(va)), flush=True)

    def tok_batches(sel):
        return (torch.from_numpy(np.asarray(TOK[sel])).to(DEV).float(),
                torch.from_numpy(LAB[sel]).to(DEV))

    def glob_batches(sel):
        return (G[sel].to(DEV), torch.from_numpy(LAB[sel]).to(DEV))

    def tok_train(sel):
        return tok_batches(trs[sel])

    def glob_train(sel):
        return glob_batches(trs[sel])

    def build(arm):
        if arm == "TAP":
            return p8.T1(c["dim"], c["hid"])
        if arm == "CA":
            return p8.CLIPAdapter(text, c["gd"], c["ca_hid"])
        return p8.GHead(c["gd"], c["gh"])

    yva = LAB[va]
    rows = []
    for seed in c["seeds"]:
        rec = {"encoder": tag, "seed": seed}
        for arm in ("TAP", "CA", "GH"):
            batches = tok_train if arm == "TAP" else glob_train
            scorers = tok_batches if arm == "TAP" else glob_batches
            for lr in LRS:
                t1 = time.perf_counter()
                torch.manual_seed(seed)              # initialisation follows from the seed
                model = build(arm)
                m = p8.train_arm(model, batches, len(trs), lr, seed)
                p = p8.eval_preds(m, scorers, va)
                rec["%s_lr%g" % (arm, lr)] = p8.uar(yva, p)
                torch.save(m.state_dict(), OUT / ("phase12_%s_%s_seed%d_lr%g.pt" % (arm.lower(), tag, seed, lr)))
                if tag == "B16" and seed == 5 and arm == "TAP" and lr == 1e-3:
                    torch.manual_seed(seed)
                    m2 = p8.train_arm(build(arm), batches, len(trs), lr, seed)
                    ident = all(torch.equal(m.state_dict()[k], m2.state_dict()[k]) for k in m.state_dict())
                    rec["determinism_rebuild_identical"] = bool(ident)
                    print("  determinism rebuild identical:", ident, flush=True)
                    del m2
                del m
                torch.cuda.empty_cache()
                print("  %s seed %d %s lr %g probe-val %.4f (%.0fs)"
                      % (tag, seed, arm, lr, rec["%s_lr%g" % (arm, lr)], time.perf_counter() - t1), flush=True)
        rows.append(rec)
        (OUT / ("phase12_train_%s.json" % tag)).write_text(
            json.dumps({"encoder": tag, "seeds": c["seeds"], "lrs": LRS, "published": c["published"],
                        "rows": rows, "wall_seconds": round(time.perf_counter() - t0, 1)}, indent=2),
            encoding="utf-8")
    return rows, round(time.perf_counter() - t0, 1)


def main():
    out = {}
    for tag in ("B16", "L14"):
        rows, secs = run_encoder(tag)
        out[tag] = {"rows": rows, "seconds": secs}
    print("done", flush=True)


if __name__ == "__main__":
    main()

