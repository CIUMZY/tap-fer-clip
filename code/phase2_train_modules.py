"""Phase 2 stage 2/3: implement T1-T4 token modules, train on source only, evaluate on the
crossed grid (ckplus_prior / kdef_prior) and the second source (KDEF-source).

Discipline: training/hyper-parameter selection ONLY on FER2013 probe-train / probe-val.
No target labels anywhere in training or selection.  Reference column = best of the two
families (PFB vs support memory) per cell.
"""
import csv, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
TC = D / 'token_cache_vitb16'
FEAT = D / 'features'
FK = D / 'features_kdef_source'
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
S, BETA, LAM, CLIPV, MINN = 100.0, 256.0, 0.2, 1.0, 32
DEV = 'cuda'
torch.manual_seed(0)


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


class T1(nn.Module):
    def __init__(self, dim=768, hid=256, C=7):
        super().__init__()
        self.a1 = nn.Linear(dim, hid)
        self.a2 = nn.Linear(hid, 1)
        self.head = nn.Linear(dim, C)

    def forward(self, tok):
        w = torch.softmax(self.a2(torch.tanh(self.a1(tok))).squeeze(-1), dim=1)
        return self.head((tok * w.unsqueeze(-1)).sum(1))


class T2(nn.Module):
    def __init__(self, dim=768, mid=128, C=7):
        super().__init__()
        self.c1 = nn.Conv2d(dim, mid, 1)
        self.c2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.head = nn.Linear(mid, C)

    def forward(self, tok):
        B = tok.shape[0]
        x = tok.transpose(1, 2).reshape(B, 768, 14, 14)
        x = F.gelu(self.c1(x))
        x = F.gelu(self.c2(x))
        return self.head(x.mean(dim=(2, 3)))


class T3(nn.Module):
    """token projections + class patch-prototype memory (top-k patch similarity) fused with
    the cached global text branch."""

    def __init__(self, dim=768, d=128, C=7, k=8):
        super().__init__()
        self.wq = nn.Linear(dim, d)
        self.wk = nn.Linear(dim, d)
        self.k = k
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.ref = None

    def forward(self, tok, text_logits):
        q = F.normalize(self.wq(tok), dim=-1)
        kk = F.normalize(self.wk(self.ref), dim=-1)       # (7,196,d)
        sim = torch.einsum('bnd,cld->bcnl', q, kk)        # B x 7 x 196 x 196
        top = sim.topk(self.k, dim=-1).values.mean(-1)    # B x 7 x 196
        mem = top.mean(-1)                                # B x 7
        return self.beta * text_logits + self.alpha * mem, mem


class T4(nn.Module):
    def __init__(self, dim=768, parts=4, d=64, C=7):
        super().__init__()
        self.parts = parts
        self.width = dim // parts
        self.branches = nn.ModuleList([nn.Linear(self.width, d) for _ in range(parts)])
        self.head = nn.Linear(parts * d, C)

    def forward(self, tok):
        outs = []
        for i, br in enumerate(self.branches):
            seg = tok[:, :, i * self.width:(i + 1) * self.width]
            outs.append(F.gelu(br(seg)).mean(1))
        return self.head(torch.cat(outs, dim=-1))


def load_tokens(name):
    z = np.load(TC / ('%s_tokens_fp16.npz' % name), allow_pickle=False)
    return z['tokens'], z['labels'].astype(np.int64)


def prior_correct(logits, order=None):
    if order is None:
        order = np.arange(len(logits))
    out = np.empty_like(logits)
    counts = np.zeros(7, dtype=np.int64)
    for i in order:
        lg = logits[i].astype(np.float64).copy()
        if counts.sum() >= MINN:
            pi = (counts + 1.0) / (counts.sum() + 7.0)
            lg = lg - min(LAM, CLIPV) * np.log(np.maximum(pi, 1e-12))
        out[i] = lg
        counts[int(np.argmax(lg))] += 1
    return out


def family_uar(tok, y, text, SV, SY, W, b):
    """PFB (probe + prior) and support memory (text + beta*cache) cells from cached tokens."""
    V = tok.mean(1)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    sim = V @ SV.T
    cache = np.zeros((len(V), 7), dtype=np.float32)
    for c in range(7):
        m = SY == c
        if m.any():
            cache[:, c] = sim[:, m].max(axis=1)
    mem = S * (V @ text.T) + BETA * cache
    pfb = prior_correct(V @ W.T + b)
    return uar(y, pfb.argmax(1)), uar(y, mem.argmax(1)), cache


def main():
    t0 = time.perf_counter()
    split = np.load(FEAT / 'linear_probe_split_seed0.npz', allow_pickle=False)
    tr = np.asarray(split['train_indices'], dtype=np.int64)
    va = np.asarray(split['validation_indices'], dtype=np.int64)
    tr_tok, tr_y = load_tokens('fer2013_train')
    Xtr = torch.from_numpy(tr_tok[tr])
    ytr = torch.from_numpy(tr_y[tr])
    Xva = torch.from_numpy(tr_tok[va])
    yva = tr_y[va]
    gm = np.load(FEAT / 'fer2013_train.npz', allow_pickle=False)['views'].mean(axis=1)
    gm = gm / np.maximum(np.linalg.norm(gm, axis=1, keepdims=True), 1e-12)
    Gtr = torch.from_numpy(gm[tr].astype(np.float32))
    Gva = torch.from_numpy(gm[va].astype(np.float32))
    del tr_tok
    print('train %s val %s' % (tuple(Xtr.shape), tuple(Xva.shape)), flush=True)
    text = np.load(FEAT / 'text_prototypes.npz', allow_pickle=False)['prototypes'].astype(np.float32)
    text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    pr = np.load(FEAT / 'linear_probe_train80_seed0.npz', allow_pickle=False)
    Wp, bp = pr['weights'].astype(np.float32), pr['bias'].astype(np.float32)
    # class patch prototypes from the source training subset (label-free at target time)
    ref = np.zeros((7, 196, 768), dtype=np.float32)
    for c in range(7):
        idx = np.flatnonzero(tr_y[tr] == c)[:1500]
        ref[c] = Xtr[idx].float().mean(0).numpy()
    np.save(OUT / 'phase2_ref_patch_prototypes.npy', ref)

    def make(kind):
        m = {'T1': T1(), 'T2': T2(), 'T4': T4()}[kind] if kind != 'T3' else T3()
        return m.to(DEV)

    # ---- training on the source only ----
    results = {}
    for kind in ('T1', 'T2', 'T3', 'T4'):
        for lr in (1e-3, 3e-3):
            torch.manual_seed(0)
            model = T1().to(DEV) if kind == 'T1' else (T2().to(DEV) if kind == 'T2'
                                                       else (T3().to(DEV) if kind == 'T3'
                                                             else T4().to(DEV)))
            if kind == 'T3':
                model.ref = torch.from_numpy(ref).to(DEV)
                tproj = torch.from_numpy(text).to(DEV)
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
            nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
            torch.cuda.reset_peak_memory_stats()
            tt = time.perf_counter()
            steps = 0
            for ep in range(8):
                perm = torch.randperm(len(Xtr))
                for i in range(0, len(perm), 128):
                    idx = perm[i:i + 128]
                    xb = Xtr[idx].to(DEV, non_blocking=True).float()
                    yb = ytr[idx].to(DEV, non_blocking=True)
                    if kind == 'T3':
                        gb = Gtr[idx].to(DEV, non_blocking=True)
                        out, _ = model(xb, S * (gb @ tproj.T))
                    else:
                        out = model(xb)
                    loss = F.cross_entropy(out, yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    steps += 1
            with torch.inference_mode():
                preds = []
                for i in range(0, len(Xva), 256):
                    xb = Xva[i:i + 256].to(DEV).float()
                    if kind == 'T3':
                        gb = Gva[i:i + 256].to(DEV)
                        out, _ = model(xb, S * (gb @ tproj.T))
                    else:
                        out = model(xb)
                    preds.append(out.argmax(1).cpu().numpy())
            pv = np.concatenate(preds)
            rec = {'kind': kind, 'lr': lr, 'params': nparam, 'steps': steps,
                   'seconds': round(time.perf_counter() - tt, 1),
                   'peak_vram_GB': round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   'probe_val_uar': uar(yva, pv)}
            results['%s|lr%s' % (kind, lr)] = rec
            torch.save(model.state_dict(), OUT / ('model_%s_lr%s.pt' % (kind, lr)))
            print('trained', rec, flush=True)
    (OUT / 'phase2_training.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('total train time', round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()
