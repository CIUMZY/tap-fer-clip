"""Phase 4 step 1: train published-adapter baselines on the FER2013 source only.

Baselines (all trained on the source probe-train split, lr selected on the source
probe-validation split with the SAME {1e-3, 3e-3} grid used for the token modules):
  CLIP-Adapter     : two-layer bottleneck residual adapter (ratio 4, residual 0.2) on the
                     global 512-d feature, classified by the FROZEN text prototypes.
  Tip-Adapter-F    : trainable cache keys initialised from the source support features,
                     frozen text branch, logits = 100*cos_text + BETA*per-class-max cache.
  Linear probe     : source-supervised linear classifier on the global feature.
  APE-style        : learnable text-side residual on the class prototypes (supplementary).
No target labels are used for training or selection.  Writes only new files.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
FEA = D / 'features'
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
S, BETA, DEV = 100.0, 256.0, 'cuda'
EPOCHS = 8


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class CLIPAdapter(nn.Module):
    """Two-layer bottleneck residual adapter on the global feature, classified by the
    FROZEN text prototypes (adapted feature is L2-normalised before the text dot product)."""
    def __init__(self, text, d=512, hid=128, alpha=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d, hid)
        self.fc2 = nn.Linear(hid, d)
        self.alpha = alpha
        self.register_buffer('text', torch.from_numpy(text.astype(np.float32)))

    def forward(self, x):
        f = self.alpha * self.fc2(F.relu(self.fc1(x))) + (1.0 - self.alpha) * x
        return S * (F.normalize(f, dim=-1) @ self.text.T)


class LinearProbe(nn.Module):
    def __init__(self, d=512, C=7):
        super().__init__()
        self.fc = nn.Linear(d, C)

    def forward(self, x):
        return self.fc(x)


class APEText(nn.Module):
    """Text-side residual: prototypes are shifted by a learnable per-class vector."""
    def __init__(self, text, C=7, d=512):
        super().__init__()
        self.register_buffer('text', torch.from_numpy(text.astype(np.float32)))
        self.delta = nn.Parameter(torch.zeros(C, d))

    def forward(self, x):
        t = F.normalize(self.text + self.delta, dim=-1)
        return S * (x @ t.T)


class TipAdapterF(nn.Module):
    def __init__(self, text, keys, labels, beta=BETA):
        super().__init__()
        self.register_buffer('text', torch.from_numpy(text.astype(np.float32)))
        self.keys = nn.Parameter(torch.from_numpy(keys.astype(np.float32)))
        self.beta = beta
        for c in range(7):
            self.register_buffer('idx%d' % c,
                                 torch.tensor(np.flatnonzero(labels == c), dtype=torch.long))

    def cache(self, x):
        kn = F.normalize(self.keys, dim=-1)
        sim = x @ kn.T
        cols = []
        for c in range(7):
            cols.append(sim.index_select(1, getattr(self, 'idx%d' % c)).max(dim=1).values)
        return torch.stack(cols, dim=1)

    def forward(self, x):
        return S * (x @ self.text.T) + self.beta * self.cache(x)


def train_generic(name, model, Xtr, ytr, Xva, yva, lr, extra=None):
    torch.manual_seed(0)
    model = model.to(DEV)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    steps = 0
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            xb = Xtr[idx].to(DEV, non_blocking=True)
            yb = ytr[idx].to(DEV, non_blocking=True)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            steps += 1
    with torch.inference_mode():
        pv = np.concatenate([model(Xva[i:i + 512].to(DEV)).argmax(1).cpu().numpy()
                             for i in range(0, len(Xva), 512)])
    rec = {'lr': lr, 'params': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
           'steps': steps, 'seconds': round(time.perf_counter() - t0, 1),
           'peak_vram_GB': round(torch.cuda.max_memory_allocated() / 1e9, 2),
           'probe_val_uar': uar(yva, pv)}
    print(name, rec, flush=True)
    return model, rec


def main():
    t_all = time.perf_counter()
    split = np.load(FEA / 'linear_probe_split_seed0.npz', allow_pickle=False)
    tr = np.asarray(split['train_indices'], np.int64)
    va = np.asarray(split['validation_indices'], np.int64)
    gm = norm(np.load(FEA / 'fer2013_train.npz', allow_pickle=False)['views'].mean(axis=1))
    text = norm(np.load(FEA / 'text_prototypes.npz', allow_pickle=False)['prototypes'].astype(np.float32))
    z = np.load(D / 'token_cache_vitb16' / 'fer2013_train_tokens_fp16.npz', allow_pickle=False)
    y = z['labels'].astype(np.int64)
    assert len(y) == len(gm) == 28709
    Xtr = torch.from_numpy(gm[tr].astype(np.float32)); ytr = torch.from_numpy(y[tr])
    Xva = torch.from_numpy(gm[va].astype(np.float32)); yva = y[va]
    rep = {'grid': {'lr': [1e-3, 3e-3], 'epochs': EPOCHS, 'batch': 128, 'logit_scale': S},
           'runs': {}, 'selected': {}}

    # ---- CLIP-Adapter ----
    for lr in (1e-3, 3e-3):
        m, rec = train_generic('CLIPAdapter', CLIPAdapter(text), Xtr, ytr, Xva, yva, lr)
        torch.save(m.state_dict(), OUT / ('phase4_clipadapter_lr%s.pt' % lr))
        rep['runs']['CLIPAdapter|lr%s' % lr] = rec

    # ---- linear probe ----
    for lr in (1e-3, 3e-3):
        m, rec = train_generic('LinearProbe', LinearProbe(), Xtr, ytr, Xva, yva, lr)
        torch.save(m.state_dict(), OUT / ('phase4_linearprobe_lr%s.pt' % lr))
        rep['runs']['LinearProbe|lr%s' % lr] = rec

    # ---- APE-style text residual ----
    for lr in (1e-3, 3e-3):
        m, rec = train_generic('APEtext', APEText(text), Xtr, ytr, Xva, yva, lr)
        torch.save(m.state_dict(), OUT / ('phase4_apetext_lr%s.pt' % lr))
        rep['runs']['APEtext|lr%s' % lr] = rec

    # ---- Tip-Adapter-F (trainable keys from the source support) ----
    sup_keys = gm[tr]
    sup_lab = y[tr]
    for lr in (1e-3, 3e-3):
        m = TipAdapterF(text, sup_keys, sup_lab)
        m, rec = train_generic('TipAdapterF', m, Xtr, ytr, Xva, yva, lr)
        rec['cache_keys'] = int(m.keys.numel())
        torch.save(m.state_dict(), OUT / ('phase4_tipadapterf_lr%s.pt' % lr))
        rep['runs']['TipAdapterF|lr%s' % lr] = rec

    for stem in ('CLIPAdapter', 'LinearProbe', 'APEtext', 'TipAdapterF'):
        ks = ['%s|lr%s' % (stem, l) for l in (1e-3, 3e-3)]
        best = max(ks, key=lambda k: rep['runs'][k]['probe_val_uar'])
        rep['selected'][stem] = {'key': best, 'lr': rep['runs'][best]['lr'],
                                 'params': rep['runs'][best]['params'],
                                 'probe_val_uar': rep['runs'][best]['probe_val_uar']}
    rep['reference_probe_val'] = {'T1_token_attention': 0.6350518869602058,
                                  'PFB_source_probeval': 0.6155,
                                  'memory_source_probeval_beta256': 0.636}
    rep['elapsed_seconds'] = round(time.perf_counter() - t_all, 1)
    (OUT / 'phase4_baselines_training.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(json.dumps(rep['selected'], indent=1))
    print('total %.1fs' % rep['elapsed_seconds'])


if __name__ == '__main__':
    main()
