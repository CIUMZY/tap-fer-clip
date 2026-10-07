"""Phase 5 step 3 (v2): retrain TAP (dim 1024), equal-capacity global head and CLIP-Adapter on
ViT-L/14. Token columns are read from the memory-mapped .npy in chunks (host RAM is 16.2 GB).
Source-only protocol, lr grid {1e-3, 3e-3} x 8 epochs x batch 128.
"""
import importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
TC = D / 'token_cache_vitl14'
L = D / 'features_vitl14'
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
S, DEV, EPOCHS, TOK_CHUNK = 100.0, 'cuda', 8, 128
spec = importlib.util.spec_from_file_location('mod', OUT / 'phase2_train_modules.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
TOK = np.load(TC / 'fer2013_train_tokens_fp16.npy', mmap_mode='r')
LAB = np.load(TC / 'fer2013_train_meta.npz')['labels'].astype(np.int64)


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def rows(idx):
    return torch.from_numpy(np.asarray(TOK[idx]))


class GHead(nn.Module):
    def __init__(self, d=768, h=260, C=7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, C))

    def forward(self, x):
        return self.net(x)


class CLIPAdapter(nn.Module):
    def __init__(self, text, d=768, hid=192, alpha=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d, hid); self.fc2 = nn.Linear(hid, d); self.alpha = alpha
        self.register_buffer('text', torch.from_numpy(text.astype(np.float32)))

    def forward(self, x):
        f = self.alpha * self.fc2(F.relu(self.fc1(x))) + (1.0 - self.alpha) * x
        return S * (F.normalize(f, dim=-1) @ self.text.T)


def train_tokens(model, tr, ytr, va, yva, lr):
    torch.manual_seed(0)
    model = model.to(DEV)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(0)
    steps = 0
    for ep in range(EPOCHS):
        perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(perm), 128):
            sel = tr[perm[i:i + 128].numpy()]
            xb = rows(sel).to(DEV).float()
            yb = torch.from_numpy(ytr[sel]).to(DEV)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            steps += 1
    preds = []
    with torch.inference_mode():
        for i in range(0, len(va), 256):
            vb = va[i:i + 256]
            preds.append(model(rows(vb).to(DEV).float()).argmax(1).cpu().numpy())
    rec = {'lr': lr, 'params': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
           'steps': steps, 'seconds': round(time.perf_counter() - t0, 1),
           'peak_vram_GB': round(torch.cuda.max_memory_allocated() / 1e9, 2),
           'probe_val_uar': uar(yva[va], np.concatenate(preds))}
    print('TAP', rec, flush=True)
    return model, rec


def train_global(name, model, Xtr, ytr, Xva, yva, lr):
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
            loss = F.cross_entropy(model(Xtr[idx].to(DEV)), ytr[idx].to(DEV))
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


def choose_h(d, C, target):
    best = None
    for h in range(8, 900):
        p = (d * h + h) + (h * h + h) + (h * C + C)
        if best is None or abs(p - target) < best[0]:
            best = (abs(p - target), h, p)
    return best[1], best[2]


def main():
    t_all = time.perf_counter()
    split = np.load(L / 'linear_probe_split_seed0.npz', allow_pickle=False)
    tr = torch.from_numpy(np.asarray(split['train_indices'], np.int64))
    va = torch.from_numpy(np.asarray(split['validation_indices'], np.int64))
    gtr = norm(np.load(L / 'fer2013_train.npz', allow_pickle=False)['views'].mean(axis=1))
    text = norm(np.load(L / 'text_prototypes.npz', allow_pickle=False)['prototypes'].astype(np.float32))
    Xtr_g = torch.from_numpy(gtr[tr.numpy()].astype(np.float32)); ytr_g = torch.from_numpy(LAB[tr.numpy()])
    Xva_g = torch.from_numpy(gtr[va.numpy()].astype(np.float32)); yva_g = LAB[va.numpy()]
    rep = {'model': 'ViT-L-14', 'grid': {'lr': [1e-3, 3e-3], 'epochs': EPOCHS, 'batch': 128,
                                         'logit_scale': S}, 'runs': {}, 'selected': {}}
    for lr in (1e-3, 3e-3):
        m, rec = train_tokens(mod.T1(dim=1024), tr, LAB, va, LAB, lr)
        torch.save(m.state_dict(), OUT / ('phase5_tap_vitl14_lr%s.pt' % lr))
        rep['runs']['TAP|lr%s' % lr] = rec
    tap_params = rep['runs']['TAP|lr0.001']['params']
    h, hp = choose_h(768, 7, tap_params)
    rep['global_head_width'] = {'h': h, 'params': hp, 'tap_params': tap_params, 'abs_diff': abs(hp - tap_params)}
    for lr in (1e-3, 3e-3):
        m, rec = train_global('GHead_L14', GHead(768, h), Xtr_g, ytr_g, Xva_g, yva_g, lr)
        torch.save(m.state_dict(), OUT / ('phase5_ghead_vitl14_lr%s.pt' % lr))
        rep['runs']['GHead|lr%s' % lr] = rec
    for lr in (1e-3, 3e-3):
        m, rec = train_global('CLIPAdapter_L14', CLIPAdapter(text), Xtr_g, ytr_g, Xva_g, yva_g, lr)
        torch.save(m.state_dict(), OUT / ('phase5_clipadapter_vitl14_lr%s.pt' % lr))
        rep['runs']['CLIPAdapter|lr%s' % lr] = rec
    for stem in ('TAP', 'GHead', 'CLIPAdapter'):
        ks = ['%s|lr%s' % (stem, l) for l in (1e-3, 3e-3)]
        best = max(ks, key=lambda k: rep['runs'][k]['probe_val_uar'])
        rep['selected'][stem] = {'key': best, 'lr': rep['runs'][best]['lr'],
                                 'params': rep['runs'][best]['params'],
                                 'probe_val_uar': rep['runs'][best]['probe_val_uar']}
    rep['reference_probeval'] = {'TAP_vitb16': 0.6350518869602058,
                                 'CLIPAdapter_vitb16': 0.617177432363264}
    rep['elapsed_seconds'] = round(time.perf_counter() - t_all, 1)
    (OUT / 'phase5_training_vitl14.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(json.dumps(rep['selected'], indent=1), flush=True)
    print('total %.1fs' % rep['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
