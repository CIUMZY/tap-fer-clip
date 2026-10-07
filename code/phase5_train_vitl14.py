"""Phase 5 step 3: retrain TAP (dim 1024), the equal-capacity global head and CLIP-Adapter on
ViT-L/14, source-only protocol and the same lr grid {1e-3, 3e-3} x 8 epochs x batch 128.
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
S, DEV, EPOCHS = 100.0, 'cuda', 8
spec = importlib.util.spec_from_file_location('mod', OUT / 'phase2_train_modules.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


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


def train(name, model, Xtr, ytr, Xva, yva, lr):
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
            xb = Xtr[idx].to(DEV, non_blocking=True).float()
            yb = ytr[idx].to(DEV, non_blocking=True)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            steps += 1
    with torch.inference_mode():
        pv = np.concatenate([model(Xva[i:i + 256].to(DEV).float()).argmax(1).cpu().numpy()
                             for i in range(0, len(Xva), 256)])
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
    tr = np.asarray(split['train_indices'], np.int64); va = np.asarray(split['validation_indices'], np.int64)
    z = np.load(TC / 'fer2013_train_tokens_fp16.npz', allow_pickle=False)
    y = z['labels'].astype(np.int64)
    gtr = norm(np.load(L / 'fer2013_train.npz', allow_pickle=False)['views'].mean(axis=1))
    text = norm(np.load(L / 'text_prototypes.npz', allow_pickle=False)['prototypes'].astype(np.float32))
    Xtr = torch.from_numpy(z['tokens'][tr]); ytr = torch.from_numpy(y[tr])
    Xva = torch.from_numpy(z['tokens'][va]); yva = y[va]
    Xtr_g = torch.from_numpy(gtr[tr].astype(np.float32)); Xva_g = torch.from_numpy(gtr[va].astype(np.float32))
    del z
    rep = {'model': 'ViT-L-14', 'grid': {'lr': [1e-3, 3e-3], 'epochs': EPOCHS, 'batch': 128,
                                         'logit_scale': S}, 'runs': {}, 'selected': {},
           'source_probeval_reference': {'pfb_vitl14': None, 'token_TAP': None}}

    for lr in (1e-3, 3e-3):
        m, rec = train('TAP_dim1024', mod.T1(dim=1024), Xtr, ytr, Xva, yva, lr)
        torch.save(m.state_dict(), OUT / ('phase5_tap_vitl14_lr%s.pt' % lr))
        rep['runs']['TAP|lr%s' % lr] = rec
    tap_params = rep['runs']['TAP|lr0.001']['params']
    h, hp = choose_h(768, 7, tap_params)
    rep['global_head_width'] = {'h': h, 'params': hp, 'tap_params': tap_params,
                                'abs_diff': abs(hp - tap_params)}
    for lr in (1e-3, 3e-3):
        m, rec = train('GHead_L14', GHead(768, h), Xtr_g, ytr, Xva_g, yva, lr)
        torch.save(m.state_dict(), OUT / ('phase5_ghead_vitl14_lr%s.pt' % lr))
        rep['runs']['GHead|lr%s' % lr] = rec
    for lr in (1e-3, 3e-3):
        m, rec = train('CLIPAdapter_L14', CLIPAdapter(text), Xtr_g, ytr, Xva_g, yva, lr)
        torch.save(m.state_dict(), OUT / ('phase5_clipadapter_vitl14_lr%s.pt' % lr))
        rep['runs']['CLIPAdapter|lr%s' % lr] = rec
    for stem in ('TAP', 'GHead', 'CLIPAdapter'):
        ks = ['%s|lr%s' % (stem, l) for l in (1e-3, 3e-3)]
        best = max(ks, key=lambda k: rep['runs'][k]['probe_val_uar'])
        rep['selected'][stem] = {'key': best, 'lr': rep['runs'][best]['lr'],
                                 'params': rep['runs'][best]['params'],
                                 'probe_val_uar': rep['runs'][best]['probe_val_uar']}
    rep['reference_probeval'] = {'TAP_vitb16': 0.6350518869602058,
                                 'CLIPAdapter_vitb16': 0.617177432363264,
                                 'source_feature_dim': 768, 'token_dim': 1024}
    rep['elapsed_seconds'] = round(time.perf_counter() - t_all, 1)
    (OUT / 'phase5_training_vitl14.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(json.dumps(rep['selected'], indent=1), flush=True)
    print('total %.1fs' % rep['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
