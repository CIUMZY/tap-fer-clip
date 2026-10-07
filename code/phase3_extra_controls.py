"""Phase 3 extras: (A) reference-column sensitivity for the KDEF-source cells (PFB lambda and
pure-cache memory variant), (B) optional T2 equal-capacity control: 1x1-only (no 3x3 spatial
mixing) matched to T2's parameter count, trained on the FER2013 source with the same lr grid.
"""
import csv, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
TC = D / 'token_cache_vitb16'
FEA = D / 'features'
KS = D / 'features_kdef_source'
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
S, BETA, LAM, CLIPV, MINN, DEV = 100.0, 256.0, 0.2, 1.0, 32, 'cuda'


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prior_correct(logits, order, lam):
    out = np.empty_like(logits)
    counts = np.zeros(7, dtype=np.int64)
    for i in order:
        lg = logits[i].astype(np.float64).copy()
        if counts.sum() >= MINN:
            pi = (counts + 1.0) / (counts.sum() + 7.0)
            lg = lg - min(lam, CLIPV) * np.log(np.maximum(pi, 1e-12))
        out[i] = lg
        counts[int(np.argmax(lg))] += 1
    return out


def subset(lab, ratio, maj, seed):
    if ratio == 0:
        return np.arange(len(lab))
    rng = np.random.default_rng(seed)
    cnt = np.bincount(lab, minlength=7)
    base = int(cnt.min())
    return np.concatenate([rng.choice(np.flatnonzero(lab == c),
                                      size=min(int(cnt[c]), base * (ratio if c == maj else 1)),
                                      replace=False) for c in range(7)])


class T2x(nn.Module):
    """T2 with the 3x3 conv replaced by a 1x1 conv; width chosen to match T2's params."""
    def __init__(self, dim=768, mid=242, C=7):
        super().__init__()
        self.c1 = nn.Conv2d(dim, mid, 1)
        self.c2 = nn.Conv2d(mid, mid, 1)
        self.head = nn.Linear(mid, C)

    def forward(self, tok):
        B = tok.shape[0]
        x = tok.transpose(1, 2).reshape(B, 768, 14, 14)
        x = F.gelu(self.c1(x)); x = F.gelu(self.c2(x))
        return self.head(x.mean(dim=(2, 3)))


def load_tokens(name):
    z = np.load(TC / (name + '_tokens_fp16.npz'), allow_pickle=False)
    return z['tokens'], z['ids'].astype(str), z['labels'].astype(np.int64)


def load_globals(path):
    z = np.load(path, allow_pickle=False)
    return norm(z['views'].mean(axis=1)), z['labels'].astype(np.int64)


def part_a(text):
    sup, sup_lab = load_globals(KS / 'kdef_source_train.npz')
    kp = np.load(KS / 'linear_probe_kdef_source_seed0.npz', allow_pickle=False)
    Wp, bp = kp['weights'].astype(np.float32), kp['bias'].astype(np.float32)
    res = {}
    for tname, gp, kpath in (('kdef_source_test', KS / 'kdef_source_test.npz', None),
                             ('fer2013_test', FEA / 'fer2013_test.npz', None),
                             ('ckplus_test', FEA / 'ckplus_test.npz', None)):
        gv, y = load_globals(gp)
        sim = gv @ sup.T
        cache = np.zeros((len(y), 7), np.float32)
        for c in range(7):
            mm = sup_lab == c
            if mm.any():
                cache[:, c] = sim[:, mm].max(1)
        canon_mem = uar(y, (S * (gv @ text.T) + BETA * cache).argmax(1))
        pure_cache = uar(y, cache.argmax(1))
        pfb = {}
        for lam in (0.0, 0.2, 0.4):
            vals = []
            for seed in range(5):
                order = np.random.default_rng(seed).permutation(len(y))
                vals.append(uar(y, prior_correct(gv @ Wp.T + bp, order, lam).argmax(1)))
            pfb[str(lam)] = float(np.mean(vals))
        res[tname] = {'n': int(len(y)), 'pfb_lambda0.0': pfb['0.0'], 'pfb_lambda0.2': pfb['0.2'],
                      'pfb_lambda0.4': pfb['0.4'], 'memory_canonical_s100_b256': canon_mem,
                      'memory_pure_max_cache': pure_cache}
    return res


def part_b(text):
    split = np.load(FEA / 'linear_probe_split_seed0.npz', allow_pickle=False)
    tr = np.asarray(split['train_indices'], np.int64); va = np.asarray(split['validation_indices'], np.int64)
    tok, ids, Y = load_tokens('fer2013_train')
    gm = norm(np.load(FEA / 'fer2013_train.npz', allow_pickle=False)['views'].mean(axis=1))
    Xtr = torch.from_numpy(tok[tr]); ytr = torch.from_numpy(Y[tr])
    Xva = torch.from_numpy(tok[va]).to(DEV).float(); yva = Y[va]
    SV, SY = gm[tr], Y[tr]
    pw = np.load(FEA / 'linear_probe_train80_seed0.npz', allow_pickle=False)
    Wp, bp = pw['weights'].astype(np.float32), pw['bias'].astype(np.float32)
    grid = {}
    for lr in (1e-3, 3e-3):
        torch.manual_seed(0)
        m = T2x().to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        t0 = time.perf_counter()
        for ep in range(8):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), 128):
                idx = perm[i:i + 128]
                xb = Xtr[idx].to(DEV, non_blocking=True).float(); yb = ytr[idx].to(DEV, non_blocking=True)
                loss = F.cross_entropy(m(xb), yb)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.inference_mode():
            pv = np.concatenate([m(Xva[i:i + 256]).argmax(1).cpu().numpy() for i in range(0, len(Xva), 256)])
        rec = {'lr': lr, 'params': sum(p.numel() for p in m.parameters()),
               'probe_val_uar': uar(yva, pv), 'seconds': round(time.perf_counter() - t0, 1)}
        torch.save(m.state_dict(), OUT / ('model_T2x_lr%s.pt' % lr))
        grid[str(lr)] = rec
        print('T2x', rec, flush=True)
    best_lr = max(grid, key=lambda k: grid[k]['probe_val_uar'])
    m = T2x().to(DEV)
    m.load_state_dict(torch.load(OUT / ('model_T2x_lr%s.pt' % best_lr), map_location=DEV))
    m.eval()
    cells = []
    for exp, tname, maj in (('ckplus_prior', 'ckplus_test', 6), ('kdef_prior', 'kdef_test', 6)):
        zt = np.load(TC / ('%s_tokens_fp16.npz' % tname), allow_pickle=False)
        gv = norm(np.load(FEA / ('%s.npz' % tname), allow_pickle=False)['views'].mean(axis=1))
        lab = zt['labels'].astype(np.int64)
        for ratio in (0, 1, 2, 5, 10):
            for seed in range(5):
                idx = subset(lab, ratio, maj, seed)
                toks = torch.from_numpy(zt['tokens'][idx]).to(DEV).float()
                g = gv[idx]; y = lab[idx]
                s = g @ SV.T
                cache = np.zeros((len(g), 7), np.float32)
                for c in range(7):
                    mm = SY == c
                    if mm.any():
                        cache[:, c] = s[:, mm].max(1)
                bf = max(uar(y, prior_correct(g @ Wp.T + bp, np.arange(len(g)), 0.2).argmax(1)),
                         uar(y, (S * (g @ text_arr.T) + BETA * cache).argmax(1)))
                with torch.inference_mode():
                    pr = np.concatenate([m(toks[i:i + 256]).argmax(1).cpu().numpy()
                                         for i in range(0, len(toks), 256)])
                cells.append({'cell': '%s|%d|%d' % (exp, ratio, seed), 'T2x': uar(y, pr),
                              'best_family': bf, 'T2x_minus_best': uar(y, pr) - bf})
    n_ge = sum(1 for c in cells if c['T2x_minus_best'] >= -1e-12)
    return {'grid': grid, 'selected_lr': best_lr, 'params': grid[best_lr]['params'],
            'probe_val_uar': grid[best_lr]['probe_val_uar'],
            'crossed_cells': cells, 'cells_ge_best': n_ge, 'n_cells': len(cells),
            'mean_minus_best': float(np.mean([c['T2x_minus_best'] for c in cells]))}


if __name__ == '__main__':
    text_arr = norm(np.load(FEA / 'text_prototypes.npz', allow_pickle=False)['prototypes'].astype(np.float32))
    payload = {'reference_sensitivity_KDE_sourcecells': part_a(text_arr),
               'T2_1x1_equal_capacity_control': part_b(text_arr)}
    (OUT / 'phase3_extra_controls.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in payload['T2_1x1_equal_capacity_control'].items()
                      if k != 'crossed_cells'}, indent=1))
