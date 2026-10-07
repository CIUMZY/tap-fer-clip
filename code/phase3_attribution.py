"""Attribution control (equal-capacity global head) + shared-unit paired bootstrap."""
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
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
S, BETA, LAM, CLIPV, MINN, DEV = 100.0, 256.0, 0.2, 1.0, 32, 'cuda'
spec = importlib.util.spec_from_file_location('m', OUT / 'phase2_train_modules.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prior_correct(lg):
    out = np.empty_like(lg); counts = np.zeros(7, np.int64)
    for i in range(len(lg)):
        v = lg[i].astype(np.float64).copy()
        if counts.sum() >= MINN:
            pi = (counts + 1.0) / (counts.sum() + 7.0)
            v = v - min(LAM, CLIPV) * np.log(np.maximum(pi, 1e-12))
        out[i] = v; counts[int(np.argmax(v))] += 1
    return out


def subset(lab, ratio, maj, seed):
    if ratio == 0:
        return np.arange(len(lab))
    rng = np.random.default_rng(seed)
    cnt = np.bincount(lab, minlength=7); base = int(cnt.min())
    return np.concatenate([rng.choice(np.flatnonzero(lab == c),
                                      size=min(int(cnt[c]), base * (ratio if c == maj else 1)),
                                      replace=False) for c in range(7)])


class GHead(nn.Module):
    """512-d global-feature head with parameter count matched to T1 (~202k)."""
    def __init__(self, d=512, h=256, C=7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(),
                                 nn.Linear(h, C))

    def forward(self, x):
        return self.net(x)


def main():
    t0 = time.perf_counter()
    split = np.load(FEA / 'linear_probe_split_seed0.npz', allow_pickle=False)
    tr = np.asarray(split['train_indices'], np.int64)
    va = np.asarray(split['validation_indices'], np.int64)
    zsrc = np.load(TC / 'fer2013_train_tokens_fp16.npz', allow_pickle=False)
    G = norm(np.load(FEA / 'fer2013_train.npz', allow_pickle=False)['views'].mean(1))
    Y = zsrc['labels'].astype(np.int64)
    Xtr = torch.from_numpy(G[tr].astype(np.float32)); ytr = torch.from_numpy(Y[tr])
    Xva = torch.from_numpy(G[va].astype(np.float32)); yva = Y[va]
    text = norm(np.load(FEA / 'text_prototypes.npz', allow_pickle=False)['prototypes'].astype(np.float32))
    pw = np.load(FEA / 'linear_probe_train80_seed0.npz', allow_pickle=False)
    Wp, bp = pw['weights'].astype(np.float32), pw['bias'].astype(np.float32)
    SV = G[tr]; SY = Y[tr]
    res, best = {}, None
    for lr in (1e-3, 3e-3):
        torch.manual_seed(0)
        head = GHead().to(DEV)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for ep in range(8):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), 128):
                idx = perm[i:i + 128]
                loss = F.cross_entropy(head(Xtr[idx].to(DEV)), ytr[idx].to(DEV))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.inference_mode():
            pv = np.concatenate([head(Xva[i:i + 512].to(DEV)).argmax(1).cpu().numpy()
                                 for i in range(0, len(Xva), 512)])
        rec = {'lr': lr, 'params': sum(p.numel() for p in head.parameters()),
               'probe_val_uar': uar(yva, pv)}
        print('global head', rec, flush=True)
        res[str(lr)] = rec
        if best is None or rec['probe_val_uar'] > best[1]['probe_val_uar']:
            best = (lr, rec)
    lr = best[0]
    head = GHead().to(DEV)
    torch.manual_seed(0); opt = torch.optim.Adam(head.parameters(), lr=lr)
    for ep in range(8):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(head(Xtr[idx].to(DEV)), ytr[idx].to(DEV))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    head.eval()
    torch.save(head.state_dict(), OUT / ('model_GHead_lr%s.pt' % lr))
    models = {}
    for kind, l2 in (('T1', '0.003'), ('T3', '0.003')):
        m = {'T1': mod.T1, 'T3': mod.T3}[kind]().to(DEV)
        m.load_state_dict(torch.load(OUT / ('model_%s_lr%s.pt' % (kind, l2)), map_location=DEV))
        m.eval()
        if kind == 'T3':
            m.ref = torch.from_numpy(np.load(OUT / 'phase2_ref_patch_prototypes.npy')).to(DEV)
        models[kind] = m
    tproj = torch.from_numpy(text).to(DEV)
    cells = {}
    for exp, tname in (('ckplus_prior', 'ckplus_test'), ('kdef_prior', 'kdef_test')):
        zt = np.load(TC / ('%s_tokens_fp16.npz' % tname), allow_pickle=False)
        gv = norm(np.load(FEA / ('%s.npz' % tname), allow_pickle=False)['views'].mean(1))
        lab = zt['labels'].astype(np.int64)
        for ratio in (0, 1, 2, 5, 10):
            for seed in range(5):
                idx = subset(lab, ratio, 6, seed)
                tok = torch.from_numpy(zt['tokens'][idx]).to(DEV).float()
                g = gv[idx]; y = lab[idx]; g_t = torch.from_numpy(g.astype(np.float32)).to(DEV)
                sim = g @ SV.T
                cache = np.zeros((len(g), 7), np.float32)
                for c in range(7):
                    mm = SY == c
                    if mm.any():
                        cache[:, c] = sim[:, mm].max(1)
                pfb = prior_correct(g @ Wp.T + bp).argmax(1)
                mem = (S * (g @ text.T) + BETA * cache).argmax(1)
                rec = {'y': y, 'pfb': pfb, 'memory': mem}
                with torch.inference_mode():
                    gh = np.concatenate([head(g_t[i:i + 512]).argmax(1).cpu().numpy()
                                         for i in range(0, len(g_t), 512)])
                    for kind, m in models.items():
                        ps = []
                        for i in range(0, len(tok), 256):
                            xb = tok[i:i + 256]
                            if kind == 'T3':
                                o, _ = m(xb, S * (g_t[i:i + 256] @ tproj.T))
                            else:
                                o = m(xb)
                            ps.append(o.argmax(1).cpu().numpy())
                        rec[kind] = np.concatenate(ps)
                rec['GHead'] = gh
                cells['%s|%d|%d' % (exp, ratio, seed)] = rec
    rows = []
    for k, v in sorted(cells.items()):
        y = v['y']; bf = max(uar(y, v['pfb']), uar(y, v['memory']))
        rows.append({'cell': k, 'pfb': uar(y, v['pfb']), 'memory': uar(y, v['memory']),
                     'best_family': bf, 'T1': uar(y, v['T1']), 'T3': uar(y, v['T3']),
                     'GHead': uar(y, v['GHead']),
                     'T1_minus_G': uar(y, v['T1']) - uar(y, v['GHead']),
                     'T3_minus_G': uar(y, v['T3']) - uar(y, v['GHead']),
                     'T1_minus_best': uar(y, v['T1']) - bf,
                     'T3_minus_best': uar(y, v['T3']) - bf,
                     'G_minus_best': uar(y, v['GHead']) - bf})
    with (OUT / 'phase3_attribution_cells.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # shared-unit paired bootstrap: hierarchical over seeds, shared sample indices
    keys = sorted(cells)
    seeds = list(range(5))
    rng = np.random.default_rng(0)
    B = 2000
    names = ('T1', 'T3', 'GHead')
    d_best = {n: [] for n in names}
    d_T1G = []
    d_T3G = []
    for _ in range(B):
        ssel = rng.choice(seeds, len(seeds), replace=True)
        acc = {n: [] for n in names}
        acc_best = []
        for s in ssel:
            for e in ('ckplus_prior', 'kdef_prior'):
                for r in (0, 1, 2, 5, 10):
                    v = cells['%s|%d|%d' % (e, r, s)]
                    y = v['y']; n = len(y)
                    idx = rng.integers(0, n, n)
                    bf = max(uar(y[idx], v['pfb'][idx]), uar(y[idx], v['memory'][idx]))
                    acc_best.append(bf)
                    for nm in names:
                        acc[nm].append(uar(y[idx], v[nm][idx]))
        for nm in names:
            d_best[nm].append(float(np.mean(acc[nm])) - float(np.mean(acc_best)))
        d_T1G.append(float(np.mean(acc['T1'])) - float(np.mean(acc['GHead'])))
        d_T3G.append(float(np.mean(acc['T3'])) - float(np.mean(acc['GHead'])))
    out = {}
    for nm in names:
        a = np.asarray(d_best[nm])
        out['%s_minus_best_family' % nm] = {'mean': float(a.mean()),
                                            'ci': [float(np.percentile(a, 2.5)),
                                                   float(np.percentile(a, 97.5))]}
    for nm, arr in (('T1_minus_GHead', d_T1G), ('T3_minus_GHead', d_T3G)):
        a = np.asarray(arr)
        out[nm] = {'mean': float(a.mean()),
                   'ci': [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
    (OUT / 'phase3_attribution.json').write_text(json.dumps(
        {'global_head_grid': res, 'selected_lr': lr,
         'global_head_params': res[str(lr)]['params'],
         'bootstrap': out, 'n_boot': B}, indent=2), encoding='utf-8')
    print('global head params', res[str(lr)]['params'], 'probe-val', res[str(lr)]['probe_val_uar'])
    print(json.dumps(out, indent=1))
    print('cells', len(rows), 'elapsed', round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()
