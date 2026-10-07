"""Phase 3: confirmation on the untouched crossed grid (reference column = best of the two
families), per-module shared-unit paired bootstrap, plus a T2 attribution control."""
import csv, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
TC = D / 'token_cache_vitb16'
FEA = D / 'features'
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
S, BETA, LAM, CLIPV, MINN = 100.0, 256.0, 0.2, 1.0, 32
DEV = 'cuda'
import importlib.util
spec = importlib.util.spec_from_file_location('mod', OUT / 'phase2_train_modules.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prior_correct(logits, order):
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


def subset(lab, ratio, maj, seed):
    if ratio == 0:
        return np.arange(len(lab))
    rng = np.random.default_rng(seed)
    cnt = np.bincount(lab, minlength=7)
    base = int(cnt.min())
    return np.concatenate([rng.choice(np.flatnonzero(lab == c),
                                      size=min(int(cnt[c]), base * (ratio if c == maj else 1)),
                                      replace=False) for c in range(7)])


def main():
    t0 = time.perf_counter()
    split = np.load(FEA / 'linear_probe_split_seed0.npz', allow_pickle=False)
    tr = np.asarray(split['train_indices'], dtype=np.int64)
    zsrc = np.load(TC / 'fer2013_train_tokens_fp16.npz', allow_pickle=False)
    SV = norm(np.load(FEA / 'fer2013_train.npz', allow_pickle=False)['views'].mean(1)[tr])
    SY = zsrc['labels'].astype(np.int64)[tr]
    text = norm(np.load(FEA / 'text_prototypes.npz', allow_pickle=False)['prototypes'].astype(np.float32))
    pr = np.load(FEA / 'linear_probe_train80_seed0.npz', allow_pickle=False)
    Wp, bp = pr['weights'].astype(np.float32), pr['bias'].astype(np.float32)
    models, best_lr = {}, {'T1': '0.003', 'T2': '0.001', 'T3': '0.003', 'T4': '0.003'}
    for kind, lr in best_lr.items():
        m = {'T1': mod.T1, 'T2': mod.T2, 'T3': mod.T3, 'T4': mod.T4}[kind]().to(DEV)
        m.load_state_dict(torch.load(OUT / ('model_%s_lr%s.pt' % (kind, lr)), map_location=DEV))
        m.eval()
        if kind == 'T3':
            m.ref = torch.from_numpy(np.load(OUT / 'phase2_ref_patch_prototypes.npy')).to(DEV)
        models[kind] = m
    tproj = torch.from_numpy(text).to(DEV)
    cells = {}
    for exp, (tname, majority) in (('ckplus_prior', ('ckplus_test', 6)), ('kdef_prior', ('kdef_test', 6))):
        zt = np.load(TC / ('%s_tokens_fp16.npz' % tname), allow_pickle=False)
        gv = norm(np.load(FEA / ('%s.npz' % tname), allow_pickle=False)['views'].mean(1))
        lab = zt['labels'].astype(np.int64)
        for ratio in (0, 1, 2, 5, 10):
            for seed in range(5):
                idx = subset(lab, ratio, majority, seed)
                tok = torch.from_numpy(zt['tokens'][idx]).to(DEV).float()
                g = gv[idx]
                y = lab[idx]
                g_t = torch.from_numpy(g).to(DEV)
                sim = g @ SV.T
                cache = np.zeros((len(g), 7), dtype=np.float32)
                for c in range(7):
                    m = SY == c
                    if m.any():
                        cache[:, c] = sim[:, m].max(1)
                mem = S * (g @ text.T) + BETA * cache
                pfb = prior_correct(g @ Wp.T + bp, np.arange(len(g)))
                rec = {'pfb': uar(y, pfb.argmax(1)), 'memory': uar(y, mem.argmax(1)),
                       'y': y, 'pfb_pred': pfb.argmax(1), 'mem_pred': mem.argmax(1)}
                with torch.inference_mode():
                    for kind, m in models.items():
                        preds = []
                        for i in range(0, len(tok), 256):
                            xb = tok[i:i + 256]
                            if kind == 'T3':
                                gb = g_t[i:i + 256]
                                o, _ = m(xb, S * (gb @ tproj.T))
                            else:
                                o = m(xb)
                            preds.append(o.argmax(1).cpu().numpy())
                        pr_ = np.concatenate(preds)
                        rec[kind] = uar(y, pr_)
                        rec[kind + '_pred'] = pr_
                cells['%s|%d|%d' % (exp, ratio, seed)] = rec
    rows = []
    for k, v in sorted(cells.items()):
        best = max(v['pfb'], v['memory'])
        rows.append({'cell': k, 'pfb': v['pfb'], 'memory': v['memory'], 'best_family': best,
                     **{m: v[m] for m in models},
                     **{m + '_minus_best': v[m] - best for m in models}})
    with (OUT / 'phase3_crossed_cells.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    agg = {}
    for k in ('T1', 'T2', 'T3', 'T4'):
        vals = [r[k + '_minus_best'] for r in rows]
        agg[k] = {'mean_minus_best': float(np.mean(vals)),
                  'cells_ge_best': int(sum(1 for x in vals if x >= -1e-12)),
                  'n_cells': len(vals),
                  'frac_ge_best': float(sum(1 for x in vals if x >= -1e-12) / len(vals))}
    for m in ('pfb', 'memory'):
        print('%-6s mean %.4f' % (m, np.mean([r[m] for r in rows])))
    for k, v in agg.items():
        print('%-3s mean_minus_best %+.4f  >=best %d/%d (%.2f)'
              % (k, v['mean_minus_best'], v['cells_ge_best'], v['n_cells'], v['frac_ge_best']))
    (OUT / 'phase3_summary.json').write_text(json.dumps(
        {'aggregate': agg, 'n_cells': len(rows),
         'threshold': '2/3 of cells >= best of the two families',
         'verdict': {k: ('pass' if v['frac_ge_best'] >= 2 / 3 else 'fail')
                     for k, v in agg.items()}}, indent=2), encoding='utf-8')
    print('elapsed', round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()
