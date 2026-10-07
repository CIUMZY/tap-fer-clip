"""Phase 3 step 2/3: second-source (KDEF-source) confirmation for the token modules.

Source support  = KDEF-source train (2056 images, subject-held-out split).
Targets         = kdef_source_test, fer2013_test, ckplus_test.
Reference cols  = PFB (KDEF-source linear probe + online prior correction) and the
                  support memory (text + BETA * per-class-max support similarity),
                  plus best-of-two-families = max(PFB, memory).
Token modules   = T1-T4 with the FER2013-source-selected lr (no new tuning) and the
                  equal-capacity 512-d global head.
No target labels are used for training or selection.  Writes only new files.
"""
import csv, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
TC = D / 'token_cache_vitb16'
FEA = D / 'features'
KS = D / 'features_kdef_source'
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
S, BETA, LAM, CLIPV, MINN, DEV = 100.0, 256.0, 0.2, 1.0, 32, 'cuda'
SEEDS = [0, 1, 2, 3, 4]
spec = importlib.util.spec_from_file_location('mod', OUT / 'phase2_train_modules.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


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


class GHead(nn.Module):
    def __init__(self, d=512, h=256, C=7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, C))

    def forward(self, x):
        return self.net(x)


def load_tokens(name):
    z = np.load(TC / (name + '_tokens_fp16.npz'), allow_pickle=False)
    return z['tokens'], z['ids'].astype(str), z['labels'].astype(np.int64)


def kdef_source_test_tokens():
    tok, ids, lab = load_tokens('kdef_test')
    pos = {x: i for i, x in enumerate(ids)}
    ks = np.load(KS / 'kdef_source_test.npz', allow_pickle=False)
    kid = ks['ids'].astype(str)
    sel = np.asarray([pos[x] for x in kid], dtype=np.int64)
    assert np.array_equal(lab[sel], ks['labels'].astype(np.int64))
    return tok[sel], kid, ks['labels'].astype(np.int64)


def load_globals(path):
    z = np.load(path, allow_pickle=False)
    return norm(z['views'].mean(axis=1)), z['labels'].astype(np.int64)


def main():
    t0 = time.perf_counter()
    text = norm(np.load(FEA / 'text_prototypes.npz', allow_pickle=False)['prototypes'].astype(np.float32))
    sup, sup_lab = load_globals(KS / 'kdef_source_train.npz')
    kp = np.load(KS / 'linear_probe_kdef_source_seed0.npz', allow_pickle=False)
    Wp, bp = kp['weights'].astype(np.float32), kp['bias'].astype(np.float32)

    # --- models: FER2013-source-selected lr, reused as-is (no new tuning) ---
    best_lr = {'T1': '0.003', 'T2': '0.001', 'T3': '0.003', 'T4': '0.003'}
    models = {}
    for kind, lr in best_lr.items():
        m = {'T1': mod.T1, 'T2': mod.T2, 'T3': mod.T3, 'T4': mod.T4}[kind]().to(DEV)
        m.load_state_dict(torch.load(OUT / ('model_%s_lr%s.pt' % (kind, lr)), map_location=DEV))
        m.eval()
        models[kind] = m
    # T3 memory buffer rebuilt from the KDEF-source support set (ref = source patch prototypes);
    # the FER2013-frozen ref is kept as a sensitivity variant.
    ks_tok, ks_ids, ks_lab = load_tokens('kdef_source_train')
    ref_kdef = np.zeros((7, 196, 768), dtype=np.float32)
    for c in range(7):
        idx = np.flatnonzero(ks_lab == c)[:1500]
        ref_kdef[c] = ks_tok[idx].astype(np.float32).mean(0)
    ref_fer = np.load(OUT / 'phase2_ref_patch_prototypes.npy')
    models['T3'].ref = torch.from_numpy(ref_kdef).to(DEV)
    gh = GHead().to(DEV)
    gh.load_state_dict(torch.load(OUT / 'model_GHead_lr0.003.pt', map_location=DEV))
    gh.eval()
    tproj = torch.from_numpy(text).to(DEV)

    def fwd_tokens(kind, tok):
        outs = []
        with torch.inference_mode():
            for i in range(0, len(tok), 256):
                xb = tok[i:i + 256].to(DEV).float()
                if kind == 'T3':
                    gb = torch.from_numpy(gv_all[i:i + 256].astype(np.float32)).to(DEV)
                    o, _ = models[kind](xb, S * (gb @ tproj.T))
                else:
                    o = models[kind](xb)
                outs.append(o.argmax(1).cpu().numpy())
        return np.concatenate(outs)

    TARGETS = [('kdef_source_test', None), ('fer2013_test', FEA / 'fer2013_test.npz'),
               ('ckplus_test', FEA / 'ckplus_test.npz')]
    cells = {}
    for tname, gpath in TARGETS:
        if tname == 'kdef_source_test':
            tok, ids, y = kdef_source_test_tokens()
            gv_all, gy = load_globals(KS / 'kdef_source_test.npz')
        else:
            tok, ids, y = load_tokens(tname)
            gv_all, gy = load_globals(gpath)
        assert np.array_equal(y, gy)
        sim = gv_all @ sup.T
        cache = np.zeros((len(y), 7), dtype=np.float32)
        for c in range(7):
            mm = sup_lab == c
            if mm.any():
                cache[:, c] = sim[:, mm].max(1)
        mem_logits = S * (gv_all @ text.T) + BETA * cache
        mem_pred = mem_logits.argmax(1)
        probe_logits = gv_all @ Wp.T + bp
        t_pred = {k: fwd_tokens(k, torch.from_numpy(tok)) for k in ('T1', 'T2', 'T4')}
        t_pred['T3'] = fwd_tokens('T3', torch.from_numpy(tok))
        with torch.inference_mode():
            gh_pred = np.concatenate([gh(torch.from_numpy(gv_all[i:i + 512].astype(np.float32)).to(DEV)).argmax(1).cpu().numpy()
                                      for i in range(0, len(gv_all), 512)])
        for seed in SEEDS:
            order = np.random.default_rng(seed).permutation(len(y))
            pfb_pred = prior_correct(probe_logits, order).argmax(1)
            rec = {'target': tname, 'seed': seed, 'n': int(len(y)), 'pfb': uar(y, pfb_pred),
                   'memory': uar(y, mem_pred)}
            for k in ('T1', 'T2', 'T3', 'T4'):
                rec[k.lower()] = uar(y, t_pred[k])
            rec['ghead'] = uar(y, gh_pred)
            rec['pfb_pred'] = pfb_pred; rec['mem_pred'] = mem_pred; rec['y'] = y
            for k in ('T1', 'T2', 'T3', 'T4'):
                rec[k.lower() + '_pred'] = t_pred[k]
            rec['ghead_pred'] = gh_pred
            cells['%s|%d' % (tname, seed)] = rec
        # T3 sensitivity with the FER2013-frozen ref buffer
        models['T3'].ref = torch.from_numpy(ref_fer).to(DEV)
        t3_fer = fwd_tokens('T3', torch.from_numpy(tok))
        models['T3'].ref = torch.from_numpy(ref_kdef).to(DEV)
        for seed in SEEDS:
            cells['%s|%d' % (tname, seed)]['t3_ferref'] = uar(y, t3_fer)
        print('target %s done %.1fs' % (tname, time.perf_counter() - t0), flush=True)

    rows = []
    for k, v in sorted(cells.items()):
        bf = max(v['pfb'], v['memory'])
        rows.append({'cell': k, 'target': v['target'], 'seed': v['seed'], 'n': v['n'],
                     'pfb': v['pfb'], 'memory': v['memory'], 'best_family': bf,
                     'T1': v['t1'], 'T2': v['t2'], 'T3': v['t3'], 'T4': v['t4'], 'GHead': v['ghead'],
                     'T3_ferref': v['t3_ferref'],
                     'T1_minus_best': v['t1'] - bf, 'T1_minus_G': v['t1'] - v['ghead'],
                     'T3_minus_best': v['t3'] - bf, 'T3_minus_G': v['t3'] - v['ghead'],
                     'T2_minus_best': v['t2'] - bf, 'T4_minus_best': v['t4'] - bf,
                     'G_minus_best': v['ghead'] - bf})
    with (OUT / 'phase3_kdef_source_cells.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    per_target = {}
    for tname, _ in TARGETS:
        sub = [r for r in rows if r['target'] == tname]
        per_target[tname] = {
            'n': sub[0]['n'],
            'pfb': float(np.mean([r['pfb'] for r in sub])),
            'memory': float(np.mean([r['memory'] for r in sub])),
            'best_family': float(np.mean([r['best_family'] for r in sub])),
            'T1': float(np.mean([r['T1'] for r in sub])), 'T2': float(np.mean([r['T2'] for r in sub])),
            'T3': float(np.mean([r['T3'] for r in sub])), 'T4': float(np.mean([r['T4'] for r in sub])),
            'GHead': float(np.mean([r['GHead'] for r in sub])),
            'T3_ferref': float(np.mean([r['T3_ferref'] for r in sub])),
            'T1_minus_best': float(np.mean([r['T1_minus_best'] for r in sub])),
            'T1_minus_G': float(np.mean([r['T1_minus_G'] for r in sub])),
            'T3_minus_G': float(np.mean([r['T3_minus_G'] for r in sub])),
            'cells_T1_ge_best': int(sum(1 for r in sub if r['T1_minus_best'] >= -1e-12)),
            'cells': len(sub)}

    # --- shared-unit paired bootstrap: hierarchical over seeds x targets, shared indices ---
    rng = np.random.default_rng(0)
    B = 2000
    names = ('T1', 'T2', 'T3', 'T4', 'GHead')
    dbest = {n: [] for n in names}
    dT1G, dT3G, dT1T3 = [], [], []
    for _ in range(B):
        ssel = rng.choice(SEEDS, len(SEEDS), replace=True)
        acc = {n: [] for n in names}
        bf_acc = []
        for s in ssel:
            for tname, _ in TARGETS:
                v = cells['%s|%d' % (tname, s)]
                y = v['y']; n = len(y)
                idx = rng.integers(0, n, n)
                bf = max(uar(y[idx], v['pfb_pred'][idx]), uar(y[idx], v['mem_pred'][idx]))
                bf_acc.append(bf)
                for nm in names:
                    acc[nm].append(uar(y[idx], v[nm.lower() + '_pred'][idx]))
        for nm in names:
            dbest[nm].append(float(np.mean(acc[nm])) - float(np.mean(bf_acc)))
        dT1G.append(float(np.mean(acc['T1'])) - float(np.mean(acc['GHead'])))
        dT3G.append(float(np.mean(acc['T3'])) - float(np.mean(acc['GHead'])))
        dT1T3.append(float(np.mean(acc['T1'])) - float(np.mean(acc['T3'])))
    boot = {}
    for nm in names:
        a = np.asarray(dbest[nm])
        boot[nm + '_minus_best_family'] = {'mean': float(a.mean()),
                                           'ci': [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
    for nm, arr in (('T1_minus_GHead', dT1G), ('T3_minus_GHead', dT3G), ('T1_minus_T3', dT1T3)):
        a = np.asarray(arr)
        boot[nm] = {'mean': float(a.mean()),
                    'ci': [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}

    payload = {'source_support': 'KDEF-source train (2056)', 'targets': [t[0] for t in TARGETS],
               'seeds': SEEDS, 'best_lr': best_lr, 'per_target': per_target,
               'cells_T1_ge_best': int(sum(1 for r in rows if r['T1_minus_best'] >= -1e-12)),
               'n_cells': len(rows), 'bootstrap': boot, 'n_boot': B,
               't3_ref': 'KDEF-source patch prototypes (primary); FER2013-frozen ref reported as T3_ferref'}
    (OUT / 'phase3_kdef_source.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(per_target, indent=1))
    print(json.dumps(boot, indent=1))
    print('T1 cells >= best %d/%d' % (payload['cells_T1_ge_best'], len(rows)))
    print('elapsed %.1fs' % (time.perf_counter() - t0))


if __name__ == '__main__':
    main()
