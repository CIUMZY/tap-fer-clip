"""Phase 4 step 2: confirm published-adapter baselines on both untouched confirmation sets.

Set A = crossed grid (ckplus_prior / kdef_prior, ratio 0/1/2/5/10, seed 0-4), 50 cells.
Set B = second source KDEF-source (kdef_source_test, fer2013_test, ckplus_test, seeds 0-4).
Reference columns: PFB, support memory (s=100, BETA=256), best-of-two-families.
Paired shared-unit bootstrap (2000, hierarchical over seeds, shared indices, seed 0-4).
No target labels are used for training or selection.  Writes only new files.
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
SEEDS = [0, 1, 2, 3, 4]
BASELINES = ['CLIPAdapter', 'LinearProbe', 'APEtext', 'TipAdapterF']
spec = importlib.util.spec_from_file_location('mod', OUT / 'phase2_train_modules.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prior_correct(logits, order, lam=LAM):
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


class CLIPAdapter(nn.Module):
    def __init__(self, text, d=512, hid=128, alpha=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d, hid); self.fc2 = nn.Linear(hid, d); self.alpha = alpha
        self.register_buffer('text', torch.from_numpy(text.astype(np.float32)))

    def forward(self, x):
        f = self.alpha * self.fc2(F.relu(self.fc1(x))) + (1.0 - self.alpha) * x
        return S * (F.normalize(f, dim=-1) @ self.text.T)


class LinearProbe(nn.Module):
    def __init__(self, d=512, C=7):
        super().__init__(); self.fc = nn.Linear(d, C)

    def forward(self, x):
        return self.fc(x)


class APEText(nn.Module):
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
            self.register_buffer('idx%d' % c, torch.tensor(np.flatnonzero(labels == c), dtype=torch.long))

    def forward(self, x):
        kn = F.normalize(self.keys, dim=-1)
        sim = x @ kn.T
        cols = [sim.index_select(1, getattr(self, 'idx%d' % c)).max(dim=1).values for c in range(7)]
        return S * (x @ self.text.T) + self.beta * torch.stack(cols, dim=1)


def load_globals(path):
    z = np.load(path, allow_pickle=False)
    return norm(z['views'].mean(axis=1)), z['labels'].astype(np.int64)


def main():
    t0 = time.perf_counter()
    text = norm(np.load(FEA / 'text_prototypes.npz', allow_pickle=False)['prototypes'].astype(np.float32))
    gtr = norm(np.load(FEA / 'fer2013_train.npz', allow_pickle=False)['views'].mean(axis=1))
    split = np.load(FEA / 'linear_probe_split_seed0.npz', allow_pickle=False)
    tr = np.asarray(split['train_indices'], np.int64)
    SY = np.load(TC / 'fer2013_train_tokens_fp16.npz', allow_pickle=False)['labels'].astype(np.int64)
    SV = gtr[tr]; SYtr = SY[tr]
    pw = np.load(FEA / 'linear_probe_train80_seed0.npz', allow_pickle=False)
    Wp, bp = pw['weights'].astype(np.float32), pw['bias'].astype(np.float32)

    basem = {}
    sel = json.loads((OUT / 'phase4_baselines_training.json').read_text(encoding='utf-8'))['selected']
    def lrs(stem):
        return str(sel[stem]['lr'])
    ca = CLIPAdapter(text).to(DEV); ca.load_state_dict(torch.load(OUT / ('phase4_clipadapter_lr%s.pt' % lrs('CLIPAdapter')), map_location=DEV)); ca.eval()
    lp = LinearProbe().to(DEV); lp.load_state_dict(torch.load(OUT / ('phase4_linearprobe_lr%s.pt' % lrs('LinearProbe')), map_location=DEV)); lp.eval()
    ap = APEText(text).to(DEV); ap.load_state_dict(torch.load(OUT / ('phase4_apetext_lr%s.pt' % lrs('APEtext')), map_location=DEV)); ap.eval()
    tf = TipAdapterF(text, SV, SYtr).to(DEV); tf.load_state_dict(torch.load(OUT / ('phase4_tipadapterf_lr%s.pt' % lrs('TipAdapterF')), map_location=DEV)); tf.eval()
    basem = {'CLIPAdapter': ca, 'LinearProbe': lp, 'APEtext': ap, 'TipAdapterF': tf}
    t1 = mod.T1().to(DEV)
    t1.load_state_dict(torch.load(OUT / 'model_T1_lr0.003.pt', map_location=DEV)); t1.eval()
    tproj = torch.from_numpy(text).to(DEV)

    def baseline_preds(name, gv, tok=None):
        g = torch.from_numpy(gv.astype(np.float32)).to(DEV)
        with torch.inference_mode():
            if name == 'CLIPAdapter':
                logits = basem[name](g)
            elif name == 'APEtext':
                logits = basem[name](g)
            else:
                logits = basem[name](g)
            return logits.argmax(1).cpu().numpy()

    def tap_preds(tok):
        outs = []
        with torch.inference_mode():
            for i in range(0, len(tok), 256):
                outs.append(t1(tok[i:i + 256].to(DEV).float()).argmax(1).cpu().numpy())
        return np.concatenate(outs)

    # ---------------- Set A: crossed grid ----------------
    cells_a = {}
    for exp, tname, maj in (('ckplus_prior', 'ckplus_test', 6), ('kdef_prior', 'kdef_test', 6)):
        zt = np.load(TC / ('%s_tokens_fp16.npz' % tname), allow_pickle=False)
        gv_all = norm(np.load(FEA / ('%s.npz' % tname), allow_pickle=False)['views'].mean(axis=1))
        lab_all = zt['labels'].astype(np.int64)
        for ratio in (0, 1, 2, 5, 10):
            for seed in range(5):
                idx = subset(lab_all, ratio, maj, seed)
                gv = gv_all[idx]; y = lab_all[idx]
                sim = gv @ SV.T
                cache = np.zeros((len(y), 7), np.float32)
                for c in range(7):
                    mm = SYtr == c
                    if mm.any():
                        cache[:, c] = sim[:, mm].max(1)
                mem = (S * (gv @ text.T) + BETA * cache).argmax(1)
                pfb = prior_correct(gv @ Wp.T + bp, np.arange(len(y))).argmax(1)
                rec = {'cell': '%s|%d|%d' % (exp, ratio, seed), 'n': int(len(y)), 'y': y,
                       'pfb_pred': pfb, 'mem_pred': mem,
                       'TAP_pred': tap_preds(torch.from_numpy(zt['tokens'][idx]))}
                for b in BASELINES:
                    rec[b + '_pred'] = baseline_preds(b, gv)
                cells_a[rec['cell']] = rec
        print('setA %s done %.1fs' % (tname, time.perf_counter() - t0), flush=True)

    # ---------------- Set B: KDEF-source ----------------
    ksup, ksy = load_globals(KS / 'kdef_source_train.npz')
    kpw = np.load(KS / 'linear_probe_kdef_source_seed0.npz', allow_pickle=False)
    kW, kb = kpw['weights'].astype(np.float32), kpw['bias'].astype(np.float32)
    kdef_tok = np.load(TC / 'kdef_test_tokens_fp16.npz', allow_pickle=False)
    pos = {x: i for i, x in enumerate(kdef_tok['ids'].astype(str))}
    ks_t = np.load(KS / 'kdef_source_test.npz', allow_pickle=False)
    k_sel = np.asarray([pos[x] for x in ks_t['ids'].astype(str)], dtype=np.int64)
    TARGETS = [('kdef_source_test', lambda: (load_globals(KS / 'kdef_source_test.npz')[0], ks_t['labels'].astype(np.int64),
                                             kdef_tok['tokens'][k_sel])),
               ('fer2013_test', lambda: (load_globals(FEA / 'fer2013_test.npz')[0],
                                         np.load(TC / 'fer2013_test_tokens_fp16.npz', allow_pickle=False)['labels'].astype(np.int64),
                                         np.load(TC / 'fer2013_test_tokens_fp16.npz', allow_pickle=False)['tokens'])),
               ('ckplus_test', lambda: (load_globals(FEA / 'ckplus_test.npz')[0],
                                        np.load(TC / 'ckplus_test_tokens_fp16.npz', allow_pickle=False)['labels'].astype(np.int64),
                                        np.load(TC / 'ckplus_test_tokens_fp16.npz', allow_pickle=False)['tokens']))]
    cells_b = {}
    for tname, loader in TARGETS:
        gv, y, tok = loader()
        sim = gv @ ksup.T
        cache = np.zeros((len(y), 7), np.float32)
        for c in range(7):
            mm = ksy == c
            if mm.any():
                cache[:, c] = sim[:, mm].max(1)
        mem = (S * (gv @ text.T) + BETA * cache).argmax(1)
        probes = gv @ kW.T + kb
        base_pred = {'TAP_pred': tap_preds(torch.from_numpy(tok)), 'mem_pred': mem}
        for b in BASELINES:
            base_pred[b + '_pred'] = baseline_preds(b, gv)
        for seed in SEEDS:
            pfb = prior_correct(probes, np.random.default_rng(seed).permutation(len(y))).argmax(1)
            rec = {'cell': '%s|%d' % (tname, seed), 'target': tname, 'seed': seed, 'n': int(len(y)),
                   'y': y, 'pfb_pred': pfb, **base_pred}
            cells_b[rec['cell']] = rec
        print('setB %s done %.1fs' % (tname, time.perf_counter() - t0), flush=True)

    arms = ['TAP'] + BASELINES

    def rows_of(cells):
        rows = []
        for k, v in sorted(cells.items()):
            y = v['y']
            bf = max(uar(y, v['pfb_pred']), uar(y, v['mem_pred']))
            r = {'cell': k, 'n': v['n'], 'pfb': uar(y, v['pfb_pred']), 'memory': uar(y, v['mem_pred']),
                 'best_family': bf}
            for a in arms:
                r[a] = uar(y, v[a + '_pred'])
            rows.append(r)
        return rows

    rows_a = rows_of(cells_a); rows_b = rows_of(cells_b)

    def boot(cells, hier):
        rng = np.random.default_rng(0)
        B = 2000
        out_d = {}
        accs = {a: [] for a in arms}
        accs_bf = []
        keys_by_seed = {s: [k for k in cells if k.endswith('|%d' % s)] for s in SEEDS}
        for _ in range(B):
            ssel = rng.choice(SEEDS, len(SEEDS), replace=True)
            per = {a: [] for a in arms}
            bf_l = []
            for s in ssel:
                ks = keys_by_seed[s]
                for k in ks:
                    v = cells[k]; y = v['y']; n = len(y)
                    idx = rng.integers(0, n, n)
                    bf = max(uar(y[idx], v['pfb_pred'][idx]), uar(y[idx], v['mem_pred'][idx]))
                    bf_l.append(bf)
                    for a in arms:
                        per[a].append(uar(y[idx], v[a + '_pred'][idx]))
            for a in arms:
                accs[a].append(float(np.mean(per[a])))
            accs_bf.append(float(np.mean(bf_l)))
        base = {a: np.asarray(accs[a]) for a in arms}
        bf = np.asarray(accs_bf)
        for a in arms:
            for ref, arr in (('best_family', base[a] - bf),) + tuple(
                    (o, base[a] - base[o]) for o in arms if o != a):
                out_d['%s_minus_%s' % (a, ref)] = {'mean': float(arr.mean()),
                                                   'ci': [float(np.percentile(arr, 2.5)),
                                                          float(np.percentile(arr, 97.5))]}
        return out_d

    boot_a = boot(cells_a, True); boot_b = boot(cells_b, True)

    for tag, rows in (('A', rows_a), ('B', rows_b)):
        pass
    with (OUT / 'phase4_baselines_cells.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_a[0].keys())); w.writeheader(); w.writerows(rows_a)
    with (OUT / 'phase4_kdef_source_baseline_cells.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_b[0].keys())); w.writeheader(); w.writerows(rows_b)

    def mean(rows, a):
        return float(np.mean([r[a] for r in rows]))
    summary = {
        'arms': arms, 'n_cells_A': len(rows_a), 'n_cells_B': len(rows_b),
        'setA_mean_uar': {a: mean(rows_a, a) for a in arms + ['pfb', 'memory', 'best_family']},
        'setB_mean_uar': {a: mean(rows_b, a) for a in arms + ['pfb', 'memory', 'best_family']},
        'setB_per_target': {t: {a: mean([r for r in rows_b if r['cell'].startswith(t)], a)
                                for a in arms + ['pfb', 'memory', 'best_family']}
                            for t in ('kdef_source_test', 'fer2013_test', 'ckplus_test')},
        'setA_cells_above_best_family': {a: int(sum(1 for r in rows_a if r[a] >= r['best_family'] - 1e-12))
                                         for a in arms},
        'setA_cells_TAP_beats': {b: int(sum(1 for r in rows_a if r['TAP'] > r[b] + 1e-12)) for b in BASELINES},
        'setB_cells_TAP_beats': {b: int(sum(1 for r in rows_b if r['TAP'] > r[b] + 1e-12)) for b in BASELINES},
        'bootstrap_setA': boot_a, 'bootstrap_setB': boot_b, 'n_boot': 2000,
        'training': json.loads((OUT / 'phase4_baselines_training.json').read_text(encoding='utf-8'))['selected'],
    }
    (OUT / 'phase4_baselines.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'setA_mean_uar': summary['setA_mean_uar'], 'setB_mean_uar': summary['setB_mean_uar'],
                      'setA_cells_above_best_family': summary['setA_cells_above_best_family'],
                      'setA_cells_TAP_beats': summary['setA_cells_TAP_beats'],
                      'setB_cells_TAP_beats': summary['setB_cells_TAP_beats']}, indent=1))
    print('TAP vs baselines boot A:', json.dumps({k: v for k, v in boot_a.items() if k.startswith('TAP_minus_')}, indent=1))
    print('TAP vs baselines boot B:', json.dumps({k: v for k, v in boot_b.items() if k.startswith('TAP_minus_')}, indent=1))
    print('elapsed %.1fs' % (time.perf_counter() - t0))


if __name__ == '__main__':
    main()
