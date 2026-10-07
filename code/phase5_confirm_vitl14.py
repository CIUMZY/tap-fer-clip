"""Phase 5 step 4: ViT-L/14 confirmation of TAP vs the equal-capacity global head and
CLIP-Adapter on the two untouched confirmation sets, with shared-unit paired bootstrap.
"""
import csv, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
TC = D / 'token_cache_vitl14'
L = D / 'features_vitl14'
KS = D / 'features_kdef_source'
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
S, BETA, LAM, CLIPV, MINN, DEV = 100.0, 256.0, 0.2, 1.0, 32, 'cuda'
SEEDS = [0, 1, 2, 3, 4]
ARMS = ['TAP', 'GHead', 'CLIPAdapter']
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


def load_globals(path):
    z = np.load(path, allow_pickle=False)
    return norm(z['views'].mean(axis=1)), z['labels'].astype(np.int64), z['ids'].astype(str)


def fit_kdef_probe(x, y, epochs=50, lr=1e-3, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    model = nn.Linear(x.shape[1], 7).to(DEV)
    counts = np.bincount(y, minlength=7)
    w = len(y) / (7 * np.clip(counts, 1, None))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    xt = torch.tensor(x, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(xt), generator=g)
        for s in range(0, len(perm), 1024):
            idx = perm[s:s + 1024]
            loss = crit(model(xt[idx].to(DEV)), yt[idx].to(DEV))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model.eval()


def main():
    t0 = time.perf_counter()
    text = norm(np.load(L / 'text_prototypes.npz', allow_pickle=False)['prototypes'].astype(np.float32))
    split = np.load(L / 'linear_probe_split_seed0.npz', allow_pickle=False)
    gtr, ytr_all, _ = load_globals(L / 'fer2013_train.npz')
    tr = np.asarray(split['train_indices'], np.int64)
    SY = np.load(TC / 'fer2013_train_tokens_fp16.npz', allow_pickle=False)['labels'].astype(np.int64)
    SV, SYtr = gtr[tr], SY[tr]
    pw = np.load(L / 'linear_probe_train80_seed0.npz', allow_pickle=False)
    Wp, bp = pw['weights'].astype(np.float32), pw['bias'].astype(np.float32)
    sel = json.loads((OUT / 'phase5_training_vitl14.json').read_text(encoding='utf-8'))['selected']
    tap = mod.T1(dim=1024).to(DEV)
    tap.load_state_dict(torch.load(OUT / ('phase5_tap_vitl14_lr%s.pt' % str(sel['TAP']['lr'])), map_location=DEV))
    tap.eval()
    gh = GHead(768, 260).to(DEV)
    gh.load_state_dict(torch.load(OUT / ('phase5_ghead_vitl14_lr%s.pt' % str(sel['GHead']['lr'])), map_location=DEV))
    gh.eval()
    ca = CLIPAdapter(text).to(DEV)
    ca.load_state_dict(torch.load(OUT / ('phase5_clipadapter_vitl14_lr%s.pt' % str(sel['CLIPAdapter']['lr'])), map_location=DEV))
    ca.eval()
    tproj = torch.from_numpy(text).to(DEV)

    def tap_preds(tok):
        outs = []
        with torch.inference_mode():
            for i in range(0, len(tok), 128):
                outs.append(tap(tok[i:i + 128].to(DEV).float()).argmax(1).cpu().numpy())
        return np.concatenate(outs)

    def global_preds(gv):
        g = torch.from_numpy(gv.astype(np.float32)).to(DEV)
        with torch.inference_mode():
            return gh(g).argmax(1).cpu().numpy(), ca(g).argmax(1).cpu().numpy()

    def kdef_target(name):
        g, y, ids = load_globals(L / ('%s.npz' % name))
        zt = np.load(TC / ('%s_tokens_fp16.npz' % name), allow_pickle=False)
        pos = {x: i for i, x in enumerate(zt['ids'].astype(str))}
        sel_idx = np.asarray([pos[x] for x in ids], dtype=np.int64)
        assert np.array_equal(zt['labels'].astype(np.int64)[sel_idx], y)
        return g, y, zt['tokens'][sel_idx]

    # ---------------- Set A: crossed grid ----------------
    cells_a = {}
    for exp, tname, maj in (('ckplus_prior', 'ckplus_test', 6), ('kdef_prior', 'kdef_test', 6)):
        za = np.load(TC / ('%s_tokens_fp16.npz' % tname), allow_pickle=False)
        zf = np.load(L / ('%s.npz' % tname), allow_pickle=False)
        assert np.array_equal(za['ids'].astype(str), zf['ids'].astype(str)), 'id order mismatch ' + tname
        assert np.array_equal(za['labels'].astype(np.int64), zf['labels'].astype(np.int64)), 'label mismatch ' + tname
        gv_all = norm(zf['views'].mean(axis=1))
        lab_all = za['labels'].astype(np.int64)
        for ratio in (0, 1, 2, 5, 10):
            for seed in range(5):
                idx = subset(lab_all, ratio, maj, seed)
                gv = gv_all[idx]; y = lab_all[idx]
                sim = gv @ SV.T
                cache = np.zeros((len(y), 7), np.float32)
                for c in range(7):
                    m = SYtr == c
                    if m.any():
                        cache[:, c] = sim[:, m].max(1)
                mem = (S * (gv @ text.T) + BETA * cache).argmax(1)
                pfb = prior_correct(gv @ Wp.T + bp, np.arange(len(y))).argmax(1)
                gp, cp = global_preds(gv)
                cells_a['%s|%d|%d' % (exp, ratio, seed)] = {
                    'cell': '%s|%d|%d' % (exp, ratio, seed), 'n': int(len(y)), 'y': y,
                    'pfb_pred': pfb, 'mem_pred': mem, 'TAP_pred': tap_preds(torch.from_numpy(za['tokens'][idx])),
                    'GHead_pred': gp, 'CLIPAdapter_pred': cp}
        print('setA %s done %.1fs' % (tname, time.perf_counter() - t0), flush=True)

    # ---------------- Set B: KDEF-source ----------------
    kg, ky, kids = load_globals(L / 'kdef_test.npz')
    kpos = {x: i for i, x in enumerate(kids)}
    src = np.load(KS / 'kdef_source_train.npz', allow_pickle=False)
    s_idx = np.asarray([kpos[x] for x in src['ids'].astype(str)], dtype=np.int64)
    ksup, ksy = kg[s_idx], src['labels'].astype(np.int64)
    assert np.array_equal(ky[s_idx], ksy), 'KDEF-source support label mismatch'
    probe_k = fit_kdef_probe(ksup, ksy)
    val = np.load(KS / 'kdef_source_val.npz', allow_pickle=False)
    vg = norm(val['views'].mean(axis=1))
    with torch.inference_mode():
        probe_val_uar = uar(val['labels'].astype(np.int64),
                            probe_k(torch.from_numpy(vg.astype(np.float32)).to(DEV)).argmax(1).cpu().numpy())
    with torch.inference_mode():
        kW = probe_k.weight.detach().cpu().numpy().astype(np.float32)
        kb = probe_k.bias.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(OUT / 'phase5_linear_probe_kdef_source_vitl14.npz', weights=kW, bias=kb,
                        train_fraction=np.asarray(1.0), split_seed=np.asarray(0),
                        unpaired=np.asarray(False))
    cells_b = {}
    for tname in ('kdef_source_test', 'fer2013_test', 'ckplus_test'):
        gv, y, tok = kdef_target(tname)
        sim = gv @ ksup.T
        cache = np.zeros((len(y), 7), np.float32)
        for c in range(7):
            m = ksy == c
            if m.any():
                cache[:, c] = sim[:, m].max(1)
        mem = (S * (gv @ text.T) + BETA * cache).argmax(1)
        gp, cp = global_preds(gv)
        tp = tap_preds(torch.from_numpy(tok))
        probes = gv @ kW.T + kb
        for seed in SEEDS:
            pfb = prior_correct(probes, np.random.default_rng(seed).permutation(len(y))).argmax(1)
            cells_b['%s|%d' % (tname, seed)] = {
                'cell': '%s|%d' % (tname, seed), 'target': tname, 'seed': seed, 'n': int(len(y)),
                'y': y, 'pfb_pred': pfb, 'mem_pred': mem, 'TAP_pred': tp,
                'GHead_pred': gp, 'CLIPAdapter_pred': cp}
        print('setB %s done %.1fs' % (tname, time.perf_counter() - t0), flush=True)

    def rows_of(cells):
        rows = []
        for k, v in sorted(cells.items()):
            y = v['y']
            bf = max(uar(y, v['pfb_pred']), uar(y, v['mem_pred']))
            r = {'cell': k, 'n': v['n'], 'pfb': uar(y, v['pfb_pred']), 'memory': uar(y, v['mem_pred']),
                 'best_family': bf}
            for a in ARMS:
                r[a] = uar(y, v[a + '_pred'])
            rows.append(r)
        return rows

    rows_a, rows_b = rows_of(cells_a), rows_of(cells_b)

    def boot(cells):
        rng = np.random.default_rng(0)
        B = 2000
        accs = {a: [] for a in ARMS}; accs_bf = []
        kbs = {s: [k for k in cells if k.endswith('|%d' % s)] for s in SEEDS}
        for _ in range(B):
            ssel = rng.choice(SEEDS, len(SEEDS), replace=True)
            per = {a: [] for a in ARMS}; bf_l = []
            for s in ssel:
                for k in kbs[s]:
                    v = cells[k]; y = v['y']; n = len(y)
                    idx = rng.integers(0, n, n)
                    bf_l.append(max(uar(y[idx], v['pfb_pred'][idx]), uar(y[idx], v['mem_pred'][idx])))
                    for a in ARMS:
                        per[a].append(uar(y[idx], v[a + '_pred'][idx]))
            for a in ARMS:
                accs[a].append(float(np.mean(per[a])))
            accs_bf.append(float(np.mean(bf_l)))
        base = {a: np.asarray(accs[a]) for a in ARMS}; bf = np.asarray(accs_bf)
        out = {}
        for a in ARMS:
            for ref, arr in (('best_family', base[a] - bf),) + tuple(
                    (o, base[a] - base[o]) for o in ARMS if o != a):
                out['%s_minus_%s' % (a, ref)] = {
                    'mean': float(arr.mean()),
                    'ci': [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]}
        return out

    ba, bb = boot(cells_a), boot(cells_b)
    with (OUT / 'phase5_vitl14_cells.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_a[0].keys())); w.writeheader(); w.writerows(rows_a)
    with (OUT / 'phase5_vitl14_kdef_source_cells.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_b[0].keys())); w.writeheader(); w.writerows(rows_b)

    def mean(rows, a):
        return float(np.mean([r[a] for r in rows]))
    payload = {
        'model': 'ViT-L-14', 'arms': ARMS + ['pfb', 'memory', 'best_family'],
        'kdef_source_probe_val_uar': probe_val_uar,
        'setA_mean_uar': {a: mean(rows_a, a) for a in ARMS + ['pfb', 'memory', 'best_family']},
        'setB_mean_uar': {a: mean(rows_b, a) for a in ARMS + ['pfb', 'memory', 'best_family']},
        'setB_per_target': {t: {a: mean([r for r in rows_b if r['cell'].startswith(t)], a)
                                for a in ARMS + ['pfb', 'memory', 'best_family']}
                            for t in ('kdef_source_test', 'fer2013_test', 'ckplus_test')},
        'setA_cells_TAP_beats': {b: int(sum(1 for r in rows_a if r['TAP'] > r[b] + 1e-12))
                                 for b in ('GHead', 'CLIPAdapter')},
        'setB_cells_TAP_beats': {b: int(sum(1 for r in rows_b if r['TAP'] > r[b] + 1e-12))
                                 for b in ('GHead', 'CLIPAdapter')},
        'n_cells_A': len(rows_a), 'n_cells_B': len(rows_b),
        'bootstrap_setA': ba, 'bootstrap_setB': bb, 'n_boot': 2000,
        'training': sel}
    (OUT / 'phase5_vitl14.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'setA_mean_uar': payload['setA_mean_uar'], 'setB_mean_uar': payload['setB_mean_uar'],
                      'setA_cells_TAP_beats': payload['setA_cells_TAP_beats'],
                      'setB_cells_TAP_beats': payload['setB_cells_TAP_beats'],
                      'kdef_probe_val_uar': probe_val_uar}, indent=1), flush=True)
    for tag, bo in (('A', ba), ('B', bb)):
        print('bootstrap', tag, json.dumps({k: v for k, v in bo.items()
                                            if 'TAP_minus' in k or 'CLIPAdapter_minus_best' in k}, indent=1))
    print('elapsed %.1fs' % (time.perf_counter() - t0), flush=True)


if __name__ == '__main__':
    main()
