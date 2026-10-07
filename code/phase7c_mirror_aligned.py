"""POST-HOC SENSITIVITY: two-view-mean TAP trained and evaluated against the reported numbers."""
import csv, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
S, DEV, SEEDS = 100.0, 'cuda', [0, 1, 2, 3, 4]
spec = importlib.util.spec_from_file_location('mod', OUT / 'phase2_train_modules.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


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
    def __init__(self, d=512, h=256, C=7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(),
                                 nn.Linear(h, C))

    def forward(self, x):
        return self.net(x)


class CLIPAdapter(nn.Module):
    def __init__(self, text, d=512, hid=128):
        super().__init__()
        self.fc1 = nn.Linear(d, hid)
        self.fc2 = nn.Linear(hid, d)
        self.alpha = 0.2
        self.register_buffer('text', torch.from_numpy(text.astype(np.float32)))

    def forward(self, x):
        f = self.alpha * self.fc2(F.relu(self.fc1(x))) + (1.0 - self.alpha) * x
        return S * (F.normalize(f, dim=-1) @ self.text.T)


def main():
    t0all = time.perf_counter()
    MM_O = np.load(D / 'token_cache_vitb16' / 'fer2013_train_tokens_fp16.npy', mmap_mode='r')
    MM_M = np.load(D / 'token_cache_vitb16_mirror' / 'fer2013_train_tokens_fp16.npy', mmap_mode='r')
    LAB = np.load(D / 'token_cache_vitb16' / 'fer2013_train_tokens_fp16.npz',
                  allow_pickle=False)['labels'].astype(np.int64)
    sp = np.load(D / 'features' / 'linear_probe_split_seed0.npz', allow_pickle=False)
    tr = np.asarray(sp['train_indices'], np.int64)
    va = np.asarray(sp['validation_indices'], np.int64)

    GRID = 14

    def unflip(b):
        n = b.shape[0]
        g = b.reshape(n, GRID, GRID, -1)
        g = g[:, :, ::-1, :]
        return g.reshape(n, GRID * GRID, -1)

    def two_view(sel):
        a = np.asarray(MM_O[sel]).astype(np.float32)
        b = np.asarray(MM_M[sel]).astype(np.float32)
        return torch.from_numpy((a + unflip(b)) / 2.0)

    def train(lr):
        torch.manual_seed(0)
        m = mod.T1(dim=768).to(DEV)
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
        g = torch.Generator().manual_seed(0)
        t0 = time.perf_counter()
        for _ in range(8):
            perm = torch.randperm(len(tr), generator=g)
            for i in range(0, len(perm), 128):
                sel = np.sort(tr[perm[i:i + 128].numpy()])
                xb = two_view(sel).to(DEV)
                yb = torch.from_numpy(LAB[sel]).to(DEV)
                loss = F.cross_entropy(m(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        m.eval()
        ps = []
        with torch.inference_mode():
            for i in range(0, len(va), 256):
                ps.append(m(two_view(va[i:i + 256]).to(DEV)).argmax(1).cpu().numpy())  # fixed: do not sort rows; labels use requested order
        return m, uar(LAB[va], np.concatenate(ps)), round(time.perf_counter() - t0, 1)

    runs = {}
    for lr in (1e-3, 3e-3):
        m, u, sec = train(lr)
        torch.save(m.state_dict(), OUT / ('phase7c_aligned_tap_lr%s.pt' % lr))
        runs['lr%s' % lr] = {'probe_val_uar': u, 'seconds': sec,
                             'params': int(sum(p.numel() for p in m.parameters()))}
        print('mirror TAP lr%s probe-val %.4f (%.0fs)' % (lr, u, sec), flush=True)
    best_lr = float(max(runs, key=lambda k: runs[k]['probe_val_uar']).split('lr')[1])
    rep = {'scope': 'POST-HOC SENSITIVITY (not confirmatory)', 'runs': runs,
           'selected_lr': best_lr, 'params': runs['lr%s' % best_lr]['params'],
           'aggregate': {}, 'harness_check': {}, 'bootstrap': {}}

    tap_m = mod.T1(dim=768).to(DEV)
    tap_m.load_state_dict(torch.load(OUT / ('phase7c_aligned_tap_lr%s.pt' % best_lr), map_location=DEV))
    tap_m.eval()
    tap_o = mod.T1(dim=768).to(DEV)
    tap_o.load_state_dict(torch.load(OUT / 'model_T1_lr0.003.pt', map_location=DEV))
    tap_o.eval()
    text = norm(np.load(D / 'features' / 'text_prototypes.npz', allow_pickle=False)['prototypes']
                .astype(np.float32))
    ca = CLIPAdapter(text).to(DEV)
    ca.load_state_dict(torch.load(OUT / 'phase4_clipadapter_lr0.001.pt', map_location=DEV))
    ca.eval()
    gh = GHead().to(DEV)
    gh.load_state_dict(torch.load(OUT / 'model_GHead_lr0.003.pt', map_location=DEV))
    gh.eval()
    gcache = {}

    def gv(name):
        if name not in gcache:
            gcache[name] = norm(np.load(D / 'features' / ('%s.npz' % name), allow_pickle=False)
                                ['views'].mean(axis=1).astype(np.float32))
        return gcache[name]

    def two_view_named(name, sel):
        if name == 'fer2013_train':
            return two_view(sel)
        o = np.asarray(np.load(D / 'token_cache_vitb16' / ('%s_tokens_fp16.npz' % name),
                               allow_pickle=False)['tokens'][sel]).astype(np.float32)
        m = np.asarray(np.load(D / 'token_cache_vitb16_mirror' / ('%s_tokens_fp16.npy' % name),
                               mmap_mode='r')[sel]).astype(np.float32)
        return torch.from_numpy((o + unflip(m)) / 2.0)

    cells = {}

    def cell(name, sel, y, g, key):
        tk = two_view_named(name, sel)
        with torch.inference_mode():
            ps = []
            for i in range(0, len(tk), 128):
                ps.append(tap_m(tk[i:i + 128].to(DEV)).argmax(1).cpu().numpy())
            p_mir = np.concatenate(ps)
            qs = []
            for i in range(0, len(tk), 128):
                qs.append(tap_o(tk[i:i + 128].to(DEV)).argmax(1).cpu().numpy())
            p_orig_2v = np.concatenate(qs)
            gt = torch.from_numpy(g).to(DEV)
            p_ca = ca(gt).argmax(1).cpu().numpy()
            p_gh = gh(gt).argmax(1).cpu().numpy()
        cells[key] = {'n': int(len(y)), 'y': y, 'mirror': p_mir, 'orig2v': p_orig_2v,
                      'clipadapter': p_ca,
                      'ghead': p_gh, 'uar': {'mirror': uar(y, p_mir),
                                             'orig2v': uar(y, p_orig_2v),
                                             'clipadapter': uar(y, p_ca),
                                             'ghead': uar(y, p_gh)}}

    for exp, name, maj in (('ckplus_prior', 'ckplus_test', 6), ('kdef_prior', 'kdef_test', 6)):
        z = np.load(D / 'token_cache_vitb16' / ('%s_tokens_fp16.npz' % name), allow_pickle=False)
        lab = z['labels'].astype(np.int64)
        del z
        g = gv(name)
        for ratio in (0, 1, 2, 5, 10):
            for seed in range(5):
                sel = subset(lab, ratio, maj, seed)
                cell(name, sel, lab[sel], g[sel], '%s|%d|%d' % (exp, ratio, seed))
        print('setA %s %.0fs' % (exp, time.perf_counter() - t0all), flush=True)
    ks = np.load(D / 'features_kdef_source' / 'kdef_source_test.npz', allow_pickle=False)
    kz = np.load(D / 'token_cache_vitb16' / 'kdef_test_tokens_fp16.npz', allow_pickle=False)
    kpos = {x: i for i, x in enumerate(kz['ids'].astype(str))}
    del kz
    ksel = np.asarray([kpos[x] for x in ks['ids'].astype(str)], np.int64)
    kg = gv('kdef_test')[ksel]
    for seed in SEEDS:
        cell('kdef_test', ksel, ks['labels'].astype(np.int64), kg, 'kdef_source_test|%d' % seed)
    for name in ('fer2013_test', 'ckplus_test'):
        z = np.load(D / 'token_cache_vitb16' / ('%s_tokens_fp16.npz' % name), allow_pickle=False)
        lab = z['labels'].astype(np.int64)
        del z
        g = gv(name)
        for seed in SEEDS:
            cell(name, np.arange(len(lab)), lab, g, '%s|%d' % (name, seed))
    print('setB %.0fs' % (time.perf_counter() - t0all), flush=True)

    stored = {}
    with (OUT / 'phase3_crossed_cells.csv').open(newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            stored[r['cell']] = float(r['T1'])
    with (OUT / 'phase3_kdef_source_cells.csv').open(newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            stored[r['cell']] = float(r['T1'])
    keysA = sorted([k for k in cells if k.split('|')[0].endswith('_prior')])
    keysB = sorted([k for k in cells if not k.split('|')[0].endswith('_prior')])
    rep['harness_check'] = {
        'note': 'stored T1 column is single-view TAP; this sensitivity run multiplies the token '
                'convention by two views, so the stored value is the comparison baseline',
        'cells_with_stored': sum(1 for k in cells if k in stored)}
    for tag, keys in (('SetA', keysA), ('SetB', keysB)):
        rep['aggregate'][tag] = {
            'stored_single_view_TAP': float(np.mean([stored[k] for k in keys])),
            'mirror_two_view_TAP': float(np.mean([cells[k]['uar']['mirror'] for k in keys])),
            'CLIPAdapter': float(np.mean([cells[k]['uar']['clipadapter'] for k in keys])),
            'GHead': float(np.mean([cells[k]['uar']['ghead'] for k in keys])),
            'n_cells': len(keys)}
        rep['aggregate'][tag]['mirror_minus_stored'] = (
            rep['aggregate'][tag]['mirror_two_view_TAP']
            - rep['aggregate'][tag]['stored_single_view_TAP'])
        rep['aggregate'][tag]['mirror_minus_CLIPAdapter'] = (
            rep['aggregate'][tag]['mirror_two_view_TAP']
            - rep['aggregate'][tag]['CLIPAdapter'])
        rep['aggregate'][tag]['mirror_minus_GHead'] = (
            rep['aggregate'][tag]['mirror_two_view_TAP']
            - rep['aggregate'][tag]['GHead'])

    def diff_uar(c, idx, arm):
        yy = c['y'][idx]
        cnt = np.bincount(yy, minlength=7)
        ok = (c[arm][idx] == yy)
        cc = np.bincount(yy, weights=ok.astype(np.float64), minlength=7)
        with np.errstate(invalid='ignore', divide='ignore'):
            per = np.where(cnt > 0, cc / np.maximum(cnt, 1), np.nan)
        return float(np.nanmean(per))

    def boot(keys, arm, ref, n=2000):
        rng = np.random.default_rng(0)
        acc = []
        for _ in range(n):
            pick = rng.choice(len(keys), len(keys), replace=True)
            vals = []
            for i in pick:
                c = cells[keys[i]]
                idx = rng.integers(0, c['n'], c['n'])
                vals.append(diff_uar(c, idx, arm) - diff_uar(c, idx, ref))
            acc.append(float(np.mean(vals)))
        arr = np.asarray(acc)
        return {'mean': float(arr.mean()),
                'ci': [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]}

    for tag, keys in (('SetA', keysA), ('SetB', keysB)):
        rep['bootstrap'][tag] = {
            'mirror_minus_orig_same_tokens_c': boot(keys, 'mirror', 'orig2v'),
        }
        rep['bootstrap'][tag]['mirror_minus_CLIPAdapter'] = boot(keys, 'mirror', 'clipadapter')
        rep['bootstrap'][tag]['mirror_minus_GHead'] = boot(keys, 'mirror', 'ghead')
    diffs = [abs(cells[k]['uar']['orig2v'] - stored[k]) for k in cells if k in stored]
    rep['harness_check']['max_abs_diff_recomputed_single_view_vs_stored_T1'] = (
        float(max(diffs)) if diffs else None)
    rep['harness_check']['n_compared'] = len(diffs)
    rep['bootstrap']['n_boot'] = 2000
    rep['elapsed_seconds'] = round(time.perf_counter() - t0all, 1)

    with (OUT / 'phase7_mirror_cells.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['cell', 'n', 'stored_single_view_TAP', 'mirror_two_view_TAP',
                    'mirror_minus_stored', 'CLIPAdapter', 'GHead'])
        for k in sorted(cells):
            c = cells[k]
            st = stored.get(k, '')
            d = ('%.4f' % (c['uar']['mirror'] - st)) if k in stored else ''
            w.writerow([k, c['n'], st, '%.4f' % c['uar']['mirror'], d,
                        '%.4f' % c['uar']['clipadapter'], '%.4f' % c['uar']['ghead']])
    (OUT / 'phase7_mirror_sensitivity_v2.json').write_text(
        json.dumps({k: v for k, v in rep.items()}, indent=1), encoding='utf-8')
    print(json.dumps(rep['aggregate'], indent=1), flush=True)
    print(json.dumps(rep['bootstrap'], indent=1), flush=True)
    print('elapsed %.0fs' % rep['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
