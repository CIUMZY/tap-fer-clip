"""Phase 6b: run the pre-registered corruption confirmation.

Four frozen arms per encoder (TAP, CLIP-Adapter, equal-capacity global head, training-free
max(PFB, support memory)), loaded exactly as phases 3/4/5 configured them.  Nothing is trained or
selected here.

Order of operations:
  1. pipeline cross-check on the CLEAN FER2013 test cache against the stored phase 4/5 Set B numbers
     (if this fails the run stops and no H2 verdict is issued);
  2. score the 12 pre-registered corruption conditions, all reported individually;
  3. paired shared-unit bootstrap (2000 draws, seeds x conditions resampled, query indices shared
     across arms) and the fixed decision rule.
"""
import argparse, csv, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
S, BETA, LAM, MINN, CLIPV, DEV = 100.0, 256.0, 0.2, 32, 1.0, 'cuda'
SEEDS = [0, 1, 2, 3, 4]
CONDS = ['blur_0p8', 'blur_1p5', 'blur_2p5', 'jpeg_15', 'jpeg_30', 'jpeg_50',
         'lowlight_0p3', 'lowlight_0p5', 'lowlight_0p7', 'noise_10', 'noise_25', 'noise_40']
TRAINED = ['TAP', 'GHead', 'CLIPAdapter']
SD_THRESHOLD = {'B16': 0.0116, 'L14': 0.0134}
STORED_CLEAN = {'B16': {'TAP': 0.6322587306593955, 'CLIPAdapter': 0.6163,
                        'GHead': 0.6082864569034772, 'pfb': 0.43872419635912496,
                        'memory': 0.40981597980527074},
                'L14': {'TAP': 0.6836415274278039, 'CLIPAdapter': 0.6916,
                        'GHead': 0.6827944966716164, 'pfb': 0.5191414200940269,
                        'memory': 0.44077867522935554}}

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


class GHead(nn.Module):
    def __init__(self, d, h, C=7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(),
                                 nn.Linear(h, C))

    def forward(self, x):
        return self.net(x)


class CLIPAdapter(nn.Module):
    def __init__(self, text, d, hid):
        super().__init__()
        self.fc1 = nn.Linear(d, hid)
        self.fc2 = nn.Linear(hid, d)
        self.alpha = 0.2
        self.register_buffer('text', torch.from_numpy(text.astype(np.float32)))

    def forward(self, x):
        f = self.alpha * self.fc2(F.relu(self.fc1(x))) + (1.0 - self.alpha) * x
        return S * (F.normalize(f, dim=-1) @ self.text.T)


def build(tag):
    if tag == 'B16':
        feat_dir, tok_dir = D / 'features', D / 'token_cache_vitb16'
        d_tok, h, hid = 768, 256, 128
        tap_ck = OUT / 'model_T1_lr0.003.pt'
        gh_ck = OUT / 'model_GHead_lr0.003.pt'
        ca_ck = OUT / 'phase4_clipadapter_lr0.001.pt'
        src_meta = tok_dir / 'fer2013_train_tokens_fp16.npz'
    else:
        feat_dir, tok_dir = D / 'features_vitl14', D / 'token_cache_vitl14'
        d_tok, h, hid = 1024, 260, 192
        tap_ck = OUT / 'phase5_tap_vitl14_lr0.001.pt'
        gh_ck = OUT / 'phase5_ghead_vitl14_lr0.001.pt'
        ca_ck = OUT / 'phase5_clipadapter_vitl14_lr0.001.pt'
        src_meta = tok_dir / 'fer2013_train_meta.npz'
    text = norm(np.load(feat_dir / 'text_prototypes.npz', allow_pickle=False)['prototypes']
                .astype(np.float32))
    tr = np.asarray(np.load(feat_dir / 'linear_probe_split_seed0.npz',
                            allow_pickle=False)['train_indices'], np.int64)
    src = np.load(feat_dir / 'fer2013_train.npz', allow_pickle=False)
    d_glob = src['views'].shape[2]
    SV = norm(src['views'].mean(axis=1).astype(np.float32))[tr]
    SY = np.load(src_meta, allow_pickle=False)['labels'].astype(np.int64)[tr]
    pw = np.load(feat_dir / 'linear_probe_train80_seed0.npz', allow_pickle=False)
    tap = mod.T1(dim=d_tok).to(DEV)
    tap.load_state_dict(torch.load(tap_ck, map_location=DEV))
    tap.eval()
    gh = GHead(d_glob, h).to(DEV)
    gh.load_state_dict(torch.load(gh_ck, map_location=DEV))
    gh.eval()
    ca = CLIPAdapter(text, d_glob, hid).to(DEV)
    ca.load_state_dict(torch.load(ca_ck, map_location=DEV))
    ca.eval()
    return {'tag': tag, 'tap': tap, 'gh': gh, 'ca': ca, 'text': text, 'SV': SV, 'SY': SY,
            'Wp': pw['weights'].astype(np.float32), 'bp': pw['bias'].astype(np.float32),
            'd_tok': d_tok, 'd_glob': d_glob, 'feat_dir': feat_dir, 'tok_dir': tok_dir,
            'ckpt_params': {'tap': int(sum(p.numel() for p in tap.parameters())),
                            'ghead': int(sum(p.numel() for p in gh.parameters())),
                            'clipadapter': int(sum(p.numel() for p in ca.parameters()))}}


def score_cell(arms, tok_path, glob_path, chunk=256):
    gz = np.load(glob_path, allow_pickle=False)
    y = gz['labels'].astype(np.int64)
    g = norm(gz['views'].mean(axis=1).astype(np.float32))
    pred = {}
    with torch.inference_mode():
        gt = torch.from_numpy(g).to(DEV)
        pred['GHead'] = arms['gh'](gt).argmax(1).cpu().numpy()
        pred['CLIPAdapter'] = arms['ca'](gt).argmax(1).cpu().numpy()
        del gt
        # the clean ViT-B/16 cache is a .npz (the confirm caches are memory-mapped .npy)
        mm = (np.load(tok_path, mmap_mode='r') if str(tok_path).endswith('.npy')
              else np.load(tok_path, allow_pickle=False)['tokens'])
        assert mm.shape[0] == len(y), (mm.shape, len(y))
        outs = []
        for i in range(0, len(y), chunk):
            xb = torch.from_numpy(np.asarray(mm[i:i + chunk])).to(DEV).float()
            outs.append(arms['tap'](xb).argmax(1).cpu().numpy())
        pred['TAP'] = np.concatenate(outs)
        del mm
    out = {a: (pred[a] == y) for a in TRAINED}
    logits = g @ arms['Wp'].T + arms['bp']
    for s in SEEDS:
        pfb = prior_correct(logits, np.random.default_rng(s).permutation(len(y))).argmax(1)
        out['pfb|%d' % s] = (pfb == y)
    sim = g @ arms['SV'].T
    cache = np.zeros((len(y), 7), np.float32)
    for c in range(7):
        m = arms['SY'] == c
        if m.any():
            cache[:, c] = sim[:, m].max(1)
    mem = (S * (g @ arms['text'].T) + BETA * cache).argmax(1)
    out['memory'] = (mem == y)
    for s in SEEDS:
        pfb = prior_correct(logits, np.random.default_rng(s).permutation(len(y))).argmax(1)
        out['best_family|%d' % s] = (np.maximum(pfb, mem) == y)
    return {'y': y, 'correct': out}


def cell_uar(cell, key):
    y, corr = cell['y'], cell['correct'][key]
    return float(np.mean([np.mean(corr[y == c]) if np.any(y == c) else 0.0 for c in range(7)]))


def bootstrap(cells, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = ['TAP', 'GHead', 'CLIPAdapter', 'memory'] + \
        ['pfb|%d' % s for s in SEEDS] + ['best_family|%d' % s for s in SEEDS]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        ssel = rng.choice(SEEDS, len(SEEDS), replace=True)
        csel = rng.choice(len(CONDS), len(CONDS), replace=True)
        per = {k: [] for k in keys}
        for s in ssel:
            for ci in csel:
                cell = cells[CONDS[ci]]
                y = cell['y']
                n = len(y)
                idx = rng.integers(0, n, n)
                yy = y[idx]
                cnt = np.bincount(yy, minlength=7)
                for k in keys:
                    kk = k if '|' not in k else k
                    corr = cell['correct'][kk]
                    c = np.bincount(yy, weights=corr[idx].astype(np.float64), minlength=7)
                    per[k].append(float(np.nanmean(np.where(cnt > 0, c / np.maximum(cnt, 1), np.nan))))
        for k in keys:
            acc[k].append(float(np.mean(per[k])))
    res = {}
    pairs = [('TAP', 'CLIPAdapter'), ('TAP', 'GHead')]
    for a, b in pairs:
        d = np.asarray(acc[a]) - np.asarray(acc[b])
        res['%s_minus_%s' % (a, b)] = {'mean': float(d.mean()),
                                       'ci': [float(np.percentile(d, 2.5)),
                                              float(np.percentile(d, 97.5))]}
    for s in SEEDS:
        pfb_k, bf_k = 'pfb|%d' % s, 'best_family|%d' % s
        for a in ('TAP', 'CLIPAdapter', 'GHead'):
            d = np.asarray(acc[a]) - np.asarray(acc[bf_k])
            res.setdefault('%s_minus_best_family|%d' % (a, s), {'mean': float(d.mean()),
                                                                'ci': [float(np.percentile(d, 2.5)),
                                                                       float(np.percentile(d, 97.5))]})
    return res, {k: {'mean': float(np.mean(acc[k])), 'sd': float(np.std(acc[k]))} for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True, choices=['B16', 'L14'])
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    tag = args.encoder
    t_all = time.perf_counter()
    arms = build(tag)
    print('%s arms loaded: %s' % (tag, json.dumps(arms['ckpt_params'])), flush=True)
    tok_confirm = D / ('token_cache_%s_confirm' % tag.lower())
    feat_confirm = D / 'features_confirm_shift' / tag.lower()
    rep = {'encoder': tag, 'conditions': CONDS, 'seeds': SEEDS, 'arms': TRAINED + ['pfb', 'memory'],
           'sd_threshold': SD_THRESHOLD[tag], 'n_boot': args.n_boot,
           'arm_params': arms['ckpt_params']}

    # 1. clean cross-check against stored Set B numbers
    clean_tok = arms['tok_dir'] / 'fer2013_test_tokens_fp16.npy'
    if not clean_tok.exists():                      # ViT-B/16 ships the clean cache as .npz
        clean_tok = arms['tok_dir'] / 'fer2013_test_tokens_fp16.npz'
    print('clean token cache: %s' % clean_tok.name, flush=True)
    clean = score_cell(arms, clean_tok, arms['feat_dir'] / 'fer2013_test.npz')
    got = {a: cell_uar(clean, a) for a in TRAINED}
    got['pfb'] = float(np.mean([cell_uar(clean, 'pfb|%d' % s) for s in SEEDS]))
    got['memory'] = cell_uar(clean, 'memory')
    # only the three trained arms are comparable to the stored Set B numbers: the reference columns
    # here use the FER2013 source support of the pre-registration (the Set A convention), whereas the
    # stored Set B values used the KDEF-source support of that second-source experiment.
    diffs = {a: got[a] - STORED_CLEAN[tag][a] for a in TRAINED}
    rep['clean_cross_check'] = {'computed': got, 'stored': STORED_CLEAN[tag], 'diff': diffs,
                                'max_abs_diff': float(max(abs(v) for v in diffs.values())),
                                'compared_arms': TRAINED,
                                'note': 'PFB and support memory use the pre-registered FER2013 source '
                                        'support, not the KDEF-source support of the stored Set B '
                                        'reference columns, so they are reported but not compared'}
    print('clean cross-check computed: %s' % json.dumps({k: round(v, 4) for k, v in got.items()}),
          flush=True)
    print('clean cross-check stored  : %s' % json.dumps(
        {k: round(v, 4) for k, v in STORED_CLEAN[tag].items()}), flush=True)
    print('max |diff| = %.4f' % rep['clean_cross_check']['max_abs_diff'], flush=True)
    if rep['clean_cross_check']['max_abs_diff'] > 0.001:
        print('CROSS-CHECK FAILED - stopping', flush=True)
        (OUT / ('phase6b_confirm_%s.json' % tag)).write_text(json.dumps(rep, indent=1),
                                                             encoding='utf-8')
        return 2

    # 2. the 12 pre-registered conditions
    cells = {}
    rows = []
    for cond in CONDS:
        t0 = time.perf_counter()
        cell = score_cell(arms, tok_confirm / ('%s_tokens_fp16.npy' % cond),
                          feat_confirm / ('%s.npz' % cond))
        cells[cond] = cell
        u = {a: cell_uar(cell, a) for a in TRAINED + ['memory']}
        for s in SEEDS:
            u['pfb|%d' % s] = cell_uar(cell, 'pfb|%d' % s)
            u['best_family|%d' % s] = cell_uar(cell, 'best_family|%d' % s)
        u['best_family_mean5'] = float(np.mean([u['best_family|%d' % s] for s in SEEDS]))
        u['pfb_mean5'] = float(np.mean([u['pfb|%d' % s] for s in SEEDS]))
        rep.setdefault('cells', {})[cond] = {'n': int(len(cell['y'])), 'uar': u,
                                             'seconds': round(time.perf_counter() - t0, 1)}
        rows.append({'condition': cond, 'n': int(len(cell['y'])), **u})
        print('%-13s %s' % (cond, json.dumps({k: round(v, 4) for k, v in u.items() if '|' not in k})),
              flush=True)

    # aggregate over conditions
    agg = {}
    for a in TRAINED + ['memory']:
        vals = np.asarray([rep['cells'][c]['uar'][a] for c in CONDS])
        agg[a] = {'mean': float(vals.mean()), 'sd_across_conditions': float(vals.std(ddof=1)),
                  'min': float(vals.min()), 'max': float(vals.max())}
    for k in ('pfb_mean5', 'best_family_mean5'):
        vals = np.asarray([rep['cells'][c]['uar'][k] for c in CONDS])
        agg[k] = {'mean': float(vals.mean()), 'sd_across_conditions': float(vals.std(ddof=1)),
                  'min': float(vals.min()), 'max': float(vals.max())}
    rep['aggregate'] = agg

    # 3. bootstrap + decision rule
    boot, boot_marginals = bootstrap(cells, n_boot=args.n_boot)
    rep['bootstrap'] = boot
    key = 'TAP_minus_CLIPAdapter'
    point = agg['TAP']['mean'] - agg['CLIPAdapter']['mean']
    ci = boot[key]['ci']
    crit1 = bool(ci[0] > 0)
    crit2 = bool(abs(point) > SD_THRESHOLD[tag])
    rep['decision'] = {'point_TAP_minus_CLIPAdapter': float(point), 'ci95': ci,
                       'criterion1_ci_excludes_zero_and_positive': crit1,
                       'criterion2_margin_exceeds_seed_sd': crit2,
                       'seed_sd_threshold': SD_THRESHOLD[tag],
                       'H2_holds': bool(crit1 and crit2),
                       'role': 'primary H2' if tag == 'B16' else 'secondary, not part of H2'}
    rep['bootstrap_marginals'] = boot_marginals
    rep['elapsed_seconds'] = round(time.perf_counter() - t_all, 1)
    with (OUT / ('phase6b_cells_%s.csv' % tag)).open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(OUT / ('phase6b_correct_arrays_%s.npz' % tag),
                        **{('%s__%s' % (c, k.replace('|', '_'))): v
                           for c in CONDS for k, v in cells[c]['correct'].items()})
    (OUT / ('phase6b_confirm_%s.json' % tag)).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    (OUT / ('phase6b_bootstrap_%s.json' % tag)).write_text(json.dumps(
        {'bootstrap': boot, 'marginals': boot_marginals, 'n_boot': args.n_boot,
         'design': 'seeds x conditions resampled with replacement; query indices shared across arms'},
        indent=1), encoding='utf-8')
    print('\naggregate: %s' % json.dumps({k: round(v['mean'], 4) for k, v in agg.items()}), flush=True)
    print('DECISION %s: point=%+.4f ci=[%+.4f,%+.4f] crit1=%s crit2=%s -> H2_holds=%s'
          % (tag, point, ci[0], ci[1], crit1, crit2, rep['decision']['H2_holds']), flush=True)
    print('elapsed %.1fs' % rep['elapsed_seconds'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
