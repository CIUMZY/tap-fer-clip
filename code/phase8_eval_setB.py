"""Phase 8 evaluation: training-seed variance of the TAP-versus-comparator contrast on the
second-source confirmation set (Set B).

Set B rebuilds the support set and prior from the KDEF-source split, but the trained arms are
re-used as-is (phase3_kdef_source_confirm.py: "FER2013-source-selected lr, reused as-is"), so the
trained-arm predictions do not depend on the support set and can be obtained by scoring the
retrained arms directly on the three targets. Post-hoc variance probe on consumed cells.
"""
import importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
DEV = 'cuda'
KS = D / 'features_kdef_source'
spec = importlib.util.spec_from_file_location('p8', OUT / 'phase8_trainvar.py')
p8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p8)

ENC = {
    'B16': dict(dim=768, gd=512, hid=256, gh=256, ca_hid=128,
                tok=D / 'token_cache_vitb16', feat=D / 'features',
                seeds=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    'L14': dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192,
                tok=D / 'token_cache_vitl14', feat=D / 'features_vitl14',
                seeds=[5, 6, 7, 8, 9]),
}


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def build_targets(tag):
    """-> list of (name, tokens, globals, labels)"""
    c = ENC[tag]
    ks = np.load(KS / 'kdef_source_test.npz', allow_pickle=False)
    ks_ids = ks['ids'].astype(str)
    ks_y = ks['labels'].astype(np.int64)
    out = []
    if tag == 'B16':
        z = np.load(c['tok'] / 'kdef_test_tokens_fp16.npz', allow_pickle=False)
        pos = {x: i for i, x in enumerate(z['ids'].astype(str))}
        sel = np.asarray([pos[x] for x in ks_ids], dtype=np.int64)
        g = norm(np.load(KS / 'kdef_source_test.npz', allow_pickle=False)['views'].mean(axis=1).astype(np.float32))
        out.append(('kdef_source_test', np.asarray(z['tokens'][sel]), g, ks_y))
        for t in ('fer2013_test', 'ckplus_test'):
            zt = np.load(c['tok'] / ('%s_tokens_fp16.npz' % t), allow_pickle=False)
            gt = np.load(c['feat'] / ('%s.npz' % t), allow_pickle=False)
            out.append((t, np.asarray(zt['tokens']), norm(gt['views'].mean(axis=1).astype(np.float32)),
                        gt['labels'].astype(np.int64)))
    else:
        gz = np.load(c['feat'] / 'kdef_test.npz', allow_pickle=False)
        kg = norm(gz['views'].mean(axis=1).astype(np.float32))
        kpos = {x: i for i, x in enumerate(gz['ids'].astype(str))}
        idx = np.asarray([kpos[x] for x in ks_ids], dtype=np.int64)
        TOK = np.load(c['tok'] / 'kdef_test_tokens_fp16.npy', mmap_mode='r')
        out.append(('kdef_source_test', np.asarray(TOK[idx]), kg[idx], ks_y))
        for t in ('fer2013_test', 'ckplus_test'):
            gt = np.load(c['feat'] / ('%s.npz' % t), allow_pickle=False)
            out.append((t, np.load(c['tok'] / ('%s_tokens_fp16.npy' % t), mmap_mode='r'),
                        norm(gt['views'].mean(axis=1).astype(np.float32)), gt['labels'].astype(np.int64)))
    return out


def run(tag):
    c = ENC[tag]
    t0 = time.perf_counter()
    models = {}
    for s in c['seeds']:
        tap = p8.T1(c['dim'], c['hid']).to(DEV)
        tap.load_state_dict(torch.load(OUT / ('phase8_tap_%s_seed%d.pt' % (tag, s)), map_location=DEV)); tap.eval()
        ca = p8.CLIPAdapter(np.zeros((7, c['gd']), np.float32), c['gd'], c['ca_hid']).to(DEV)
        ca.load_state_dict(torch.load(OUT / ('phase8_clipadapter_%s_seed%d.pt' % (tag, s)), map_location=DEV)); ca.eval()
        gh = p8.GHead(c['gd'], c['gh']).to(DEV)
        gh.load_state_dict(torch.load(OUT / ('phase8_ghead_%s_seed%d.pt' % (tag, s)), map_location=DEV)); gh.eval()
        models[s] = (tap, ca, gh)

    per_seed = {s: {'TAP': [], 'CA': [], 'GH': []} for s in c['seeds']}
    rows_t = []
    for tname, TOK, gv, y in build_targets(tag):
        gt = torch.from_numpy(np.ascontiguousarray(gv)).to(DEV)
        n = len(y)
        row = {'target': tname, 'n': int(n)}
        with torch.inference_mode():
            for s, (tap, ca, gh) in models.items():
                outs = []
                for i in range(0, n, 256):
                    xb = torch.from_numpy(np.asarray(TOK[i:i + 256])).to(DEV).float()
                    outs.append(tap(xb).argmax(1).cpu().numpy())
                tp = np.concatenate(outs)
                cp = ca(gt).argmax(1).cpu().numpy()
                gp = gh(gt).argmax(1).cpu().numpy()
                u = {'TAP': uar(y, tp), 'CA': uar(y, cp), 'GH': uar(y, gp)}
                for a in ('TAP', 'CA', 'GH'):
                    per_seed[s][a].append(u[a])
                row['seed%d' % s] = u
        del TOK, gt
        torch.cuda.empty_cache()
        rows_t.append(row)
        print('%s %-18s done %.1fs' % (tag, tname, time.perf_counter() - t0), flush=True)

    rows = []
    for s in c['seeds']:
        d = per_seed[s]
        rows.append({'encoder': tag, 'train_seed': s, 'n_targets': len(d['TAP']),
                     'TAP_mean': float(np.mean(d['TAP'])), 'CA_mean': float(np.mean(d['CA'])),
                     'GH_mean': float(np.mean(d['GH'])),
                     'TAP_minus_CA': float(np.mean(d['TAP']) - np.mean(d['CA'])),
                     'TAP_minus_GH': float(np.mean(d['TAP']) - np.mean(d['GH']))})
    d1 = np.array([r['TAP_minus_CA'] for r in rows])
    d2 = np.array([r['TAP_minus_GH'] for r in rows])
    return ({'rows': rows, 'targets': [r['target'] for r in rows_t], 'n_seeds': len(rows),
             'TAP_minus_CA': {'mean': float(d1.mean()), 'sd': float(d1.std(ddof=1)),
                              'min': float(d1.min()), 'max': float(d1.max())},
             'TAP_minus_GH': {'mean': float(d2.mean()), 'sd': float(d2.std(ddof=1)),
                              'min': float(d2.min()), 'max': float(d2.max())},
             'seconds': round(time.perf_counter() - t0, 1)}, rows_t)


def main():
    out = {}; per = {}
    for tag in ('B16', 'L14'):
        res, rows_t = run(tag)
        out[tag] = res; per[tag] = rows_t
        print(tag, json.dumps({k: v for k, v in res.items() if k != 'rows'}), flush=True)
    (OUT / 'phase8_setB_trainvar.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    (OUT / 'phase8_setB_pertarget.json').write_text(json.dumps(per, indent=2), encoding='utf-8')
    print('done', flush=True)


if __name__ == '__main__':
    main()

