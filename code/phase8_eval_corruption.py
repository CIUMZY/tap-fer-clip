"""Phase 8 evaluation: training-seed variance of the TAP-versus-comparator contrast on the
pre-specified 12-condition corruption axis.

Same post-hoc variance probe as phase8_eval_setA.py, applied to the corrupted FER2013-test cache.
The axis is already consumed; nothing is selected on it.
"""
import importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
DEV = 'cuda'
spec = importlib.util.spec_from_file_location('p8', OUT / 'phase8_trainvar.py')
p8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p8)

CONDS = ['blur_0p8', 'blur_1p5', 'blur_2p5', 'jpeg_15', 'jpeg_30', 'jpeg_50',
         'lowlight_0p3', 'lowlight_0p5', 'lowlight_0p7', 'noise_10', 'noise_25', 'noise_40']

ENC = {
    'B16': dict(dim=768, gd=512, hid=256, gh=256, ca_hid=128,
                seeds=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    'L14': dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192,
                seeds=[5, 6, 7, 8, 9]),
}


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def run(tag):
    c = ENC[tag]
    low = tag.lower()
    tok_dir = D / ('token_cache_%s_confirm' % low)
    feat_dir = D / 'features_confirm_shift' / low
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
    cond_rows = []
    for cond in CONDS:
        gz = np.load(feat_dir / ('%s.npz' % cond), allow_pickle=False)
        y = gz['labels'].astype(np.int64)
        g = norm(gz['views'].mean(axis=1).astype(np.float32))
        TOK = np.load(tok_dir / ('%s_tokens_fp16.npy' % cond), mmap_mode='r')
        assert len(TOK) == len(y), (len(TOK), len(y))
        gt = torch.from_numpy(g).to(DEV)
        row = {'condition': cond, 'n': int(len(y))}
        with torch.inference_mode():
            for s, (tap, ca, gh) in models.items():
                outs = []
                for i in range(0, len(y), 256):
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
        cond_rows.append(row)
        print('%s %-12s done %.1fs' % (tag, cond, time.perf_counter() - t0), flush=True)

    rows = []
    for s in c['seeds']:
        d = per_seed[s]
        rows.append({'encoder': tag, 'train_seed': s, 'n_cond': len(d['TAP']),
                     'TAP_mean': float(np.mean(d['TAP'])), 'CA_mean': float(np.mean(d['CA'])),
                     'GH_mean': float(np.mean(d['GH'])),
                     'TAP_minus_CA': float(np.mean(d['TAP']) - np.mean(d['CA'])),
                     'TAP_minus_GH': float(np.mean(d['TAP']) - np.mean(d['GH'])),
                     'cells_TAP_gt_CA': int(sum(1 for a, b in zip(d['TAP'], d['CA']) if a > b))})
    d1 = np.array([r['TAP_minus_CA'] for r in rows])
    d2 = np.array([r['TAP_minus_GH'] for r in rows])
    return ({'rows': rows, 'conditions': CONDS, 'n_seeds': len(rows),
             'TAP_minus_CA': {'mean': float(d1.mean()), 'sd': float(d1.std(ddof=1)), 'min': float(d1.min()), 'max': float(d1.max())},
             'TAP_minus_GH': {'mean': float(d2.mean()), 'sd': float(d2.std(ddof=1)), 'min': float(d2.min()), 'max': float(d2.max())},
             'seconds': round(time.perf_counter() - t0, 1)}, cond_rows)


def main():
    out = {}
    conds = {}
    for tag in ('B16', 'L14'):
        res, cond_rows = run(tag)
        out[tag] = res
        conds[tag] = cond_rows
        print(tag, json.dumps({k: v for k, v in res.items() if k != 'rows'}), flush=True)
    (OUT / 'phase8_corruption_trainvar.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    (OUT / 'phase8_corruption_percond.json').write_text(json.dumps(conds, indent=2), encoding='utf-8')
    print('done', flush=True)


if __name__ == '__main__':
    main()

