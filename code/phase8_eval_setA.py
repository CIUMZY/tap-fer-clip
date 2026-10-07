"""Phase 8 evaluation: training-seed variance of the TAP-versus-comparator contrast on the
crossed-grid confirmation cells (Set A).

Re-uses the freshly retrained source-domain arms (phase8_trainvar.py) and evaluates every training
seed on the same 50 Set A cells used in the paper. Post-hoc variance probe, not a new confirmation:
the cells are already consumed and nothing is selected on them.
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

ENC = {
    'B16': dict(dim=768, gd=512, hid=256, gh=256, ca_hid=128,
                tokdir=D / 'token_cache_vitb16', feat=D / 'features',
                seeds=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    'L14': dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192,
                tokdir=D / 'token_cache_vitl14', feat=D / 'features_vitl14',
                seeds=[5, 6, 7, 8, 9]),
}


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


def run(tag):
    c = ENC[tag]
    t0 = time.perf_counter()
    models = {}
    for s in c['seeds']:
        tap = p8.T1(c['dim'], c['hid']).to(DEV)
        tap.load_state_dict(torch.load(OUT / ('phase8_tap_%s_seed%d.pt' % (tag, s)), map_location=DEV))
        tap.eval()
        ca = p8.CLIPAdapter(np.zeros((7, c['gd']), np.float32), c['gd'], c['ca_hid']).to(DEV)
        ca.load_state_dict(torch.load(OUT / ('phase8_clipadapter_%s_seed%d.pt' % (tag, s)), map_location=DEV))
        ca.eval()
        gh = p8.GHead(c['gd'], c['gh']).to(DEV)
        gh.load_state_dict(torch.load(OUT / ('phase8_ghead_%s_seed%d.pt' % (tag, s)), map_location=DEV))
        gh.eval()
        models[s] = (tap, ca, gh)

    per_seed = {s: {'TAP': [], 'CA': [], 'GH': []} for s in c['seeds']}
    n_cells = 0
    for tname, maj in (('ckplus_test', 6), ('kdef_test', 6)):
        zf = np.load(c['feat'] / ('%s.npz' % tname), allow_pickle=False)
        gv_all = norm(zf['views'].mean(axis=1))
        lab_all = zf['labels'].astype(np.int64)
        if tag == 'B16':
            zt = np.load(c['tokdir'] / ('%s_tokens_fp16.npz' % tname), allow_pickle=False)
            TOK = zt['tokens']
        else:
            TOK = np.load(c['tokdir'] / ('%s_tokens_fp16.npy' % tname), mmap_mode='r')
        for ratio in (0, 1, 2, 5, 10):
            for cseed in range(5):
                idx = subset(lab_all, ratio, maj, cseed)
                y = lab_all[idx]
                g = torch.from_numpy(gv_all[idx].astype(np.float32)).to(DEV)
                tok = torch.from_numpy(np.asarray(TOK[idx])).to(DEV).float()
                with torch.inference_mode():
                    for s, (tap, ca, gh) in models.items():
                        tpred = torch.cat([tap(tok[i:i + 256]).argmax(1) for i in range(0, len(tok), 256)]).cpu().numpy()
                        cpred = torch.cat([ca(g[i:i + 512]).argmax(1) for i in range(0, len(g), 512)]).cpu().numpy()
                        gpred = torch.cat([gh(g[i:i + 512]).argmax(1) for i in range(0, len(g), 512)]).cpu().numpy()
                        per_seed[s]['TAP'].append(uar(y, tpred))
                        per_seed[s]['CA'].append(uar(y, cpred))
                        per_seed[s]['GH'].append(uar(y, gpred))
                n_cells += 1
                del tok, g
                torch.cuda.empty_cache()
        print('%s %s done %.1fs' % (tag, tname, time.perf_counter() - t0), flush=True)

    rows = []
    for s in c['seeds']:
        d = per_seed[s]
        rows.append({'encoder': tag, 'train_seed': s, 'n_cells': len(d['TAP']),
                     'TAP_mean': float(np.mean(d['TAP'])),
                     'CA_mean': float(np.mean(d['CA'])),
                     'GH_mean': float(np.mean(d['GH'])),
                     'TAP_minus_CA': float(np.mean(d['TAP']) - np.mean(d['CA'])),
                     'TAP_minus_GH': float(np.mean(d['TAP']) - np.mean(d['GH']))})
    d1 = np.array([r['TAP_minus_CA'] for r in rows])
    d2 = np.array([r['TAP_minus_GH'] for r in rows])
    return ({'rows': rows, 'n_cells': n_cells, 'n_seeds': len(rows),
             'TAP_minus_CA': {'mean': float(d1.mean()), 'sd': float(d1.std(ddof=1)), 'min': float(d1.min()), 'max': float(d1.max())},
             'TAP_minus_GH': {'mean': float(d2.mean()), 'sd': float(d2.std(ddof=1)), 'min': float(d2.min()), 'max': float(d2.max())},
             'seconds': round(time.perf_counter() - t0, 1)}, per_seed)


def main():
    allseeds = {}
    out = {}
    for tag in ('B16', 'L14'):
        res, per_seed = run(tag)
        out[tag] = res
        allseeds[tag] = {str(k): {a: list(map(float, v)) for a, v in d.items()} for k, d in per_seed.items()}
        print(tag, json.dumps({k: v for k, v in res.items() if k != 'rows'}), flush=True)
    (OUT / 'phase8_setA_trainvar.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    (OUT / 'phase8_setA_percell.json').write_text(json.dumps(allseeds, indent=2), encoding='utf-8')
    print('done', flush=True)


if __name__ == '__main__':
    main()

