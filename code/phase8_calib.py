"""Calibration: run the phase-8 Set A evaluator on the ORIGINAL seed-0 checkpoints and compare
with the numbers reported in the manuscript (B/16: TAP 0.6784, CLIP-Adapter 0.6396, GHead 0.6158).
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
spec2 = importlib.util.spec_from_file_location('p8e', OUT / 'phase8_eval_setA.py')
p8e = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(p8e)

ORIG = {
    'B16': dict(dim=768, gd=512, hid=256, gh=256, ca_hid=128,
                tokdir=D / 'token_cache_vitb16', feat=D / 'features',
                tap='model_T1_lr0.003.pt', ca='phase4_clipadapter_lr0.001.pt', ghck='model_GHead_lr0.003.pt'),
    'L14': dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192,
                tokdir=D / 'token_cache_vitl14', feat=D / 'features_vitl14',
                tap='phase5_tap_vitl14_lr0.001.pt', ca='phase5_clipadapter_vitl14_lr0.001.pt',
                ghck='phase5_ghead_vitl14_lr0.001.pt'),
}


def run(tag):
    c = ORIG[tag]
    tap = p8.T1(c['dim'], c['hid']).to(DEV); tap.load_state_dict(torch.load(OUT / c['tap'], map_location=DEV)); tap.eval()
    ca = p8.CLIPAdapter(np.zeros((7, c['gd']), np.float32), c['gd'], c['ca_hid']).to(DEV)
    ca.load_state_dict(torch.load(OUT / c['ca'], map_location=DEV)); ca.eval()
    gh = p8.GHead(c['gd'], c['gh']).to(DEV); gh.load_state_dict(torch.load(OUT / c['ghck'], map_location=DEV)); gh.eval()
    acc = {'TAP': [], 'CA': [], 'GH': []}
    for tname, maj in (('ckplus_test', 6), ('kdef_test', 6)):
        zf = np.load(c['feat'] / ('%s.npz' % tname), allow_pickle=False)
        gv_all = p8e.norm(zf['views'].mean(axis=1)); lab_all = zf['labels'].astype(np.int64)
        if tag == 'B16':
            TOK = np.load(c['tokdir'] / ('%s_tokens_fp16.npz' % tname), allow_pickle=False)['tokens']
        else:
            TOK = np.load(c['tokdir'] / ('%s_tokens_fp16.npy' % tname), mmap_mode='r')
        for ratio in (0, 1, 2, 5, 10):
            for cseed in range(5):
                idx = p8e.subset(lab_all, ratio, maj, cseed)
                y = lab_all[idx]
                g = torch.from_numpy(gv_all[idx].astype(np.float32)).to(DEV)
                tok = torch.from_numpy(np.asarray(TOK[idx])).to(DEV).float()
                with torch.inference_mode():
                    tp = torch.cat([tap(tok[i:i + 256]).argmax(1) for i in range(0, len(tok), 256)]).cpu().numpy()
                    cp = torch.cat([ca(g[i:i + 512]).argmax(1) for i in range(0, len(g), 512)]).cpu().numpy()
                    gp = torch.cat([gh(g[i:i + 512]).argmax(1) for i in range(0, len(g), 512)]).cpu().numpy()
                acc['TAP'].append(p8e.uar(y, tp)); acc['CA'].append(p8e.uar(y, cp)); acc['GH'].append(p8e.uar(y, gp))
                del tok, g; torch.cuda.empty_cache()
    mu = {k: float(np.mean(v)) for k, v in acc.items()}
    return {'TAP_mean': mu['TAP'], 'CA_mean': mu['CA'], 'GH_mean': mu['GH'],
            'TAP_minus_CA': mu['TAP'] - mu['CA'], 'TAP_minus_GH': mu['TAP'] - mu['GH'],
            'n_cells': len(acc['TAP'])}


res = {}
for tag in ('B16', 'L14'):
    res[tag] = run(tag)
    print(tag, json.dumps(res[tag]), flush=True)
(OUT / 'phase8_calib_seed0.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
print('expected B16 from manuscript: TAP 0.6784, CA 0.6396, GHead 0.6158', flush=True)
print('expected L14 from manuscript: TAP 0.7570, CA 0.7767, GHead 0.7681', flush=True)

