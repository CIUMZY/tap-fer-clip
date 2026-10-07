"""Calibration of the Set B retraining evaluator against the single run (seed-0 checkpoints).

Manuscript Table 3 means: B/16 TAP 0.6722, CLIP-Adapter 0.6375, matched head 0.6202;
L/14 TAP 0.7405, CLIP-Adapter 0.7543, matched head 0.7412.
"""
import importlib.util, json, sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
DEV = 'cuda'
spec = importlib.util.spec_from_file_location('p8', OUT / 'phase8_trainvar.py')
p8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p8)
spec2 = importlib.util.spec_from_file_location('p8b', OUT / 'phase8_eval_setB.py')
p8b = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(p8b)

ORIG = {
    'B16': dict(dim=768, gd=512, hid=256, gh=256, ca_hid=128,
                tap='model_T1_lr0.003.pt', ca='phase4_clipadapter_lr0.001.pt', ghck='model_GHead_lr0.003.pt'),
    'L14': dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192,
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
    per = {}
    for tname, TOK, gv, y in p8b.build_targets(tag):
        gt = torch.from_numpy(np.ascontiguousarray(gv)).to(DEV)
        n = len(y)
        with torch.inference_mode():
            outs = []
            for i in range(0, n, 256):
                xb = torch.from_numpy(np.asarray(TOK[i:i + 256])).to(DEV).float()
                outs.append(tap(xb).argmax(1).cpu().numpy())
            tp = np.concatenate(outs)
            cp = ca(gt).argmax(1).cpu().numpy(); gp = gh(gt).argmax(1).cpu().numpy()
        u = {'TAP': p8b.uar(y, tp), 'CA': p8b.uar(y, cp), 'GH': p8b.uar(y, gp)}
        per[tname] = u
        for a in ('TAP', 'CA', 'GH'):
            acc[a].append(u[a])
        del TOK, gt
        torch.cuda.empty_cache()
    mu = {k: float(np.mean(v)) for k, v in acc.items()}
    return {'per_target': per, 'TAP_mean': mu['TAP'], 'CA_mean': mu['CA'], 'GH_mean': mu['GH'],
            'TAP_minus_CA': mu['TAP'] - mu['CA'], 'TAP_minus_GH': mu['TAP'] - mu['GH']}


res = {}
for tag in ('B16', 'L14'):
    res[tag] = run(tag)
    print(tag, json.dumps(res[tag]), flush=True)
(OUT / 'phase8_calib_setB_seed0.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
print('expected B16: TAP 0.6722 CA 0.6375 GH 0.6202 | L14: TAP 0.7405 CA 0.7543 GH 0.7412', flush=True)

