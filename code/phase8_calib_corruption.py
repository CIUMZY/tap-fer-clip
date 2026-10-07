"""Calibration for the corruption-axis evaluator: run it with the ORIGINAL seed-0 checkpoints and
compare with the manuscript (B/16 TAP-CA +0.0181; L/14 TAP-CA +0.0320)."""
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

CONDS = ['blur_0p8', 'blur_1p5', 'blur_2p5', 'jpeg_15', 'jpeg_30', 'jpeg_50',
         'lowlight_0p3', 'lowlight_0p5', 'lowlight_0p7', 'noise_10', 'noise_25', 'noise_40']
ORIG = {
    'B16': dict(dim=768, gd=512, hid=256, gh=256, ca_hid=128,
                tap='model_T1_lr0.003.pt', ca='phase4_clipadapter_lr0.001.pt', ghck='model_GHead_lr0.003.pt'),
    'L14': dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192,
                tap='phase5_tap_vitl14_lr0.001.pt', ca='phase5_clipadapter_vitl14_lr0.001.pt',
                ghck='phase5_ghead_vitl14_lr0.001.pt'),
}


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def run(tag):
    c = ORIG[tag]; low = tag.lower()
    tok_dir = D / ('token_cache_%s_confirm' % low)
    feat_dir = D / 'features_confirm_shift' / low
    tap = p8.T1(c['dim'], c['hid']).to(DEV); tap.load_state_dict(torch.load(OUT / c['tap'], map_location=DEV)); tap.eval()
    ca = p8.CLIPAdapter(np.zeros((7, c['gd']), np.float32), c['gd'], c['ca_hid']).to(DEV)
    ca.load_state_dict(torch.load(OUT / c['ca'], map_location=DEV)); ca.eval()
    gh = p8.GHead(c['gd'], c['gh']).to(DEV); gh.load_state_dict(torch.load(OUT / c['ghck'], map_location=DEV)); gh.eval()
    acc = {'TAP': [], 'CA': [], 'GH': []}
    for cond in CONDS:
        gz = np.load(feat_dir / ('%s.npz' % cond), allow_pickle=False)
        y = gz['labels'].astype(np.int64)
        g = norm(gz['views'].mean(axis=1).astype(np.float32))
        TOK = np.load(tok_dir / ('%s_tokens_fp16.npy' % cond), mmap_mode='r')
        gt = torch.from_numpy(g).to(DEV)
        with torch.inference_mode():
            outs = []
            for i in range(0, len(y), 256):
                xb = torch.from_numpy(np.asarray(TOK[i:i + 256])).to(DEV).float()
                outs.append(tap(xb).argmax(1).cpu().numpy())
            tp = np.concatenate(outs)
            cp = ca(gt).argmax(1).cpu().numpy(); gp = gh(gt).argmax(1).cpu().numpy()
        acc['TAP'].append(uar(y, tp)); acc['CA'].append(uar(y, cp)); acc['GH'].append(uar(y, gp))
        del TOK, gt; torch.cuda.empty_cache()
    mu = {k: float(np.mean(v)) for k, v in acc.items()}
    return {'TAP_mean': mu['TAP'], 'CA_mean': mu['CA'], 'GH_mean': mu['GH'],
            'TAP_minus_CA': mu['TAP'] - mu['CA'], 'TAP_minus_GH': mu['TAP'] - mu['GH'],
            'n_cond': len(acc['TAP'])}


res = {}
for tag in ('B16', 'L14'):
    res[tag] = run(tag)
    print(tag, json.dumps(res[tag]), flush=True)
(OUT / 'phase8_calib_corruption_seed0.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
print('manuscript: B16 TAP-CA +0.0181 (TAP .4740 / CA .4559); L14 TAP-CA +0.0320 (TAP .5360 / CA .5040)', flush=True)

