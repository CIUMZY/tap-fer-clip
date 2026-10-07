"""Temporary instrumented probe of the R3 (attention-pooling) training path.
Prints token statistics, batch class composition, loss trajectory and val UAR for
(a) the current TAP training and (b) a variant whose attention starts exactly uniform.
"""
import importlib.util, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
spec = importlib.util.spec_from_file_location('dg', OUT / 'diagnostic_token_vs_global_run.py')
dg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dg)

enc = dg.ENC['B16']
z = np.load(enc['glob'], allow_pickle=False)
nz = np.load(enc['npz'], allow_pickle=False)
labs, gids = nz['labels'].astype(np.int64), nz['ids'].astype(str)
assert np.array_equal(gids, z['ids'].astype(str))
dg.build_b16_cache(enc)
tok = dg.Tokens(enc)
split = np.load(enc['glob'].parent / 'linear_probe_split_seed0.npz', allow_pickle=False)
tr = np.asarray(split['train_indices'], np.int64)
va = np.asarray(split['validation_indices'], np.int64)
y_all = labs
g_all = dg.norm(z['views'].mean(axis=1)).astype(np.float32)

# --- 1. is the stored row order class-grouped?  (sorted-within-batch would then matter) -------
chg = int(np.sum(np.diff(y_all) != 0))
print('row-order class changes: %d over %d rows (grouped if small)  counts=%s'
      % (chg, len(y_all), np.bincount(y_all, minlength=7).tolist()), flush=True)
rng = np.random.default_rng(0)
b = rng.choice(tr, 128, replace=False)
print('random 128-row batch class counts  :', np.bincount(y_all[b], minlength=7).tolist(), flush=True)
print('sorted 128-row batch class counts  :',
      np.bincount(y_all[np.sort(b)], minlength=7).tolist(), flush=True)

# --- 2. token statistics of an actual training batch ----------------------------------------
sel = np.sort(b)
x = tok.rows(sel).astype(np.float32)
print('batch tokens: shape %s finite=%s norm mean %.3f min %.3f max %.3f'
      % (x.shape, bool(np.isfinite(x).all()), float(np.linalg.norm(x, axis=-1).mean()),
         float(np.linalg.norm(x, axis=-1).min()), float(np.linalg.norm(x, axis=-1).max())), flush=True)
P = dg.pooled_tokens(tok, tr[:2000], 512)[0]
print('mean-pooled probe-train sample: norm mean %.3f' % float(np.linalg.norm(P, axis=1).mean()),
      flush=True)


def run(tag, zero_attn, steps=300, lr=1e-3):
    torch.manual_seed(0)
    m = dg.TAP(enc['tok_dim']).to(dg.DEV)
    if zero_attn:
        nn.init.zeros_(m.a2.weight)
        nn.init.zeros_(m.a2.bias)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator().manual_seed(0)
    done = 0
    while done < steps:
        perm = torch.randperm(len(tr), generator=gen)
        for i in range(0, len(perm), 128):
            s = np.sort(np.asarray(tr)[perm[i:i + 128].numpy()])
            xb = torch.from_numpy(tok.rows(s)).to(dg.DEV).float()
            yb = torch.from_numpy(y_all[s]).to(dg.DEV)
            out = m(xb)
            loss = F.cross_entropy(out, yb)
            if done % 50 == 0:
                w = torch.softmax(m.a2(torch.tanh(m.a1(xb))).squeeze(-1), dim=1)
                print('  [%s] step %4d loss %.4f | logit_std %.3f | attn entropy %.4f (max %.4f)'
                      % (tag, done, float(loss), float(out.std()), float(-(w * torch.log(w + 1e-12)).sum(1).mean()),
                         float(w.max(1).values.mean())), flush=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            done += 1
            if done >= steps:
                break
    m.eval()
    rec = dg.eval_stream(m, y_all, g_all, tok, va, 'R3')
    print('  [%s] after %d steps: val UAR %.4f (prior %.4f)'
          % (tag, steps, rec['uar'], rec['uar_prior']), flush=True)


print('--- (a) current TAP training ---', flush=True)
run('plain', False)
print('--- (b) attention initialised exactly uniform ---', flush=True)
run('uniform', True)
