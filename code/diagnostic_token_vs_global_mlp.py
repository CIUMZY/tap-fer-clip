"""Extension of the token-vs-global diagnostic: the same frozen representations, but probed with
a nonlinear head instead of a linear one.

Why this is needed: in the main diagnostic every representation is read out by a linear probe, but
the comparators TAP lost to at ViT-L/14 in phase 5 (the equal-capacity global head, CLIP-Adapter)
are nonlinear.  Without a nonlinear-head row, a null result for R4 cannot be attributed to the
representation rather than to the head class.

Head: Linear(d, 256) -> Tanh -> Linear(256, 256) -> Tanh -> Linear(256, 7), which is exactly the
phase 5 global-head form and is capacity-matched to TAP (269,832 params at ViT-L/14, 198,919 at
ViT-B/16 for the 512-d global input).  Same budget and grid as everything else: Adam, 8 epochs,
batch 128, lr {1e-3, 3e-3}, selection on source probe-val only, seeds 0/1/2.

Representations probed: R1 global (frozen), R2 mean-pooled raw tokens (frozen),
R4a concat(global, mean tokens) (frozen), and the attention-pooled representation of the selected
R3 checkpoint (frozen pooling).  Source domain only; no confirmation set is touched.
"""
import argparse, csv, importlib.util, json, sys, time
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
DEV = dg.DEV


class MLP(nn.Module):
    def __init__(self, d, h=256, C=7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(),
                                 nn.Linear(h, C))

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, Xva, yva, lr, seed):
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for _ in range(dg.EPOCHS):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm), dg.BATCH):
            idx = perm[i:i + dg.BATCH]
            loss = F.cross_entropy(model(Xt[idx].to(DEV)), yt[idx].to(DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    rec = dg.eval_linear(model, Xva, yva)
    rec.update({'lr': lr, 'seed': seed, 'seconds': round(time.perf_counter() - t0, 1),
                'params': int(sum(p.numel() for p in model.parameters()))})
    return model, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True, choices=['B16', 'L14'])
    args = ap.parse_args()
    tag = args.encoder
    enc = dg.ENC[tag]
    t_all = time.perf_counter()
    print('=== MLP extension %s (%s) ===' % (tag, enc['name']), flush=True)

    z = np.load(enc['glob'], allow_pickle=False)
    if enc['npy'] is not None and tag == 'B16':
        dg.build_b16_cache(enc)
        meta = np.load(enc['meta'], allow_pickle=False)
        labs = meta['labels'].astype(np.int64)
    elif tag == 'L14':
        meta = np.load(enc['meta'], allow_pickle=False)
        labs = meta['labels'].astype(np.int64)
    assert np.array_equal(z['labels'].astype(np.int64), labs)
    split = np.load(enc['glob'].parent / 'linear_probe_split_seed0.npz', allow_pickle=False)
    tr = np.asarray(split['train_indices'], np.int64)
    va = np.asarray(split['validation_indices'], np.int64)
    y_all = labs
    g_all = dg.norm(z['views'].mean(axis=1)).astype(np.float32)
    tok = dg.Tokens(enc)

    t0 = time.perf_counter()
    mean_raw, _, _ = dg.pooled_tokens(tok, np.arange(len(labs)), enc['chunk'])
    print('mean-pooled in %.1fs' % (time.perf_counter() - t0), flush=True)
    reps = {'R1_global': g_all,
            'R2_mean_raw': mean_raw,
            'R4a_concat_raw': np.concatenate([g_all, mean_raw], axis=1)}

    # attention-pooled representation from the selected R3 checkpoint (seed 0, selected lr)
    run = json.loads((OUT / ('diagnostic_token_vs_global_%s.json' % tag)).read_text(encoding='utf-8'))
    sel = run['reps']['R3_attn_pool']['selected']
    ckpt = OUT / ('diagnostic_%s_R3_attn_pool_lr%s_seed0.pt' % (tag, sel['lr']))
    m = dg.TAP(enc['tok_dim']).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV))
    m.eval()
    t0 = time.perf_counter()
    Ptr = dg.pooled_from_tap(m, tok, tr)
    Pva = dg.pooled_from_tap(m, tok, va)
    print('R3 pooling for train+val in %.1fs (checkpoint lr%s)'
          % (time.perf_counter() - t0, sel['lr']), flush=True)
    reps['R3_attn_pool'] = None    # handled per-split below

    out = {'encoder': tag, 'model': enc['name'], 'head': 'MLP 256-256 tanh (phase 5 global-head form)',
           'budget': {'epochs': dg.EPOCHS, 'batch': dg.BATCH, 'lr': list(dg.LRS), 'seeds': [0, 1, 2]},
           'reps': {}, 'runs': []}
    probes = []
    for name in ('R1_global', 'R2_mean_raw', 'R4a_concat_raw', 'R3_attn_pool'):
        if name == 'R3_attn_pool':
            # pooled_from_tap returns rows positionally aligned with the requested row list, so
            # Ptr already corresponds to y_all[tr] and Pva to y_all[va] (indexing them again with
            # the global row ids would be out of bounds and, worse, silently wrong).
            Xtr, Xva = Ptr, Pva
        else:
            Xtr, Xva = reps[name][tr], reps[name][va]
        runs = []
        for lr in dg.LRS:
            for seed in (0, 1, 2):
                _, rec = train_mlp(Xtr, y_all[tr], Xva, y_all[va], lr, seed)
                rec.update({'rep': name, 'encoder': tag, 'kind': 'mlp'})
                runs.append(rec)
                probes.append({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in rec.items()})
                print('  %-16s mlp lr%-6s seed%d uar %.4f prior %.4f (%.1fs)'
                      % (name, lr, seed, rec['uar'], rec['uar_prior'], rec['seconds']), flush=True)
        out['reps'][name] = {'dim': int(Xtr.shape[1]), 'params': runs[0]['params'], 'runs': runs,
                             'selected': dg.select(runs)}
        s = out['reps'][name]['selected']
        print('  -> %s selected %s' % (name, json.dumps({k: s[k] for k in ('lr', 'seed', 'uar', 'uar_prior')})),
              flush=True)
        del Xtr, Xva
    out['elapsed_seconds'] = round(time.perf_counter() - t_all, 1)
    (OUT / ('diagnostic_token_vs_global_mlp_%s.json' % tag)).write_text(json.dumps(out, indent=1),
                                                                       encoding='utf-8')
    with (OUT / ('diagnostic_token_vs_global_mlp_probes_%s.csv' % tag)).open('w', newline='',
                                                                             encoding='utf-8') as fh:
        cols = ['encoder', 'rep', 'kind', 'lr', 'seed', 'params', 'seconds', 'uar', 'uar_prior']
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(probes)
    print('=== MLP extension %s done in %.1fs ===' % (tag, out['elapsed_seconds']), flush=True)


if __name__ == '__main__':
    main()
