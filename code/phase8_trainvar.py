"""Phase 8: training-variance re-run.

Retrains the three source-domain arms (TAP, CLIP-Adapter, capacity-matched global head) with
fresh random seeds at the already-selected learning rates, and records the paired source-domain
probe-validation difference for the TAP-versus-comparator contrast.

Discipline unchanged: training and learning-rate selection stay on FER2013 probe-train /
probe-val; the learning rates are the ones selected in the original runs and are NOT re-selected
here. No target label is used for training or selection.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
DEV = 'cuda'
S, EPOCHS, BATCH = 100.0, 8, 128

CFG = {
    'B16': dict(dim=768, gd=512, hid=256, gh=256, ca_hid=128, lr_tap=3e-3, lr_ca=1e-3, lr_gh=3e-3,
                tokdir=D / 'token_cache_vitb16', feat=D / 'features',
                seeds=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    'L14': dict(dim=1024, gd=768, hid=256, gh=260, ca_hid=192, lr_tap=1e-3, lr_ca=1e-3, lr_gh=1e-3,
                tokdir=D / 'token_cache_vitl14', feat=D / 'features_vitl14',
                seeds=[5, 6, 7, 8, 9]),
}


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class T1(nn.Module):
    def __init__(self, dim, hid, C=7):
        super().__init__()
        self.a1 = nn.Linear(dim, hid)
        self.a2 = nn.Linear(hid, 1)
        self.head = nn.Linear(dim, C)

    def forward(self, tok):
        w = torch.softmax(self.a2(torch.tanh(self.a1(tok))).squeeze(-1), dim=1)
        return self.head((tok * w.unsqueeze(-1)).sum(1))


class GHead(nn.Module):
    def __init__(self, d, h, C=7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, C))

    def forward(self, x):
        return self.net(x)


class CLIPAdapter(nn.Module):
    def __init__(self, text, d, hid, alpha=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d, hid)
        self.fc2 = nn.Linear(hid, d)
        self.alpha = alpha
        self.register_buffer('text', torch.from_numpy(text.astype(np.float32)))

    def forward(self, x):
        f = self.alpha * self.fc2(F.relu(self.fc1(x))) + (1.0 - self.alpha) * x
        return S * (F.normalize(f, dim=-1) @ self.text.T)


def train_arm(model, batches, n, lr, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    model = model.to(DEV)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            sel = perm[i:i + BATCH].numpy()
            xb, yb = batches(sel)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def eval_preds(model, batches, va):
    preds = []
    with torch.inference_mode():
        for i in range(0, len(va), 256):
            sel = va[i:i + 256]
            xb, _ = batches(sel)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def run_encoder(tag):
    c = CFG[tag]
    t0 = time.perf_counter()
    Fe = c['feat']
    split = np.load(Fe / 'linear_probe_split_seed0.npz', allow_pickle=False)
    tr = np.asarray(split['train_indices'], dtype=np.int64)
    va = np.asarray(split['validation_indices'], dtype=np.int64)
    text = norm(np.load(Fe / 'text_prototypes.npz', allow_pickle=False)['prototypes'].astype(np.float32))
    gtr = norm(np.load(Fe / 'fer2013_train.npz', allow_pickle=False)['views'].mean(axis=1))
    G = torch.from_numpy(gtr.astype(np.float32))
    if tag == 'B16':
        z = np.load(c['tokdir'] / 'fer2013_train_tokens_fp16.npz', allow_pickle=False)
        TOK = z['tokens']
        LAB = z['labels'].astype(np.int64)

        def tok_batches(sel):
            return (torch.from_numpy(TOK[sel].astype(np.float32)).to(DEV),
                    torch.from_numpy(LAB[sel]).to(DEV))
        n_all = len(TOK)
    else:
        TOK = np.load(c['tokdir'] / 'fer2013_train_tokens_fp16.npy', mmap_mode='r')
        LAB = np.load(c['tokdir'] / 'fer2013_train_meta.npz')['labels'].astype(np.int64)

        def tok_batches(sel):
            return (torch.from_numpy(np.asarray(TOK[sel])).to(DEV).float(),
                    torch.from_numpy(LAB[sel]).to(DEV))
        n_all = len(LAB)

    def glob_batches(sel):
        return (G[sel].to(DEV), torch.from_numpy(LAB[sel]).to(DEV))

    # training batches must cover the probe-train subset only
    tr_sorted = np.sort(tr)

    def tok_train(sel):
        return tok_batches(tr_sorted[sel])

    def glob_train(sel):
        return glob_batches(tr_sorted[sel])

    yva = LAB[va]
    rows = []
    for seed in c['seeds']:
        rec = {'encoder': tag, 'seed': seed}
        m = train_arm(T1(c['dim'], c['hid']), tok_train, len(tr), c['lr_tap'], seed)
        rec['TAP_probeval'] = uar(yva, eval_preds(m, tok_batches, va))
        torch.save(m.state_dict(), OUT / ('phase8_tap_%s_seed%d.pt' % (tag, seed)))
        m = train_arm(CLIPAdapter(text, c['gd'], c['ca_hid']), glob_train, len(tr), c['lr_ca'], seed)
        rec['CLIPAdapter_probeval'] = uar(yva, eval_preds(m, glob_batches, va))
        torch.save(m.state_dict(), OUT / ('phase8_clipadapter_%s_seed%d.pt' % (tag, seed)))
        m = train_arm(GHead(c['gd'], c['gh']), glob_train, len(tr), c['lr_gh'], seed)
        rec['GHead_probeval'] = uar(yva, eval_preds(m, glob_batches, va))
        torch.save(m.state_dict(), OUT / ('phase8_ghead_%s_seed%d.pt' % (tag, seed)))
        rec['TAP_minus_CLIPAdapter'] = rec['TAP_probeval'] - rec['CLIPAdapter_probeval']
        rec['TAP_minus_GHead'] = rec['TAP_probeval'] - rec['GHead_probeval']
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    return rows, round(time.perf_counter() - t0, 1)


def main():
    out = {}
    for tag in ('B16', 'L14'):
        rows, secs = run_encoder(tag)
        d1 = np.array([r['TAP_minus_CLIPAdapter'] for r in rows])
        d2 = np.array([r['TAP_minus_GHead'] for r in rows])
        out[tag] = {
            'rows': rows, 'seconds': secs, 'n_seeds': len(rows),
            'TAP_minus_CLIPAdapter': {'mean': float(d1.mean()), 'sd': float(d1.std(ddof=1)), 'n': len(d1)},
            'TAP_minus_GHead': {'mean': float(d2.mean()), 'sd': float(d2.std(ddof=1)), 'n': len(d2)},
            'note': 'source probe-validation paired training variance; lr fixed at the originally selected value',
        }
        print(tag, json.dumps({k: v for k, v in out[tag].items() if k != 'rows'}), flush=True)
    (OUT / 'phase8_trainvar.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('done', flush=True)


if __name__ == '__main__':
    main()

