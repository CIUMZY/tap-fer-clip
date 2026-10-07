"""Source-domain diagnostic: token pooling versus the global feature, at both encoders.

Question: why does TAP's gain over the equal-capacity global head vanish at ViT-L/14?
  (i)  the ViT-L global feature already carries the class information the token branch adds at
       ViT-B/16, or
  (ii) the TAP pooling itself loses information because it is a convex combination of patch
       tokens with no direct path from the global feature.

Representations, all trained and selected on the FER2013 source split only (probe-train /
probe-val, seed 0 split), same budget as every earlier arm (Adam, 8 epochs, batch 128, lr grid
{1e-3, 3e-3}):
  R1     frozen L2-normalised global feature (512-d ViT-B/16, 768-d ViT-L/14)
  R2     frozen uniform mean pooling of patch tokens (raw), plus mean-of-L2-normalised tokens
         and L2-normalised mean as robustness variants
  R3     the TAP head (attention pooling over tokens + linear head), architecture identical to
         phase 2 T1; R3refit freezes the learned pooling and refits a fresh linear probe
  R4a    frozen concat(R1, R2) + linear probe (does the token field add information at all?)
  R4b    concat(R1, attention-pooled tokens) trained jointly (does a direct global path repair
         the learned pooling?)

No target label is used anywhere.  The two confirmation sets (Set A crossed grid, Set B
KDEF-source) are NOT touched; this is a source-domain diagnostic.

Usage:  python diagnostic_token_vs_global_run.py --encoder B16
        python diagnostic_token_vs_global_run.py --encoder L14
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
DEV = 'cuda'
EPOCHS, BATCH = 8, 128
LRS = (1e-3, 3e-3)
LAM, MINN, CLIPV = 0.2, 32, 1.0

ENC = {
    'B16': {'name': 'ViT-B/16', 'glob': D / 'features' / 'fer2013_train.npz',
            'glob_dim': 512, 'tok_dim': 768, 'n_tok': 196,
            # the shard-granular reader amplifies badly for random 128-row batches (a batch of 128
            # scattered rows touches ~44 of the 57 shards), so ViT-B/16 is assembled once into the
            # same memory-mapped .npy layout the ViT-L/14 cache already uses.  See build_b16_cache.
            'npy': D / 'token_cache_vitb16' / 'fer2013_train_tokens_fp16.npy',
            'shards': D / 'token_cache_vitb16' / 'fer2013_train',
            'npz': D / 'token_cache_vitb16' / 'fer2013_train_tokens_fp16.npz',
            'meta': D / 'token_cache_vitb16' / 'fer2013_train_tokens_fp16.npz',
            'shard_rows': 512, 'chunk': 512},
    'L14': {'name': 'ViT-L/14', 'glob': D / 'features_vitl14' / 'fer2013_train.npz',
            'glob_dim': 768, 'tok_dim': 1024, 'n_tok': 256,
            'npy': D / 'token_cache_vitl14' / 'fer2013_train_tokens_fp16.npy',
            'shards': None, 'npz': None,
            'meta': D / 'token_cache_vitl14' / 'fer2013_train_meta.npz',
            'shard_rows': None, 'chunk': 256},
}


def uar(y, p):
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in range(7)]))


def recalls(y, p):
    return [float(np.mean(p[y == c] == c)) if np.any(y == c) else float('nan') for c in range(7)]


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prior_correct(logits, order, lam=LAM, minn=MINN, clipv=CLIPV):
    out = np.empty_like(logits)
    counts = np.zeros(7, dtype=np.int64)
    for i in order:
        lg = logits[i].astype(np.float64).copy()
        if counts.sum() >= minn:
            pi = (counts + 1.0) / (counts.sum() + 7.0)
            lg = lg - min(lam, clipv) * np.log(np.maximum(pi, 1e-12))
        out[i] = lg
        counts[int(np.argmax(lg))] += 1
    return out


# ----------------------------------------------------------------- token reading (chunked) --
class Tokens:
    """Row-indexed reader over the existing on-disk caches.  Never materialises the whole array:
    ViT-L/14 goes through the memory-mapped .npy, ViT-B/16 through the 512-row shard files with a
    small LRU shard cache.  Rows are requested in ascending order, which turns each block into
    near-sequential disk access."""

    def __init__(self, enc, cache_shards=6):
        self.enc = enc
        self.mm = np.load(enc['npy'], mmap_mode='r') if enc['npy'] is not None else None
        self.cache_shards = cache_shards
        self._cache = {}
        self._order = []
        if self.mm is not None:
            assert tuple(self.mm.shape[1:]) == (enc['n_tok'], enc['tok_dim']), self.mm.shape

    def _shard(self, sid):
        if sid in self._cache:
            return self._cache[sid]
        a = np.load(self.enc['shards'] / ('shard_%04d.npz' % sid), allow_pickle=False)['tokens']
        self._cache[sid] = a
        self._order.append(sid)
        while len(self._order) > self.cache_shards:
            self._cache.pop(self._order.pop(0), None)
        return a

    def rows(self, rows):
        rows = np.asarray(rows, np.int64)
        if self.mm is not None:
            return np.asarray(self.mm[rows])
        sr = self.enc['shard_rows']
        out = np.empty((len(rows), self.enc['n_tok'], self.enc['tok_dim']), np.float16)
        for sid in np.unique(rows // sr):
            m = (rows // sr) == sid
            a = self._shard(int(sid))
            out[m] = a[rows[m] - sid * sr]
        return out


def pooled_tokens(tok, rows, chunk):
    """Uniform mean pooling and two robustness variants, computed chunk by chunk."""
    n, dt = len(rows), tok.enc['tok_dim']
    mean_raw = np.empty((n, dt), np.float32)
    mean_normed = np.empty((n, dt), np.float32)
    for i in range(0, n, chunk):
        block = rows[i:i + chunk]
        x = tok.rows(block).astype(np.float32)
        mean_raw[i:i + len(block)] = x.mean(axis=1)
        xn = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
        mean_normed[i:i + len(block)] = xn.mean(axis=1)
        del x, xn
    return mean_raw, norm(mean_raw), mean_normed


def build_b16_cache(enc):
    """Assemble the ViT-B/16 train tokens into one memory-mapped .npy, reading the existing shards
    strictly sequentially (one 154 MB shard in flight at a time).  Skipped when the file already
    exists with the expected shape.  Nothing is overwritten: the file is created only if absent."""
    p = enc['npy']
    shard_files = sorted(enc['shards'].glob('shard_*.npz'))
    n_rows = sum(np.load(sp, allow_pickle=False)['tokens'].shape[0] for sp in shard_files)
    want = (n_rows, enc['n_tok'], enc['tok_dim'])
    if p.exists():
        a = np.load(p, mmap_mode='r')
        assert tuple(a.shape) == want, (a.shape, want)
        assert a.dtype == np.float16, a.dtype
        print('reusing existing %s %s' % (p.name, a.shape), flush=True)
        return n_rows, False
    print('assembling %s %s fp16 from %d shards (sequential read, one shard in flight)'
          % (p.name, want, len(shard_files)), flush=True)
    out = np.lib.format.open_memmap(p, mode='w+', dtype=np.float16, shape=want)
    off = 0
    for sp in shard_files:
        a = np.load(sp, allow_pickle=False)['tokens']
        out[off:off + a.shape[0]] = a
        off += a.shape[0]
        del a
    assert off == n_rows, (off, n_rows)
    out.flush()
    del out
    mm = np.load(p, mmap_mode='r')
    rng = np.random.default_rng(0)
    for row in rng.integers(0, n_rows, 12):
        sid, local = int(row) // enc['shard_rows'], int(row) % enc['shard_rows']
        ref = np.load(enc['shards'] / ('shard_%04d.npz' % sid), allow_pickle=False)['tokens'][local]
        assert np.array_equal(np.asarray(mm[int(row)]), ref), 'cache mismatch at row %d' % row
    print('  assembled %d rows; 12 random rows verified against their source shards' % n_rows,
          flush=True)
    return n_rows, True


# ----------------------------------------------------------------------------- models -------
class TAP(nn.Module):
    """Identical to phase 2 T1 / phase 5 TAP."""

    def __init__(self, dim, hid=256, C=7):
        super().__init__()
        self.a1 = nn.Linear(dim, hid)
        self.a2 = nn.Linear(hid, 1)
        self.head = nn.Linear(dim, C)
        self.attn_params = self.a1.weight.numel() + self.a1.bias.numel() + \
            self.a2.weight.numel() + self.a2.bias.numel()

    def pooled(self, tok):
        w = torch.softmax(self.a2(torch.tanh(self.a1(tok))).squeeze(-1), dim=1)
        return (tok * w.unsqueeze(-1)).sum(1)

    def forward(self, tok):
        return self.head(self.pooled(tok))


class TAPResidual(nn.Module):
    """Attention pooling plus a direct path from the global feature to the classifier."""

    def __init__(self, dim, dglob, hid=256, C=7):
        super().__init__()
        self.a1 = nn.Linear(dim, hid)
        self.a2 = nn.Linear(hid, 1)
        self.head = nn.Linear(dim + dglob, C)

    def pooled(self, tok):
        w = torch.softmax(self.a2(torch.tanh(self.a1(tok))).squeeze(-1), dim=1)
        return (tok * w.unsqueeze(-1)).sum(1)

    def forward(self, tok, g):
        return self.head(torch.cat([self.pooled(tok), g], dim=-1))


class Lin(nn.Module):
    def __init__(self, d, C=7):
        super().__init__()
        self.fc = nn.Linear(d, C)

    def forward(self, x):
        return self.fc(x)


# ----------------------------------------------------------------------------- training -----
def eval_linear(model, X, y, prior=True):
    preds = []
    with torch.inference_mode():
        for i in range(0, len(X), 1024):
            preds.append(model(torch.from_numpy(X[i:i + 1024]).to(DEV)).cpu().numpy())
    logits = np.concatenate(preds)
    pred = logits.argmax(1)
    out = {'uar': uar(y, pred), 'recall': recalls(y, pred)}
    if prior:
        pc = prior_correct(logits, np.arange(len(y))).argmax(1)
        out['uar_prior'] = uar(y, pc)
        out['recall_prior'] = recalls(y, pc)
    return out


def train_frozen_probe(Xtr, ytr, Xva, yva, lr, seed):
    """Linear probe on a frozen representation: same budget as every other arm."""
    torch.manual_seed(seed)
    model = Lin(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr)
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(Xt[idx].to(DEV)), yt[idx].to(DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    rec = eval_linear(model, Xva, yva)
    rec.update({'lr': lr, 'seed': seed, 'params': int(sum(p.numel() for p in model.parameters())),
                'seconds': round(time.perf_counter() - t0, 1)})
    return model, rec


def train_stream(factory, y_all, g_all, tok, tr, va, lr, seed, kind):
    """Train a token-streaming model (R3 attention pooling, or R4b attention + global residual).

    The model is built *inside* the seeded scope so that its initial weights, and not just the
    batch order, are a function of the seed.  Building it outside (as an earlier version of this
    script did) leaves the init to whatever RNG state is current, which is how one R3 run
    collapsed to chance level."""
    torch.manual_seed(seed)
    model = factory().to(DEV)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    steps = 0
    for _ in range(EPOCHS):
        perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(perm), BATCH):
            sel = np.sort(np.asarray(tr)[perm[i:i + BATCH].numpy()])
            xb = torch.from_numpy(tok.rows(sel)).to(DEV).float()
            yb = torch.from_numpy(y_all[sel]).to(DEV)
            if kind == 'R3':
                loss = F.cross_entropy(model(xb), yb)
            else:
                gb = torch.from_numpy(g_all[sel]).to(DEV)
                loss = F.cross_entropy(model(xb, gb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
    model.eval()
    rec = eval_stream(model, y_all, g_all, tok, va, kind)
    rec.update({'lr': lr, 'seed': seed, 'steps': steps,
                'params': int(sum(p.numel() for p in model.parameters())),
                'seconds': round(time.perf_counter() - t0, 1),
                'peak_vram_GB': round(torch.cuda.max_memory_allocated() / 1e9, 2)})
    return model, rec


def eval_stream(model, y_all, g_all, tok, va, kind, chunk=256):
    preds, logits = [], []
    with torch.inference_mode():
        for i in range(0, len(va), chunk):
            block = np.asarray(va)[i:i + chunk]
            order = np.argsort(block)
            sel = block[order]
            xb = torch.from_numpy(tok.rows(sel)).to(DEV).float()
            if kind == 'R3':
                o = model(xb)
            else:
                o = model(xb, torch.from_numpy(g_all[sel]).to(DEV))
            o = o.cpu().numpy()
            # rows were read in ascending order for sequential I/O; restore the requested order so
            # that predictions stay aligned with y_all[va]
            back = np.empty_like(o)
            back[order] = o
            logits.append(back)
            preds.append(back.argmax(1))
    logits = np.concatenate(logits)
    pred = np.concatenate(preds)
    y = y_all[np.asarray(va)]
    out = {'uar': uar(y, pred), 'recall': recalls(y, pred)}
    pc = prior_correct(logits, np.arange(len(y))).argmax(1)
    out['uar_prior'] = uar(y, pc)
    out['recall_prior'] = recalls(y, pc)
    return out


def pooled_from_tap(model, tok, rows, chunk=256):
    rows = np.asarray(rows)
    out = None
    with torch.inference_mode():
        for i in range(0, len(rows), chunk):
            block = rows[i:i + chunk]
            order = np.argsort(block)
            sel = block[order]
            xb = torch.from_numpy(tok.rows(sel)).to(DEV).float()
            p = model.pooled(xb).cpu().numpy().astype(np.float32)
            if out is None:
                out = np.empty((len(rows), p.shape[1]), np.float32)
            out[i + order] = p          # scatter back into the requested row order
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True, choices=['B16', 'L14'])
    args = ap.parse_args()
    enc = ENC[args.encoder]
    tag = args.encoder
    t_all = time.perf_counter()
    print('=== %s (%s) ===' % (tag, enc['name']), flush=True)

    z = np.load(enc['glob'], allow_pickle=False)
    if enc['npy'] is not None:
        meta = np.load(enc['meta'], allow_pickle=False)
        ids, labs = meta['ids'].astype(str), meta['labels'].astype(np.int64)
    else:
        nz = np.load(enc['npz'], allow_pickle=False)
        ids, labs = nz['ids'].astype(str), nz['labels'].astype(np.int64)
        del nz
    assert np.array_equal(z['ids'].astype(str), ids), 'global/token id mismatch'
    assert np.array_equal(z['labels'].astype(np.int64), labs), 'global/token label mismatch'
    split = np.load(enc['glob'].parent / 'linear_probe_split_seed0.npz', allow_pickle=False)
    tr = np.asarray(split['train_indices'], np.int64)
    va = np.asarray(split['validation_indices'], np.int64)
    # both encoders must use the identical source split for the two encoder columns to be comparable
    split_check = {'n_train': int(len(tr)), 'n_val': int(len(va)), 'identical_across_encoders': None}
    other = ENC['L14'] if args.encoder == 'B16' else ENC['B16']
    op = other['glob'].parent / 'linear_probe_split_seed0.npz'
    if op.exists():
        oz = np.load(op, allow_pickle=False)
        same = (np.array_equal(tr, np.asarray(oz['train_indices'], np.int64))
                and np.array_equal(va, np.asarray(oz['validation_indices'], np.int64)))
        split_check['identical_across_encoders'] = bool(same)
        assert same, 'the source split differs between the two encoders'
    print('split train=%d val=%d | identical_across_encoders=%s'
          % (len(tr), len(va), split_check['identical_across_encoders']), flush=True)
    g_all = norm(z['views'].mean(axis=1)).astype(np.float32)
    y_all = labs
    cache_info = None
    if tag == 'B16':
        n_rows, built_now = build_b16_cache(enc)
        assert n_rows == len(labs), (n_rows, len(labs))
        cache_info = {'path': str(enc['npy']), 'rows': int(n_rows), 'built_now': bool(built_now),
                      'verified': '12 random rows equal to their source shard rows'}
        print('token cache: %s' % json.dumps(cache_info), flush=True)
    tok = Tokens(enc)

    rep = {'encoder': tag, 'model': enc['name'], 'glob_dim': enc['glob_dim'],
           'tok_dim': enc['tok_dim'], 'n_tok': enc['n_tok'],
           'grid': {'lr': list(LRS), 'epochs': EPOCHS, 'batch': BATCH},
           'split': split_check,
           'token_cache': cache_info,
           'n_train': int(len(tr)), 'n_val': int(len(va)),
           'prior_rule': {'lambda': LAM, 'min_prior_samples': MINN, 'prior_clip': CLIPV},
           'reps': {}, 'selected': {}, 'runs': []}
    probes, recall_rows = [], []

    # ---------------- frozen representations: R1, R2 family, R4a family ----------------
    t0 = time.perf_counter()
    mean_raw, mean_norm, mean_normtok = pooled_tokens(tok, np.arange(len(labs)), enc['chunk'])
    rep['pooling_seconds'] = round(time.perf_counter() - t0, 1)
    print('mean-pooled tokens in %.1fs: raw %s normed %s normtok %s'
          % (rep['pooling_seconds'], mean_raw.shape, mean_norm.shape, mean_normtok.shape), flush=True)
    np.savez_compressed(OUT / ('diagnostic_pooled_tokens_%s_valonly.npz' % tag),
                        val_raw=mean_raw[va], val_norm=mean_norm[va], val_normtok=mean_normtok[va])

    FROZEN = {
        'R1_global': g_all,
        'R2_mean_raw': mean_raw,
        'R2_mean_normed': mean_norm,
        'R2_mean_of_normed': mean_normtok,
        'R4a_concat_raw': np.concatenate([g_all, mean_raw], axis=1),
        'R4a_concat_normed': np.concatenate([g_all, mean_norm], axis=1),
        'R4a_concat_normtok': np.concatenate([g_all, mean_normtok], axis=1),
    }
    for name, X in FROZEN.items():
        Xtr, Xva = X[tr], X[va]
        runs = []
        for lr in LRS:
            for seed in (0, 1, 2):
                _, rec = train_frozen_probe(Xtr, y_all[tr], Xva, y_all[va], lr, seed)
                rec.update({'rep': name, 'encoder': tag, 'kind': 'frozen',
                            'dim': int(X.shape[1])})
                runs.append(rec)
                probes.append({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in rec.items()})
                print('  %-20s lr%-6s seed%d  uar %.4f  prior %.4f  (%.1fs)'
                      % (name, lr, seed, rec['uar'], rec['uar_prior'], rec['seconds']), flush=True)
        rep['reps'][name] = {'dim': int(X.shape[1]), 'kind': 'frozen', 'runs': runs,
                             'selected': select(runs)}
        s = rep['reps'][name]['selected']
        recall_rows.append({'encoder': tag, 'rep': name, 'kind': 'frozen', 'selected_lr': s['lr'],
                            'selected_seed': s['seed'], 'uar': s['uar'], 'uar_prior': s['uar_prior'],
                            **{'recall_c%d' % c: s['recall'][c] for c in range(7)},
                            **{'recall_prior_c%d' % c: s['recall_prior'][c] for c in range(7)}})
        del Xtr, Xva
        print('  -> %s selected %s' % (name, json.dumps({k: s[k] for k in ('lr', 'seed', 'uar', 'uar_prior')})),
              flush=True)

    # ---------------- R3: attention pooling (TAP head), full lr grid at seed 0 ----------------
    for kind, name in (('R3', 'R3_attn_pool'), ('R4b', 'R4b_attn_plus_global')):
        mk = ((lambda: TAP(enc['tok_dim'])) if kind == 'R3'
              else (lambda: TAPResidual(enc['tok_dim'], enc['glob_dim'])))
        runs = []
        for lr in LRS:
            m, rec = train_stream(mk, y_all, g_all, tok, tr, va, lr, 0, kind)
            rec.update({'rep': name, 'encoder': tag, 'kind': kind})
            runs.append(rec)
            probes.append({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in rec.items()})
            print('  %-20s lr%-6s seed0  uar %.4f  prior %.4f  (%.1fs)'
                  % (name, lr, rec['uar'], rec['uar_prior'], rec['seconds']), flush=True)
            torch.save(m.state_dict(), OUT / ('diagnostic_%s_%s_lr%s_seed0.pt' % (tag, name, lr)))
        sel0 = select(runs)
        for seed in (1, 2):
            m, rec = train_stream(mk, y_all, g_all, tok, tr, va, sel0['lr'], seed, kind)
            rec.update({'rep': name, 'encoder': tag, 'kind': kind})
            runs.append(rec)
            probes.append({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in rec.items()})
            print('  %-20s lr%-6s seed%d  uar %.4f  prior %.4f  (%.1fs)'
                  % (name, sel0['lr'], seed, rec['uar'], rec['uar_prior'], rec['seconds']), flush=True)
        rep['reps'][name] = {'kind': kind, 'runs': runs, 'selected': select(runs)}
        s = rep['reps'][name]['selected']
        recall_rows.append({'encoder': tag, 'rep': name, 'kind': kind, 'selected_lr': s['lr'],
                            'selected_seed': s['seed'], 'uar': s['uar'], 'uar_prior': s['uar_prior'],
                            **{'recall_c%d' % c: s['recall'][c] for c in range(7)},
                            **{'recall_prior_c%d' % c: s['recall_prior'][c] for c in range(7)}})
        print('  -> %s selected %s' % (name, json.dumps({k: s[k] for k in ('lr', 'seed', 'uar', 'uar_prior')})),
              flush=True)
        if kind == 'R3':
            m = TAP(enc['tok_dim']).to(DEV)
            m.load_state_dict(torch.load(OUT / ('diagnostic_%s_%s_lr%s_seed0.pt' % (tag, name, sel0['lr'])),
                                         map_location=DEV))
            m.eval()
            t0 = time.perf_counter()
            Ptr = pooled_from_tap(m, tok, tr)
            Pva = pooled_from_tap(m, tok, va)
            rep['R3_refit_pooling_seconds'] = round(time.perf_counter() - t0, 1)
            np.savez_compressed(OUT / ('diagnostic_pooled_tokens_%s_r3pool_valonly.npz' % tag),
                                val_ptr=Pva, val_labels=y_all[va])
            nruns = []
            for lr in LRS:
                for seed in (0, 1, 2):
                    _, rec = train_frozen_probe(Ptr, y_all[tr], Pva, y_all[va], lr, seed)
                    rec.update({'rep': 'R3refit_frozen_attn', 'encoder': tag, 'kind': 'frozen',
                                'dim': int(Ptr.shape[1])})
                    nruns.append(rec)
                    probes.append({k: (json.dumps(v) if isinstance(v, list) else v)
                                   for k, v in rec.items()})
                    print('  %-20s lr%-6s seed%d  uar %.4f  prior %.4f  (%.1fs)'
                          % ('R3refit_frozen_attn', lr, seed, rec['uar'], rec['uar_prior'],
                             rec['seconds']), flush=True)
            rep['reps']['R3refit_frozen_attn'] = {'dim': int(Ptr.shape[1]), 'kind': 'frozen',
                                                 'runs': nruns, 'selected': select(nruns),
                                                 'pooling_source': 'R3_attn_pool lr%s seed0'
                                                                   % sel0['lr']}
            s = rep['reps']['R3refit_frozen_attn']['selected']
            recall_rows.append({'encoder': tag, 'rep': 'R3refit_frozen_attn', 'kind': 'refit',
                                'selected_lr': s['lr'], 'selected_seed': s['seed'], 'uar': s['uar'],
                                'uar_prior': s['uar_prior'],
                                **{'recall_c%d' % c: s['recall'][c] for c in range(7)},
                                **{'recall_prior_c%d' % c: s['recall_prior'][c] for c in range(7)}})
            del Ptr, Pva

    rep['reference_phase5_probeval'] = {'TAP': 0.6350518869602058 if tag == 'B16' else 0.6754398614199143,
                                        'global_head': 0.6144336662833177 if tag == 'B16' else 0.6720116028706133,
                                        'clip_adapter': 0.617177432363264 if tag == 'B16' else 0.679862246940411,
                                        'linear_probe_phase4': 0.5691 if tag == 'B16' else None,
                                        'pfb_prior_corrected': 0.6155 if tag == 'B16' else None}
    rep['elapsed_seconds'] = round(time.perf_counter() - t_all, 1)
    (OUT / ('diagnostic_token_vs_global_%s.json' % tag)).write_text(json.dumps(rep, indent=1),
                                                                    encoding='utf-8')
    with (OUT / ('diagnostic_token_vs_global_probes_%s.csv' % tag)).open('w', newline='',
                                                                        encoding='utf-8') as fh:
        cols = ['encoder', 'rep', 'kind', 'dim', 'lr', 'seed', 'steps', 'params', 'seconds',
                'peak_vram_GB', 'uar', 'uar_prior']
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(probes)
    with (OUT / ('diagnostic_token_vs_global_recall_%s.csv' % tag)).open('w', newline='',
                                                                        encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(recall_rows[0].keys()))
        w.writeheader()
        w.writerows(recall_rows)
    print('=== %s done in %.1fs ===' % (tag, rep['elapsed_seconds']), flush=True)
    for name, d in rep['reps'].items():
        print('  %-22s uar %.4f (prior %.4f)  lr %s seed %d'
              % (name, d['selected']['uar'], d['selected']['uar_prior'], d['selected']['lr'],
                 d['selected']['seed']), flush=True)


def select(runs):
    """Selection by probe-val UAR only, then by uar_prior as a tie-break; never by target data."""
    best = max(runs, key=lambda r: (r['uar'], r['uar_prior'], -r['lr'], -r['seed']))
    return dict(best)


if __name__ == '__main__':
    main()
