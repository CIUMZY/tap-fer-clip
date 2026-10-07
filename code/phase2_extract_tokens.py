"""Phase 2 step 1: full patch-token extraction (fp16, 512-image shards -> merged)."""
import hashlib, json, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
M = Path('D:/ResearchVault/99system/models/open_clip/ViT-B-16.pt')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
CACHE = D / 'token_cache_vitb16'
CACHE.mkdir(parents=True, exist_ok=True)
SHARD = 512


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained=str(M), device='cuda', weights_only=False)
model.eval()
model.visual.output_tokens = True
print('free GB on D:', round(shutil.disk_usage('D:/').free / 1e9, 1), flush=True)

JOBS = [('fer2013_train', 'fer2013_train.csv'), ('fer2013_test', 'fer2013_test.csv'),
        ('ckplus_test', 'ckplus_test.csv'), ('kdef_test', 'kdef_test.csv')]
man = {'shard_size': SHARD, 'dtype': 'float16', 'datasets': {},
       'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
       'vitl14_weights': str(sorted(Path('D:/ResearchVault/99system/models').rglob('*L-14*'))[:5])}
for name, csv_name in JOBS:
    rows = (D / 'manifests' / csv_name).open(encoding='utf-8').read().splitlines()[1:]
    paths = [r.split(',')[0] for r in rows]
    labels = np.asarray([int(r.split(',')[1]) for r in rows], dtype=np.int16)
    outdir = CACHE / name
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    shard_hashes = []
    n_shards = (len(paths) + SHARD - 1) // SHARD
    for s in range(n_shards):
        sp = outdir / ('shard_%04d.npz' % s)
        chunk = paths[s * SHARD:(s + 1) * SHARD]
        if not sp.exists():
            toks = []
            with torch.inference_mode():
                for i in range(0, len(chunk), 128):
                    x = torch.stack([preprocess(Image.open(p).convert('RGB'))
                                     for p in chunk[i:i + 128]]).to('cuda')
                    _, feats = model.visual(x)
                    toks.append(feats.half().cpu().numpy())
            np.savez(sp, tokens=np.concatenate(toks), labels=labels[s * SHARD:(s + 1) * SHARD],
                     ids=np.asarray(chunk))
        shard_hashes.append(sha(sp))
        if s % 10 == 0:
            print('  %s shard %d/%d %.1fs' % (name, s + 1, n_shards,
                                              time.perf_counter() - t0), flush=True)
    merged = CACHE / ('%s_tokens_fp16.npz' % name)
    if not merged.exists():
        toks, labs, ids = [], [], []
        for s in range(n_shards):
            z = np.load(outdir / ('shard_%04d.npz' % s), allow_pickle=False)
            toks.append(z['tokens'])
            labs.append(z['labels'])
            ids.append(z['ids'])
        np.savez(merged, tokens=np.concatenate(toks), labels=np.concatenate(labs),
                 ids=np.concatenate(ids))
    man['datasets'][name] = {
        'n': len(paths), 'shards': n_shards, 'tokens_shape': list(
            np.load(merged, allow_pickle=False)['tokens'].shape),
        'merged_MB': round(merged.stat().st_size / 1e6, 1),
        'sha256_shards': shard_hashes, 'sha256_merged': sha(merged),
        'seconds': round(time.perf_counter() - t0, 1),
        'peak_vram_GB': round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print('DONE %s %s' % (name, json.dumps(man['datasets'][name])[:260]), flush=True)
man['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
(OUT / 'phase2_extraction.json').write_text(json.dumps(man, indent=2), encoding='utf-8')
print('vitl14 weights seen:', man['vitl14_weights'])
