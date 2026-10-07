"""Phase 3 step 1: extract patch tokens for the KDEF-source support set (2056 images).

Follows the same conventions as phase2_extract_tokens.py: open_clip ViT-B/16 with
visual.output_tokens = True (tokens already CLS-stripped internally), fp16, 512-image
shards written first then merged. Writes ONLY new files; never overwrites existing caches.
"""
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
KS = D / 'features_kdef_source'
NAME = 'kdef_source_train'
SHARD = 512


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    src = np.load(KS / 'kdef_source_train.npz', allow_pickle=False)
    ids = src['ids'].astype(str)
    labels = src['labels'].astype(np.int16)
    assert len(ids) == 2056, len(ids)
    merged = CACHE / (NAME + '_tokens_fp16.npz')
    assert not merged.exists(), 'refusing to overwrite ' + str(merged)
    outdir = CACHE / NAME
    outdir.mkdir(parents=True, exist_ok=True)
    n_shards = (len(ids) + SHARD - 1) // SHARD
    for s in range(n_shards):
        assert not (outdir / ('shard_%04d.npz' % s)).exists(), 'shard already exists'

    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained=str(M), device='cuda', weights_only=False)
    model.eval()
    model.visual.output_tokens = True
    man = {'dataset': NAME, 'n': int(len(ids)), 'shards': n_shards, 'shard_size': SHARD,
           'dtype': 'float16', 'device': 'NVIDIA GeForce RTX 3080',
           'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'free_GB_on_D': round(shutil.disk_usage('D:/').free / 1e9, 1)}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    shard_hashes = []
    for s in range(n_shards):
        sp = outdir / ('shard_%04d.npz' % s)
        chunk = ids[s * SHARD:(s + 1) * SHARD]
        toks = []
        with torch.inference_mode():
            for i in range(0, len(chunk), 128):
                x = torch.stack([preprocess(Image.open(p).convert('RGB')) for p in chunk[i:i + 128]]).to('cuda')
                _, feats = model.visual(x)
                toks.append(feats.half().cpu().numpy())
        np.savez(sp, tokens=np.concatenate(toks), labels=labels[s * SHARD:(s + 1) * SHARD],
                 ids=np.asarray(chunk))
        shard_hashes.append(sha(sp))
        print('  shard %d/%d %.1fs' % (s + 1, n_shards, time.perf_counter() - t0), flush=True)
    toks, labs, cids = [], [], []
    for s in range(n_shards):
        z = np.load(outdir / ('shard_%04d.npz' % s), allow_pickle=False)
        toks.append(z['tokens'])
        labs.append(z['labels'])
        cids.append(z['ids'])
    np.savez(merged, tokens=np.concatenate(toks), labels=np.concatenate(labs), ids=np.concatenate(cids))
    got = np.load(merged, allow_pickle=False)
    man['tokens_shape'] = list(got['tokens'].shape)
    man['sha256_shards'] = shard_hashes
    man['sha256_merged'] = sha(merged)
    man['merged_GB'] = round(merged.stat().st_size / 1e9, 3)
    man['seconds'] = round(time.perf_counter() - t0, 1)
    man['peak_vram_GB'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    man['ids_equal_source'] = bool(np.array_equal(got['ids'].astype(str), ids))
    man['labels_equal_source'] = bool(np.array_equal(got['labels'].astype(np.int16), labels))
    man['token_norm_mean'] = float(np.linalg.norm(got['tokens'].astype(np.float32), axis=-1).mean())
    man['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    (OUT / 'phase3_kdef_source_extraction.json').write_text(json.dumps(man, indent=2), encoding='utf-8')
    print(json.dumps(man, indent=2)[:900], flush=True)


if __name__ == '__main__':
    main()
