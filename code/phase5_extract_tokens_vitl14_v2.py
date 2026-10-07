"""Phase 5 step 2 (v2): ViT-L/14 patch-token extraction with memory-mapped storage.

v1 wrote 512-image shards and then tried to concatenate them into one npz; the host has 16.2 GB
of RAM, so the 15 GB concatenation failed. v2 keeps the shards (per-shard sha256 provenance) and
writes the merged token array as an uncompressed .npy that is memory-mapped, plus a small
meta npz holding labels and ids. Existing shards are reused; nothing is overwritten.
"""
import hashlib, json, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
M = Path('D:/ResearchVault/99system/models/open_clip/ViT-L-14.pt')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
CACHE = D / 'token_cache_vitl14'
KS = D / 'features_kdef_source'
SHARD = 512
JOBS = [('fer2013_train', 'fer2013_train.csv', None), ('fer2013_test', 'fer2013_test.csv', None),
        ('ckplus_test', 'ckplus_test.csv', None), ('kdef_test', 'kdef_test.csv', None),
        ('kdef_source_train', None, KS / 'kdef_source_train.npz')]
BATCH = 96


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    man = {'model': 'ViT-L-14', 'pretrained': str(M), 'shard_size': SHARD, 'dtype': 'float16',
           'storage': 'shards + uncompressed .npy (memory-mapped) + meta npz',
           'storage_reason': 'host RAM is 16.2 GB; a 15 GB npz concatenation failed',
           'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'datasets': {},
           'free_GB_on_D': round(shutil.disk_usage('D:/').free / 1e9, 1)}
    model = preprocess = None
    for name, csv_name, npz_path in JOBS:
        if npz_path is not None:
            z = np.load(npz_path, allow_pickle=False)
            paths = [str(x) for x in z['ids']]
            labels = z['labels'].astype(np.int16)
        else:
            rows = (D / 'manifests' / csv_name).open(encoding='utf-8').read().splitlines()[1:]
            paths = [r.split(',')[0] for r in rows]
            labels = np.asarray([int(r.split(',')[1]) for r in rows], dtype=np.int16)
        outdir = CACHE / name
        outdir.mkdir(parents=True, exist_ok=True)
        n_shards = (len(paths) + SHARD - 1) // SHARD
        tok_npy = CACHE / ('%s_tokens_fp16.npy' % name)
        meta_npz = CACHE / ('%s_meta.npz' % name)
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        missing = [s for s in range(n_shards) if not (outdir / ('shard_%04d.npz' % s)).exists()]
        if missing:
            if model is None:
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-L-14', pretrained=str(M), device='cuda', weights_only=False)
                model.eval(); model.visual.output_tokens = True
                man['logit_scale_exp'] = float(model.logit_scale.exp().detach())
            for s in missing:
                sp = outdir / ('shard_%04d.npz' % s)
                chunk = paths[s * SHARD:(s + 1) * SHARD]
                toks = []
                with torch.inference_mode():
                    for i in range(0, len(chunk), BATCH):
                        x = torch.stack([preprocess(Image.open(p).convert('RGB'))
                                         for p in chunk[i:i + BATCH]]).to('cuda')
                        _, feats = model.visual(x)
                        toks.append(feats.half().cpu().numpy())
                np.savez(sp, tokens=np.concatenate(toks), labels=labels[s * SHARD:(s + 1) * SHARD],
                         ids=np.asarray(chunk))
                if s % 5 == 0 or s == n_shards - 1:
                    print('  %s shard %d/%d %.1fs' % (name, s + 1, n_shards,
                                                      time.perf_counter() - t0), flush=True)
        shard_hashes = [sha(outdir / ('shard_%04d.npz' % s)) for s in range(n_shards)]
        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if model else None
        if not tok_npy.exists():
            mm = np.lib.format.open_memmap(tok_npy, mode='w+', dtype=np.float16,
                                           shape=(len(paths), 256, 1024))
            for s in range(n_shards):
                z = np.load(outdir / ('shard_%04d.npz' % s), allow_pickle=False)
                mm[s * SHARD:s * SHARD + z['tokens'].shape[0]] = z['tokens']
            mm.flush(); del mm
            np.savez_compressed(meta_npz, labels=labels, ids=np.asarray(paths))
        got = np.load(tok_npy, mmap_mode='r')
        chk = np.load(outdir / 'shard_0000.npz', allow_pickle=False)
        man['datasets'][name] = {
            'n': len(paths), 'shards': n_shards, 'tokens_shape': list(got.shape),
            'dtype': str(got.dtype), 'npy_GB': round(tok_npy.stat().st_size / 1e9, 3),
            'sha256_shards': shard_hashes, 'sha256_npy': sha(tok_npy),
            'seconds': round(time.perf_counter() - t0, 1), 'peak_vram_GB': peak,
            'shard0_matches_npy': bool(np.array_equal(np.asarray(got[:chk['tokens'].shape[0]]),
                                                      chk['tokens'])),
            'ids_equal_source': bool(np.array_equal(np.load(meta_npz)['ids'].astype(str),
                                                    np.asarray(paths))),
            'labels_equal_source': bool(np.array_equal(np.load(meta_npz)['labels'].astype(np.int16),
                                                       labels)),
            'token_norm_mean': float(np.linalg.norm(np.asarray(got[:64]).astype(np.float32), axis=-1).mean())}
        print('DONE %s %s' % (name, json.dumps(man['datasets'][name])[:280]), flush=True)
    man['total_npy_GB'] = round(sum(v['npy_GB'] for v in man['datasets'].values()), 3)
    man['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    (OUT / 'phase5_extraction_vitl14.json').write_text(json.dumps(man, indent=2), encoding='utf-8')
    print('total npy GB', man['total_npy_GB'], flush=True)


if __name__ == '__main__':
    main()
