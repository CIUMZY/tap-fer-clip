"""POST-HOC SENSITIVITY: extract MIRROR-view ViT-B/16 patch tokens (single mirrored view).

Storage convention copies the clean caches: tokens = model.visual(x)[1] for the mirrored image,
fp16, memory-mapped .npy + meta npz.  ids/labels are taken from the EXISTING clean cache, so
alignment is guaranteed by construction; the validation here is shape/rows/id-label alignment
(not numerical equality with the original view, which must differ).
"""
import hashlib, json, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
MIR = D / 'token_cache_vitb16_mirror'
WEIGHTS = Path('D:/ResearchVault/99system/models/open_clip/ViT-B-16.pt')
SRC = D / 'token_cache_vitb16'
JOBS = ['fer2013_train', 'fer2013_test', 'ckplus_test', 'kdef_test']
ROWS = 128


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    MIR.mkdir(parents=True, exist_ok=True)
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained=str(WEIGHTS), device='cuda', weights_only=False)
    model.eval()
    model.visual.output_tokens = True
    rep = {'scope': 'POST-HOC SENSITIVITY (not confirmatory)',
           'what': 'mirror-view patch tokens for the two-view-mean sensitivity check',
           'weights': str(WEIGHTS), 'weights_sha256': sha(WEIGHTS), 'rows_per_batch': ROWS,
           'datasets': {}, 'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'free_GB_on_D': round(shutil.disk_usage('D:/').free / 1e9, 1)}

    def forward(paths):
        imgs = [ImageOps.mirror(Image.open(p).convert('RGB')) for p in paths]
        x = torch.stack([preprocess(im) for im in imgs]).to('cuda')
        with torch.inference_mode():
            pooled, tokens = model.visual(x)
            pooled = F.normalize(pooled.float(), dim=-1)
        return tokens.half().cpu().numpy(), pooled.cpu().numpy().astype(np.float32)

    for name in JOBS:
        z_src = np.load(SRC / ('%s_tokens_fp16.npz' % name), allow_pickle=False)
        ids = z_src['ids'].astype(str)
        labels = z_src['labels'].astype(np.int64)
        del z_src
        npy = MIR / ('%s_tokens_fp16.npy' % name)
        meta = MIR / ('%s_meta.npz' % name)
        gpath = MIR / ('%s_mirror_globals.npz' % name)
        if npy.exists() and meta.exists() and gpath.exists():
            got = np.load(npy, mmap_mode='r')
            rep['datasets'][name] = {'reused': True, 'n': int(len(ids)), 'shape': list(got.shape),
                                     'sha256_npy': sha(npy)}
            print('%-15s reused %s' % (name, got.shape), flush=True)
            continue
        mm = np.lib.format.open_memmap(npy, mode='w+', dtype=np.float16, shape=(len(ids), 196, 768))
        g_all = np.empty((len(ids), 512), np.float32)
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for i in range(0, len(ids), ROWS):
            t, g = forward([str(p) for p in ids[i:i + ROWS]])
            mm[i:i + t.shape[0]] = t
            g_all[i:i + g.shape[0]] = g
        mm.flush()
        del mm
        np.savez_compressed(meta, ids=ids, labels=labels)
        np.savez_compressed(gpath, views=g_all)
        # alignment + sanity checks
        got = np.load(npy, mmap_mode='r')
        m2 = np.load(meta, allow_pickle=False)
        orig_g = np.load((D / 'features' / ('%s.npz' % name)), allow_pickle=False)['views'][:, 0]
        cos = np.sum(g_all * orig_g, axis=-1) / np.maximum(
            np.linalg.norm(g_all, axis=-1) * np.linalg.norm(orig_g, axis=-1), 1e-12)
        rep['datasets'][name] = {
            'n': int(len(ids)), 'shape': list(got.shape), 'dtype': str(got.dtype),
            'npy_GB': round(npy.stat().st_size / 1e9, 3), 'sha256_npy': sha(npy),
            'seconds': round(time.perf_counter() - t0, 1),
            'peak_vram_GB': round(torch.cuda.max_memory_allocated() / 1e9, 2),
            'ids_equal_source': bool(np.array_equal(m2['ids'].astype(str), ids)),
            'labels_equal_source': bool(np.array_equal(m2['labels'].astype(np.int64), labels)),
            'ids_unique': bool(len(set(ids.tolist())) == len(ids)),
            'tokens_finite': bool(np.isfinite(np.asarray(got[:8]).astype(np.float32)).all()),
            'mirror_global_vs_original_view0_cosine': {
                'mean': float(cos.mean()), 'min': float(cos.min()), 'max': float(cos.max()),
                'frac_below_0p999': float((cos < 0.999).mean())},
            'token_norm_mean': float(np.linalg.norm(np.asarray(got[:64]).astype(np.float32), axis=-1).mean()),
        }
        print('%-15s n=%d %s %.1fs peak=%.2fGB cos(mean)=%.4f below0.999=%.3f'
              % (name, len(ids), tuple(got.shape), rep['datasets'][name]['seconds'],
                 rep['datasets'][name]['peak_vram_GB'],
                 rep['datasets'][name]['mirror_global_vs_original_view0_cosine']['mean'],
                 rep['datasets'][name]['mirror_global_vs_original_view0_cosine']['frac_below_0p999']),
              flush=True)
        del got
    rep['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    (OUT / 'phase7_mirror_extraction.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print('wrote phase7_mirror_extraction.json')


if __name__ == '__main__':
    main()
