"""Phase 6b extraction: patch tokens + global features for the 12 pre-registered corruption conditions.

Convention is copied from the clean caches:
  tokens  = model.visual(x)[1][0::2]           (original view only, fp16)
  globals = normalize(model.visual(x)[0])      (two views: image, mirror; fp32, unit norm)
with x built by the open_clip preprocess, exactly as OpenCLIPEncoder.encode_images / the phase-5
token extraction do.  Memory-mapped .npy output, batches of rows, nothing concatenated in host RAM.

Writes token_cache_<enc>_confirm/ and features_confirm_shift/<enc>/ plus phase6b_extraction_<enc>.json.
"""
import argparse, csv, hashlib, json, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
MAN = D / 'manifests'
SWEEP = D / 'features' / 'shift_sweep'
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
CONDS = ['blur_0p8', 'blur_1p5', 'blur_2p5', 'jpeg_15', 'jpeg_30', 'jpeg_50',
         'lowlight_0p3', 'lowlight_0p5', 'lowlight_0p7', 'noise_10', 'noise_25', 'noise_40']
ENC = {
    'B16': {'model': 'ViT-B-16', 'weights': D / 'models' / 'open_clip' / 'ViT-B-16.pt' if False
            else Path('D:/ResearchVault/99system/models/open_clip/ViT-B-16.pt'),
            'd_glob': 512, 'n_tok': 196, 'd_tok': 768, 'rows': 128,
            'clean_feat': D / 'features' / 'fer2013_test.npz'},
    'L14': {'model': 'ViT-L-14', 'weights': Path('D:/ResearchVault/99system/models/open_clip/ViT-L-14.pt'),
            'd_glob': 768, 'n_tok': 256, 'd_tok': 1024, 'rows': 64,
            'clean_feat': D / 'features_vitl14' / 'fer2013_test.npz'},
}


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def read_manifest(path):
    with path.open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    paths = [r['path'] for r in rows]
    labels = np.asarray([int(r['label']) for r in rows], np.int64)
    return paths, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True, choices=['B16', 'L14'])
    ap.add_argument('--conditions', nargs='*', default=CONDS)
    args = ap.parse_args()
    enc = ENC[args.encoder]
    tag = args.encoder
    tokdir = D / ('token_cache_%s_confirm' % tag.lower())
    featdir = D / 'features_confirm_shift' / tag.lower()
    tokdir.mkdir(parents=True, exist_ok=True)
    featdir.mkdir(parents=True, exist_ok=True)
    clean = np.load(enc['clean_feat'], allow_pickle=False)
    clean_lab = clean['labels'].astype(np.int64)
    rep = {'encoder': tag, 'model': enc['model'], 'weights': str(enc['weights']),
           'weights_sha256': sha(enc['weights']), 'conditions': {},
           'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'free_GB_on_D': round(shutil.disk_usage('D:/').free / 1e9, 1)}

    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        enc['model'], pretrained=str(enc['weights']), device='cuda', weights_only=False)
    model.eval()
    model.visual.output_tokens = True
    print('%s loaded; logit_scale_exp=%.4f' % (enc['model'],
                                               float(model.logit_scale.exp())), flush=True)

    def forward(paths):
        """-> tokens fp16 (n, n_tok, d_tok), globals fp32 (n, 2, d_glob)."""
        imgs = [Image.open(p).convert('RGB') for p in paths]
        pairs = []
        for im in imgs:
            pairs.extend([im, ImageOps.mirror(im)])
        x = torch.stack([preprocess(im) for im in pairs]).to('cuda')
        with torch.inference_mode():
            pooled, tokens = model.visual(x)
            pooled = F.normalize(pooled.float(), dim=-1)
        g = pooled.cpu().numpy().astype(np.float32).reshape(len(paths), 2, -1)
        t = tokens[0::2].half().cpu().numpy()
        del x, pooled, tokens
        return t, g

    def run_condition(paths, labels, npy_path, meta_path, feat_path, rows):
        mm = np.lib.format.open_memmap(npy_path, mode='w+', dtype=np.float16,
                                       shape=(len(paths), enc['n_tok'], enc['d_tok']))
        g_all = np.empty((len(paths), 2, enc['d_glob']), np.float32)
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for i in range(0, len(paths), rows):
            t, g = forward(paths[i:i + rows])
            mm[i:i + t.shape[0]] = t
            g_all[i:i + g.shape[0]] = g
        mm.flush()
        del mm
        np.savez_compressed(meta_path, labels=labels, ids=np.asarray(paths))
        np.savez_compressed(feat_path, ids=np.asarray(paths), views=g_all, labels=labels,
                            meta_json=np.asarray(json.dumps(
                                {'model': enc['model'], 'pretrained': str(enc['weights']),
                                 'projection': None, 'manifest': str(MAN / ('fer2013_%s.csv' % npy_path.stem
                                                                            .replace('_tokens_fp16', ''))),
                                 'num_samples': len(paths), 'source': 'phase6b_corruption_confirmation',
                                 'convention': 'tokens single original view fp16; globals two views '
                                               '(image, mirror) L2-normalised fp32'})))
        got = np.load(npy_path, mmap_mode='r')
        return {'n': len(paths), 'tokens_shape': list(got.shape), 'npy_GB': round(
                    npy_path.stat().st_size / 1e9, 3),
                'globals_shape': list(g_all.shape),
                'unit_norm_views': bool(np.allclose(np.linalg.norm(g_all, axis=-1), 1.0, atol=1e-4)),
                'seconds': round(time.perf_counter() - t0, 1),
                'peak_vram_GB': round(torch.cuda.max_memory_allocated() / 1e9, 2),
                'npy_sha256': sha(npy_path)}

    total_t0 = time.perf_counter()
    for cond in args.conditions:
        npy = tokdir / ('%s_tokens_fp16.npy' % cond)
        meta = tokdir / ('%s_meta.npz' % cond)
        feat = featdir / ('%s.npz' % cond)
        if npy.exists() and meta.exists() and feat.exists():
            rep['conditions'][cond] = {
                'reused': True,
                'npy_sha256': sha(npy),
                'npy_GB': round(npy.stat().st_size / 1e9, 3),
                'peak_vram_GB': None,
                'n': int(len(np.load(meta, allow_pickle=False)['labels']))}
            print('%s reused' % cond, flush=True)
            continue
        paths, labels = read_manifest(MAN / ('fer2013_%s.csv' % cond))
        assert len(paths) == 7178, (cond, len(paths))
        assert np.array_equal(labels, clean_lab), 'labels differ from the clean test cache: ' + cond
        missing = [p for p in paths[:50] if not Path(p).exists()]
        assert not missing, missing
        info = run_condition(paths, labels, npy, meta, feat, enc['rows'])
        info['condition'] = cond
        rep['conditions'][cond] = info
        print('%-13s n=%d %.1fs peak=%.2fGB npy=%.1fs norm_ok=%s'
              % (cond, info['n'], info['seconds'], info['peak_vram_GB'], info['npy_GB'],
                 info['unit_norm_views']), flush=True)

    # ---------------- validation gate ----------------
    gate = {}
    # chunk this: one forward of 1024 images needs about 4 GB just for the ViT-L/14 MLP
    # intermediate (1024 x 256 x 4096 float32), which overflows the 10 GB card.
    cpaths, clabs = read_manifest(MAN / 'fer2013_test.csv')
    cos_min = 1.0
    tok_cos_min = None
    tok_ref = None
    tref_path = D / ('token_cache_%s' % tag.lower()) / 'fer2013_test_tokens_fp16.npy'
    if tref_path.exists():
        tok_ref = np.load(tref_path, mmap_mode='r')
        tok_cos_min = 1.0
    for start in range(0, 512, 32):
        sel = np.arange(start, min(start + 32, 512))
        t, g = forward([cpaths[i] for i in sel])
        ref = clean['views'][sel]
        cos = np.sum(g * ref, axis=-1) / np.maximum(
            np.linalg.norm(g, axis=-1) * np.linalg.norm(ref, axis=-1), 1e-12)
        cos_min = min(cos_min, float(cos.min()))
        if tok_ref is not None:
            r = np.asarray(tok_ref[sel]).astype(np.float32)
            c2 = np.sum(t.astype(np.float32) * r, axis=-1) / np.maximum(
                np.linalg.norm(t.astype(np.float32), axis=-1) * np.linalg.norm(r, axis=-1), 1e-12)
            tok_cos_min = min(tok_cos_min, float(c2.min()))
            del r, c2
        del t, g, ref
    gate['clean_globals_cosine_min'] = float(cos_min)
    gate['clean_rows_checked'] = 512
    gate['clean_token_cosine_min'] = tok_cos_min
    torch.cuda.empty_cache()
    cond0 = args.conditions[0]
    mine = np.load(featdir / ('%s.npz' % cond0), allow_pickle=False)['views']
    stored_cond = SWEEP / ('%s.npz' % cond0)
    ref0 = np.load(stored_cond, allow_pickle=False)['views'] if stored_cond.exists() else None
    if ref0 is not None and ref0.shape == mine.shape:
        cos0 = np.sum(mine * ref0, axis=-1) / np.maximum(
            np.linalg.norm(mine, axis=-1) * np.linalg.norm(ref0, axis=-1), 1e-12)
        gate['condition_%s_globals_cosine_min' % cond0] = float(cos0.min())
        gate['condition_compared'] = cond0
        gate['condition_check'] = 'stored features/shift_sweep cache (ViT-B/16)'
    else:
        gate['condition_compared'] = None
        gate['condition_check'] = ('not applicable: features/shift_sweep/*.npz are ViT-B/16 (512-d) '
                                   'caches and no stored ViT-L/14 corrupted cache exists; the clean '
                                   'token-cache comparison above is used instead')
    ok = gate['clean_globals_cosine_min'] >= 0.9999
    if gate.get('clean_token_cosine_min') is not None:
        ok = ok and gate['clean_token_cosine_min'] >= 0.999
    if gate.get('condition_compared'):
        ok = ok and gate['condition_%s_globals_cosine_min' % cond0] >= 0.9999
    gate['pass'] = bool(ok)
    gate['criterion'] = ('elementwise cosine >= 0.9999 against the stored clean cache'
                         + (' and the stored corrupted cache' if gate.get('condition_compared') else '')
                         + ('; clean token cache cosine >= 0.999' if gate.get('clean_token_cosine_min')
                            is not None else ''))
    rep['validation_gate'] = gate
    rep['elapsed_seconds'] = round(time.perf_counter() - total_t0, 1)
    rep['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    (OUT / ('phase6b_extraction_%s.json' % tag)).write_text(json.dumps(rep, indent=2),
                                                            encoding='utf-8')
    print('\nGATE clean_glob=%.6f clean_tok=%s cond=%s -> %s'
          % (gate['clean_globals_cosine_min'], gate.get('clean_token_cosine_min'),
             gate.get('condition_%s_globals_cosine_min' % cond0) if gate.get('condition_compared')
             else 'n/a', 'PASS' if gate['pass'] else 'FAIL'), flush=True)
    return 0 if gate['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
