"""Verify the extracted confirm caches: structural completeness, meta/id/label alignment, and an
independent content recomputation on sampled rows (tokens compared elementwise in fp16, globals by
cosine).  Read-only apart from phase6b_verify_extraction.json.

This is the check that decides whether the 12 reused ViT-L/14 conditions may be evaluated at all.
"""
import csv, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
CONDS = ['blur_0p8', 'blur_1p5', 'blur_2p5', 'jpeg_15', 'jpeg_30', 'jpeg_50',
         'lowlight_0p3', 'lowlight_0p5', 'lowlight_0p7', 'noise_10', 'noise_25', 'noise_40']
ENC = {'B16': ('ViT-B-16', 'D:/ResearchVault/99system/models/open_clip/ViT-B-16.pt', 196, 768, 512),
       'L14': ('ViT-L-14', 'D:/ResearchVault/99system/models/open_clip/ViT-L-14.pt', 256, 1024, 768)}
NROW = 7178


def npy_offset(p):
    """Data offset of a .npy file, so file_size - offset can be checked against nbytes."""
    with p.open('rb') as fh:
        assert fh.read(6) == b'\x93NUMPY', 'not a npy file'
        ver = fh.read(2)
        hlen = int.from_bytes(fh.read(2 if ver[0] == 1 else 4), 'little')
        hdr = fh.read(hlen).decode('latin1')
    import ast
    d = ast.literal_eval(hdr.strip())
    return 6 + 2 + (2 if ver[0] == 1 else 4) + hlen, d


def main():
    rep = {'conditions': {}, 'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
    models = {}
    for tag in ('B16', 'L14'):
        name, weights, n_tok, d_tok, d_glob = ENC[tag]
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            name, pretrained=weights, device='cuda', weights_only=False)
        model.eval()
        model.visual.output_tokens = True
        models[tag] = (model, preprocess)
        tokdir = D / ('token_cache_%s_confirm' % tag.lower())
        featdir = D / 'features_confirm_shift' / tag.lower()
        clean = np.load((D / 'features' if tag == 'B16' else D / 'features_vitl14')
                        / 'fer2013_test.npz', allow_pickle=False)
        clab = clean['labels'].astype(np.int64)
        for cond in CONDS:
            with (D / 'manifests' / ('fer2013_%s.csv' % cond)).open(newline='', encoding='utf-8') as fh:
                rows = list(csv.DictReader(fh))
            mpaths = [r['path'] for r in rows]
            mlab = np.asarray([int(r['label']) for r in rows], np.int64)
            npy = tokdir / ('%s_tokens_fp16.npy' % cond)
            meta = np.load(tokdir / ('%s_meta.npz' % cond), allow_pickle=False)
            feat = np.load(featdir / ('%s.npz' % cond), allow_pickle=False)
            off, hdr = npy_offset(npy)
            nbytes = NROW * n_tok * d_tok * 2
            mm = np.load(npy, mmap_mode='r')
            v = feat['views']
            info = {
                'file_bytes': npy.stat().st_size,
                'expected_bytes': off + nbytes,
                'size_complete': bool(npy.stat().st_size == off + nbytes),
                'header_shape': [int(x) for x in hdr['shape']], 'header_dtype': str(hdr['descr']),
                'memmap_shape': list(mm.shape), 'memmap_dtype': str(mm.dtype),
                'meta_labels_len': int(len(meta['labels'])), 'meta_ids_len': int(len(meta['ids'])),
                'meta_labels_equal_manifest': bool(np.array_equal(meta['labels'].astype(np.int64), mlab)),
                'meta_labels_equal_clean_test': bool(np.array_equal(meta['labels'].astype(np.int64), clab)),
                'meta_ids_equal_manifest': bool(np.array_equal(meta['ids'].astype(str), np.asarray(mpaths))),
                'meta_ids_unique': bool(len(set(meta['ids'].astype(str).tolist())) == NROW),
                'feat_views_shape': list(v.shape),
                'feat_unit_norm': bool(np.allclose(np.linalg.norm(v, axis=-1), 1.0, atol=1e-4)),
                'feat_labels_equal_manifest': bool(np.array_equal(feat['labels'].astype(np.int64), mlab)),
                'feat_ids_equal_manifest': bool(np.array_equal(feat['ids'].astype(str), np.asarray(mpaths))),
                'tokens_finite_sampled': bool(np.isfinite(
                    np.asarray(mm[np.r_[0:4, NROW - 4:NROW]]).astype(np.float32)).all()),
                'tokens_std_sampled': float(np.asarray(
                    mm[np.r_[0:4, NROW - 4:NROW]]).astype(np.float32).std()),
            }
            # independent recomputation of 8 rows (first 4 and last 4)
            sel = list(range(4)) + list(range(NROW - 4, NROW))
            toks_parts, glob_parts = [], []
            for grp in (sel[:4], sel[4:]):
                imgs = [Image.open(mpaths[i]).convert('RGB') for i in grp]
                pairs = []
                for im in imgs:
                    pairs.extend([im, ImageOps.mirror(im)])
                x = torch.stack([preprocess(im) for im in pairs]).to('cuda')
                with torch.inference_mode():
                    p, t = model.visual(x)
                    pooled = F.normalize(p.float(), dim=-1)
                toks_parts.append(t[0::2].half().cpu().numpy())
                glob_parts.append(pooled.cpu().numpy().reshape(len(grp), 2, -1))
                del x, p, t, pooled
            toks = np.concatenate(toks_parts, axis=0)
            g = np.concatenate(glob_parts, axis=0)
            ref_t = np.asarray(mm[sel])
            d = np.abs(toks.astype(np.float32) - ref_t.astype(np.float32))
            info['tokens_bitwise_equal_recompute'] = bool(np.array_equal(toks, ref_t))
            info['tokens_max_abs_diff_recompute'] = float(d.max())
            # fp16 storage means a recomputation in a different batch shape differs by a few ULPs;
            # ULP is 0.015625 at magnitude 16-31, so allow 2 ULP (0.03125) and also require cosine.
            a_flat = toks.astype(np.float32).reshape(len(sel), -1)
            b_flat = ref_t.astype(np.float32).reshape(len(sel), -1)
            c_tok = np.sum(a_flat * b_flat, axis=-1) / np.maximum(
                np.linalg.norm(a_flat, axis=-1) * np.linalg.norm(b_flat, axis=-1), 1e-12)
            info['tokens_cosine_min_recompute'] = float(c_tok.min())
            info['tokens_ulps_max'] = round(info['tokens_max_abs_diff_recompute'] / 0.015625, 2)
            ref_g = v[sel]
            cos = np.sum(g * ref_g, axis=-1) / np.maximum(
                np.linalg.norm(g, axis=-1) * np.linalg.norm(ref_g, axis=-1), 1e-12)
            info['globals_cosine_min_recompute'] = float(cos.min())
            info['verified'] = bool(info['size_complete'] and info['meta_labels_equal_manifest']
                                    and info['meta_ids_equal_manifest'] and info['feat_unit_norm']
                                    and info['tokens_max_abs_diff_recompute'] <= 0.03125
                                    and info['tokens_cosine_min_recompute'] >= 0.9999
                                    and info['globals_cosine_min_recompute'] >= 0.9999)
            rep['conditions']['%s/%s' % (tag, cond)] = info
            print('%-4s %-13s bytes_ok=%s labels_ok=%s ids_ok=%s tok_ulp=%s tok_cos=%.6f glob_cos=%.6f -> %s'
                  % (tag, cond, info['size_complete'], info['meta_labels_equal_manifest'],
                     info['meta_ids_equal_manifest'], info['tokens_ulps_max'],
                     info['tokens_cosine_min_recompute'],
                     info['globals_cosine_min_recompute'], 'OK' if info['verified'] else 'FAIL'),
                  flush=True)
            del toks, g, mm, v
        del models[tag]
        torch.cuda.empty_cache()
    rep['n_verified'] = sum(1 for v in rep['conditions'].values() if v['verified'])
    rep['n_total'] = len(rep['conditions'])
    rep['all_verified'] = bool(rep['n_verified'] == rep['n_total'])
    rep['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    (OUT / 'phase6b_verify_extraction.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print('\nverified %d/%d -> %s' % (rep['n_verified'], rep['n_total'],
                                       'ALL OK' if rep['all_verified'] else 'FAILURES PRESENT'))
    return 0 if rep['all_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
