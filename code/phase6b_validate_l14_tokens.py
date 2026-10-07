"""Standalone ViT-L/14 extraction validation: 512 clean FER2013 test rows, globals AND patch tokens,
compared against the stored clean caches.  Repairs the skipped token check of the L/14 gate (the
in-script path used tag.lower() = 'l14' instead of the real directory name 'token_cache_vitl14')."""
import csv, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
NROW = 512


def main():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained=str(Path('D:/ResearchVault/99system/models/open_clip/ViT-L-14.pt')),
        device='cuda', weights_only=False)
    model.eval()
    model.visual.output_tokens = True
    with (D / 'manifests' / 'fer2013_test.csv').open(newline='', encoding='utf-8') as fh:
        paths = [r['path'] for r in list(csv.DictReader(fh))[:NROW]]
    tref = np.load(D / 'token_cache_vitl14' / 'fer2013_test_tokens_fp16.npy', mmap_mode='r')
    gref = np.load(D / 'features_vitl14' / 'fer2013_test.npz', allow_pickle=False)['views'][:NROW]
    gc, tc = 1.0, 1.0
    for start in range(0, NROW, 32):
        chunk = paths[start:start + 32]
        imgs = [Image.open(p).convert('RGB') for p in chunk]
        pairs = []
        for im in imgs:
            pairs.extend([im, ImageOps.mirror(im)])
        x = torch.stack([preprocess(im) for im in pairs]).to('cuda')
        with torch.inference_mode():
            pooled, tokens = model.visual(x)
            pooled = F.normalize(pooled.float(), dim=-1)
        g = pooled.cpu().numpy().reshape(len(chunk), 2, -1)
        t = tokens[0::2].float().cpu().numpy()
        r = np.asarray(tref[start:start + 32]).astype(np.float32)
        c2 = np.sum(t * r, axis=-1) / np.maximum(
            np.linalg.norm(t, axis=-1) * np.linalg.norm(r, axis=-1), 1e-12)
        tc = min(tc, float(c2.min()))
        rf = gref[start:start + 32]
        c1 = np.sum(g * rf, axis=-1) / np.maximum(
            np.linalg.norm(g, axis=-1) * np.linalg.norm(rf, axis=-1), 1e-12)
        gc = min(gc, float(c1.min()))
        del x, pooled, tokens, g, t, r, c2, c1
    rep = {'encoder': 'L14', 'rows': NROW,
           'clean_globals_cosine_min': gc, 'clean_token_cosine_min': tc,
           'tolerance': {'globals': 0.9999, 'tokens': 0.999},
           'pass': bool(gc >= 0.9999 and tc >= 0.999),
           'note': 'standalone repair of the L/14 gate token check; the corrupted-condition check of '
                   'the pre-registration applies to ViT-B/16 only, because features/shift_sweep/*.npz '
                   'are 512-d ViT-B/16 caches and no stored ViT-L/14 corrupted cache exists'}
    (OUT / 'phase6b_validation_l14_tokens.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print('L14 validation: globals cos min=%.6f tokens cos min=%.6f -> %s'
          % (gc, tc, 'PASS' if rep['pass'] else 'FAIL'), flush=True)
    return 0 if rep['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
