"""8-row smoke test of the phase-6b extraction convention, both encoders. Read-only."""
import csv, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')


def rows(p):
    with p.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def main():
    import open_clip
    for tag, name, weights, clean_path in (
            ('B16', 'ViT-B-16', 'D:/ResearchVault/99system/models/open_clip/ViT-B-16.pt',
             D / 'features' / 'fer2013_test.npz'),
            ('L14', 'ViT-L-14', 'D:/ResearchVault/99system/models/open_clip/ViT-L-14.pt',
             D / 'features_vitl14' / 'fer2013_test.npz')):
        assert Path(weights).exists(), weights
        model, _, preprocess = open_clip.create_model_and_transforms(
            name, pretrained=weights, device='cuda', weights_only=False)
        model.eval()
        model.visual.output_tokens = True
        cs = rows(D / 'manifests' / 'fer2013_test.csv')[:4]
        ws = rows(D / 'manifests' / 'fer2013_blur_2p5.csv')[:4]
        for label, recs in (('clean', cs), ('blur_2p5', ws)):
            imgs = [Image.open(r['path']).convert('RGB') for r in recs]
            pairs = []
            for im in imgs:
                pairs.extend([im, ImageOps.mirror(im)])
            x = torch.stack([preprocess(im) for im in pairs]).to('cuda')
            with torch.inference_mode():
                pooled, tokens = model.visual(x)
                pooled = F.normalize(pooled.float(), dim=-1)
            g = pooled.cpu().numpy().reshape(len(recs), 2, -1)
            t = tokens[0::2].half().cpu().numpy()
            print('%-4s %-9s pooled%s tokens%s  tok_norm=%.2f  glob_norm=%.4f/%.4f'
                  % (tag, label, pooled.shape, tuple(t.shape),
                     float(np.linalg.norm(t[0].astype(np.float32), axis=-1).mean()),
                     np.linalg.norm(g[0, 0]), np.linalg.norm(g[0, 1])), flush=True)
            if label == 'clean':
                ref = np.load(clean_path, allow_pickle=False)['views'][:4]
                cos = np.sum(g * ref, axis=-1) / np.maximum(
                    np.linalg.norm(g, axis=-1) * np.linalg.norm(ref, axis=-1), 1e-12)
                print('      cosine vs stored clean cache: min=%.6f' % cos.min(), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
