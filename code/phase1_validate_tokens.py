"""Phase 1 validation: patch-token extraction feasibility + alignment with cached embeddings."""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
M = Path('D:/ResearchVault/99system/models/open_clip/ViT-B-16.pt')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
OUT.mkdir(parents=True, exist_ok=True)
N = 200

import open_clip
t0 = time.perf_counter()
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained=str(M), device='cuda', weights_only=False)
model.eval()
model.visual.output_tokens = True
print('loaded in %.1fs | output_tokens=%s' % (time.perf_counter() - t0,
                                              model.visual.output_tokens))


def tokens_for(paths):
    toks, pooled, proj = [], [], []
    with torch.inference_mode():
        for i in range(0, len(paths), 128):
            batch = [preprocess(Image.open(p).convert('RGB')) for p in paths[i:i + 128]]
            x = torch.stack(batch).to('cuda')
            pooled_proj, feats = model.visual(x)           # (B,512), (B,197,768)
            toks.append(feats.half().cpu().numpy())
            pooled.append(pooled_proj.float().cpu().numpy())
            pr = pooled_proj.float()
            pr = pr / pr.norm(dim=-1, keepdim=True)
            proj.append(pr.cpu().numpy())
    return (np.concatenate(toks), np.concatenate(pooled), np.concatenate(proj))


report = {'device': torch.cuda.get_device_name(0), 'n_per_dataset': N, 'datasets': {}}
for name, man, cache in (('ckplus', D / 'manifests/ckplus_test.csv', D / 'features/ckplus_test.npz'),
                         ('kdef', D / 'manifests/kdef_test.csv', D / 'features/kdef_test.npz')):
    lines = man.open(encoding='utf-8').read().splitlines()[1:N + 1]
    paths = [l.split(',')[0] for l in lines]
    t = time.perf_counter()
    toks, pooled, proj = tokens_for(paths)
    dt = time.perf_counter() - t
    z = np.load(cache, allow_pickle=False)
    ids = [str(x) for x in z['ids']]
    views = z['views']
    idx = [ids.index(p) for p in paths]
    cached0 = views[idx, 0, :]
    cached0 = cached0 / np.maximum(np.linalg.norm(cached0, axis=1, keepdims=True), 1e-12)
    cos = np.sum(proj * cached0, axis=1)
    report['datasets'][name] = {
        'n': len(paths), 'token_shape': list(toks.shape), 'dtype': str(toks.dtype),
        'grid': [int(np.sqrt(toks.shape[1])), int(np.sqrt(toks.shape[1]))],
        'token_norm_mean': float(np.linalg.norm(toks.astype(np.float32), axis=-1).mean()),
        'proj_vs_cached_view0_cosine': {'min': float(cos.min()), 'mean': float(cos.mean()),
                                        'max': float(cos.max())},
        'seconds_for_%d' % N: round(dt, 2),
        'images_per_second': round(N / dt, 1),
        'fp16_MB_per_image': float(toks.itemsize * np.prod(toks.shape[1:]) / 1e6),
    }
    np.save(OUT / ('phase1_tokens_%s_sample.npy' % name), toks[:20])
    print(name, report['datasets'][name])

tot_imgs = 28709 + 7178 + 902 + 2938
per_img_mb = report['datasets']['ckplus']['fp16_MB_per_image']
report['cost_estimate'] = {
    'total_images': tot_imgs, 'fp16_MB_per_image': per_img_mb,
    'disk_GB_single_view': round(tot_imgs * per_img_mb / 1024, 2),
    'disk_GB_with_mirror': round(2 * tot_imgs * per_img_mb / 1024, 2),
    'throughput_img_per_s_observed': report['datasets']['kdef']['images_per_second'],
    'extract_minutes_observed_rate': round(tot_imgs /
                                           report['datasets']['kdef']['images_per_second'] / 60, 1),
    'ferplus_note': 'FER+ has no image manifest; its images are fer2013_test images '
                    '(features built by build_ferplus_features.py)',
}
(OUT / 'phase1_feasibility.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report['cost_estimate'], indent=2))
