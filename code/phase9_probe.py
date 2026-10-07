import os, shutil
import numpy as np

D = 'D:/ResearchVault/99system/data/rb-tta-fer-fg2027'
for d in ['token_cache_vitl14', 'token_cache_vitb16_mirror', 'token_cache_vitl14_mirror',
          'token_cache_l14_confirm', 'token_cache_b16_confirm', 'features_confirm_shift']:
    p = os.path.join(D, d)
    print('==', d, os.path.isdir(p))
    if os.path.isdir(p):
        print('   first:', sorted(os.listdir(p))[:8], 'n=', len(os.listdir(p)))

p = D + '/token_cache_vitl14/fer2013_train_tokens_fp16.npy'
print('l14 train npy exists', os.path.exists(p), round(os.path.getsize(p) / 1e9, 2) if os.path.exists(p) else None)
a = np.load(p, mmap_mode='r')
print('shape', a.shape, a.dtype)
m = np.load(D + '/token_cache_vitl14/fer2013_train_meta.npz', allow_pickle=False)
print('meta files', m.files, m['labels'].shape)
q = D + '/token_cache_vitb16_mirror/fer2013_train_tokens_fp16.npy'
print('b16 mirror train', os.path.exists(q), np.load(q, mmap_mode='r').shape if os.path.exists(q) else None)
print(sorted(os.listdir(D + '/token_cache_vitb16_mirror')))
sp = np.load(D + '/features_vitl14/linear_probe_split_seed0.npz', allow_pickle=False)
print('split', [(k, sp[k].shape) for k in sp.files])
spb = np.load(D + '/features/linear_probe_split_seed0.npz', allow_pickle=False)
print('split B16', [(k, spb[k].shape) for k in spb.files])
f = np.load(D + '/features_confirm_shift/l14/blur_0p8.npz', allow_pickle=False)
print('feat l14', f.files, f['views'].shape, f['labels'].shape)
man = D + '/manifests'
print('manifests sample', [x for x in sorted(os.listdir(man)) if 'blur_0p8' in x])
import csv
with open(man + '/fer2013_blur_0p8.csv', newline='', encoding='utf-8') as fh:
    rows = list(csv.DictReader(fh))
print('manifest rows', len(rows), rows[0])
print('disk free GB', round(shutil.disk_usage('D:/').free / 1e9, 1))
