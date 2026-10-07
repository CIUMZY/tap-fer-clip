import os
import numpy as np

D = 'D:/ResearchVault/99system/data/rb-tta-fer-fg2027'

m14 = np.load(D + '/token_cache_vitl14/fer2013_train_meta.npz', allow_pickle=False)
ids14 = m14['ids']
print('L14 ids dtype', ids14.dtype, ids14.shape)
print('first ids', ids14[:3])
print('exists(first)', [os.path.exists(str(x)) for x in ids14[:3]])

b = np.load(D + '/token_cache_vitb16/fer2013_train_tokens_fp16.npz', allow_pickle=False)
print('b16 clean keys', b.files)
mb = np.load(D + '/token_cache_vitb16_mirror/fer2013_train_meta.npz', allow_pickle=False)
print('b16 mirror keys', mb.files)
for k in mb.files:
    print('  ', k, mb[k].dtype, mb[k].shape, str(mb[k].ravel()[:2]))
if 'ids' in mb.files and 'ids' in b.files:
    print('ids identical', bool(np.array_equal(mb['ids'], b['ids'])))
if 'labels' in mb.files and 'labels' in b.files:
    print('labels identical', bool(np.array_equal(mb['labels'], b['labels'])))
