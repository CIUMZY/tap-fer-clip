"""POST-HOC SENSITIVITY - NOT CONFIRMATORY.
Position-correspondence self-check: is the mirror token at index k the mirror of the original
token at index k (raw pairing) or at the horizontally reversed index (unflip pairing)?
Read-only; prints per-position cosine statistics for both pairings.
"""
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
N = 256


def main():
    for name in ('ckplus_test', 'kdef_test', 'fer2013_test'):
        o = np.asarray(np.load(D / 'token_cache_vitb16' / ('%s_tokens_fp16.npz' % name),
                               allow_pickle=False)['tokens'][:N]).astype(np.float32)
        m = np.asarray(np.load(D / 'token_cache_vitb16_mirror' / ('%s_tokens_fp16.npy' % name),
                               mmap_mode='r')[:N]).astype(np.float32)
        grid = int(round(o.shape[1] ** 0.5))
        ol = o / np.maximum(np.linalg.norm(o, axis=-1, keepdims=True), 1e-12)
        ml = m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)
        raw = (ol * ml).sum(-1)                        # (N, 196) raw index pairing
        mf = ml.reshape(len(ml), grid, grid, -1)[:, :, ::-1, :].reshape(len(ml), -1, ml.shape[-1])
        unf = (ol * mf).sum(-1)                        # (N, 196) after horizontal unflip
        print('%-13s grid=%dx%d  mean per-position cosine: raw %.4f | unflipped %.4f'
              % (name, grid, grid, raw.mean(), unf.mean()), flush=True)
        print('%-13s   frac(cos>0.9): raw %.3f | unflipped %.3f ; frac(cos<0.5): raw %.3f | unflipped %.3f'
              % ('', (raw > 0.9).mean(), (unf > 0.9).mean(), (raw < 0.5).mean(), (unf < 0.5).mean()),
              flush=True)
        # diagonal-consistency: does index k of the mirror match index (grid-1-c) of the original?
        rev = ol.reshape(len(ol), grid, grid, -1)[:, :, ::-1, :].reshape(len(ol), -1, ol.shape[-1])
        cross = (rev * ml).sum(-1).mean()
        print('%-13s   cross-check mean cosine (original unflipped vs mirror raw) %.4f'
              % ('', cross), flush=True)


if __name__ == '__main__':
    main()
