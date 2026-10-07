"""Phase 6b: read the per-condition provenance metadata of the corruption / FER+ feature caches."""
import json, sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
F = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027/features')
RAW = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027/raw')


def main():
    out = {'conditions': {}, 'ferplus': None, 'image_dirs': {}}
    paths = ([F / 'corruptions' / ('%s.npz' % n) for n in ('blur', 'jpeg', 'lowlight', 'noise')]
             + sorted((F / 'shift_sweep').glob('*.npz')))
    for p in paths:
        z = np.load(p, allow_pickle=False)
        meta = str(z['meta_json'])
        out['conditions'][p.name] = {'meta_json': meta, 'n': int(len(z['ids']))}
        print('%-20s %s' % (p.name, meta), flush=True)

    print('\n--- is corruptions/<c>.npz identical to one shift_sweep level?', flush=True)
    out['identity_corruptions_vs_sweep'] = {}
    for n in ('blur', 'jpeg', 'lowlight', 'noise'):
        a = np.load(F / 'corruptions' / ('%s.npz' % n), allow_pickle=False)['views']
        matches = []
        for q in sorted((F / 'shift_sweep').glob(n + '_*.npz')):
            b = np.load(q, allow_pickle=False)['views']
            if a.shape == b.shape and np.array_equal(a, b):
                matches.append(q.name)
        out['identity_corruptions_vs_sweep'][n] = matches
        print('   %-9s == %s' % (n, matches), flush=True)

    z = np.load(F / 'ferplus_test.npz', allow_pickle=False)
    out['ferplus'] = {'meta_json': str(z['meta_json']), 'n': int(len(z['ids'])),
                      'labels_bincount': np.bincount(z['labels'].astype(np.int64), minlength=7).tolist()}
    print('\nFER+ meta_json: %s' % out['ferplus']['meta_json'], flush=True)

    print('\n--- image folders on disk', flush=True)
    for sub in ('fer2013_corruptions', 'fer2013_shift_sweep'):
        root = RAW / sub
        info = {'exists': root.exists(), 'per_condition': {}}
        if root.exists():
            for c in sorted(root.iterdir()):
                if c.is_dir():
                    info['per_condition'][c.name] = sum(1 for _ in c.rglob('*.png'))
        out['image_dirs'][sub] = info
        print('%-22s exists=%s conditions=%s' % (sub, info['exists'],
                                                 json.dumps(info['per_condition'])), flush=True)
    print('\nfer2013 raw test images: %d'
          % sum(1 for _ in (RAW / 'fer2013' / 'test').rglob('*.png')), flush=True)
    (Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
     / 'phase6b_provenance.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('wrote phase6b_provenance.json')


if __name__ == '__main__':
    main()
