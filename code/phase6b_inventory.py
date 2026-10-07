"""Phase 6b step 1: inventory candidate confirmation materials (read-only)."""
import json, sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
F = D / 'features'
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')

TARGETS = (
    [F / 'corruptions' / ('%s.npz' % n) for n in ('blur', 'jpeg', 'lowlight', 'noise')]
    + sorted((F / 'shift_sweep').glob('*.npz'))
    + [F / 'ferplus_test.npz', F / 'fer2013_test.npz', F / 'fer2013_probe_val_seed0.npz',
       F / 'ckplus48_all.npz', F / 'kdef_test.npz', F / 'ckplus_test.npz']
)


def main():
    rep = {'files': {}}
    for p in TARGETS:
        if not p.exists():
            rep['files'][p.name] = {'missing': True}
            continue
        z = np.load(p, allow_pickle=False)
        info = {'path': str(p), 'bytes': p.stat().st_size, 'keys': {}}
        for k in z.files:
            a = z[k]
            if a.dtype.kind in 'OUS':
                info['keys'][k] = {'shape': list(a.shape), 'dtype': str(a.dtype),
                                   'first': str(a.flat[0])[:110],
                                   'n_unique': int(len(set(a.astype(str).tolist())))}
            else:
                info['keys'][k] = {'shape': list(a.shape), 'dtype': str(a.dtype),
                                   'min': float(np.min(a)), 'max': float(np.max(a))}
                if k == 'labels':
                    lab = a.astype(np.int64)
                    info['keys'][k]['bincount'] = np.bincount(lab[lab >= 0], minlength=7).tolist()
                    info['keys'][k]['n_negative'] = int((lab < 0).sum())
        rep['files'][p.name] = info
        print('%-26s %-9s %s' % (p.name, 'MB=%.1f' % (p.stat().st_size / 1e6),
                                 json.dumps({k: (v.get('shape'), v.get('dtype')) for k, v in info['keys'].items()})),
              flush=True)
        if 'labels' in info['keys']:
            print('    labels bincount: %s' % info['keys']['labels'].get('bincount'), flush=True)
        if 'ids' in info['keys']:
            print('    ids: n_unique=%s first=%s' % (info['keys']['ids']['n_unique'],
                                                     info['keys']['ids']['first']), flush=True)

    # do the corrupted/shifted views line up with the clean fer2013_test rows?
    clean = np.load(F / 'fer2013_test.npz', allow_pickle=False)
    cid, clab = clean['ids'].astype(str), clean['labels'].astype(np.int64)
    rep['alignment_with_fer2013_test'] = {}
    for p in ([F / 'corruptions' / ('%s.npz' % n) for n in ('blur', 'jpeg', 'lowlight', 'noise')]
              + sorted((F / 'shift_sweep').glob('*.npz'))):
        z = np.load(p, allow_pickle=False)
        same_ids = np.array_equal(z['ids'].astype(str), cid) if 'ids' in z.files else None
        same_lab = np.array_equal(z['labels'].astype(np.int64), clab) if 'labels' in z.files else None
        rep['alignment_with_fer2013_test'][p.name] = {'ids_equal': same_ids, 'labels_equal': same_lab,
                                                     'n': int(len(z['ids'])) if 'ids' in z.files else None}
        print('alignment %-18s ids_equal=%s labels_equal=%s n=%s'
              % (p.name, same_ids, same_lab, len(z['ids']) if 'ids' in z.files else None), flush=True)

    fs = F / 'ferplus_stats.json'
    rep['ferplus_stats'] = json.loads(fs.read_text(encoding='utf-8')) if fs.exists() else None
    print('ferplus_stats.json:', json.dumps(rep['ferplus_stats']), flush=True)
    (OUT / 'phase6b_inventory.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print('wrote phase6b_inventory.json')


if __name__ == '__main__':
    main()
