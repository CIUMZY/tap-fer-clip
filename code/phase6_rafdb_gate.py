"""Phase 6 gate: can the local RAF-DB copy be used as a confirmation set?  Read-only.

Checks, in order:
  1. what the local copy declares about itself (PROVENANCE.md),
  2. whether an official RAF-DB split/label file exists anywhere on disk,
  3. per-class image counts and file sizes,
  4. image format (mode, size) on the full tree,
  5. exact-duplicate structure inside the tree,
  6. pixel-level overlap between the tree and the FER2013 images this project trains on
     (the decisive contamination test),
  7. whether the class-imbalance ratios of the phase 3 protocol are realisable.

Writes phase6_rafdb_gate.json.  Nothing outside this results directory is written.
"""
import hashlib, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
RAF = D / 'downloads' / 'rafdb_processed' / 'processed_data'
FER = D / 'raw' / 'fer2013'
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
CLASSES = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')


def h48(path):
    """Content hash in the common space: 48x48 8-bit grayscale, plus the original mode/size."""
    with Image.open(path) as im:
        mode, size = im.mode, im.size
        g = np.asarray(im.convert('L').resize((48, 48), Image.BILINEAR), dtype=np.uint8)
    return hashlib.sha1(g.tobytes()).hexdigest(), mode, size


def main():
    t0 = time.perf_counter()
    rep = {'raf_root': str(RAF), 'fer_root': str(FER), 'checks': [], 'fail': []}

    def chk(name, ok, detail=''):
        rep['checks'].append({'check': name, 'ok': bool(ok), 'detail': str(detail)})
        if not ok:
            rep['fail'].append(name)
        print('%-58s %s %s' % (name, 'OK  ' if ok else 'GATE-FAIL', detail), flush=True)

    prov = (RAF.parent / 'PROVENANCE.md')
    rep['provenance'] = prov.read_text(encoding='utf-8') if prov.exists() else None
    chk('local copy ships a provenance note', prov.exists())
    chk('provenance is not a quarantine notice',
        bool(rep['provenance']) and 'QUARANTINED' not in rep['provenance'],
        'PROVENANCE.md status: ' + ('QUARANTINED' if rep['provenance'] and 'QUARANTINED' in rep['provenance'] else 'n/a'))

    # 2. official split / label files anywhere on the data volume
    hits = []
    for pat in ('list_patition_label.txt', 'list_patition*.txt', '*rafdb*label*.txt'):
        hits += [str(p) for p in Path('D:/ResearchVault').rglob(pat)]
    rep['official_split_files_found'] = sorted(set(hits))
    chk('an official RAF-DB split/label file exists on disk', bool(hits),
        'searched D:/ResearchVault for list_patition_label.txt and variants')

    # 3. per-class counts and file sizes
    per_class, sizes = {}, {}
    for c in CLASSES:
        fs = sorted((RAF / c).glob('*'))
        per_class[c] = len(fs)
        sz = [f.stat().st_size for f in fs]
        sizes[c] = {'n': len(sz), 'min': int(min(sz)), 'median': int(np.median(sz)),
                    'max': int(max(sz)), 'mean': float(np.mean(sz))}
    rep['per_class_counts'] = per_class
    rep['per_class_file_bytes'] = sizes
    total = sum(per_class.values())
    rep['total_images'] = total
    print('per-class counts:', per_class, 'total', total, flush=True)
    dup_counts = [c for c, n in per_class.items() if list(per_class.values()).count(n) > 1]
    chk('class counts are not an artificial balancing signature',
        not dup_counts,
        'classes sharing the identical count %s: %s' % (set(per_class.values()), dup_counts))

    # 4/5. decode everything: format distribution, exact and normalized duplicates
    modes, dims = Counter(), Counter()
    h_norm = {}
    n_dup_in_tree = 0
    t1 = time.perf_counter()
    for c in CLASSES:
        for f in sorted((RAF / c).glob('*')):
            h, mode, size = h48(f)
            modes[mode] += 1
            dims['%dx%d' % size] += 1
            if h in h_norm:
                n_dup_in_tree += 1
            else:
                h_norm[h] = str(f)
    rep['modes'] = dict(modes)
    rep['dimensions_top'] = dict(dims.most_common(8))
    rep['unique_normalized_images'] = len(h_norm)
    rep['duplicate_images_in_tree'] = n_dup_in_tree
    rep['decode_seconds'] = round(time.perf_counter() - t1, 1)
    print('decoded %d images in %.1fs; modes %s; dims %s' % (total, rep['decode_seconds'],
                                                             dict(modes), rep['dimensions_top']),
          flush=True)
    chk('no duplicated images inside the candidate set', n_dup_in_tree == 0,
        '%d of %d images are duplicates of another image after 48x48 normalisation'
        % (n_dup_in_tree, total))

    # 6. contamination: overlap with the FER2013 images this project trains on
    fer = {}
    for split in ('train', 'test'):
        for c in CLASSES:
            for f in sorted((FER / split / c).glob('*')):
                h, _mode, _size = h48(f)
                fer.setdefault(h, []).append('%s/%s/%s' % (split, c, f.name))
    rep['fer2013_images_hashed'] = sum(len(v) for v in fer.values())
    inter = sorted(set(h_norm) & set(fer))
    rep['overlap_with_fer2013'] = {
        'n_unique_normalized_images_overlapping': len(inter),
        'examples': [(h_norm[h].replace(str(D), '<data>'),
                      [x for x in fer[h]][:3]) for h in inter[:10]],
        'overlap_fraction_of_candidate': round(len(inter) / max(len(h_norm), 1), 4),
    }
    print('overlap with FER2013 (48x48 normalised): %d unique images (%.2f%% of the candidate set)'
          % (len(inter), 100 * len(inter) / max(len(h_norm), 1)), flush=True)
    for h in inter[:5]:
        print('   %s  ==  %s' % (h_norm[h].replace(str(D), '<data>'), fer[h][:2]), flush=True)
    chk('candidate set does not overlap the FER2013 source images',
        len(inter) == 0,
        '%d candidate images are pixel-identical (48x48 normalised) to a FER2013 image' % len(inter))

    # 7. ratio feasibility under the phase 3 construction (base = smallest class count)
    base = min(per_class.values())
    maj = max(per_class, key=lambda c: per_class[c])
    feas = {}
    for r in (0, 1, 2, 5, 10):
        need = 0 if r == 0 else base * r
        feas[r] = {'needed_majority': need, 'available_majority': per_class[maj],
                   'feasible': bool(r == 0 or need <= per_class[maj])}
    rep['ratio_feasibility'] = {'base_minority_count': base, 'majority_class': maj, 'ratios': feas}
    ok_ratios = [r for r in feas if feas[r]['feasible']]
    chk('all pre-registered ratios 0/1/2/5/10 are realisable',
        len(ok_ratios) == 5,
        'base=%d majority=%s(%d); realisable ratios %s; blocked %s'
        % (base, maj, per_class[maj], ok_ratios, [r for r in feas if not feas[r]['feasible']]))

    # class-space mapping onto this project's label indices, read from the existing FER2013 cache
    meta = np.load(D / 'token_cache_vitb16' / 'fer2013_train_tokens_fp16.npz', allow_pickle=False)
    ids = meta['ids'].astype(str)
    labs = meta['labels'].astype(np.int64)
    mapping = {}
    for i, p in enumerate(ids):
        folder = Path(p.replace('\\', '/')).parent.name
        mapping.setdefault(folder, set()).add(int(labs[i]))
    rep['project_class_index_mapping'] = {k: sorted(v) for k, v in sorted(mapping.items())}
    chk('project label mapping is a clean folder->index bijection',
        all(len(v) == 1 for v in mapping.values()) and len({list(v)[0] for v in mapping.values()}) == len(mapping),
        json.dumps(rep['project_class_index_mapping']))

    rep['verdict'] = ('GATE FAILED - experiment not run'
                      if rep['fail'] else 'GATE PASSED')
    rep['failed_checks'] = rep['fail']
    rep['elapsed_seconds'] = round(time.perf_counter() - t0, 1)
    (OUT / 'phase6_rafdb_gate.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print('\n%s  failed: %s' % (rep['verdict'], rep['fail']), flush=True)
    return 0 if not rep['fail'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
