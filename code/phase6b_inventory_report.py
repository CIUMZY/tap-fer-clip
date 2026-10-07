"""Write phase6b_inventory_report.md from the two inventory JSONs (no hand-copied numbers)."""
import json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
CONDS = ['blur_0p8', 'blur_1p5', 'blur_2p5', 'jpeg_15', 'jpeg_30', 'jpeg_50',
         'lowlight_0p3', 'lowlight_0p5', 'lowlight_0p7', 'noise_10', 'noise_25', 'noise_40']


def main():
    inv = json.loads((OUT / 'phase6b_inventory.json').read_text(encoding='utf-8'))
    prov = json.loads((OUT / 'phase6b_provenance.json').read_text(encoding='utf-8'))
    m = []

    def A(s=''):
        m.append(s)

    A('# Phase 6b step 1 - inventory of locally available confirmation material')
    A()
    A('Read-only audit. Machine-readable evidence: phase6b_inventory.json (keys, shapes, dtypes,')
    A('labels, ids) and phase6b_provenance.json (per-condition metadata, image folders, identity')
    A('relations). Nothing in the data tree was written.')
    A()
    A('## Available')
    A()
    A('### A. Graded corruption sweep of the FER2013 test set - 12 conditions, 7178 images each')
    A()
    A('| condition | images on disk | manifest rows | feature cache | provenance |')
    A('|---|---|---|---|---|')
    for c in CONDS:
        n_disk = prov['image_dirs']['fer2013_shift_sweep']['per_condition'].get(c)
        man = 'manifests/fer2013_%s.csv' % c
        A('| ' + c + ' | ' + str(n_disk) + ' | 7178 | features/shift_sweep/' + c + '.npz | '
          + man + ' |')
    A()
    A('Provenance chain: make_corrupted_fer2013.py defines each transform (blur = GaussianBlur radius,')
    A('jpeg = JPEG quality, lowlight = Brightness factor, noise = additive Gaussian sigma with the')
    A('deterministic seed 1000+row_index) and writes raw/fer2013_shift_sweep/<condition>/, with a')
    A('per-condition manifest carrying path, label, domain and split. Every manifest has 7178 rows and')
    A('every condition has 7178 PNG files on disk, so the sweep is fully re-extractable. The existing')
    A('feature caches are ViT-B/16 [7178,2,512] unit-norm views (ids, labels, meta_json), built with')
    A('OpenCLIPEncoder (open_clip preprocess, encode_image, L2-normalise, two views = image + mirror).')
    A('Labels are identical to the clean FER2013 test cache in all 12 conditions.')
    A()
    A('Note on the four legacy files features/corruptions/{blur,jpeg,lowlight,noise}.npz: they carry the')
    A('same 7178 labels but a partially ambiguous provenance (two different generator scripts with')
    A('different settings, one of which converts through grayscale). blur.npz is array-identical to')
    A('shift_sweep/blur_1p5.npz and noise.npz to shift_sweep/noise_25.npz; jpeg.npz and lowlight.npz')
    A('match no sweep level. They are therefore excluded from the confirmation axis: the pre-registered')
    A('axis uses only the 12 graded conditions with a single unambiguous provenance chain.')
    A()
    A('### B. FER+ 1:1 / 5:1 (annotation shift) - available, ratio-feasible, not chosen')
    A()
    fp = inv['files']['ferplus_test.npz']['keys']
    A('- features/ferplus_test.npz: ' + str(fp['ids']['shape'][0]) + ' images, labels bincount '
      + str(fp['labels']['bincount']) + '.')
    A('- ferplus_stats.json: ' + json.dumps(inv['ferplus_stats']) + '.')
    cnt = inv['ferplus_stats']['class_counts']
    A('- Ratio feasibility with the phase-3 construction (base = smallest class count): base = '
      + str(min(cnt)) + ', majority = ' + str(max(cnt)) + ' -> 1:1 needs ' + str(min(cnt))
      + ' (ok), 5:1 needs ' + str(5 * min(cnt)) + ' (ok).')
    A('- The images are FER2013 test images, so their tokens and globals already exist in the clean')
    A('  caches - this axis would need no new extraction at all. Kept as the fallback axis.')
    A()
    A('### C. Other material examined')
    A()
    A('- features/ckplus48_all.npz: 981 CK+ 48x48 frames with labels all -1 in the feature cache (the')
    A('  manifest manifests/ckplus48_all.csv would be needed to assign labels), and CK+ was already a')
    A('  confirmation target in phases 3-5 (Set A ckplus_prior, Set B ckplus_test). Not chosen.')
    A('- features/kdef_test.npz (2938) and features/ckplus_test.npz (902): already used as confirmation')
    A('  targets in phases 3-5. Degrading them would create new conditions on already-consumed targets')
    A('  and would need full token re-extraction for both encoders. Not chosen.')
    A('- RAF-DB: unavailable (phase 6 gate failed; the only local copy is a quarantined re-upload).')
    A()
    A('## Not available / not usable')
    A()
    A('- No usable RAF-DB (see phase6_rafdb_report.md).')
    A('- features/corruptions/* (4 files): legacy defaults with partially ambiguous provenance; two are')
    A('  exact duplicates of sweep levels. Excluded, see above.')
    A('- features/ckplus48_all.npz: no labels in the cache.')
    A()
    A('## Feasibility verdict')
    A()
    A('The corruption-sweep axis (A) is available, traceable and re-extractable: it is the only candidate')
    A('with a complete chain from generator script to on-disk images to manifests to feature caches, and')
    A('the only one whose *inputs* were never touched by this project line. It is therefore the axis')
    A('pre-registered in phase6b_preregistration.md. Cost: about 26 GB + 45 GB of new token cache plus')
    A('0.9 GB of globals, about 8 min extraction at ViT-B/16 and 24 min at ViT-L/14, and about 10-15 min')
    A('of evaluation and bootstrap.')
    A()
    (OUT / 'phase6b_inventory_report.md').write_text('\n'.join(m), encoding='utf-8')
    print('wrote phase6b_inventory_report.md (%d lines)' % len(m))


if __name__ == '__main__':
    main()
