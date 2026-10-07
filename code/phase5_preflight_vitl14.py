"""Phase 5 preflight: verify that every input the ViT-L/14 training and confirmation scripts
read exists, has the expected shape/dtype and is consistent across caches.  Read-only.
"""
import json, sys, time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
TC = D / 'token_cache_vitl14'
L = D / 'features_vitl14'
KS = D / 'features_kdef_source'
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
shapes = {200: 196, 768: 1024}


def hdr(p, mmap=True):
    a = np.load(p, mmap_mode='r' if mmap else None, allow_pickle=False)
    return a


def main():
    t0 = time.perf_counter()
    rep = {'checks': [], 'fail': []}

    def chk(name, ok, detail=''):
        rep['checks'].append({'check': name, 'ok': bool(ok), 'detail': str(detail)})
        if not ok:
            rep['fail'].append(name)
        print('%-46s %s %s' % (name, 'OK ' if ok else 'FAIL', detail), flush=True)

    names = ('fer2013_train', 'fer2013_test', 'ckplus_test', 'kdef_test', 'kdef_source_train')
    tok, meta, glob, ids, labs = {}, {}, {}, {}, {}
    for n in names:
        tp = TC / ('%s_tokens_fp16.npy' % n)
        mp = TC / ('%s_meta.npz' % n)
        chk('%s: tokens npy + meta npz present' % n, tp.exists() and mp.exists())
        a = hdr(tp)
        tok[n] = a
        m = np.load(mp, allow_pickle=False)
        meta[n] = m
        ids[n] = m['ids'].astype(str)
        labs[n] = m['labels'].astype(np.int64)
        chk('%s: token shape (n,256,1024) fp16' % n,
            a.ndim == 3 and a.shape[1] == 256 and a.shape[2] == 1024 and a.dtype == np.float16,
            '%s %s' % (a.shape, a.dtype))
        chk('%s: len(ids)==len(labels)==n' % n,
            len(ids[n]) == len(labs[n]) == a.shape[0],
            '%d %d %d' % (len(ids[n]), len(labs[n]), a.shape[0]))
        chk('%s: unique ids' % n, len(set(ids[n].tolist())) == len(ids[n]))
        chk('%s: labels within 0..6, all 7 present' % n,
            labs[n].min() == 0 and labs[n].max() == 6 and len(set(labs[n].tolist())) == 7,
            str(np.bincount(labs[n], minlength=7).tolist()))
        # touch first and last token row to prove the npy is readable through the memmap
        _ = float(np.asarray(a[0, 0, :4], dtype=np.float32).sum())
        _ = float(np.asarray(a[-1, -1, :4], dtype=np.float32).sum())
        # kdef_source_train has no ViT-L global cache: its support globals are taken from
        # kdef_test rows mapped by id (that is what phase5_confirm_vitl14_v2.py does).
        if n != 'kdef_source_train':
            g = hdr(L / ('%s.npz' % n), mmap=False)
            glob[n] = g
            chk('%s: global views (n,R,768)' % n,
                g['views'].ndim == 3 and g['views'].shape[0] == a.shape[0]
                and g['views'].shape[2] == 768, '%s' % (g['views'].shape,))
            chk('%s: features ids == token ids' % n,
                np.array_equal(g['ids'].astype(str), ids[n]),
                'first=%s' % ids[n][0])
            chk('%s: features labels == token labels' % n,
                np.array_equal(g['labels'].astype(np.int64), labs[n]))

    split = np.load(L / 'linear_probe_split_seed0.npz', allow_pickle=False)
    tr = np.asarray(split['train_indices'], np.int64)
    va = np.asarray(split['validation_indices'], np.int64)
    chk('linear_probe_split_seed0: disjoint train/val inside fer2013_train',
        len(np.intersect1d(tr, va)) == 0 and tr.max() < 28709 and va.max() < 28709
        and len(tr) + len(va) == 28709, 'train=%d val=%d' % (len(tr), len(va)))

    pw = np.load(L / 'linear_probe_train80_seed0.npz', allow_pickle=False)
    chk('linear_probe_train80: weights (7,768) bias (7,)',
        pw['weights'].shape == (7, 768) and pw['bias'].shape == (7,),
        '%s %s' % (pw['weights'].shape, pw['bias'].shape))
    text = np.load(L / 'text_prototypes.npz', allow_pickle=False)['prototypes'].astype(np.float32)
    chk('text_prototypes: (7,768)', text.shape == (7, 768), '%s' % (text.shape,))
    chk('text_prototypes: no zero rows', np.all(np.linalg.norm(text, axis=1) > 1e-6))

    kpos = {x: i for i, x in enumerate(ids['kdef_test'].tolist())}
    for tag in ('kdef_source_train', 'kdef_source_val', 'kdef_source_test'):
        z = hdr(KS / ('%s.npz' % tag), mmap=False)
        zid = z['ids'].astype(str)
        missing = [x for x in zid.tolist() if x not in kpos]
        chk('KS/%s: %d ids all present in kdef_test cache' % (tag, len(zid)),
            len(missing) == 0, ('missing=%s' % missing[:3]) if missing else '')
        if not missing:
            sel = np.asarray([kpos[x] for x in zid.tolist()], np.int64)
            chk('KS/%s: labels match kdef_test labels at mapped rows' % tag,
                np.array_equal(z['labels'].astype(np.int64), labs['kdef_test'][sel]))
            chk('KS/%s: global views are 512-d (ViT-L globals come from kdef_test rows)' % tag,
                z['views'].shape[2] == 512, '%s' % (z['views'].shape,))
        chk('KS/%s: labels within 0..6' % tag,
            int(z['labels'].min()) == 0 and int(z['labels'].max()) == 6)

    zm = meta['kdef_source_train']
    chk('token_cache kdef_source_train rows == KS ids order',
        np.array_equal(zm['ids'].astype(str),
                       hdr(KS / 'kdef_source_train.npz', mmap=False)['ids'].astype(str)))
    chk('token_cache kdef_source_train labels == KS labels',
        np.array_equal(zm['labels'].astype(np.int64),
                       hdr(KS / 'kdef_source_train.npz', mmap=False)['labels'].astype(np.int64)))
    chk('KS global features are 512-d (not used by the ViT-L scripts)',
        hdr(KS / 'kdef_source_train.npz', mmap=False)['views'].shape[2] == 512,
        '%s' % (hdr(KS / 'kdef_source_train.npz', mmap=False)['views'].shape,))

    torch_ok, torch_detail = False, ''
    try:
        import torch
        torch_ok = torch.cuda.is_available()
        torch_detail = '%s | %s | free %.2f GB / %.2f GB' % (
            torch.__version__, torch.cuda.get_device_name(0),
            *[x / 1e9 for x in (torch.cuda.mem_get_info()[0], torch.cuda.mem_get_info()[1])])
    except Exception as exc:  # pragma: no cover
        torch_detail = repr(exc)
    chk('cuda available', torch_ok, torch_detail)

    rep['target_label_counts'] = {n: np.bincount(labs[n], minlength=7).tolist() for n in names}
    rep['probe_split'] = {'train': int(len(tr)), 'val': int(len(va))}
    rep['elapsed_seconds'] = round(time.perf_counter() - t0, 1)
    rep['pass'] = len(rep['fail']) == 0
    (OUT / 'phase5_preflight_vitl14.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print('\nPASS' if rep['pass'] else '\nFAIL %s' % rep['fail'])
    return 0 if rep['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
