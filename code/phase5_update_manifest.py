"""Append the Phase 5 (ViT-L/14) record to MANIFEST.json and hash every Phase 5 artifact.

MANIFEST.json is the artifact's own manifest and is updated in place, exactly as
phase3_update_manifest.py and phase4_update_manifest.py did; no phase 1-4 output is rewritten.
"""
import hashlib, json
from pathlib import Path

OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
DATA = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
MODELS = Path('D:/ResearchVault/99system/models/open_clip')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


FILES = ['phase5_download_vitl14.py', 'phase5_download_vitl14_v2.py', 'phase5_download.json',
         'phase5_download_v1_attempts.json', 'phase5_extract_tokens_vitl14.py',
         'phase5_extract_tokens_vitl14_v2.py', 'phase5_extraction_vitl14.json',
         'phase5_preflight_vitl14.py', 'phase5_preflight_vitl14.json',
         'phase5_train_vitl14.py', 'phase5_train_vitl14_v2.py', 'phase5_train_vitl14_v2.log',
         'phase5_confirm_vitl14.py', 'phase5_confirm_vitl14_v2.py', 'phase5_confirm_vitl14_v3.py',
         'phase5_confirm_vitl14_v2.log', 'phase5_confirm_vitl14_v3.log',
         'phase5_training_vitl14.json', 'phase5_vitl14.json', 'phase5_vitl14_cells.csv',
         'phase5_vitl14_kdef_source_cells.csv', 'phase5_bootstrap_vitl14.csv',
         'phase5_linear_probe_kdef_source_vitl14.npz',
         'phase5_report.py', 'phase5_report.md', 'phase5_update_manifest.py',
         'phase5_tap_vitl14_lr0.001.pt', 'phase5_tap_vitl14_lr0.003.pt',
         'phase5_ghead_vitl14_lr0.001.pt', 'phase5_ghead_vitl14_lr0.003.pt',
         'phase5_clipadapter_vitl14_lr0.001.pt', 'phase5_clipadapter_vitl14_lr0.003.pt',
         'phase5_tap_vitl14_lr0.001.interrupted_20261006T1938.pt']


def main():
    m = json.loads((OUT / 'MANIFEST.json').read_text(encoding='utf-8'))
    m['phase'] = 5
    missing = [f for f in FILES if not (OUT / f).exists()]
    m['output_hashes'].update({f: sha(OUT / f) for f in FILES if (OUT / f).exists()})
    train = json.loads((OUT / 'phase5_training_vitl14.json').read_text(encoding='utf-8'))
    conf = json.loads((OUT / 'phase5_vitl14.json').read_text(encoding='utf-8'))
    cache = json.loads((OUT / 'phase5_extraction_vitl14.json').read_text(encoding='utf-8'))
    dl = json.loads((OUT / 'phase5_download.json').read_text(encoding='utf-8'))
    pre = json.loads((OUT / 'phase5_preflight_vitl14.json').read_text(encoding='utf-8'))

    m['stage5_vitl14'] = {
        'executed': True,
        'weights': {
            'path': str(MODELS / 'ViT-L-14.pt'),
            'sha256': dl['actual_sha256'],
            'bytes': dl['bytes'],
            'source': dl['url'],
            'verified': bool(dl['sha256_matches']),
        },
        'token_cache': {
            'root': str(DATA / 'token_cache_vitl14'),
            'layout': 'per-dataset .npy (memory-mapped) + meta npz + shards',
            'per_image': '256 x 1024 fp16',
            'total_GB': cache['total_npy_GB'],
            'datasets': {k: {'n': v['n'], 'shape': v['tokens_shape'], 'dtype': v['dtype'],
                             'sha256_npy': v['sha256_npy'], 'shards': v['shards']}
                         for k, v in cache['datasets'].items()},
            'extraction_record': 'phase5_extraction_vitl14.json',
        },
        'commands': [
            {'cmd': 'python phase5_preflight_vitl14.py', 'exit_code': 0, 'seconds': 8.8,
             'note': '%d input-consistency checks, all pass' % len(pre['checks'])},
            {'cmd': 'python phase5_train_vitl14_v2.py', 'exit_code': 0,
             'seconds': train['elapsed_seconds'],
             'note': 'source-only, 8 epochs, batch 128, Adam, lr grid {1e-3, 3e-3}; log phase5_train_vitl14_v2.log'},
            {'cmd': 'python phase5_confirm_vitl14_v3.py', 'exit_code': 0, 'seconds': 84.3,
             'note': 'Set A 50 cells + Set B 15 cells; shared-unit paired bootstrap 2000; log phase5_confirm_vitl14_v3.log'},
            {'cmd': 'python phase5_report.py', 'exit_code': 0, 'seconds': 0.4,
             'note': 'writes phase5_report.md and phase5_bootstrap_vitl14.csv from the stored JSON/CSV'},
        ],
        'failed_then_fixed': [
            {'cmd': 'python phase5_confirm_vitl14_v2.py', 'exit_code': 1,
             'error': 'RuntimeError: mat1 and mat2 shapes cannot be multiplied (441x512 and 768x7)',
             'cause': 'the KDEF-source held-out diagnostic read the 512-d ViT-B views in '
                      'data/features_kdef_source while the retrained probe expects the 768-d ViT-L rows '
                      'of kdef_test that the same ids map to',
             'resolution': 'phase5_confirm_vitl14_v3.py maps the validation ids into the same 768-d rows '
                           'the support uses; v2 is left unmodified. Affects only the reported held-out '
                           'source UAR of that probe; no prediction, cell, reference column or bootstrap draw '
                           'depends on it. Set A had been computed but nothing had been written yet.'},
        ],
        'environment_incident': {
            'observed': 'a second process tree had already launched phase5_train_vitl14_v2.py at '
                        '2026-10-06T19:49:41 against the same output paths, before this run started at '
                        '19:51:04',
            'detected_by': 'Get-CimInstance Win32_Process command lines, created at 19:49:41, 83 s before '
                           'this run; pid 4356/34252 under pwsh parent 37132',
            'action': 'the duplicate tree was stopped at about 19:57 after it had completed only its TAP '
                      'lr 1e-3 checkpoint; the run described here has START/EXIT log capture and exit code 0',
            'retained': 'the superseded 19:38 run is kept as '
                        'phase5_tap_vitl14_lr0.001.interrupted_20261006T1938.pt',
        },
        'canonical': {'logit_scale': 100.0, 'affinity': 'max', 'beta': 256.0, 'prior_clip': 1.0,
                      'pfb_lambda': 0.2, 'min_prior_samples': 32},
        'seeds': [0, 1, 2, 3, 4],
        'grid': {'lr': [1e-3, 3e-3], 'epochs': 8, 'batch': 128, 'optimizer': 'Adam'},
        'source_selection': train['selected'],
        'global_head_width': train['global_head_width'],
        'setA': {'n_cells': conf['n_cells_A'], 'mean_uar': conf['setA_mean_uar'],
                 'cells_TAP_beats': conf['setA_cells_TAP_beats'],
                 'bootstrap': conf['bootstrap_setA']},
        'setB': {'n_cells': conf['n_cells_B'], 'mean_uar': conf['setB_mean_uar'],
                 'per_target': conf['setB_per_target'],
                 'cells_TAP_beats': conf['setB_cells_TAP_beats'],
                 'bootstrap': conf['bootstrap_setB'],
                 'kdef_source_probe_val_uar': conf['kdef_source_probe_val_uar']},
        'n_boot': conf['n_boot'],
        'answer': 'The two ViT-B/16 quantities do NOT replicate at ViT-L/14. TAP - CLIP-Adapter = '
                  '-1.96pp [-2.59,-1.31] on Set A and -1.37pp [-2.38,-0.39] on Set B (sign reversed, CI '
                  'excludes zero against TAP). TAP - equal-capacity global head = -1.10pp [-1.82,-0.38] on '
                  'Set A (sign reversed) and -0.07pp [-1.08,+0.92] on Set B (wash). TAP still beats '
                  'best-of-two-families by +9.03pp / +6.18pp. Source-only probe-val selection already '
                  'ranked CLIP-Adapter first at ViT-L/14 (0.6799 vs TAP 0.6754).',
        'read_list': [
            'D:/ResearchVault/99system/models/open_clip/ViT-L-14.pt',
            str(DATA / 'token_cache_vitl14' / 'fer2013_train_tokens_fp16.npy'),
            str(DATA / 'token_cache_vitl14' / 'fer2013_test_tokens_fp16.npy'),
            str(DATA / 'token_cache_vitl14' / 'ckplus_test_tokens_fp16.npy'),
            str(DATA / 'token_cache_vitl14' / 'kdef_test_tokens_fp16.npy'),
            str(DATA / 'token_cache_vitl14' / 'kdef_source_train_tokens_fp16.npy'),
            str(DATA / 'token_cache_vitl14' / 'fer2013_train_meta.npz'),
            str(DATA / 'token_cache_vitl14' / 'fer2013_test_meta.npz'),
            str(DATA / 'token_cache_vitl14' / 'ckplus_test_meta.npz'),
            str(DATA / 'token_cache_vitl14' / 'kdef_test_meta.npz'),
            str(DATA / 'token_cache_vitl14' / 'kdef_source_train_meta.npz'),
            str(DATA / 'features_vitl14' / 'fer2013_train.npz'),
            str(DATA / 'features_vitl14' / 'fer2013_test.npz'),
            str(DATA / 'features_vitl14' / 'ckplus_test.npz'),
            str(DATA / 'features_vitl14' / 'kdef_test.npz'),
            str(DATA / 'features_vitl14' / 'text_prototypes.npz'),
            str(DATA / 'features_vitl14' / 'linear_probe_train80_seed0.npz'),
            str(DATA / 'features_vitl14' / 'linear_probe_split_seed0.npz'),
            str(DATA / 'features_kdef_source' / 'kdef_source_train.npz'),
            str(DATA / 'features_kdef_source' / 'kdef_source_val.npz'),
            str(DATA / 'features_kdef_source' / 'kdef_source_test.npz'),
        ],
        'boundary': {'target_labels_in_training': False, 'target_labels_in_selection': False,
                     'locked_test_read': False, 'old_2250_read': False,
                     'main_rev_tex_modified': False, 'submission_or_contact': False,
                     'paid': False, 'manuscript_transferred': False},
    }
    m['unfinished'] = [
        'The ViT-L/14 confirmation returned a null replication of the ViT-B/16 TAP margins; the '
        'encoder-dependence question is now open (phase5_report.md sections 5-6).',
        'ViT-L/14 variants of the Phase 4 published adapters (APE-style, Tip-Adapter-F, the 512-d linear '
        'probe) were not retrained; only the four required reference columns were produced.',
        'The frozen-feature diagnostic that would separate "the ViT-L global feature is already strong" '
        'from "the TAP head is underfitted" has not been run (recommendation 2 in the report).',
        'Any post-hoc ViT-L TAP variant needs a fresh untouched confirmation set: Set A and Set B have now '
        'been evaluated at two encoders.',
        'KDEF-source variant of the T2 3x3-vs-1x1 control not run.',
        'workspace scratch copies _p3.py/_a1.py retained (identical to phase3_confirm_crossed.py/'
        'phase3_attribution.py)',
    ]
    if missing:
        m['stage5_vitl14']['missing_files'] = missing
    (OUT / 'MANIFEST.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    print('manifest phase 5 written; hashed files: %d; missing: %s' % (len(m['output_hashes']), missing))


if __name__ == '__main__':
    main()
