"""Append the Phase 3 (KDEF-source confirmation) record to MANIFEST.json."""
import hashlib, json, sys
from pathlib import Path

OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
TC = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027/token_cache_vitb16')
KS = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027/features_kdef_source')
FEA = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027/features')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    m = json.loads((OUT / 'MANIFEST.json').read_text(encoding='utf-8'))
    m['phase'] = 3
    new_files = ['phase3_confirm_crossed.py', 'phase3_attribution.py', 'phase3_extract_kdef_source.py',
                 'phase3_kdef_source_confirm.py', 'phase3_extra_controls.py',
                 'phase3_kdef_source_extraction.json', 'phase3_kdef_source.json',
                 'phase3_kdef_source_cells.csv', 'phase3_extra_controls.json',
                 'phase3_crossed_cells.csv', 'phase3_summary.json', 'phase3_attribution_cells.csv',
                 'phase3_attribution.json', 'phase3_report.md',
                 'model_T2x_lr0.001.pt', 'model_T2x_lr0.003.pt', 'model_GHead_lr0.003.pt']
    hashes = {f: sha(OUT / f) for f in new_files if (OUT / f).exists()}
    m['output_hashes'].update(hashes)
    m['stage3_kdef_source_confirmation'] = {
        'commands': [
            {'cmd': 'python phase3_extract_kdef_source.py', 'exit_code': 0,
             'seconds': 18.9, 'peak_vram_GB': 1.7, 'out': 'kdef_source_train_tokens_fp16.npz (2056,196,768) fp16',
             'sha256_merged': '1f1da19ec1f56795c5c9d542a59262d80be14e1b50a2e9f60c7f27a56724d627',
             'merged_GB': 0.62, 'ids_equal_source': True, 'labels_equal_source': True},
            {'cmd': 'python phase3_kdef_source_confirm.py', 'exit_code': 0, 'seconds': 23.9,
             'out': 'phase3_kdef_source_cells.csv, phase3_kdef_source.json'},
            {'cmd': 'python phase3_extra_controls.py', 'exit_code': 0, 'seconds': 92.9,
             'out': 'phase3_extra_controls.json, model_T2x_lr0.001.pt, model_T2x_lr0.003.pt'},
        ],
        'seeds': [0, 1, 2, 3, 4],
        'canonical': {'logit_scale': 100.0, 'affinity': 'max', 'beta': 256.0, 'prior_clip': 1.0,
                      'pfb_lambda': 0.2, 'min_prior_samples': 32},
        'read_list': [str(TC / 'fer2013_train_tokens_fp16.npz'), str(TC / 'fer2013_test_tokens_fp16.npz'),
                      str(TC / 'ckplus_test_tokens_fp16.npz'), str(TC / 'kdef_test_tokens_fp16.npz'),
                      str(TC / 'kdef_source_train_tokens_fp16.npz'),
                      str(FEA / 'fer2013_train.npz'), str(FEA / 'fer2013_test.npz'),
                      str(FEA / 'ckplus_test.npz'), str(FEA / 'text_prototypes.npz'),
                      str(FEA / 'linear_probe_train80_seed0.npz'), str(FEA / 'linear_probe_split_seed0.npz'),
                      str(KS / 'kdef_source_train.npz'), str(KS / 'kdef_source_test.npz'),
                      str(KS / 'linear_probe_kdef_source_seed0.npz')],
        'gate_set_A_crossed': {'T1_cells_ge_best': 50, 'n_cells': 50,
                               'T1_minus_GHead': [0.0625, 0.0561, 0.0687], 'pass': True},
        'gate_set_B_kdef_source': {'T1_cells_ge_best': 10, 'n_cells': 15, 'frac': 2.0 / 3.0,
                                   'T1_minus_GHead': [0.0521, 0.0418, 0.0628], 'pass': True},
        'verdict': 'T1 passes both halves of the gate on both confirmation sets; T2/T3/T4 do not',
        'net_token_contribution': {'set_A_pp': 6.25, 'set_B_pp': 5.21,
                                   'head_share_of_surface_gain': 'about 45% (set A)'},
    }
    m['boundary'].update({'vitl14_tokens_read': False, 'locked_test_read': False,
                          'old_2250_read': False, 'main_rev_tex_modified': False,
                          'submission_or_contact': False, 'new_hparam_grid': False})
    m['unfinished'] = ['ViT-L/14 token confirmation left blank (patch-token weights absent locally)',
                       'T2 3x3 vs 1x1 control completed on the crossed grid only; KDEF-source variant not run',
                       'workspace scratch copies _p3.py/_a1.py retained (identical to phase3_confirm_crossed.py/phase3_attribution.py)']
    (OUT / 'MANIFEST.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    print('manifest updated, files hashed:', len(hashes))


if __name__ == '__main__':
    main()
