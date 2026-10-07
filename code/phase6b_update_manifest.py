"""Append the phase 6b (corruption confirmation) record to MANIFEST.json and hash its artifacts.

Updated in place, as phases 3-6 did; no earlier output is rewritten.
"""
import hashlib, json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def dir_inventory(p):
    if not p.exists():
        return None
    fs = [f for f in sorted(p.rglob('*')) if f.is_file()]
    return {'path': str(p), 'n_files': len(fs), 'GB': round(sum(f.stat().st_size for f in fs) / 1e9, 2)}


def main():
    m = json.loads((OUT / 'MANIFEST.json').read_text(encoding='utf-8'))
    files = sorted(p.name for p in OUT.glob('phase6b_*') if p.is_file())
    m['output_hashes'].update({f: sha(OUT / f) for f in files
                               if f != 'phase6b_update_manifest.py'})
    m['output_hashes']['phase6b_update_manifest.py'] = sha(OUT / 'phase6b_update_manifest.py')
    b16 = json.loads((OUT / 'phase6b_confirm_B16.json').read_text(encoding='utf-8'))
    l14 = json.loads((OUT / 'phase6b_confirm_L14.json').read_text(encoding='utf-8'))
    ver = json.loads((OUT / 'phase6b_verify_extraction.json').read_text(encoding='utf-8'))
    val = json.loads((OUT / 'phase6b_validation_l14_tokens.json').read_text(encoding='utf-8'))

    m['phase6b_corruption_confirmation'] = {
        'status': 'run and complete',
        'axis': 'graded synthetic corruption of the FER2013 test set (within-domain robustness), '
                '4 corruption types x 3 severities, 7178 images per condition',
        'conditions': b16['conditions'],
        'hypothesis_H2': 'on frozen CLIP ViT-B/16, TAP beats CLIP-Adapter on the 12-condition '
                         'corruption set',
        'decision_rule': 'paired 95% CI of TAP - CLIP-Adapter excludes 0 and is positive, AND the '
                         'point estimate exceeds the token-arm between-seed sd (0.0116 at ViT-B/16, '
                         '0.0134 at ViT-L/14)',
        'seeds': b16['seeds'],
        'arms': {'TAP': 'frozen, phase 2 / phase 5 checkpoints',
                 'CLIP-Adapter': 'frozen, phase 4 / phase 5 checkpoints',
                 'GHead': 'frozen equal-capacity global head, phase 3 / phase 5 checkpoints',
                 'reference': 'training-free max(PFB, support memory) with the FER2013 source support'},
        'arm_params': b16['arm_params'],
        'result': {
            'B16_primary_H2_holds': b16['decision']['H2_holds'],
            'B16_point_TAP_minus_CLIPAdapter': b16['decision']['point_TAP_minus_CLIPAdapter'],
            'B16_ci95': b16['decision']['ci95'],
            'B16_aggregate_uar': {k: v['mean'] for k, v in b16['aggregate'].items()},
            'B16_conditions_TAP_above_CLIPAdapter': sum(
                1 for c in b16['conditions'] if b16['cells'][c]['uar']['TAP']
                > b16['cells'][c]['uar']['CLIPAdapter']),
            'L14_secondary_point': l14['decision']['point_TAP_minus_CLIPAdapter'],
            'L14_secondary_ci95': l14['decision']['ci95'],
            'L14_secondary_criteria_met': l14['decision']['H2_holds'],
            'L14_role': 'secondary; does not enter the H2 decision',
            'L14_aggregate_uar': {k: v['mean'] for k, v in l14['aggregate'].items()},
            'contrast': 'under this within-domain degradation axis TAP leads CLIP-Adapter at BOTH '
                        'encoders (+1.81pp / +3.20pp), which is the opposite of the phase-4/5 '
                        'cross-database ViT-L/14 result (-1.37pp); the phase-5 reversal is therefore '
                        'specific to the cross-database setting, not a general property of ViT-L/14',
        },
        'commands': [
            {'cmd': 'python phase6b_inventory.py', 'exit_code': 0, 'note': 'step-1 inventory'},
            {'cmd': 'python phase6b_provenance.py', 'exit_code': 0, 'note': 'per-condition provenance'},
            {'cmd': 'python phase6b_inventory_report.py', 'exit_code': 0},
            {'cmd': 'python phase6b_smoke.py', 'exit_code': 0,
             'note': '8-row convention check, clean-cache cosine 1.000000 both encoders'},
            {'cmd': 'python phase6b_extract.py --encoder B16', 'exit_code': 0, 'seconds': 724.0,
             'note': '12 conditions; validation gate PASS (clean 1.000000, blur_0p8 1.000000)'},
            {'cmd': 'python phase6b_extract.py --encoder L14', 'exit_code': 1,
             'note': 'first attempt: all 12 conditions written, then the validation step OOMed '
                     '(one 1024-image forward needs ~4 GB for the ViT-L/14 MLP intermediate)'},
            {'cmd': 'python phase6b_extract.py --encoder L14 (rerun)', 'exit_code': 0,
             'note': 'reused the 12 verified conditions; validation gate PASS on the clean cache'},
            {'cmd': 'python phase6b_validate_l14_tokens.py', 'exit_code': 0,
             'note': 'repairs the skipped L/14 token check: clean globals and tokens cosine 1.000000'},
            {'cmd': 'python phase6b_verify_extraction.py', 'exit_code': 0,
             'note': '24/24 condition caches byte-complete, aligned and recomputation-verified'},
            {'cmd': 'python phase6b_confirm.py --encoder B16', 'exit_code': 0, 'seconds': 100.7,
             'note': 'clean cross-check max |diff| 0.0000 vs stored Set B, then 12 conditions + 2000-draw bootstrap'},
            {'cmd': 'python phase6b_confirm.py --encoder L14', 'exit_code': 0, 'seconds': 121.0,
             'note': 'same, secondary'},
            {'cmd': 'python phase6b_report.py', 'exit_code': 0},
        ],
        'extraction_validation': {
            'B16_gate': 'clean globals cosine 1.000000 (512 rows) and blur_0p8 vs the stored '
                        'features/shift_sweep cache cosine 1.000000 (all 7178 rows)',
            'L14_gate_actually_verified': ['clean globals cosine 1.000000 on 512 rows',
                                           'clean patch tokens cosine 1.000000 on 512 rows '
                                           '(standalone repair run)',
                                           '24/24 condition caches recomputation-verified: tokens '
                                           'within 0.12-1.5 fp16 ULP, globals cosine 1.000000, '
                                           'byte-complete, ids/labels aligned'],
            'L14_not_verifiable': 'agreement with a pre-existing corrupted L/14 reference: none '
                                  'exists (features/shift_sweep/*.npz are 512-d ViT-B/16 caches)',
            'L14_evidence_class': 'recomputation- and clean-cache-verified, not cross-artifact '
                                  'verified; acceptable because L/14 is secondary and does not enter H2',
            'pipeline_cross_check': {
                'B16': 'TAP 0.6323 / GHead 0.6083 / CLIP-Adapter 0.6163 reproduce the stored '
                       'phase-4 Set B fer2013_test numbers with max |diff| 0.0000',
                'L14': 'TAP 0.6836 / GHead 0.6828 / CLIP-Adapter 0.6916 reproduce the stored '
                       'phase-5 Set B numbers with max |diff| 0.0000',
                'reference_columns': 'PFB and support memory here use the pre-registered FER2013 '
                                     'source support (Set A convention), not the KDEF-source support '
                                     'of the stored Set B reference columns'},
            'verification_record': {'phase6b_verify_extraction.json': ver['n_verified'],
                                    'total': ver['n_total'],
                                    'L14_token_check': val['pass']},
        },
        'bootstrap': {'n_boot': 2000,
                      'design': 'seeds (0-4) and conditions (12) resampled with replacement; the 7178 '
                                'query indices are resampled within each cell and shared across arms',
                      'honest_scope': 'because the arms are frozen, seeds 0-4 change only the PFB '
                                      'online prior order; the interval covers query and condition '
                                      'sampling, and the training-seed variance is carried by '
                                      'decision criterion 2'},
        'derived_artifacts': {k: dir_inventory(D / k) for k in
                              ('token_cache_b16_confirm', 'token_cache_l14_confirm',
                               'features_confirm_shift')},
        'boundary': {'cross_database_claim': False, 'locked_test_read': False, 'old_2250_read': False,
                     'set_A_or_B_read': False, 'rafdb_tree_read': False,
                     'main_rev_tex_modified': False, 'submission_or_contact': False, 'paid': False,
                     'existing_files_overwritten': False},
        'read_list': [
            'manifests/fer2013_<condition>.csv (12) and manifests/fer2013_test.csv',
            'raw/fer2013_shift_sweep/<condition>/<class>/*.png (12 x 7178)',
            'features/shift_sweep/<condition>.npz (12, ViT-B/16 validation only)',
            'features/fer2013_test.npz, features/fer2013_train.npz, features/text_prototypes.npz, '
            'features/linear_probe_train80_seed0.npz, features/linear_probe_split_seed0.npz',
            'features_vitl14/ (same five files)',
            'token_cache_vitb16/fer2013_test_tokens_fp16.npz, token_cache_vitl14/fer2013_test_tokens_fp16.npy',
            'token_cache_vitb16/fer2013_train_tokens_fp16.npz, token_cache_vitl14/fer2013_train_meta.npz',
            'models/open_clip/ViT-B-16.pt, models/open_clip/ViT-L-14.pt',
            'frozen arms: model_T1_lr0.003.pt, model_GHead_lr0.003.pt, '
            'phase4_clipadapter_lr0.001.pt, phase5_tap_vitl14_lr0.001.pt, '
            'phase5_ghead_vitl14_lr0.001.pt, phase5_clipadapter_vitl14_lr0.001.pt',
        ],
        'weights_sha256': {v['weights'].split('/')[-1]: v['weights_sha256']
                           for v in (json.loads((OUT / 'phase6b_extraction_B16.json')
                                                .read_text(encoding='utf-8')),
                                     json.loads((OUT / 'phase6b_extraction_L14.json')
                                                .read_text(encoding='utf-8')))},
        'unfinished': [
            'RAF-DB remains unconsumed but locally unavailable (phase 6 gate).',
            'The corruption conditions are now a consumed confirmation set; any further variant '
            'designed with knowledge of these results needs a fresh untouched set.',
            'No pre-existing corrupted ViT-L/14 reference exists, so the L/14 half is verified by '
            'recomputation and clean-cache agreement only (documented in the report).',
        ],
    }
    m['unfinished'] = list(dict.fromkeys(list(m.get('unfinished', [])) + [
        'Phase 6b corruption confirmation ran: H2 holds at ViT-B/16 (+1.81pp, CI [+0.75,+2.92]) and '
        'the secondary ViT-L/14 comparison also favours TAP (+3.20pp). The corruption conditions are '
        'now consumed; further variants need a fresh untouched confirmation set.',
    ]))
    (OUT / 'MANIFEST.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    print('manifest phase 6b written; phase6b files hashed: %d; total hashed: %d'
          % (len(files), len(m['output_hashes'])))


if __name__ == '__main__':
    main()
