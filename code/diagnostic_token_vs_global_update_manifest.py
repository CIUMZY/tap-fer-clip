"""Append the token-vs-global diagnostic record to MANIFEST.json and hash every diagnostic artifact.

MANIFEST.json is the artifact's own manifest and is updated in place, exactly as
phase3/phase4/phase5_update_manifest.py did; no earlier output is rewritten and no phase output
directory outside this one is touched.
"""
import hashlib, json
from pathlib import Path

OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
DATA = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
DERIVED = DATA / 'token_cache_vitb16' / 'fer2013_train_tokens_fp16.npy'


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    m = json.loads((OUT / 'MANIFEST.json').read_text(encoding='utf-8'))
    files = sorted(p.name for p in OUT.glob('diagnostic_*') if p.is_file())
    files = [f for f in files if f != 'diagnostic_token_vs_global_update_manifest.py']
    files.append('diagnostic_token_vs_global_update_manifest.py')
    m['output_hashes'].update({f: sha(OUT / f) for f in files if (OUT / f).exists()
                               and f != 'diagnostic_token_vs_global_update_manifest.py'})
    m['output_hashes']['diagnostic_token_vs_global_update_manifest.py'] = \
        sha(OUT / 'diagnostic_token_vs_global_update_manifest.py')
    discarded = sorted(p.name for p in (OUT / 'discarded_buggy_run').glob('*') if p.is_file())

    b16 = json.loads((OUT / 'diagnostic_token_vs_global_B16.json').read_text(encoding='utf-8'))
    l14 = json.loads((OUT / 'diagnostic_token_vs_global_L14.json').read_text(encoding='utf-8'))
    mb16 = json.loads((OUT / 'diagnostic_token_vs_global_mlp_B16.json').read_text(encoding='utf-8'))
    ml14 = json.loads((OUT / 'diagnostic_token_vs_global_mlp_L14.json').read_text(encoding='utf-8'))
    extraction = json.loads((OUT / 'phase5_extraction_vitl14.json').read_text(encoding='utf-8'))
    stage1 = m.get('stage1_extraction', {}).get('datasets', {})

    m['diagnostic_token_vs_global'] = {
        'question': 'Why does the ViT-B/16 TAP advantage over an equal-capacity global head and over '
                    'CLIP-Adapter fail to replicate at ViT-L/14?  (i) global feature already saturated, '
                    'or (ii) the convex token pooling has no direct path from the global feature.',
        'status': 'source-domain diagnostic only; not a new method and not a confirmatory claim',
        'representations': {
            'R1_global': 'frozen L2-normalised global feature (512-d ViT-B/16, 768-d ViT-L/14)',
            'R2_mean_raw': 'frozen uniform mean of the patch tokens (headline R2)',
            'R2_mean_normed': 'frozen L2-normalised mean tokens (robustness)',
            'R2_mean_of_normed': 'frozen mean of L2-normalised tokens (robustness)',
            'R3_attn_pool': 'TAP head, identical to phase 2 T1, trained jointly',
            'R3refit_frozen_attn': 'learned pooling frozen, linear head refit',
            'R4a_concat_raw': 'frozen concat(global, mean tokens) + linear probe',
            'R4a_concat_normed': 'frozen concat(global, L2-normalised mean tokens)',
            'R4a_concat_normtok': 'frozen concat(global, mean of L2-normalised tokens)',
            'R4b_attn_plus_global': 'concat(global, attention-pooled vector), trained jointly',
            'MLP extension': 'same representations with Linear(d,256)-Tanh-Linear(256,256)-Tanh-Linear(256,7), '
                             'i.e. the phase 5 global-head form (198,919 params on 512-d, 264,455 on 768-d)',
        },
        'training': {'optimizer': 'Adam', 'epochs': 8, 'batch': 128, 'lr_grid': [1e-3, 3e-3],
                     'seeds': [0, 1, 2], 'selection': 'probe-val UAR on the FER2013 source split only'},
        'prior_rule': {'lambda': 0.2, 'min_prior_samples': 32, 'prior_clip': 1.0},
        'source_split': {'file': 'features/linear_probe_split_seed0.npz',
                         'note': 'identical for both encoders, verified element-wise',
                         'n_train': b16['n_train'], 'n_val': b16['n_val']},
        'commands': [
            {'cmd': 'python diagnostic_token_vs_global_run.py --encoder B16', 'exit_code': 0,
             'seconds': b16['elapsed_seconds'], 'log': 'diagnostic_token_vs_global_B16.log'},
            {'cmd': 'python diagnostic_token_vs_global_run.py --encoder L14', 'exit_code': 0,
             'seconds': l14['elapsed_seconds'], 'log': 'diagnostic_token_vs_global_L14.log'},
            {'cmd': 'python diagnostic_token_vs_global_mlp.py --encoder B16', 'exit_code': 0,
             'seconds': mb16['elapsed_seconds'], 'log': 'diagnostic_token_vs_global_mlp.log'},
            {'cmd': 'python diagnostic_token_vs_global_mlp.py --encoder L14', 'exit_code': 0,
             'seconds': ml14['elapsed_seconds'], 'log': 'diagnostic_token_vs_global_mlp.log'},
            {'cmd': 'python diagnostic_token_vs_global_report.py', 'exit_code': 0, 'seconds': 0.5,
             'note': 'writes diagnostic_token_vs_global_report.md and the merged summary json/csv'},
            {'cmd': 'python diagnostic_debug_r3.py', 'exit_code': 0, 'seconds': 167.0,
             'note': 'instrumented probe that diagnosed the evaluator ordering bug; output in '
                     'diagnostic_debug_r3.out'},
        ],
        'aborted_runs_kept': [
            {'log': 'diagnostic_token_vs_global_B16_aborted_shardreader.log',
             'reason': 'shard-granular reader copied about 6.8 GB per batch; stopped after the frozen '
                       'probes and one partial R3 run, before any result file was written'},
            {'log': 'diagnostic_token_vs_global_B16_aborted_unseeded_init.log',
             'reason': 'streaming model was constructed before seeding'},
            {'log': 'diagnostic_token_vs_global_B16_aborted_eval_order_bug.log',
             'reason': 'evaluator wrote predictions back in ascending row order while the class-grouped '
                       'row order misaligned them with labels (chance-level UAR)',
             'affected': 'streaming arms only (R3, R4b, R3refit); frozen probes R1/R2/R4a were unaffected'},
            {'log': 'diagnostic_token_vs_global_mlp_first_attempt_indexerror.log',
             'reason': 'pooled attention matrix was indexed again with global row ids (IndexError)'},
        ],
        'discarded_artifacts': {'dir': 'discarded_buggy_run', 'files': discarded,
                                'reason': 'checkpoints and pooled matrices from the pre-fix streaming runs'},
        'derived_inputs': {
            'vitb16_train_token_npy': {'path': str(DERIVED),
                                       'shape': [28709, 196, 768], 'dtype': 'float16',
                                       'sha256': sha(DERIVED),
                                       'why': 'a random 128-row batch touches about 44 of the 57 shards, so '
                                              'the shard reader copied about 6.8 GB per step',
                                       'verification': '12 random rows equal to their source shard rows; '
                                                       'frozen R2/R4a results bit-identical to the '
                                                       'shard-based run'},
            'pooled_token_val_matrices': sorted(p.name for p in OUT.glob('diagnostic_pooled_tokens_*')),
        },
        'pipeline_cross_checks': [
            'R1 linear ViT-B/16 lr 3e-3 seed 0 = 0.5691, matching the stored phase 4 linear probe',
            'MLP head on the frozen global feature = 0.6201 (ViT-B/16, lr 3e-3 seed mean) vs stored '
            'phase 5 global head 0.6144, and 0.6714 (ViT-L/14, lr 1e-3 seed mean) vs stored 0.6720',
            'R3 at ViT-L/14 = 0.6720 (lr 1e-3 seed 0) and 0.6861 (seed 1), bracketing the stored '
            'phase 5 TAP value 0.6754',
        ],
        'answers': {
            'Q1_R2_vs_R1_at_L14_linear': {
                'R2': 0.6513, 'R1': 0.6172, 'delta_pp': 3.41,
                'verdict': 'token pooling does add linearly available information at L/14; hypothesis (i) '
                           'is not supported in the form "mean-pooled tokens cannot beat the global '
                           'feature"'},
            'Q1_R2_MLP_vs_R1_MLP_at_L14': {
                'R1_MLP': 0.6714, 'R2_MLP': 0.6656, 'delta_pp': -0.58,
                'verdict': 'under a nonlinear readout the global feature is at least as good; this is the '
                           'narrow form of (i) that the data do support'},
            'Q2_ordering': 'not reversed in general; only the nonlinear-head comparison flips '
                           '(tokens +0.80pp at ViT-B/16, -0.58pp at ViT-L/14). Linear token-over-global '
                           'margin is +3.36pp/+3.41pp, attention over mean is +2.00pp/+2.12pp at the '
                           'selected lrs (about zero at lr 3e-3 on both encoders).',
            'Q3_residual': 'R4a is +0.77pp/+0.49pp over R2 and R4b is -0.96pp/-0.46pp vs R3: no form of '
                           'the direct path beats the token-only arm',
            'Q4_verdict': '(ii) is excluded; (i) is supported only in its narrow nonlinear-readout form. '
                          'The driver of the phase 5 reversal is the global-feature route under a '
                          'nonlinear readout, plus seed noise of the same order as the matched-head '
                          'margin.',
        },
        'seed_noise_floor': 'R3 spans 0.6066-0.6280 at ViT-B/16 and 0.6592-0.6861 at ViT-L/14 across '
                            'seeds 0/1/2; frozen R1/R2/R4a are stable to about 0.2-1.8pp',
        'boundary': {'target_labels_used': False, 'confirmation_sets_touched': False,
                     'locked_test_read': False, 'old_2250_read': False, 'main_rev_tex_modified': False,
                     'submission_or_contact': False, 'paid': False, 'manuscript_transferred': False},
        'follow_up_discipline': 'A variant designed from this diagnostic needs a fresh untouched set; '
                                'Set A and Set B have now been evaluated at two encoders. Ranked '
                                'candidates: RAF-DB, corruption sweep, FER+ 1:1 / 5:1.',
        'read_list': [
            str(DATA / 'features' / 'linear_probe_split_seed0.npz'),
            str(DATA / 'features' / 'fer2013_train.npz'),
            str(DATA / 'features_vitl14' / 'linear_probe_split_seed0.npz'),
            str(DATA / 'features_vitl14' / 'fer2013_train.npz'),
            str(DATA / 'token_cache_vitb16' / 'fer2013_train_tokens_fp16.npz'),
            str(DATA / 'token_cache_vitb16' / 'fer2013_train' / 'shard_%04d.npz (57 shards)'),
            str(DATA / 'token_cache_vitl14' / 'fer2013_train_tokens_fp16.npy'),
            str(DATA / 'token_cache_vitl14' / 'fer2013_train_meta.npz'),
            str(DERIVED),
        ],
        'token_cache_hashes_referenced': {
            'vitb16_fer2013_train_merged_sha256': stage1.get('fer2013_train', {}).get('sha256_merged'),
            'vitl14_fer2013_train_sha256': extraction['datasets']['fer2013_train']['sha256_npy'],
        },
    }
    extra = [
        'Diagnostic follow-up that was not run: an R4 variant with a *nonlinear* head on '
        'concat(global, token pooling) (CLIP-Adapter-style route plus token pooling); the two R4 forms '
        'tested bound the linear-head version of hypothesis (ii) only.',
        'Diagnostic follow-up that was not run: corruption sweep, RAF-DB, FER+ 1:1 and 5:1.',
    ]
    m['unfinished'] = list(dict.fromkeys(list(m.get('unfinished', [])) + extra))
    (OUT / 'MANIFEST.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    print('manifest updated: %d hashed files, %d diagnostic files, derived cache sha256 %s'
          % (len(m['output_hashes']), len(files), m['diagnostic_token_vs_global']['derived_inputs']
             ['vitb16_train_token_npy']['sha256'][:16]))


if __name__ == '__main__':
    main()
