"""Append the phase 6 (RAF-DB gate) record to MANIFEST.json and hash every phase 6 artifact.

Updated in place, as phase3/phase4/phase5 and the diagnostic did; no earlier output is rewritten.
"""
import hashlib, json
from pathlib import Path

OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
DATA = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    m = json.loads((OUT / 'MANIFEST.json').read_text(encoding='utf-8'))
    files = sorted(p.name for p in OUT.glob('phase6_rafdb_*') if p.is_file())
    m['output_hashes'].update({f: sha(OUT / f) for f in files
                               if f != 'phase6_rafdb_update_manifest.py'})
    m['output_hashes']['phase6_rafdb_update_manifest.py'] = sha(OUT / 'phase6_rafdb_update_manifest.py')
    gate = json.loads((OUT / 'phase6_rafdb_gate.json').read_text(encoding='utf-8'))
    ans = json.loads((OUT / 'phase6_rafdb_answer.json').read_text(encoding='utf-8'))

    m['phase6_rafdb_confirmation'] = {
        'status': 'not run - step-1 availability gate failed',
        'H1': ans['H1'],
        'H1_statement': ans['H1_statement'],
        'pre_registration': {
            'file': 'phase6_rafdb_preregistration.md',
            'written_before': 'any RAF-DB token extraction or evaluation',
            'arms': ['TAP', 'CLIP-Adapter', 'equal-capacity global head',
                     'training-free max(PFB, support memory)'],
            'decision_rule': 'H1 holds iff the paired 95% CI of TAP - CLIP-Adapter excludes 0 and is '
                             'positive, AND the point estimate exceeds the token-arm between-seed sd '
                             '0.0116',
            'seeds': [0, 1, 2, 3, 4],
            'protocol': 'identical to phase 3/4: s=100, affinity=max, beta=256, prior_clip=1.0, '
                        'PFB lambda=0.2, min_prior_samples=32; RAF-DB full target plus ratios '
                        '0/1/2/5/10 with the phase 3 ratio construction',
            'modified_after_the_fact': False,
        },
        'gate': {
            'script': 'phase6_rafdb_gate.py',
            'exit_code': 1,
            'exit_code_meaning': 'the audit ran to completion and reports a failed gate (not a crash)',
            'n_checks': len(gate['checks']),
            'n_failed': len(gate['failed_checks']),
            'failed_checks': gate['failed_checks'],
            'elapsed_seconds': gate['elapsed_seconds'],
            'evidence': {
                'provenance': 'downloads/rafdb_processed/PROVENANCE.md: Kaggle re-upload '
                              'fahadullaha/facial-emotion-recognition-dataset, "preprocessed FER2013 + '
                              'RAF-DB JPEG files, 49,779 images", status QUARANTINED, "do not use these '
                              'images for manuscript results"',
                'official_split_file_found': gate['official_split_files_found'],
                'per_class_counts': gate['per_class_counts'],
                'total_images': gate['total_images'],
                'image_format': {'modes': gate['modes'], 'dimensions': gate['dimensions_top']},
                'identical_counts_among_classes': ['angry', 'disgust', 'fear', 'surprise'],
                'duplicates_within_candidate_after_48x48': gate['duplicate_images_in_tree'],
                'overlap_with_FER2013_source': gate['overlap_with_fer2013'],
                'ratio_feasibility': gate['ratio_feasibility'],
                'class_index_mapping': gate['project_class_index_mapping'],
            },
            'why_disqualifying': [
                'the copy is a quarantined third-party re-upload whose own note forbids manuscript use',
                'no official split/label file exists; the RAF-DB subset cannot be separated from the '
                'FER2013 images the same bundle declares it contains',
                'four classes share an identical count (5,920), a re-balancing/augmentation signature',
                'only ratios 0 and 1 of the pre-registered 0/1/2/5/10 grid are realisable, so the '
                'pre-registered protocol cannot be executed as written',
            ],
            'not_disqualifying_but_recorded': [
                '0 of 49,779 candidate images duplicate another candidate after 48x48 normalisation',
                '0 of 49,779 candidate images are pixel-identical to any of the 35,887 FER2013 images '
                'used as source; this rules out only the crudest contamination route',
            ],
        },
        'commands': [
            {'cmd': 'python phase6_rafdb_gate.py', 'exit_code': 1, 'seconds': gate['elapsed_seconds'],
             'log': 'phase6_rafdb_gate.log',
             'note': 'read-only audit; decodes 49,779 candidate and 35,887 FER2013 images'},
            {'cmd': 'python phase6_rafdb_report.py', 'exit_code': 0, 'seconds': 0.4,
             'note': 'composes the report from the frozen pre-registration and the gate json'},
        ],
        'not_run': ['ViT-B/16 token extraction on RAF-DB', 'ViT-L/14 token extraction on RAF-DB',
                    'training of any arm on RAF-DB', 'per-cell evaluation', 'paired bootstrap',
                    'H1 verdict'],
        'deliverables_empty_by_design': {
            'phase6_rafdb_cells.csv': 'header with the intended schema, zero rows',
            'phase6_rafdb_bootstrap.json': 'status "not_run"',
        },
        'rafdb_consumed': False,
        'boundary': {'training_or_selection_on_rafdb': False, 'rafdb_written_to': False,
                     'locked_test_read': False, 'old_2250_read': False,
                     'set_A_or_B_read': False, 'main_rev_tex_modified': False,
                     'submission_or_contact': False, 'paid': False},
        'read_list': [
            str(DATA / 'downloads' / 'rafdb_processed' / 'PROVENANCE.md'),
            str(DATA / 'downloads' / 'rafdb_processed' / 'processed_data')
            + '  (49,779 jpg files across 7 class folders)',
            str(DATA / 'raw' / 'fer2013' / 'train') + '  (28,709 png files)',
            str(DATA / 'raw' / 'fer2013' / 'test') + '  (7,178 png files)',
            str(DATA / 'token_cache_vitb16' / 'fer2013_train_tokens_fp16.npz')
            + '  (labels and ids only, to read the project class-index mapping)',
        ],
        'next_step': 'obtain the official RAF-DB via the official request process, re-verify the failed '
                     'gate checks against it, then run phase6_rafdb_preregistration.md unchanged (data '
                     'path is the only permitted change)',
    }
    m['unfinished'] = list(dict.fromkeys(list(m.get('unfinished', [])) + [
        'Phase 6 RAF-DB confirmation: not run, blocked by the availability gate. H1 remains open and '
        'RAF-DB remains unconsumed; the blocker is acquisition of the official RAF-DB (the only local '
        'copy is a quarantined Kaggle re-upload that mixes FER2013 and RAF-DB content).',
    ]))
    (OUT / 'MANIFEST.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    print('manifest phase 6 written; phase6 files hashed: %d; total hashed: %d'
          % (len(files), len(m['output_hashes'])))


if __name__ == '__main__':
    main()
