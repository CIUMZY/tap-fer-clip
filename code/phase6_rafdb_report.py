"""Compose the phase 6 deliverable from the frozen pre-registration and the gate audit.

The pre-registration is embedded verbatim, read from phase6_rafdb_preregistration.md, so the text
in the report cannot drift from the file that was written before any RAF-DB model output existed.
Writes: phase6_rafdb_report.md, phase6_rafdb_cells.csv (schema only, no cells),
phase6_rafdb_bootstrap.json (explicit not-run record), phase6_rafdb_answer.json.
"""
import csv, json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')


def main():
    prereg = (OUT / 'phase6_rafdb_preregistration.md').read_text(encoding='utf-8')
    gate = json.loads((OUT / 'phase6_rafdb_gate.json').read_text(encoding='utf-8'))
    counts = gate['per_class_counts']
    ratio = gate['ratio_feasibility']
    md = []

    def A(s=''):
        md.append(s)

    A('# Phase 6 report - RAF-DB confirmation of H1: **GATE FAILED, experiment not run**')
    A()
    A('Executor: DeepSeek (deepseek-flash). Date: 2026-10-06. Interpreter')
    A('D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe.')
    A()
    A('**H1 has no verdict.** The pre-registered confirmation was blocked by its own step-1 gate: the')
    A('only RAF-DB copy on this machine cannot support the pre-registered protocol. No RAF-DB token was')
    A('extracted, no arm was trained on RAF-DB, no cell and no bootstrap interval exists. RAF-DB remains')
    A('an unconsumed confirmation set - nothing in this phase was selected, tuned or thresholded on it.')
    A()
    A('## 0. Pre-registration (verbatim; written before any RAF-DB output existed)')
    A()
    A('Frozen file: phase6_rafdb_preregistration.md (embedded here byte-for-byte).')
    A()
    A('~~~markdown')
    md.extend(prereg.rstrip('\n').split('\n'))
    A('~~~')
    A()
    A('## 1. Gate verdict')
    A()
    A('Step 1 of the task requires verifying that the local RAF-DB copy is usable - original images with')
    A('labels, a known class mapping, per-class counts, and no contamination of the source training set -')
    A('and requires stopping honestly if that cannot be established. Of eight checks, four failed:')
    A()
    A('| check | result | evidence |')
    A('|---|---|---|')
    for c in gate['checks']:
        A('| ' + c['check'] + ' | ' + ('pass' if c['ok'] else '**GATE FAIL**') + ' | '
          + (c['detail'].replace('|', '/') if c['detail'] else '') + ' |')
    A()
    A('Verdict: **' + gate['verdict'] + '**.')
    A()
    A('## 2. Evidence')
    A()
    A('**2.1 The copy declares itself unusable.** downloads/rafdb_processed/PROVENANCE.md records the')
    A('source as the Kaggle re-upload fahadullaha/facial-emotion-recognition-dataset, states the content')
    A('is "preprocessed FER2013 + RAF-DB JPEG files, 49,779 images", marks the status QUARANTINED, and')
    A('says: "Do not use these images for manuscript results until the redistribution rights and original')
    A('RAF-DB license are verified. Prefer the official RAF-DB request process for the paper."')
    A()
    A('**2.2 No official split or label file exists.** A recursive search of D:/ResearchVault for')
    A('list_patition_label.txt and variants returns nothing. The RAF-DB train/test split and its labels')
    A('cannot be recovered from this copy, and the images are re-indexed (angry_00000.jpg), not the')
    A('official train_00001_aligned.jpg / test_00001_aligned.jpg naming, so the RAF-DB subset cannot be')
    A('separated from the FER2013 images the same bundle contains.')
    A()
    A('**2.3 Class counts carry a re-balancing signature.**')
    A()
    A('| class | images | project index |')
    A('|---|---|---|')
    for c in ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'):
        idx = gate['project_class_index_mapping'].get(c, ['-'])
        A('| ' + c + ' | ' + str(counts[c]) + ' | ' + str(idx[0]) + ' |')
    A('| **total** | **' + str(gate['total_images']) + '** | |')
    A()
    A('angry, disgust, fear and surprise all hold exactly 5,920 images. A natural facial-expression')
    A('distribution does not have four identical class counts; this is the signature of oversampling or')
    A('synthesis applied to the minority classes, which would make any accuracy measured on this copy')
    A('uninterpretable. All ' + str(gate['total_images']) + ' images are 96x96 RGB, so they are not a')
    A('faithful copy of either RAF-DB (100x100 aligned colour) or FER2013 (48x48 grayscale) either.')
    A()
    A('**2.4 The pre-registered protocol is literally inexecutable on this copy.** The phase 3 ratio')
    A('construction takes base = smallest class count and gives the majority class base*ratio samples.')
    A('Here base = ' + str(ratio['base_minority_count']) + ' and the majority class is '
      + ratio['majority_class'] + ' with ' + str(counts[ratio['majority_class']]) + ' images, so:')
    A()
    A('| ratio | majority samples needed | available | realisable |')
    A('|---|---|---|---|')
    for r in ('0', '1', '2', '5', '10'):
        d = ratio['ratios'][r]
        A('| ' + r + ' | ' + str(d['needed_majority']) + ' | ' + str(d['available_majority']) + ' | '
          + ('yes' if d['feasible'] else '**no**') + ' |')
    A()
    A('Only ratios 0 and 1 can be built; 2, 5 and 10 are blocked. The pre-registration explicitly requires')
    A('reporting a ratio as skipped rather than approximating it, but a run over ratios {0,1} only would no')
    A('longer be the pre-registered experiment, so it was not run.')
    A()
    A('**2.5 Two checks did pass, and they are worth recording honestly.**')
    A()
    A('- No duplicated images inside the candidate set: 0 of ' + str(gate['total_images'])
      + ' images repeat after 48x48')
    A('  normalisation. So the identical class counts are not explained by exact duplicates within this')
    A('  copy; they would be explained by augmentation or by images drawn from other sources.')
    A('- No detected contamination of the source training set: 0 of ' + str(gate['total_images'])
      + ' candidate images are')
    A('  pixel-identical (48x48 normalised) to any of the ' + str(gate['fer2013_images_hashed'])
      + ' FER2013 images this project trains on.')
    A('  This rules out the crudest contamination route only. It does not rule out that the bundle mixes')
    A('  FER2013 and RAF-DB content as its own provenance note states, because our FER2013 copy is the')
    A('  official 48x48 grayscale release while this tree is a 96x96 colour re-render of unnamed images.')
    A('  The decisive point is that the source membership of these 49,779 images is not determinable from')
    A('  anything on disk, which is disqualifying for a confirmation set on its own.')
    A()
    A('**2.6 Class space mapping.** From the existing FER2013 token cache, the project label order is')
    A('angry 0, disgust 1, fear 2, happy 3, sad 4, surprise 5, neutral 6. The candidate tree uses the same')
    A('seven folder names, so the class space itself is not the blocker.')
    A()
    A('## 3. What was not run, and what that means for H1')
    A()
    A('- No ViT-B/16 or ViT-L/14 token extraction on RAF-DB.')
    A('- No training, no evaluation, no per-cell results, no bootstrap intervals.')
    A('- **H1 (TAP beats CLIP-Adapter at ViT-B/16 on RAF-DB) is therefore neither confirmed nor')
    A('  refuted. It stays open.**')
    A('- RAF-DB was not used for any selection, tuning or threshold choice, so it is still an unconsumed')
    A('  confirmation set in the sense that matters for the pre-registration.')
    A()
    A('Two deliverables requested for the confirmation run are delivered as explicit empty records rather')
    A('than omitted, so their absence cannot be mistaken for a missing file: phase6_rafdb_cells.csv carries')
    A('the intended schema and zero rows, and phase6_rafdb_bootstrap.json records status "not_run".')
    A()
    A('## 4. Conditions under which this exact pre-registered test can still be run')
    A()
    A('The pre-registration file is unmodified and remains valid. To execute it once and only once:')
    A()
    A('1. Obtain the official RAF-DB through the official request process (the same route the quarantine')
    A('   note recommends), or any copy that ships the official list_patition_label.txt with the')
    A('   official train/test split and its official filenames.')
    A('2. Re-verify the four gate checks that failed here, plus the class counts, from that copy - in')
    A('   particular that the ratio grid is realisable, which must be read off the official labels rather')
    A('   than assumed (the official basic-emotion set is far more imbalanced than this copy, so the')
    A('   minority-class count matters).')
    A('3. Run the frozen pre-registration unchanged: same hypothesis, same four arms, same decision rule')
    A('   (paired 95% CI excludes 0 and the margin exceeds the token-arm seed sd of 0.0116), same seeds')
    A('   0-4, same protocol constants. The only thing that changes is the data path.')
    A()
    A('If no official copy is obtained, the honest end state is the current one: H1 open, RAF-DB')
    A('unconsumed, and the encoder-dependence question answered only by the source-domain diagnostic of')
    A('the previous phase.')
    A()
    A('## 5. Scope, boundaries, unfinished')
    A()
    A('- All RAF-DB access in this phase was read-only enumeration and decoding for the gate audit;')
    A('  nothing was written inside downloads/rafdb_processed.')
    A('- The locked test set, the old 2250 holdout, Set A and Set B were not read. No main*.tex file was')
    A('  modified; nothing was submitted, no editor was contacted, nothing was paid, nothing transferred.')
    A('- Unfinished: the confirmation itself, for the reason above; the official RAF-DB acquisition is')
    A('  the only blocking precondition. The audit also did not attempt to determine which of the 49,779')
    A('  images originate from RAF-DB, because no on-disk information supports that determination.')
    A('- Runtime: gate audit ' + str(gate['elapsed_seconds']) + ' s ('
      + str(gate['decode_seconds']) + ' s to decode both image sets).')
    A()
    (OUT / 'phase6_rafdb_report.md').write_text('\n'.join(md), encoding='utf-8')

    with (OUT / 'phase6_rafdb_cells.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['cell', 'target', 'ratio', 'seed', 'n', 'pfb', 'memory', 'best_family',
                    'TAP', 'GHead', 'CLIPAdapter', 'status'])
    (OUT / 'phase6_rafdb_bootstrap.json').write_text(json.dumps({
        'status': 'not_run',
        'reason': 'phase 6 gate failed: the only local RAF-DB copy is a quarantined Kaggle re-upload '
                  'that mixes FER2013 and RAF-DB content, ships no official split/label file, shows a '
                  'class re-balancing signature, and supports only ratios 0 and 1 of the pre-registered '
                  '0/1/2/5/10 grid',
        'plan': 'paired shared-unit bootstrap, 2000 draws, hierarchical over seeds 0-4, shared query '
                'indices, identical to phases 3-5',
        'n_boot': 2000,
        'pairs': {},
    }, indent=1), encoding='utf-8')
    (OUT / 'phase6_rafdb_answer.json').write_text(json.dumps({
        'H1': 'no verdict - gate failed before the experiment could run',
        'H1_statement': 'on frozen CLIP ViT-B/16, TAP beats CLIP-Adapter on the RAF-DB target',
        'gate': gate['verdict'],
        'failed_checks': gate['failed_checks'],
        'rafdb_consumed': False,
        'arms_trained': [], 'cells': 0, 'bootstrap_intervals': 0,
        'viL14_secondary': 'not attempted (gate failed before extraction)',
        'next_step': 'obtain the official RAF-DB (official request process) and re-run the unchanged '
                     'pre-registration',
    }, indent=1), encoding='utf-8')
    print('wrote phase6_rafdb_report.md (%d lines), cells csv (header only), bootstrap json, answer json'
          % len(md))


if __name__ == '__main__':
    main()
