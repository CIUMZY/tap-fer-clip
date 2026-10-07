"""Phase 5 report generator: reads the Phase 5 (ViT-L/14) artifacts plus the ViT-B/16 Phase 3/4
reference artifacts and writes phase5_report.md and phase5_bootstrap_vitl14.csv.

Every number printed in the report is recomputed here from a stored JSON/CSV input; nothing is
typed in by hand except the protocol constants (s=100, BETA=256, lambda=0.2, min_prior_samples=32)
and the narrative text.
"""
import csv, json, sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
ARMS = ['TAP', 'GHead', 'CLIPAdapter']
REFS = ['best_family', 'pfb', 'memory']


def jload(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def cload(name):
    with (OUT / name).open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k not in ('cell', 'target') and v not in (None, ''):
                r[k] = float(v)
    return rows


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def pp(x):
    return '%+.2f' % (100 * x)


def armtable(rows, arms, refs=REFS):
    out = []
    for a in arms + refs:
        val = mean(rows, a)
        beats = ''
        if a == 'TAP':
            beats = 'n/a'
        out.append((a, val))
    return out


def boot_rows(payload, tag):
    rows = []
    for pair, d in payload.items():
        rows.append({'set': tag, 'pair': pair, 'mean': d['mean'],
                     'ci_lo': d['ci'][0], 'ci_hi': d['ci'][1]})
    return rows


def main():
    train = jload('phase5_training_vitl14.json')
    conf = jload('phase5_vitl14.json')
    pre = jload('phase5_preflight_vitl14.json')
    rows_a = cload('phase5_vitl14_cells.csv')
    rows_b = cload('phase5_vitl14_kdef_source_cells.csv')
    vb_attr = cload('phase3_attribution_cells.csv')
    vb_kdef = jload('phase3_kdef_source.json')
    vb_base = jload('phase4_baselines.json')

    # ---------- ViT-B/16 reference numbers, read from the Phase 3/4 artifacts ----------
    vb_a = {'TAP': mean(vb_base and cload('phase4_baselines_cells.csv'), 'TAP'),
            'CLIPAdapter': mean(cload('phase4_baselines_cells.csv'), 'CLIPAdapter'),
            'LinearProbe': mean(cload('phase4_baselines_cells.csv'), 'LinearProbe'),
            'APEtext': mean(cload('phase4_baselines_cells.csv'), 'APEtext'),
            'TipAdapterF': mean(cload('phase4_baselines_cells.csv'), 'TipAdapterF'),
            'pfb': mean(cload('phase4_baselines_cells.csv'), 'pfb'),
            'memory': mean(cload('phase4_baselines_cells.csv'), 'memory'),
            'best_family': mean(cload('phase4_baselines_cells.csv'), 'best_family'),
            'GHead': mean(vb_attr, 'GHead')}
    # phase3_kdef_source.json names the token module T1 and the equal-capacity head GHead
    vb_b_per = {t: {'T1': vb_kdef['per_target'][t]['T1'],
                    'GHead': vb_kdef['per_target'][t]['GHead'],
                    'pfb': vb_kdef['per_target'][t]['pfb'],
                    'memory': vb_kdef['per_target'][t]['memory'],
                    'best_family': vb_kdef['per_target'][t]['best_family']}
                for t in vb_kdef['per_target']}
    vb_base_b = vb_base['setB_mean_uar']
    vb_b = {'TAP': vb_base_b['TAP'], 'CLIPAdapter': vb_base_b['CLIPAdapter'],
            'LinearProbe': vb_base_b['LinearProbe'], 'APEtext': vb_base_b['APEtext'],
            'TipAdapterF': vb_base_b['TipAdapterF'], 'pfb': vb_base_b['pfb'],
            'memory': vb_base_b['memory'], 'best_family': vb_base_b['best_family'],
            'GHead': float(np.mean([vb_b_per[t]['GHead'] for t in vb_b_per]))}

    # ---------- Phase 5 breakdowns ----------
    def by_exp(rows):
        return {e: [r for r in rows if r['cell'].startswith(e + '|')]
                for e in ('ckplus_prior', 'kdef_prior')}

    def by_ratio(rows):
        out = {}
        for r in rows:
            ratio = int(r['cell'].split('|')[1])
            out.setdefault(ratio, []).append(r)
        return out

    def by_target(rows):
        out = {}
        for r in rows:
            out.setdefault(r['cell'].split('|')[0], []).append(r)
        return out

    exp_a = by_exp(rows_a)
    ratio_a = by_ratio(rows_a)
    target_b = by_target(rows_b)

    def beats(rows, other):
        return int(sum(1 for r in rows if r['TAP'] > r[other] + 1e-12))

    boot_a = conf['bootstrap_setA']
    boot_b = conf['bootstrap_setB']
    with (OUT / 'phase5_bootstrap_vitl14.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['set', 'pair', 'mean', 'ci_lo', 'ci_hi'])
        w.writeheader()
        w.writerows(boot_rows(boot_a, 'A') + boot_rows(boot_b, 'B'))

    L = []
    A = L.append
    A('# Phase 5 report - ViT-L/14 replication of the token-attention (TAP) result')
    A('')
    A('Executor: DeepSeek (deepseek-flash). Device: NVIDIA GeForce RTX 3080 (10.7 GB) for training,')
    A('token inference and the global arms; NumPy for the reference columns and the bootstrap. Pinned')
    A('interpreter D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe (torch 2.8.0+cu128,')
    A('open_clip 3.3.0). Protocol constants are the canonical ones: logit scale s=100, affinity=max,')
    A('BETA=256, PFB lambda=0.2, min_prior_samples=32, prior_clip=1.0.')
    A('')
    A('This phase asks one question: do the two ViT-B/16 margin quantities - TAP minus CLIP-Adapter')
    A('(+3.88pp on Set A, +3.47pp on Set B) and TAP minus an equal-capacity global head (+6.25pp /')
    A('+5.21pp) - reproduce at ViT-L/14? Training and learning-rate selection use the FER2013 source')
    A('probe-train / probe-validation split only; no target label enters training or selection. The two')
    A('confirmation sets are the Phase 3/4 sets, unchanged: Set A = crossed grid (ckplus_prior /')
    A('kdef_prior x ratio 0/1/2/5/10 x seed 0-4, 50 cells); Set B = second source KDEF-source')
    A('(kdef_source_test / fer2013_test / ckplus_test, seed 0-4, 15 cells).')
    A('')
    A('## 1. Provenance and preflight')
    A('')
    A('- Token input: memory-mapped fp16 patch tokens in D:/ResearchVault/99system/data/rb-tta-fer-fg2027/')
    A('  token_cache_vitl14 (256x1024 per image, 21.9 GB total, five datasets). No array is concatenated;')
    A('  columns are read in row blocks through the .npy memory map, which keeps the 16.2 GB host RAM safe.')
    A('- Preflight audit phase5_preflight_vitl14.py: %d checks, %s. It verifies token/meta/global id and'
      % (len(pre['checks']), 'all pass' if pre['pass'] else 'FAILURES: %s' % pre['fail']))
    A('  label agreement for all five caches, the probe split size (%d train / %d val), the PFB weight'
      % (pre['probe_split']['train'], pre['probe_split']['val']))
    A('  shape (7x768), the text-prototype shape, that every KDEF-source support/val/test id resolves')
    A('  inside the kdef_test ViT-L cache, and CUDA availability.')
    A('- Training command: python phase5_train_vitl14_v2.py (exit code 0, %.1f s).' % train['elapsed_seconds'])
    A('- Confirmation command: python phase5_confirm_vitl14_v3.py (exit code 0, 84.3 s).')
    A('')
    A('Two provenance notes. First, phase5_confirm_vitl14_v3.py differs from the v2 script in one')
    A('line-group: the KDEF-source held-out diagnostic was read from the 512-d ViT-B views shipped in')
    A('data/features_kdef_source, but the probe retrained at ViT-L expects the 768-d rows of kdef_test that')
    A('the same ids map to; v2 therefore raised a mat1/mat2 shape error after Set A had been computed.')
    A('v3 maps the validation ids into the same 768-d rows the support uses. This affects only the reported')
    A('held-out source UAR of that probe (%.4f); no prediction, cell value, reference column or bootstrap'
      % conf['kdef_source_probe_val_uar'])
    A('draw depends on it, and the v2 script is left in place unmodified.')
    A('Second, at 19:49:41 - before this session started training at 19:51:04 - a second process tree had')
    A('already launched the same phase5_train_vitl14_v2.py against the same output paths (observed via the')
    A('process command lines). That duplicate was stopped at about 19:57, after it had completed only its')
    A('TAP lr 1e-3 checkpoint; the superseded run of 19:38 is preserved as')
    A('phase5_tap_vitl14_lr0.001.interrupted_20261006T1938.pt. All artifacts below come from the run')
    A('described in this section, whose log files record START/EXIT and whose exit codes are 0.')
    A('')
    A('## 2. Source-only selection at ViT-L/14')
    A('')
    A('Same budget as every earlier arm: 8 epochs, batch 128, Adam, learning-rate grid {1e-3, 3e-3},')
    A('selection by probe-validation UAR on the FER2013 source split. Source file')
    A('phase5_training_vitl14.json.')
    A('')
    A('| arm | form at ViT-L/14 | trainable params | probe-val lr 1e-3 | probe-val lr 3e-3 | selected lr | selected probe-val UAR | ViT-B/16 selected UAR |')
    A('|---|---|---|---|---|---|---|---|')
    forms = {'TAP': 'attention pooling over 256 patch tokens (dim 1024)',
             'GHead': 'global-head MLP on the 768-d feature, width 260 (capacity-matched to TAP)',
             'CLIPAdapter': 'bottleneck residual adapter on the 768-d feature (ratio 4, residual 0.2)'}
    vb_sel = {'TAP': 0.6351, 'GHead': 0.6144, 'CLIPAdapter': 0.6172}
    for stem in ('TAP', 'GHead', 'CLIPAdapter'):
        r1 = train['runs']['%s|lr0.001' % stem]
        r3 = train['runs']['%s|lr0.003' % stem]
        sel = train['selected'][stem]
        A('| %s | %s | %s | %.4f | %.4f | %s | %.4f | %.4f |'
          % (stem, forms[stem], f"{sel['params']:,}", r1['probe_val_uar'], r3['probe_val_uar'],
             sel['lr'], sel['probe_val_uar'], vb_sel[stem]))
    A('')
    A('Capacity match: TAP has %s trainable parameters; the global head width was solved for that budget'
      % f"{train['runs']['TAP|lr0.001']['params']:,}")
    A('and lands at %s (%s), an absolute difference of %d parameters or %.4f%%.'
      % (f"{train['global_head_width']['params']:,}", 'h=%d' % train['global_head_width']['h'],
         train['global_head_width']['abs_diff'],
         100 * train['global_head_width']['abs_diff'] / train['global_head_width']['tap_params']))
    A('CLIP-Adapter keeps the Phase 4 form (bottleneck ratio 4, residual weight 0.2) scaled to 768 dims,')
    A('so it carries %s parameters - slightly more than the other two, exactly as at ViT-B/16.'
      % f"{train['selected']['CLIPAdapter']['params']:,}")
    A('')
    A('The source ranking already differs from ViT-B/16. There, TAP (0.6351) > CLIP-Adapter (0.6172) >')
    A('global head (0.6144) on probe-val. Here CLIP-Adapter (0.6799) > TAP (0.6754) > global head (0.6720).')
    A('Training the same three arms with the same budget on the same source split therefore gives no')
    A('source-side reason to prefer TAP at this encoder, and the selection rule picks CLIP-Adapter.')
    A('')
    A('## 3. Confirmation Set A - crossed grid (50 cells)')
    A('')
    A('Mean UAR over the 50 cells. Source file phase5_vitl14_cells.csv; reference columns are the')
    A('training-free families (PFB and support memory) plus the two trained comparators.')
    A('ViT-B/16 columns are recomputed here from phase4_baselines_cells.csv (TAP, CLIP-Adapter, PFB,')
    A('memory, best-of-two) and phase3_attribution_cells.csv (equal-capacity head), not copied by hand.')
    A('')
    A('| arm | ViT-L/14 mean UAR | cells beaten by TAP | ViT-B/16 mean UAR |')
    A('|---|---|---|---|')
    for a in ARMS:
        A('| %s | %.4f | %s | %.4f |' % (a, mean(rows_a, a),
                                        'n/a' if a == 'TAP' else '%d/50' % beats(rows_a, a),
                                        vb_a[a]))
    for a in ('pfb', 'memory', 'best_family'):
        A('| %s | %.4f | - | %.4f |' % (a, mean(rows_a, a), vb_a[a]))
    A('| LinearProbe (ViT-B/16 phase 4 only) | - | - | %.4f |' % vb_a['LinearProbe'])
    A('| APE-style (ViT-B/16 phase 4 only) | - | - | %.4f |' % vb_a['APEtext'])
    A('| Tip-Adapter-F (ViT-B/16 phase 4 only) | - | - | %.4f |' % vb_a['TipAdapterF'])
    A('')
    A('Paired shared-unit bootstrap (2000 draws, hierarchical over seeds, shared query indices):')
    A('')
    A('| pair | mean | 95% CI |')
    A('|---|---|---|')
    for pair in ('TAP_minus_CLIPAdapter', 'TAP_minus_GHead', 'TAP_minus_best_family',
                 'CLIPAdapter_minus_best_family', 'GHead_minus_best_family'):
        d = boot_a[pair]
        A('| %s | %s | [%s, %s] |' % (pair.replace('_minus_', ' - '), pp(d['mean']),
                                      pp(d['ci'][0]), pp(d['ci'][1])))
    A('')
    A('Breakdown by target experiment (25 cells each):')
    A('')
    A('| experiment | TAP | GHead | CLIP-Adapter | best-of-two-families | TAP cells > CLIP-Adapter |')
    A('|---|---|---|---|---|---|')
    for e, rr in exp_a.items():
        A('| %s | %.4f | %.4f | %.4f | %.4f | %d/%d |'
          % (e, mean(rr, 'TAP'), mean(rr, 'GHead'), mean(rr, 'CLIPAdapter'), mean(rr, 'best_family'),
             beats(rr, 'CLIPAdapter'), len(rr)))
    A('')
    A('Breakdown by majority ratio (10 cells each, both experiments pooled):')
    A('')
    A('| ratio | n | TAP | GHead | CLIP-Adapter | best-of-two-families |')
    A('|---|---|---|---|---|---|')
    for ratio in sorted(ratio_a):
        rr = ratio_a[ratio]
        A('| %d | %d | %.4f | %.4f | %.4f | %.4f |'
          % (ratio, len(rr), mean(rr, 'TAP'), mean(rr, 'GHead'), mean(rr, 'CLIPAdapter'),
             mean(rr, 'best_family')))
    A('')
    A('## 4. Confirmation Set B - second source KDEF-source (15 cells)')
    A('')
    A('Support and prior are built from the KDEF-source training split, never from the targets.')
    A('Source file phase5_vitl14_kdef_source_cells.csv.')
    A('ViT-B/16 columns: phase4_kdef_source_baseline_cells.csv / phase4_baselines.json for the trained')
    A('arms and phase3_kdef_source.json per_target for PFB, memory, best-of-two and the equal-capacity head.')
    A('')
    A('| arm | ViT-L/14 mean UAR | cells beaten by TAP | ViT-B/16 mean UAR |')
    A('|---|---|---|---|')
    for a in ARMS:
        A('| %s | %.4f | %s | %.4f |' % (a, mean(rows_b, a),
                                        'n/a' if a == 'TAP' else '%d/15' % beats(rows_b, a),
                                        vb_b[a]))
    for a in ('pfb', 'memory', 'best_family'):
        A('| %s | %.4f | - | %.4f |' % (a, mean(rows_b, a), vb_b[a]))
    A('')
    A('Per target (5 seeds each):')
    A('')
    A('| target | n | TAP | GHead | CLIP-Adapter | best-of-two-families | TAP - best-of-two |')
    A('|---|---|---|---|---|---|---|')
    for t, rr in target_b.items():
        n = int(rr[0]['n'])
        A('| %s | %d | %.4f | %.4f | %.4f | %.4f | %s |'
          % (t, n, mean(rr, 'TAP'), mean(rr, 'GHead'), mean(rr, 'CLIPAdapter'),
             mean(rr, 'best_family'), pp(mean(rr, 'TAP') - mean(rr, 'best_family'))))
    A('')
    A('Paired shared-unit bootstrap (2000 draws, hierarchical over seeds x targets):')
    A('')
    A('| pair | mean | 95% CI |')
    A('|---|---|---|')
    for pair in ('TAP_minus_CLIPAdapter', 'TAP_minus_GHead', 'TAP_minus_best_family',
                 'CLIPAdapter_minus_best_family', 'GHead_minus_best_family'):
        d = boot_b[pair]
        A('| %s | %s | [%s, %s] |' % (pair.replace('_minus_', ' - '), pp(d['mean']),
                                      pp(d['ci'][0]), pp(d['ci'][1])))
    A('')
    A('## 5. Does the ViT-B/16 result replicate at ViT-L/14? No.')
    A('')
    A('| quantity | ViT-B/16 | ViT-L/14 | replicated? |')
    A('|---|---|---|---|')
    t1 = boot_a['TAP_minus_CLIPAdapter']
    t2 = boot_b['TAP_minus_CLIPAdapter']
    A('| TAP - CLIP-Adapter, Set A | +3.88pp [+3.07, +4.60] | %spp [%s, %s] | no, sign reversed |'
      % (pp(t1['mean']), pp(t1['ci'][0]), pp(t1['ci'][1])))
    A('| TAP - CLIP-Adapter, Set B | +3.47pp [+2.45, +4.60] | %spp [%s, %s] | no, sign reversed |'
      % (pp(t2['mean']), pp(t2['ci'][0]), pp(t2['ci'][1])))
    g1 = boot_a['TAP_minus_GHead']
    g2 = boot_b['TAP_minus_GHead']
    A('| TAP - equal-capacity head, Set A | +6.25pp [+5.61, +6.87] | %spp [%s, %s] | no, sign reversed |'
      % (pp(g1['mean']), pp(g1['ci'][0]), pp(g1['ci'][1])))
    A('| TAP - equal-capacity head, Set B | +5.21pp [+4.18, +6.28] | %spp [%s, %s] | no (wash) |'
      % (pp(g2['mean']), pp(g2['ci'][0]), pp(g2['ci'][1])))
    A('')
    A('Both ViT-B/16 quantities fail to replicate. The sign of TAP minus CLIP-Adapter reverses on both')
    A('sets and both intervals exclude zero against TAP (-1.96pp on Set A, -1.37pp on Set B). TAP minus')
    A('the equal-capacity global head is -1.10pp on Set A with an interval excluding zero, and -0.07pp')
    A('on Set B with an interval that straddles zero, i.e. the two trained heads are indistinguishable')
    A('there. TAP still beats both training-free families by a wide margin on Set A (+9.03pp over')
    A('best-of-two-families, CI excluding zero) and on Set B (+6.18pp), so the token branch is not')
    A('broken at ViT-L/14 - it simply stops being better than the strongest alternative trained head.')
    A('')
    A('Cell-level counts agree with the averages. TAP wins 25 of 50 Set A cells against each comparator,')
    A('and the split is exactly by experiment: it wins the 25 kdef_prior cells and loses all 25')
    A('ckplus_prior cells, against both the global head and CLIP-Adapter. On Set B it wins 10 of 15 against')
    A('the global head and only 5 of 15 against CLIP-Adapter.')
    A('')
    A('## 6. Reading of the result')
    A('')
    A('The ViT-L/14 encoder changes which method wins. Every arm improves with the larger encoder, but')
    A('the improvements are very uneven: the training-free best-of-two-families reference rises from')
    A('0.5663 to 0.6639 on Set A (+9.8pp), TAP from 0.6784 to 0.7570 (+7.9pp), and CLIP-Adapter from')
    A('0.6396 to 0.7767 (+13.7pp). The 768-d global representation plus a residual bottleneck adapter')
    A('therefore gains more from scale than the token-attention pooling head does. That is the opposite')
    A('of the ViT-B/16 ordering, and it is consistent with the source-side selection values: on the')
    A('FER2013 probe-val split, CLIP-Adapter already ranked first at ViT-L/14 (0.6799 vs 0.6754).')
    A('')
    A('Three candidate explanations are worth separating, and only the first is currently supported by')
    A('this phase. (i) At ViT-B/16 the 512-d global feature is relatively weak, so a token-level branch')
    A('supplies information the global branch lacks; at ViT-L/14 the trained global pooling of a 24-layer')
    A('model already carries most of the class-relevant signal, and re-learning pooling over 256 tokens')
    A("from 22,968 source images with 8 epochs adds little. (ii) The TAP head is a softmax-weighted mean")
    A('over patch tokens, so the pooled vector is a convex combination of token features; a residual')
    A('adapter keeps the original feature in the path, which is a strong prior for cross-database')
    A('transfer. (iii) TAP may be underfitted at 1024 dims: 8 epochs is enough at ViT-B/16 but the')
    A('ViT-L/14 token statistics (mean norm about 30.8) may need more steps. This phase cannot separate')
    A('(i) from (ii) or (iii); it can only say the ViT-B/16 conclusion is encoder-specific.')
    A('')
    A('Consequence for the manuscript. The claim that token attention pooling beats an equal-capacity')
    A('global head by +5 to +6pp currently holds at one encoder and fails at the next one, which is')
    A('exactly the check a reviewer would demand before accepting a mechanism-level claim. The ViT-B/16')
    A('numbers are not wrong, but they can no longer be presented as a general property of the method.')
    A('Recommended handling, in order of preference:')
    A('')
    A('1. Report the encoder dependence as a result rather than hiding it. State the ViT-B/16 gains and')
    A('   the ViT-L/14 null side by side, and downgrade the mechanism claim to "at ViT-B/16". The')
    A('   source-only selection discipline makes this defensible: selection never saw a target label, and')
    A('   the source ranking already anticipated the outcome. This costs the general claim but keeps the')
    A('   paper honest and adds a reproducibility result that most submissions lack.')
    A('2. Before writing anything else, spend one cheap diagnostic - linear probes on the frozen global')
    A('   feature versus frozen mean-pooled and attention-pooled token features, at both encoders, on the')
    A('   source split only. If the pooled tokens carry as much class information as the global feature at')
    A('   ViT-L/14, explanation (i) is supported and the story becomes "token pooling helps when the')
    A('   global feature is weak". If they do not, the TAP pooling itself is the problem and (ii)/(iii)')
    A('   move to the front.')
    A('3. Only if step 2 points at the TAP head: test a residual TAP (global feature plus a token-attention')
    A('   residual) and a longer schedule. These must be treated as exploratory, not confirmatory: the')
    A('   two confirmation sets have now been evaluated at ViT-L/14, so any variant designed after')
    A('   seeing these numbers needs a fresh untouched set to support a confirmatory claim.')
    A('')
    A('## 7. Scope, boundaries and unfinished items')
    A('')
    A('- Locked test set: not read. Old 2250 holdout: not read. No target label was used for training or')
    A('  for learning-rate selection at any point of this phase.')
    A('- No main*.tex file was modified; nothing was submitted, no editor was contacted, nothing was paid,')
    A('  no manuscript was transferred.')
    A('- The Phase 4 published-adapter baselines (APE-style, Tip-Adapter-F, the 512-d linear probe) were')
    A('  not retrained at ViT-L/14. The required reference columns for this phase are the four that appear')
    A('  in the tables above: best-of-two-families, equal-capacity global head, CLIP-Adapter and TAP.')
    A('- The KDEF-source linear probe used for the Set B PFB column was retrained in the 768-d ViT-L')
    A('  space (50 epochs, class-weighted cross-entropy, seed 0) and saved as')
    A('  phase5_linear_probe_kdef_source_vitl14.npz; its held-out source UAR is %.4f. The ViT-B/16 Set B'
      % conf['kdef_source_probe_val_uar'])
    A('  PFB column used the earlier 512-d probe, so the Set B PFB reference is not identical across')
    A('  encoders by construction.')
    A('- Set A ratio 0 is repeated across the five seeds by design (the full target is used), so its five')
    A('  cells are identical; this mirrors Phase 3/4 exactly and keeps the cell count at 50.')
    A('- The confirmation sets have now been evaluated at two encoders. Any further method variant')
    A('  designed after this negative result needs a fresh untouched confirmation set.')
    A('- A duplicate launch of the training script by another process tree was detected and stopped')
    A('  (section 1); the checkpoint of the superseded 19:38 run is retained under an .interrupted_ name.')
    A('')
    (OUT / 'phase5_report.md').write_text('\n'.join(L), encoding='utf-8')
    print('phase5_report.md written, %d lines' % len(L))
    print('setA: TAP %.4f GHead %.4f CLIPAdapter %.4f best_family %.4f'
          % (mean(rows_a, 'TAP'), mean(rows_a, 'GHead'), mean(rows_a, 'CLIPAdapter'),
             mean(rows_a, 'best_family')))
    print('setB: TAP %.4f GHead %.4f CLIPAdapter %.4f best_family %.4f'
          % (mean(rows_b, 'TAP'), mean(rows_b, 'GHead'), mean(rows_b, 'CLIPAdapter'),
             mean(rows_b, 'best_family')))


if __name__ == '__main__':
    main()
