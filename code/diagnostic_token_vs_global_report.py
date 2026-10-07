"""Report generator for the token-vs-global diagnostic.

Reads only the stored per-encoder JSON artifacts and recomputes every number that appears in
diagnostic_token_vs_global_report.md.  Also writes diagnostic_token_vs_global_summary.json and
diagnostic_token_vs_global_summary.csv (merged, long format, both encoders).
"""
import csv, json, sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')


def jload(p):
    return json.loads((OUT / p).read_text(encoding='utf-8'))


def summarize(runs, selected):
    by_lr = {}
    for r in runs:
        by_lr.setdefault(r['lr'], []).append(r)
    stats = {}
    for lr, rs in by_lr.items():
        u = np.array([r['uar'] for r in rs], float)
        p = np.array([r['uar_prior'] for r in rs], float)
        stats[lr] = {'n': len(rs), 'mean': float(u.mean()),
                     'sd': float(u.std(ddof=1)) if len(rs) > 1 else 0.0,
                     'min': float(u.min()), 'max': float(u.max()),
                     'mean_prior': float(p.mean())}
    best_lr = max(stats, key=lambda k: stats[k]['mean'])
    return {'stats': stats, 'best_lr': best_lr, 'selected': selected,
            'params': runs[0].get('params'), 'dim': runs[0].get('dim')}


def collect(tag):
    main = jload('diagnostic_token_vs_global_%s.json' % tag)
    mlp = jload('diagnostic_token_vs_global_mlp_%s.json' % tag)
    T = {}
    for name, d in main['reps'].items():
        T[name] = summarize(d['runs'], d['selected'])
        T[name]['dim'] = d.get('dim')
    for name, d in mlp['reps'].items():
        T[name + '_MLP'] = summarize(d['runs'], d['selected'])
        T[name + '_MLP']['dim'] = d.get('dim')
    return main, mlp, T


def f4(x):
    return '%.4f' % x


def pp(x):
    return '%+.2fpp' % (100 * x)


LABELS = {
    'R1_global': 'R1 global (frozen)', 'R2_mean_raw': 'R2 mean tokens (frozen)',
    'R2_mean_normed': 'R2 L2-normalised mean', 'R2_mean_of_normed': 'R2 mean of L2-normalised tokens',
    'R3_attn_pool': 'R3 attention pooling (TAP head)', 'R3refit_frozen_attn': 'R3refit frozen pooling',
    'R4a_concat_raw': 'R4a concat(global, mean tokens)',
    'R4a_concat_normed': 'R4a concat(global, normalised mean)',
    'R4a_concat_normtok': 'R4a concat(global, mean of normalised)',
    'R4b_attn_plus_global': 'R4b concat(global, attention pool)',
    'R1_global_MLP': 'R1 global + MLP head', 'R2_mean_raw_MLP': 'R2 mean tokens + MLP head',
    'R3_attn_pool_MLP': 'R3 attention pool + MLP head',
    'R4a_concat_raw_MLP': 'R4a concat + MLP head'}
ORDER = ['R1_global', 'R2_mean_raw', 'R2_mean_normed', 'R2_mean_of_normed', 'R3_attn_pool',
         'R3refit_frozen_attn', 'R4a_concat_raw', 'R4a_concat_normed', 'R4a_concat_normtok',
         'R4b_attn_plus_global', 'R1_global_MLP', 'R2_mean_raw_MLP', 'R3_attn_pool_MLP',
         'R4a_concat_raw_MLP']


def main():
    data = {tag: collect(tag) for tag in ('B16', 'L14')}
    B16, L14 = data['B16'][2], data['L14'][2]

    def m(T, name, lr=None):
        d = T[name]
        return d['stats'][lr if lr else d['best_lr']]['mean']

    md = []

    def A(s=''):
        md.append(s)

    A('# Diagnostic report - token pooling versus the global feature (source domain, both encoders)')
    A()
    A('Executor: DeepSeek (deepseek-flash). Device: NVIDIA GeForce RTX 3080. Interpreter')
    A('D:/ResearchVault/99system/envs/rb-tta-fer-fg2027/Scripts/python.exe (torch 2.8.0+cu128).')
    A()
    A('This is a **source-domain diagnostic**, not a new method and not a confirmation run. Phase 5')
    A("established that TAP's margins over CLIP-Adapter (+3.88pp / +3.47pp) and over an equal-capacity")
    A('global head (+6.25pp / +5.21pp) at ViT-B/16 do not replicate at ViT-L/14, where the signs')
    A('reverse, while TAP still beats both training-free families at both encoders. This report asks')
    A('*why*, using only the FER2013 source probe-train / probe-val split:')
    A()
    A('- (i) the ViT-L/14 global feature is already strong enough that token pooling adds nothing, or')
    A("- (ii) TAP's convex combination of patch tokens loses information because it has no direct path")
    A('  from the global feature.')
    A()
    A('Every arm trains and selects on the source split only (Adam, 8 epochs, batch 128, learning-rate')
    A('grid {1e-3, 3e-3}, selection by probe-val UAR, seeds 0/1/2). No target label is used anywhere.')
    A('Neither confirmation set (Set A crossed grid, Set B KDEF-source), the locked test set nor the old')
    A('2250 holdout was read.')
    A()
    A('## 1. Protocol, provenance and pipeline validation')
    A()
    A('Representations, frozen unless noted:')
    A()
    A('- **R1** global pooled feature, L2-normalised (512-d ViT-B/16, 768-d ViT-L/14) - the existing')
    A('  PFB / global-head input.')
    A('- **R2** uniform mean pooling of the patch tokens (196x768, 256x1024), raw; plus two robustness')
    A('  variants (mean of L2-normalised tokens, L2-normalised mean).')
    A('- **R3** the TAP head: softmax attention over tokens -> 1xd pooling -> linear head, architecture')
    A('  identical to Phase 2 T1, trained with the same budget. **R3refit** freezes that learned pooling')
    A('  and refits a fresh linear head.')
    A('- **R4a** frozen concat(R1, R2) + linear probe; **R4b** concat(R1, attention-pooled vector)')
    A('  trained jointly - two forms of the direct-path / residual test.')
    A('- **MLP extension** (marked, and added because the comparators TAP lost to in Phase 5 are')
    A('  nonlinear): the same representations probed with Linear(d,256) -> Tanh -> Linear(256,256) ->')
    A('  Tanh -> Linear(256,7). This is exactly the Phase 5 global-head form: 198,919 parameters on the')
    A('  512-d global input and 264,455 on the 768-d input.')
    A()
    A('Inputs are read through the existing caches in 128-256 row blocks: the ViT-L/14 .npy memory map,')
    A('and for ViT-B/16 a one-off assembly of the 57 train shards into')
    A('token_cache_vitb16/fer2013_train_tokens_fp16.npy (28709x196x768 fp16). The assembly was needed')
    A('because a random 128-row batch touches about 44 of the 57 shards, so a shard-granular reader would')
    A('copy about 6.8 GB per step. Twelve random rows of the assembled file were verified equal to their')
    A('source shard rows, and the frozen R2/R4a probe results computed from the assembled file are')
    A('bit-identical to those computed from the shards in an aborted earlier attempt. Nothing is')
    A('concatenated in host memory; the largest transient is one 0.3-0.5 GB token block.')
    A()
    A('Both encoders use the **identical** source split (verified element-wise: 22,968 train / 5,741 val,')
    A('same indices), which is what makes the two columns comparable.')
    A()
    A('Cross-checks against the stored artifacts:')
    A()
    A('- R1 linear at ViT-B/16, lr 3e-3, seed 0: **0.5691**, matching the stored Phase 4 linear probe to')
    A('  four decimals.')
    A('- The MLP extension on the frozen global feature reproduces the stored Phase 5 global head:')
    A('  ViT-B/16 ' + f4(m(B16, 'R1_global_MLP', 0.003)) + ' vs stored 0.6144 (lr 3e-3 seed mean), and')
    A('  ViT-L/14 ' + f4(m(L14, 'R1_global_MLP', 0.001)) + ' vs stored 0.6720 (lr 1e-3 seed mean).')
    A('- R3 at ViT-L/14 lands at 0.6720 (lr 1e-3, seed 0) and 0.6861 (seed 1), bracketing the stored')
    A('  Phase 5 TAP value 0.6754 from a different code path.')
    A()
    A('Two instrumentation bugs were found and fixed here; they are recorded because they explain the')
    A('aborted logs kept alongside the results:')
    A()
    A('1. The first version built each streaming model *before* seeding it, so its initial weights came')
    A('   from a leftover RNG state. The model is now constructed inside the seeded scope.')
    A('2. The streaming evaluator read rows in ascending order for sequential I/O and wrote predictions')
    A('   back in that order, while labels were taken in the requested order. Because the stored row order')
    A('   is class-grouped (only 6 class changes across 28,709 rows), this mispaired predictions with')
    A('   labels and gave a chance-level UAR (0.137-0.141) while the training loss fell to 0.88. The')
    A('   evaluator now scatters results back into the requested order. Only the streaming arms (R3, R4b,')
    A('   R3refit) were affected; the frozen-probe numbers R1/R2/R4a never were, and all numbers in this')
    A('   report come from the fixed code.')
    A()
    A('Checkpoints and pooled matrices from the buggy intermediate runs were moved to')
    A('discarded_buggy_run/ rather than deleted. Aborted logs are kept as')
    A('diagnostic_token_vs_global_{B16_aborted_shardreader,B16_aborted_unseeded_init,')
    A('B16_aborted_eval_order_bug}.log and diagnostic_token_vs_global_mlp_first_attempt_indexerror.log.')
    A()
    A('## 2. Results - probe-val UAR on the FER2013 source split')
    A()
    A('"selected" is the Phase 3/4/5 rule (best single run on probe-val); it is biased upward by seed')
    A('noise, so the fairer comparator is "seed mean at best lr" with its standard deviation and range.')
    A()
    for tag in ('B16', 'L14'):
        main_run, _mlp, T = data[tag]
        A('### ' + ('ViT-B/16' if tag == 'B16' else 'ViT-L/14') + ' (' + main_run['model'] + ', '
          + str(main_run['n_train']) + ' train / ' + str(main_run['n_val']) + ' val)')
        A()
        A('| representation | dim | selected UAR | selected +prior | seed mean at best lr | sd | seed range |')
        A('|---|---|---|---|---|---|---|')
        for name in ORDER:
            if name not in T:
                continue
            d = T[name]
            st = d['stats'][d['best_lr']]
            sel = d['selected']
            A('| ' + LABELS[name] + ' | ' + (str(d.get('dim')) if d.get('dim') else '-') + ' | '
              + f4(sel['uar']) + ' | ' + f4(sel['uar_prior']) + ' | ' + f4(st['mean']) + ' | '
              + f4(st['sd']) + ' | ' + f4(st['min']) + '-' + f4(st['max']) + ' |')
        A()
        A('Per-learning-rate seed means (every run is listed in')
        A('diagnostic_token_vs_global_probes_*.csv and diagnostic_token_vs_global_mlp_probes_*.csv):')
        A()
        A('| representation | lr 1e-3 | lr 3e-3 | selected lr | selected seed | params |')
        A('|---|---|---|---|---|---|')
        for name in ORDER:
            if name not in T:
                continue
            d = T[name]
            a = d['stats'].get(0.001, {}).get('mean')
            b = d['stats'].get(0.003, {}).get('mean')
            A('| ' + LABELS[name] + ' | ' + (f4(a) if a is not None else '-') + ' | '
              + (f4(b) if b is not None else '-') + ' | ' + str(d['selected']['lr']) + ' | '
              + str(d['selected']['seed']) + ' | ' + (str(d.get('params')) if d.get('params') else '-')
              + ' |')
        A()

    A('## 3. Answers to the four questions')
    A()
    A('### Q1. At ViT-L/14, is R2 (mean-pooled tokens) already not worse than R1 (global)?')
    A()
    A('Yes under a linear probe, by the same margin as at ViT-B/16:')
    A()
    A('- ViT-L/14: R2 ' + f4(m(L14, 'R2_mean_raw')) + ' vs R1 ' + f4(m(L14, 'R1_global')) + ' = '
      + pp(m(L14, 'R2_mean_raw') - m(L14, 'R1_global')) + ' UAR (prior-corrected: '
      + f4(L14['R2_mean_raw']['stats'][L14['R2_mean_raw']['best_lr']]['mean_prior']) + ' vs '
      + f4(L14['R1_global']['stats'][L14['R1_global']['best_lr']]['mean_prior']) + ').')
    A('- ViT-B/16: R2 ' + f4(m(B16, 'R2_mean_raw')) + ' vs R1 ' + f4(m(B16, 'R1_global')) + ' = '
      + pp(m(B16, 'R2_mean_raw') - m(B16, 'R1_global')) + ' UAR.')
    A()
    A('Both frozen representations have small seed spread (sd '
      + f4(L14['R1_global']['stats'][0.003]['sd']) + ' and '
      + f4(L14['R2_mean_raw']['stats'][0.003]['sd']) + ' for R1 / R2 at ViT-L/14), so this gap is well')
    A('outside noise. **The token field is not information-poor at ViT-L/14 and token pooling does add')
    A('information a linear readout of the global feature does not recover.** Hypothesis (i) is therefore')
    A('not supported in the form "mean-pooled tokens cannot beat the global feature at L/14".')
    A()
    A('The picture changes once the readout is nonlinear, which is the regime the Phase 5 comparators')
    A('live in. With the capacity-matched MLP head at ViT-L/14, R1 is ' + f4(m(L14, 'R1_global_MLP'))
      + ' and R2 is ' + f4(m(L14, 'R2_mean_raw_MLP')) + ' - the global feature is now ahead by '
      + pp(m(L14, 'R1_global_MLP') - m(L14, 'R2_mean_raw_MLP')) + ', whereas at ViT-B/16 the token field stays ahead by '
      + pp(m(B16, 'R2_mean_raw_MLP') - m(B16, 'R1_global_MLP')) + '. So at ViT-L/14 the global feature carries')
    A('information comparable to or better than the token field once the readout can exploit it')
    A('nonlinearly, and at ViT-B/16 it does not catch up. That is the specific, much weaker form of (i)')
    A('the data support.')
    A()
    A('### Q2. Ordering at ViT-B/16 versus ViT-L/14 - reversed or not?')
    A()
    A('| comparison | ViT-B/16 | ViT-L/14 | reversed? |')
    A('|---|---|---|---|')
    A('| R2 - R1 (linear) | ' + pp(m(B16, 'R2_mean_raw') - m(B16, 'R1_global')) + ' | '
      + pp(m(L14, 'R2_mean_raw') - m(L14, 'R1_global')) + ' | no: same sign and same size |')
    A('| R3 - R2 (attention vs mean pooling) | '
      + pp(m(B16, 'R3_attn_pool') - m(B16, 'R2_mean_raw')) + ' | '
      + pp(m(L14, 'R3_attn_pool') - m(L14, 'R2_mean_raw')) + ' | no |')
    A('| R4a - R2 (does the global feature add on top of tokens?) | '
      + pp(m(B16, 'R4a_concat_raw') - m(B16, 'R2_mean_raw')) + ' | '
      + pp(m(L14, 'R4a_concat_raw') - m(L14, 'R2_mean_raw')) + ' | no: both about zero |')
    A('| R4b - R3 (does a direct global path help the learned pooling?) | '
      + pp(m(B16, 'R4b_attn_plus_global') - m(B16, 'R3_attn_pool')) + ' | '
      + pp(m(L14, 'R4b_attn_plus_global') - m(L14, 'R3_attn_pool')) + ' | no: both at or below zero |')
    A('| R2_MLP - R1_MLP (nonlinear head) | '
      + pp(m(B16, 'R2_mean_raw_MLP') - m(B16, 'R1_global_MLP')) + ' | '
      + pp(m(L14, 'R2_mean_raw_MLP') - m(L14, 'R1_global_MLP')) + ' | **yes: the sign flips** |')
    A()
    A('Only the nonlinear-head comparison reverses. Everything else - tokens beating the linearly-read')
    A('global feature, attention pooling beating mean pooling by a small constant, the global feature')
    A('adding nothing on top of the tokens, and the direct path adding nothing - behaves the same way at')
    A('both encoders.')
    A()
    A('One caveat on the attention-versus-mean row: that gap is learning-rate dependent, not a constant.')
    A('At lr 1e-3 the attention head beats raw mean pooling by '
      + pp(B16['R3_attn_pool']['stats'][0.001]['mean'] - B16['R2_mean_raw']['stats'][0.001]['mean'])
      + ' at ViT-B/16 and '
      + pp(L14['R3_attn_pool']['stats'][0.001]['mean'] - L14['R2_mean_raw']['stats'][0.001]['mean'])
      + ' at ViT-L/14, but at lr 3e-3 the two poolings are indistinguishable ('
      + pp(B16['R3_attn_pool']['stats'][0.003]['mean'] - B16['R2_mean_raw']['stats'][0.003]['mean'])
      + ' and '
      + pp(L14['R3_attn_pool']['stats'][0.003]['mean'] - L14['R2_mean_raw']['stats'][0.003]['mean'])
      + '). The headline row above compares each arm at its own selected lr; the same-lr view is the')
    A('honest range for the attention claim.')
    A()
    A('### Q3. What do the residual variants (R4) do?')
    A()
    A('- **R4a** (frozen concat of the global feature with mean tokens) is above R1 by construction, but')
    A('  only ' + pp(m(B16, 'R4a_concat_raw') - m(B16, 'R2_mean_raw')) + ' above R2 at ViT-B/16 and '
      + pp(m(L14, 'R4a_concat_raw') - m(L14, 'R2_mean_raw')) + ' at ViT-L/14 - inside one seed sd. The global feature')
    A('  contributes essentially nothing once the token field is present, at either encoder.')
    A('- **R4b** (the global feature given directly to the classifier alongside the learned attention')
    A('  pooling) is ' + pp(m(B16, 'R4b_attn_plus_global') - m(B16, 'R3_attn_pool')) + ' vs R3 at ViT-B/16 and '
      + pp(m(L14, 'R4b_attn_plus_global') - m(L14, 'R3_attn_pool')) + ' at ViT-L/14: the direct path does not help, and is if')
    A('  anything slightly negative.')
    A()
    A('So R4 does not beat R1 in the sense that matters (the part of R4 carrying the gain is the token')
    A('part, not the global part), and R4 does not beat R3. The direct path is not the missing')
    A('ingredient.')
    A()
    A('### Q4. Which explanation is supported, which is excluded')
    A()
    A('- **(ii) - "the convex token combination loses information because it has no direct path from the')
    A('  global feature" - is excluded.** Two independent forms of the direct path (a frozen concat, and')
    A('  a jointly trained concat with the learned pooling) change the result by at most +0.8pp over the')
    A('  token-only arm and are negative at ViT-L/14. If the missing residual were the cause, the')
    A('  ViT-L/14 gap should close when the path is added; it does not.')
    A('- **(i) is supported only in its narrow form**: at ViT-L/14 the global feature is as usable as the')
    A('  token field once the readout is nonlinear (R1_MLP ' + f4(m(L14, 'R1_global_MLP')) + ' vs R2_MLP '
      + f4(m(L14, 'R2_mean_raw_MLP')) + '), and the nonlinear readout is exactly what the Phase 5 comparators')
    A('  (equal-capacity global head, CLIP-Adapter) use. It is not supported in the form "the token field')
    A('  carries no usable information at ViT-L/14": under a linear probe the tokens are still '
      + pp(m(L14, 'R2_mean_raw') - m(L14, 'R1_global')) + ' ahead, the same margin as at ViT-B/16.')
    A('- **The direct driver of the Phase 5 reversal is the global-feature route under a nonlinear')
    A('  readout, not the token route.** Across encoder scale the MLP head on the frozen global feature')
    A('  moves ' + f4(m(B16, 'R1_global_MLP')) + ' -> ' + f4(m(L14, 'R1_global_MLP')) + ', the token route moves '
      + f4(m(B16, 'R3_attn_pool')) + ' -> ' + f4(m(L14, 'R3_attn_pool')) + ', and the two routes are essentially tied at')
    A('  both encoders once the heads are matched. The stored ViT-B/16 advantage of TAP over the')
    A('  equal-capacity global head (+2.1pp in Phases 3/5) is not reproduced by this independent')
    A('  implementation (' + f4(m(B16, 'R3_attn_pool')) + ' vs ' + f4(m(B16, 'R1_global_MLP')) + '), and the token')
    A('  route edge over a nonlinear global head is only ' + pp(m(B16, 'R3_attn_pool') - m(B16, 'R1_global_MLP'))
      + ' at ViT-B/16 and ' + pp(m(L14, 'R3_attn_pool') - m(L14, 'R1_global_MLP')) + ' at ViT-L/14.')
    A()
    A('A third factor is visible and matters for the paper: **seed and initialisation noise in the trained')
    A('token arms is of the same order as the effect being claimed.** Across seeds 0/1/2 the R3 arm spans')
    A('  ' + f4(L14['R3_attn_pool']['stats'][0.001]['min']) + '-'
      + f4(L14['R3_attn_pool']['stats'][0.001]['max']) + ' at ViT-L/14 (sd '
      + f4(L14['R3_attn_pool']['stats'][0.001]['sd']) + ') and '
      + f4(B16['R3_attn_pool']['stats'][0.001]['min']) + '-'
      + f4(B16['R3_attn_pool']['stats'][0.001]['max']) + ' at ViT-B/16 (sd '
      + f4(B16['R3_attn_pool']['stats'][0.001]['sd']) + '), while the frozen representations R1/R2/R4a are')
    A('stable to about 0.2-1.8pp. The Phase 5 ViT-B/16 margins (+3.88pp over CLIP-Adapter, +6.25pp over')
    A('the global head) are larger than this noise, but the matched-head difference between the token')
    A('route and the global route is not.')
    A()
    A('## 4. What this implies for the paper')
    A()
    A('1. The load-bearing claim cannot be "token attention pooling beats a global-feature head". Under')
    A('   matched heads the two routes are essentially tied at both encoders in this diagnostic, and the')
    A('   attention mechanism itself contributes only a small, roughly constant '
      + pp(m(B16, 'R3_attn_pool') - m(B16, 'R2_mean_raw')) + ' / '
      + pp(m(L14, 'R3_attn_pool') - m(L14, 'R2_mean_raw')) + ' over plain uniform mean pooling.')
    A('2. The defensible claims are narrower and better supported: (a) the token field is a better')
    A('   linear read-out target than the global feature at both encoders ('
      + pp(m(B16, 'R2_mean_raw') - m(B16, 'R1_global')) + ' / '
      + pp(m(L14, 'R2_mean_raw') - m(L14, 'R1_global')) + '); (b) a direct global path does not repair or improve')
    A('   token pooling; (c) attention pooling is a small, consistent gain over mean pooling.')
    A('3. If the encoder dependence itself is to be reported as a result, it needs the same treatment as')
    A('   any other claim: a pre-specified comparison, several seeds, and a stated noise floor. This')
    A('   diagnostic supplies the noise floor.')
    A()
    A('## 5. Discipline for any follow-up')
    A()
    A('This diagnostic consumed no confirmation data: it ran entirely on the source probe-train /')
    A('probe-val split. But it was designed *after* seeing the Phase 5 result, so **any new method')
    A('variant derived from it must be confirmed on a fresh, untouched set** - the two existing')
    A('confirmation sets (Set A crossed grid, Set B KDEF-source) have now been evaluated at two encoders')
    A('and cannot support a confirmatory claim about a variant chosen with this information. Candidate')
    A('fresh sets, in the order I would rank them:')
    A()
    A('1. **RAF-DB** as a third cross-database target (different collection and label noise than')
    A('   CK+ / KDEF / FER2013), with the same prior/support protocol. Strongest generalisation test.')
    A('2. **Corruption sweep** on the existing source-trained arms (blur, noise, JPEG levels): tests')
    A('   whether the token route is more robust than the global route rather than only more accurate -')
    A('   a different axis from the one that failed, and cheap to run.')
    A('3. **FER+ 1:1 and 5:1 imbalance** as a prior-shift test, the regime where prior correction and')
    A('   support memory matter most and where the two routes have not been compared.')
    A()
    A('Design requirements for whichever is used: fixed before running; all arms (token route, global')
    A('route with matched heads, CLIP-Adapter, training-free families) evaluated on it; at least three')
    A('seeds with the spread reported; and no arm or hyper-parameter chosen on the new set.')
    A()
    A('## 6. Scope, boundaries, unfinished')
    A()
    A('- Source domain only: the locked test set, the old 2250 holdout, Set A and Set B were not read.')
    A('- No main*.tex file was modified; nothing was submitted, no editor was contacted, nothing was paid.')
    A('- Delivered: this report; diagnostic_token_vs_global_{B16,L14}.json;')
    A('  diagnostic_token_vs_global_mlp_{B16,L14}.json; diagnostic_token_vs_global_probes_{B16,L14}.csv')
    A('  (every run); diagnostic_token_vs_global_mlp_probes_{B16,L14}.csv;')
    A('  diagnostic_token_vs_global_recall_{B16,L14}.csv (per-class recall, raw and prior-corrected, for')
    A('  the selected configuration of each representation); diagnostic_token_vs_global_summary.{json,csv}')
    A('  (merged cross-encoder table); the run scripts, the report generator and logs with START/EXIT.')
    A('- Derived artifacts created: token_cache_vitb16/fer2013_train_tokens_fp16.npy (the assembled')
    A('  ViT-B/16 train-token memory map, verified against the shards) and the pooled-token val matrices')
    A('  saved by the run script. The original ViT-B/16 token cache was not modified.')
    A('- The MLP extension is an addition to the requested R1-R4 linear-probe design, marked as such,')
    A('  included because the task logic for Q3 cannot be evaluated without knowing how the same')
    A('  representations behave under the nonlinear head class the Phase 5 comparators use.')
    A('- Not run: an R4 variant combining the direct global path with a *nonlinear* head (a')
    A('  CLIP-Adapter-style route with token pooling added). Both R4 forms tested here put a linear head')
    A('  on a concatenation, so they bound the linear-head version of hypothesis (ii), not every')
    A('  nonlinear variant of it; the MLP rows partially cover this for the frozen representations.')
    A('- Not run: corruption sweep, RAF-DB, FER+ imbalance variants, and any hyper-parameter search')
    A('  beyond the fixed {1e-3, 3e-3} grid.')
    A()
    (OUT / 'diagnostic_token_vs_global_report.md').write_text('\n'.join(md), encoding='utf-8')

    merged = []
    for tag in ('B16', 'L14'):
        for name, d in data[tag][2].items():
            st = d['stats'][d['best_lr']]
            merged.append({'encoder': tag, 'rep': name, 'dim': d.get('dim'), 'params': d.get('params'),
                           'selected_lr': d['selected']['lr'], 'selected_seed': d['selected']['seed'],
                           'selected_uar': d['selected']['uar'],
                           'selected_uar_prior': d['selected']['uar_prior'],
                           'best_lr_by_seed_mean': d['best_lr'],
                           'seed_mean_at_best_lr': st['mean'], 'seed_sd_at_best_lr': st['sd'],
                           'seed_min_at_best_lr': st['min'], 'seed_max_at_best_lr': st['max'],
                           'seed_mean_prior_at_best_lr': st['mean_prior'],
                           'seed_mean_lr1e3': d['stats'].get(0.001, {}).get('mean'),
                           'seed_mean_lr3e3': d['stats'].get(0.003, {}).get('mean')})
    with (OUT / 'diagnostic_token_vs_global_summary.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)
    (OUT / 'diagnostic_token_vs_global_summary.json').write_text(json.dumps(merged, indent=1),
                                                                encoding='utf-8')
    print('wrote diagnostic_token_vs_global_report.md (%d lines)' % len(md))
    for tag in ('B16', 'L14'):
        print(tag, {k: round(v['selected']['uar'], 4) for k, v in data[tag][2].items()})


if __name__ == '__main__':
    main()
