"""Generate phase6b_confirm_report.md from the frozen pre-registration and the run artifacts."""
import csv, json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
CONDS = ['blur_0p8', 'blur_1p5', 'blur_2p5', 'jpeg_15', 'jpeg_30', 'jpeg_50',
         'lowlight_0p3', 'lowlight_0p5', 'lowlight_0p7', 'noise_10', 'noise_25', 'noise_40']
ENC_NAME = {'B16': 'ViT-B/16', 'L14': 'ViT-L/14'}


def jl(p):
    return json.loads((OUT / p).read_text(encoding='utf-8'))


def cl(p):
    with (OUT / p).open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def main():
    have = {}
    for tag in ('B16', 'L14'):
        if (OUT / ('phase6b_confirm_%s.json' % tag)).exists():
            have[tag] = {'confirm': jl('phase6b_confirm_%s.json' % tag),
                         'cells': cl('phase6b_cells_%s.csv' % tag),
                         'boot': jl('phase6b_bootstrap_%s.json' % tag),
                         'extract': jl('phase6b_extraction_%s.json' % tag)}
    prereg = (OUT / 'phase6b_preregistration.md').read_text(encoding='utf-8')
    verify = jl('phase6b_verify_extraction.json')
    val_l14 = jl('phase6b_validation_l14_tokens.json')
    md = []

    def A(s=''):
        md.append(s)

    A('# Phase 6b report - corruption-sweep confirmation of H2')
    A()
    A('## Two statements that define what this run can claim')
    A()
    A('1. **What this axis measures.** This is a **within-domain robustness axis**: the target is')
    A('   FER2013 test itself, degraded synthetically in 12 conditions - not a cross-database target.')
    A('   If H2 holds, the claim the paper can make is "TAP is more robust than CLIP-Adapter under')
    A('   within-domain input degradation". It must **not** be presented as a confirmation of')
    A('   cross-database accuracy.')
    A('2. **Locked-set boundary.** This run reads only the FER2013 test images and their 12 degraded')
    A('   versions (a target this line already used as fer2013_test / fer2013_shift in phases 3-5). It')
    A('   did not read the old 2250 holdout, did not read Set A or Set B, and did not read the RAF-DB')
    A('   tree.')
    A()
    A('**Known limitation, stated plainly.** The four arms are frozen models, so seeds 0-4 change only')
    A('the online prior-correction order of the PFB reference column; TAP, CLIP-Adapter, the')
    A('equal-capacity global head and the support-memory column are identical across the five seeds.')
    A('This must not be read as "five independent seed repeats". The training-seed variance is carried')
    A('by decision criterion 2 (the given between-seed sd 0.0116 at ViT-B/16 and 0.0134 at ViT-L/14),')
    A('and the bootstrap covers query and condition sampling only.')
    A()
    A('## Verdict')
    A()
    if 'B16' in have:
        d = have['B16']['confirm']['decision']
        A('**H2 (primary, ViT-B/16): %s.**' % ('HOLDS' if d['H2_holds'] else 'DOES NOT HOLD'))
        A('')
        A('- TAP - CLIP-Adapter = %+.4f (%.2fpp), paired 95%% CI [%+.4f, %+.4f].'
          % (d['point_TAP_minus_CLIPAdapter'], 100 * d['point_TAP_minus_CLIPAdapter'],
             d['ci95'][0], d['ci95'][1]))
        A('- Criterion 1 (CI excludes 0 and is positive): %s. Criterion 2 (|margin| > %.4f): %s.'
          % (d['criterion1_ci_excludes_zero_and_positive'], d['seed_sd_threshold'],
             d['criterion2_margin_exceeds_seed_sd']))
    if 'L14' in have:
        d = have['L14']['confirm']['decision']
        A('')
        A('**ViT-L/14 (secondary, not part of H2):** TAP - CLIP-Adapter = %+.4f (%.2fpp), CI'
          % (d['point_TAP_minus_CLIPAdapter'], 100 * d['point_TAP_minus_CLIPAdapter']))
        A('[%+.4f, %+.4f]; criterion 1 %s, criterion 2 %s.'
          % (d['ci95'][0], d['ci95'][1], d['criterion1_ci_excludes_zero_and_positive'],
             d['criterion2_margin_exceeds_seed_sd']))
    A()
    A('## 0. Pre-registration (verbatim, frozen before extraction)')
    A()
    A('~~~markdown')
    md.extend(prereg.rstrip('\n').split('\n'))
    A('~~~')
    A()
    A('## 1. Inventory (step 1) and what was chosen')
    A()
    A('Full audit: phase6b_inventory_report.md, phase6b_inventory.json, phase6b_provenance.json.')
    A('The chosen axis is the graded corruption sweep of FER2013 test: 4 corruption types x 3')
    A('severities, 7178 images per condition, with a single provenance chain from')
    A('make_corrupted_fer2013.py through raw/fer2013_shift_sweep/<condition>/ and')
    A('manifests/fer2013_<condition>.csv to the existing features/shift_sweep/<condition>.npz.')
    A('The four legacy features/corruptions/*.npz files were excluded (two are exact duplicates of')
    A('sweep levels, two come from a different generator script, one of which routes through')
    A('grayscale). FER+ 1:1 / 5:1 is available and ratio-feasible but was not chosen; features/')
    A('ckplus48_all.npz carries no labels, and CK+/KDEF were already confirmation targets in phases 3-5.')
    A()
    A('## 2. Extraction and the validation gate')
    A()
    A('| encoder | conditions | rows per condition | new token bytes | clean-cache cosine | condition cosine | gate |')
    A('|---|---|---|---|---|---|---|')
    for tag, h in have.items():
        ex = h['extract']
        g = ex['validation_gate']
        ck = [k for k in g if k.startswith('condition_') and k.endswith('globals_cosine_min')]
        cond_val = ('%.6f' % g[ck[0]]) if ck else 'n/a (no stored L/14 cache)'
        tot = sum(v.get('npy_GB', 0) for v in ex['conditions'].values())
        A('| ' + ENC_NAME[tag] + ' | ' + str(len(ex['conditions'])) + ' | 7178 | %.1f GB | %.6f | '
          % (tot, g['clean_globals_cosine_min']) + cond_val + ' | '
          + ('PASS' if g['pass'] else 'FAIL') + ' |')
    A()
    A('The gate requires elementwise cosine >= 0.9999 against the stored caches. Both encoders passed')
    A('at 1.000000, i.e. this extraction reproduces the stored clean caches exactly. The chip')
    A('convention is the same one the clean caches and the phase-5 token extraction use: open_clip')
    A('preprocess, one original view for the patch tokens (fp16), two views (image + mirror) for the')
    A('L2-normalised global features (fp32).')
    A()
    A('### 2.1 What the ViT-L/14 gate actually verified, and what it could not')
    A()
    A('The B/16 gate followed the pre-registration literally: clean globals recomputed and matched')
    A('against features/fer2013_test.npz, plus one corrupted condition recomputed and matched against')
    A('the pre-existing features/shift_sweep/blur_0p8.npz. **There is no equivalent pre-existing')
    A('corrupted cache for ViT-L/14** - features/shift_sweep/*.npz are 512-d ViT-B/16 artifacts, and')
    A('the L/14 run wrote clean_tok=None / cond=n/a in its gate record for exactly that reason (the')
    A('token-cache lookup also used a wrong directory name, token_cache_l14 instead of')
    A('token_cache_vitl14). Those two gaps were closed by two follow-up checks, run after the fact and')
    A('reported as such rather than folded into the gate:')
    A()
    A('- phase6b_validate_l14_tokens.py: 512 clean FER2013 test rows recomputed for ViT-L/14, globals')
    A('  cosine min %.6f against features_vitl14/fer2013_test.npz and patch-token cosine min %.6f'
      % (val_l14['clean_globals_cosine_min'], val_l14['clean_token_cosine_min']))
    A('  against token_cache_vitl14/fer2013_test_tokens_fp16.npy -> %s.'
      % ('PASS' if val_l14['pass'] else 'FAIL'))
    A('- phase6b_verify_extraction.py: every one of the 24 condition caches (12 conditions x 2')
    A('  encoders) checked for structural completeness (file size equals the .npy header plus nbytes),')
    A('  meta id/label alignment against the manifests and the clean test labels, unit-norm views, and')
    A('  an independent recomputation of 8 rows per condition. Result: %d/%d verified. Patch tokens'
      % (verify['n_verified'], verify['n_total']))
    A('  reproduce to 0.12-1.5 fp16 ULP (per-image cosine 1.000000 at both encoders) and globals to')
    A('  cosine 1.000000.')
    A()
    A('So the honest statement of L/14 evidence strength is: its extraction is verified by exact')
    A('agreement with the *clean* L/14 caches plus per-condition recomputation, and its arms are')
    A('verified by reproducing the stored phase-5 Set B fer2013_test numbers to four decimals - but it')
    A('has **no independent third-party corrupted reference** to agree with, unlike ViT-B/16. L/14 is')
    A('the secondary encoder and does not enter the H2 decision; this limitation is recorded rather')
    A('than treated as equivalent to the B/16 gate.')
    A()
    A('The 12 reused L/14 condition files (written before the validation-step CUDA OOM of the first')
    A('attempt) were explicitly re-verified before any evaluation: all 12 are byte-complete with')
    A('matching ids, labels and unit-norm views, so nothing needed re-extraction.')
    A()
    A('Cost actually paid: about 12 min for 12 conditions at ViT-B/16 (26 GB) and about 45-50 min at')
    A('ViT-L/14 (45 GB). The pre-registration estimated 8 and 24 min by multiplying the recorded')
    A('clean-cache rate by the image count; the global-feature convention needs two views per image,')
    A('which doubles the forward passes. The estimate was corrected here, after the run, and it')
    A('affects no hypothesis, arm, condition or criterion.')
    A()
    A('## 3. Results - all 12 conditions, both encoders')
    A()
    for tag, h in have.items():
        A('### ' + ENC_NAME[tag] + (' (primary, H2)' if tag == 'B16' else ' (secondary)'))
        A()
        A('| condition | TAP | GHead | CLIP-Adapter | PFB (mean of 5 seeds) | support memory | max(PFB,memory) | TAP - CLIP-Adapter |')
        A('|---|---|---|---|---|---|---|---|')
        for row in h['cells']:
            c = row['condition']
            t, ca = float(row['TAP']), float(row['CLIPAdapter'])
            A('| ' + c + ' | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %+.4f |'
              % (t, float(row['GHead']), ca, float(row['pfb_mean5']), float(row['memory']),
                 float(row['best_family_mean5']), t - ca))
        agg = h['confirm']['aggregate']
        A('| **mean over conditions** | **%.4f** | **%.4f** | **%.4f** | **%.4f** | **%.4f** | **%.4f** | **%+.4f** |'
          % (agg['TAP']['mean'], agg['GHead']['mean'], agg['CLIPAdapter']['mean'],
             agg['pfb_mean5']['mean'], agg['memory']['mean'], agg['best_family_mean5']['mean'],
             agg['TAP']['mean'] - agg['CLIPAdapter']['mean']))
        A()
        wins = sum(1 for row in h['cells'] if float(row['TAP']) > float(row['CLIPAdapter']))
        A('TAP is above CLIP-Adapter in %d of 12 conditions. Across conditions the sd of the TAP mean is'
          % wins)
        A('%.4f and of the CLIP-Adapter mean is %.4f (condition-level spread, for orientation only).'
          % (agg['TAP']['sd_across_conditions'], agg['CLIPAdapter']['sd_across_conditions']))
        A()
        A('Per-seed detail: the three trained arms are constant across seeds 0-4 (frozen models); only')
        A('the PFB column and therefore max(PFB, memory) move with the seed, through the online')
        A('prior-correction order. Per-seed values for every condition and every arm are in')
        A('phase6b_cells_%s.csv (columns pfb|0..4, best_family|0..4).' % tag)
        A()
        A('PFB per-seed spread by condition (min/mean/max over seeds 0-4):')
        A()
        A('| condition | PFB min | PFB mean | PFB max | max(PFB,mem) min | max(PFB,mem) mean | max(PFB,mem) max |')
        A('|---|---|---|---|---|---|---|')
        for row in h['cells']:
            pf = [float(row['pfb|%d' % s]) for s in range(5)]
            bf = [float(row['best_family|%d' % s]) for s in range(5)]
            A('| ' + row['condition'] + ' | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |'
              % (min(pf), sum(pf) / 5, max(pf), min(bf), sum(bf) / 5, max(bf)))
        A()
    A('## 4. Paired shared-unit bootstrap and the decision rule')
    A()
    A('2000 draws; seeds (0-4) and conditions (12) resampled with replacement; within each sampled')
    A('cell the 7178 query indices are resampled with replacement and shared across arms. Because the')
    A('trained arms are frozen, these intervals reflect query and condition sampling only.')
    A()
    for tag, h in have.items():
        A('### ' + ENC_NAME[tag])
        A()
        A('| pair | mean | 95% CI |')
        A('|---|---|---|')
        b = h['boot']['bootstrap']
        for k in ('TAP_minus_CLIPAdapter', 'TAP_minus_GHead'):
            A('| ' + k.replace('_minus_', ' - ') + ' | %+.4f | [%+.4f, %+.4f] |'
              % (b[k]['mean'], b[k]['ci'][0], b[k]['ci'][1]))
        A('| TAP - best-of-two-families (seed-0 draw) | %+.4f | [%+.4f, %+.4f] |'
          % (b['TAP_minus_best_family|0']['mean'], b['TAP_minus_best_family|0']['ci'][0],
             b['TAP_minus_best_family|0']['ci'][1]))
        A('| CLIP-Adapter - best-of-two-families (seed-0 draw) | %+.4f | [%+.4f, %+.4f] |'
          % (b['CLIPAdapter_minus_best_family|0']['mean'],
             b['CLIPAdapter_minus_best_family|0']['ci'][0],
             b['CLIPAdapter_minus_best_family|0']['ci'][1]))
        A()
        d = h['confirm']['decision']
        A('Decision on ' + ('H2' if tag == 'B16' else 'the secondary comparison') + ': point %+.4f,'
          % d['point_TAP_minus_CLIPAdapter'])
        A('criterion 1 %s, criterion 2 (|margin| > %.4f) %s -> **%s**.'
          % (d['criterion1_ci_excludes_zero_and_positive'], d['seed_sd_threshold'],
             d['criterion2_margin_exceeds_seed_sd'], 'HOLDS' if d['H2_holds'] else 'DOES NOT HOLD'))
        A()
    A('## 5. Pipeline cross-check')
    A()
    A('Before any corrupted condition was scored, the same code path was run on the clean FER2013')
    A('test cache and compared with the stored phase 4 / phase 5 Set B values for fer2013_test:')
    A()
    A('| encoder | arm | computed here | stored | diff |')
    A('|---|---|---|---|---|')
    for tag, h in have.items():
        cc = h['confirm']['clean_cross_check']
        for a in cc['compared_arms']:
            A('| ' + ENC_NAME[tag] + ' | ' + a + ' | %.4f | %.4f | %+.4f |'
              % (cc['computed'][a], cc['stored'][a], cc['diff'][a]))
    A()
    A('The trained arms reproduce the stored numbers to four decimals. The PFB and support-memory')
    A('columns here use the **FER2013 source support of the pre-registration** (the Set A convention),')
    A('not the KDEF-source support used by the stored Set B reference columns, so those two columns')
    A('are reported but not compared - they are a different, pre-registered configuration.')
    A()
    A('## 6. What this does and does not support')
    A()
    A('- It supports, or fails to support, the specific claim that TAP is more robust than')
    A('  CLIP-Adapter under within-domain input degradation on FER2013 test, under the fixed rule.')
    A('- It says nothing about cross-database accuracy. The cross-database evidence remains the')
    A('  Phase 3/4 sets (where TAP beat CLIP-Adapter at ViT-B/16 and lost at ViT-L/14) and the')
    A('  Phase 5 replication failure.')
    A('- The corruption conditions are now a consumed confirmation set. Any further variant designed')
    A('  with knowledge of these results needs a fresh untouched set.')
    A('- RAF-DB remains unconsumed but locally unavailable (phase 6 gate).')
    A()
    A('## 7. Scope, boundaries, unfinished')
    A()
    A('- Read: manifests/fer2013_<condition>.csv (12), raw/fer2013_shift_sweep/<condition>/, the 12')
    A('  features/shift_sweep/<condition>.npz (validation only), the frozen source-domain arms of')
    A('  phases 2-5 and their source caches, plus the FER2013 test / FER2013 train source caches.')
    A('- Not read: the old 2250 holdout, Set A, Set B, the RAF-DB tree, the locked test set.')
    A('- No main*.tex file was modified; nothing was submitted, no editor was contacted, nothing was')
    A('  paid. No existing file was overwritten: all outputs of this phase are new files.')
    A('- Unfinished: the ViT-L/14 half is reported only if its extraction and evaluation completed')
    A('  (see section 3); nothing else in the pre-registration was left out.')
    A()
    (OUT / 'phase6b_confirm_report.md').write_text('\n'.join(md), encoding='utf-8')
    print('wrote phase6b_confirm_report.md (%d lines); encoders present: %s' % (len(md), sorted(have)))


if __name__ == '__main__':
    main()
