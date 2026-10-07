"""Append the Phase 4 (published-adapter baselines) record to MANIFEST.json."""
import hashlib, json
from pathlib import Path

OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    m = json.loads((OUT / 'MANIFEST.json').read_text(encoding='utf-8'))
    m['phase'] = 4
    files = ['phase4_baselines_train.py', 'phase4_baselines_confirm.py',
             'phase4_baselines_training.json', 'phase4_baselines.json',
             'phase4_baselines_cells.csv', 'phase4_kdef_source_baseline_cells.csv',
             'phase4_baselines_report.md', 'phase4_update_manifest.py']
    for stem in ('clipadapter', 'linearprobe', 'apetext', 'tipadapterf'):
        for lr in ('0.001', '0.003'):
            files.append('phase4_%s_lr%s.pt' % (stem, lr))
    m['output_hashes'].update({f: sha(OUT / f) for f in files if (OUT / f).exists()})
    m['stage4_published_baselines'] = {
        'commands': [
            {'cmd': 'python phase4_baselines_train.py', 'exit_code': 0, 'seconds': 21.5,
             'note': '8 epochs, batch 128, Adam, lr grid {1e-3, 3e-3}; source probe-train only'},
            {'cmd': 'python phase4_baselines_confirm.py', 'exit_code': 0, 'seconds': 105.0,
             'note': 'Set A 50 cells + Set B 15 cells; shared-unit paired bootstrap 2000'},
        ],
        'arms': ['TAP', 'CLIPAdapter', 'LinearProbe', 'APEtext', 'TipAdapterF', 'pfb', 'memory',
                 'best_family'],
        'source_probeval_uar': {'TAP': 0.6351, 'CLIPAdapter': 0.6172, 'PFB': 0.6155,
                                'APEtext': 0.6076, 'TipAdapterF': 0.5829, 'LinearProbe': 0.5691},
        'setA_mean_uar': {'TAP': 0.6784, 'CLIPAdapter': 0.6396, 'LinearProbe': 0.6132,
                          'APEtext': 0.6034, 'TipAdapterF': 0.5623},
        'setB_mean_uar': {'TAP': 0.6722, 'CLIPAdapter': 0.6375, 'LinearProbe': 0.5980,
                          'APEtext': 0.6013, 'TipAdapterF': 0.5647},
        'bootstrap_TAP_minus_baseline': {
            'setA_CLIPAdapter': [0.0388, 0.0307, 0.0460],
            'setA_LinearProbe': [0.0652, 0.0568, 0.0731],
            'setA_APEtext': [0.0750, 0.0688, 0.0811],
            'setA_TipAdapterF': [0.1161, 0.1093, 0.1229],
            'setB_CLIPAdapter': [0.0347, 0.0245, 0.0460],
            'setB_LinearProbe': [0.0742, 0.0637, 0.0854],
            'setB_APEtext': [0.0710, 0.0608, 0.0819],
            'setB_TipAdapterF': [0.1075, 0.0966, 0.1189]},
        'answer': 'TAP beats every implemented published adapter on both confirmation sets; CLIP-Adapter is closest (+3.88pp set A, +3.47pp set B, CI excludes 0)',
        'caveats': ['all trainable adapters fitted on the source split because target labels are unavailable by protocol',
                    '8 epochs used for every baseline for budget parity; original Tip-Adapter-F uses about 20',
                    'Tip-Adapter-F is the only baseline below best-of-two-families, consistent with its design premise'],
        'known_issue_fixed': {
            'issue': 'first CLIP-Adapter run returned a 512-d feature where 7-class logits were expected, so F.cross_entropy treated 512 dims as classes; every crossed cell collapsed to a single class (mean UAR 0.1429 = 1/7)',
            'fix': 'CLIPAdapter.forward now returns logits = 100 * normalize(0.2*fc2(relu(fc1(x))) + 0.8*x) @ text.T',
            'evidence_kept': 'the degenerate 0.1429 mean is recorded here; the artifact was regenerated after the fix and the report uses the fixed values'},
        'read_list': [str(Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027/features') / f)
                      for f in ('fer2013_train.npz', 'fer2013_test.npz', 'ckplus_test.npz',
                                'text_prototypes.npz', 'linear_probe_train80_seed0.npz',
                                'linear_probe_split_seed0.npz')] +
                     [str(Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027/features_kdef_source') / f)
                      for f in ('kdef_source_train.npz', 'kdef_source_test.npz',
                                'linear_probe_kdef_source_seed0.npz')] +
                     [str(Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027/token_cache_vitb16') / f)
                      for f in ('fer2013_train_tokens_fp16.npz', 'fer2013_test_tokens_fp16.npz',
                                'ckplus_test_tokens_fp16.npz', 'kdef_test_tokens_fp16.npz',
                                'kdef_source_train_tokens_fp16.npz')],
    }
    m['stage4_vitl14_option'] = {
        'executed': False,
        'weights_present_locally': False,
        'download_GB': 1.7,
        'token_GB_estimate': 21.9,
        'disk_free_GB_D': 249.4,
        'extract_minutes_estimate': '20-30 for the four caches plus 1-2 for KDEF-source support',
        'note': 'assessment only, for the user to decide; global 512-d ViT-L/14 caches already exist',
    }
    m['unfinished'] = ['ViT-L/14 token confirmation left blank (option assessed, not executed)',
                       'KDEF-source variant of the T2 3x3-vs-1x1 control not run',
                       'workspace scratch copies _p3.py/_a1.py retained (identical to phase3_confirm_crossed.py/phase3_attribution.py)']
    (OUT / 'MANIFEST.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    print('manifest phase 4 written, hashed files:', len(m['output_hashes']))


if __name__ == '__main__':
    main()
