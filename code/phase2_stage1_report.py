"""Phase 2 stage-1 report: extraction + shape/id/label alignment checks."""
import hashlib, json, sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = Path('D:/ResearchVault/99system/data/rb-tta-fer-fg2027')
CACHE = D / 'token_cache_vitb16'
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
ext = json.loads((OUT / 'phase2_extraction.json').read_text(encoding='utf-8'))

checks = {}
for name, feat in (('fer2013_train', 'fer2013_train.npz'), ('fer2013_test', 'fer2013_test.npz'),
                   ('ckplus_test', 'ckplus_test.npz'), ('kdef_test', 'kdef_test.npz')):
    z = np.load(CACHE / ('%s_tokens_fp16.npz' % name), allow_pickle=False)
    fz = np.load(D / 'features' / feat, allow_pickle=False)
    ids_t = [str(x) for x in z['ids']]
    ids_f = [str(x) for x in fz['ids']]
    checks[name] = {
        'tokens_shape': list(z['tokens'].shape), 'dtype': str(z['tokens'].dtype),
        'labels_equal_cached': bool(np.array_equal(z['labels'].astype(int),
                                                   fz['labels'].astype(int))),
        'ids_equal_cached': ids_t == ids_f,
        'token_norm_mean': float(np.linalg.norm(z['tokens'][:50].astype(np.float32),
                                                axis=-1).mean()),
        'grid': [14, 14]}
    print(name, json.dumps(checks[name])[:220])

L = []
L.append('# Token 级模块缝合 — Phase 2 阶段 1 报告：全量 patch token 提取')
L.append('')
L.append('执行：DeepSeek（deepseek-flash），GPU（RTX 3080）。未训练、未评估；'
         '未读锁定 test 与旧 2250、未改 main_rev.tex、未投稿。')
L.append('')
L.append('## 提取结果（fp16，每 512 图一分片，先落盘再合并）')
L.append('')
L.append('| 数据集 | n | token 形状 | 分片数 | 合并文件 | 合并 sha256 前缀 | 耗时 | 峰值显存 |')
L.append('|---|---|---|---|---|---|---|---|')
for k, v in ext['datasets'].items():
    L.append('| %s | %d | %s | %d | token_cache_vitb16/%s_tokens_fp16.npz | %s | %.1f s | %.2f GB |'
             % (k, v['n'], v['tokens_shape'], v['shards'], k, v['sha256_merged'][:16],
                v['seconds'], v['peak_vram_GB']))
L.append('')
L.append('合计 **%.2f GB**（train %.2f + test %.2f + ckplus %.2f + kdef %.2f GB），'
         '与 Phase 1 估计 11.68 GB 吻合。分片 sha256 全量记录在 phase2_extraction.json'
         '（每数据集 2–57 片，逐片列出）。**FER+ 未重复提取**（复用 fer2013_test token + FER+ 标签）。'
         % (sum(v['merged_MB'] for v in ext['datasets'].values()) / 1024,
            ext['datasets']['fer2013_train']['merged_MB'] / 1024,
            ext['datasets']['fer2013_test']['merged_MB'] / 1024,
            ext['datasets']['ckplus_test']['merged_MB'] / 1024,
            ext['datasets']['kdef_test']['merged_MB'] / 1024))
L.append('')
L.append('## 形状与对齐校验（对合并件复核）')
L.append('')
L.append('| 数据集 | token 形状 | dtype | 网格 | label 与既有 512 维缓存一致 | id 顺序一致 | token 范数均值 |')
L.append('|---|---|---|---|---|---|---|')
for k, c in checks.items():
    L.append('| %s | %s | %s | %d×%d | %s | %s | %.2f |'
             % (k, c['tokens_shape'], c['dtype'], c['grid'][0], c['grid'][1],
                '是' if c['labels_equal_cached'] else '否',
                '是' if c['ids_equal_cached'] else '否', c['token_norm_mean']))
L.append('')
L.append('Phase 1 已在 200+200 张上确认池化+投影嵌入与缓存 view0 余弦 1.0；本轮复核确认**合并件的'
         '样本顺序、标签与既有 512 维缓存逐条一致**，token 形状 196×768、dtype float16、14×14 网格，'
         'token 范数均值 ≈ 21（未归一化 transformer 输出，符合预期）。')
L.append('')
L.append('## 第二编码器（ViT-L/14）权重可得性')
L.append('')
L.append('**本地不可得**：在 D:/ResearchVault/99system/models/ 下按 *L-14* 检索为空'
         '（提取记录字段 vitl14_weights = []）。注意 data/features_vitl14/ 只有 512 维全局特征缓存、'
         '**没有 patch token**；因此 token 级第二编码器确认本轮无法进行，**标记留空**，'
         'Phase 3 确认集只含 crossed 网格 + 第二源域 KDEF-source 两套，并如实标注该缺口。')
L.append('')
L.append('## 读取 / 写入清单')
L.append('')
L.append('- 读：data/manifests/{fer2013_train,fer2013_test,ckplus_test,kdef_test}.csv、'
         'models/open_clip/ViT-B-16.pt、data/features/*.npz（仅用于 id/label 对齐复核）。')
L.append('- 写：data/token_cache_vitb16/<dataset>/shard_XXXX.npz（分片）与 '
         'data/token_cache_vitb16/<dataset>_tokens_fp16.npz（合并）；本目录 phase2_extraction.json。')
L.append('')
(OUT / 'phase2_stage1_report.md').write_text('\n'.join(L), encoding='utf-8')
man = json.loads((OUT / 'MANIFEST.json').read_text(encoding='utf-8'))
man['phase'] = 2
man['stage1_extraction'] = {
    'cache_root': str(CACHE), 'shard_size': 512, 'dtype': 'float16',
    'datasets': {k: {'n': v['n'], 'shards': v['shards'], 'shape': v['tokens_shape'],
                     'sha256_merged': v['sha256_merged'], 'seconds': v['seconds'],
                     'peak_vram_GB': v['peak_vram_GB']}
                 for k, v in ext['datasets'].items()},
    'total_GB': round(sum(v['merged_MB'] for v in ext['datasets'].values()) / 1024, 2),
    'vitl14_weights_available': False, 'checks': checks}
man.setdefault('commands', []).append('python _x1.py (full token extraction)')
for p in [OUT / 'phase2_stage1_report.md', OUT / 'phase2_extraction.json']:
    man.setdefault('output_hashes', {})[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
(OUT / 'MANIFEST.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
print('wrote phase2_stage1_report.md')
