"""Stage-2 report: token modules trained on the source only (T1-T4)."""
import hashlib, json, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
tr = json.loads((OUT / 'phase2_training.json').read_text(encoding='utf-8'))
REF = {'pfb_source_probeval': 0.6155, 'memory_source_probeval_beta256': 0.6360}
L = []
L.append('# Token 级模块缝合 — Phase 2 阶段 2 报告：T1–T4 实现、源域训练与选择')
L.append('')
L.append('执行：DeepSeek（deepseek-flash），GPU（RTX 3080）。**训练与选择只在源域**'
         '（FER2013 probe-train 22,968 训练 / probe-val 5,741 选择），**未使用任何目标标签**；'
         'CLIP 权重冻结（输入为已缓存 token），只训练各模块 head。'
         '未读锁定 test 与旧 2250、未改 main_rev.tex、未投稿。')
L.append('')
L.append('## 训练配置（预登记小网格，未扩）')
L.append('')
L.append('| 模块 | lr | 参数量 | 步数 | 耗时(s) | 峰值显存(GB) | probe-val UAR |')
L.append('|---|---|---|---|---|---|---|')
for k, v in tr.items():
    L.append('| %s | %s | %d | %d | %.1f | %.2f | %.4f |'
             % (v['kind'], k.split('|lr')[1], v['params'], v['steps'], v['seconds'],
                v['peak_vram_GB'], v['probe_val_uar']))
L.append('')
L.append('网格：lr ∈ {1e-3, 3e-3} × epochs 8 × batch 128（每模块 2 个候选，逐条上报）。'
         '总训练耗时 179.7 s。')
L.append('')
L.append('## 与两族参照的对照（同一 probe-validation）')
L.append('')
L.append('源侧参照：**PFB = %.4f、支持记忆(beta=256) = %.4f**（此前 canonical 校准值，'
         '同一 split）。' % (REF['pfb_source_probeval'], REF['memory_source_probeval_beta256']))
L.append('')
L.append('| 模块（最优 lr） | probe-val UAR | −PFB | −支持记忆 | 选中的 lr |')
L.append('|---|---|---|---|---|')
best = {}
for kind in ('T1', 'T2', 'T3', 'T4'):
    cands = {k: v for k, v in tr.items() if v['kind'] == kind}
    k, v = max(cands.items(), key=lambda kv: kv[1]['probe_val_uar'])
    best[kind] = (k, v)
    L.append('| %s | %.4f | %+.4f | %+.4f | %s |'
             % (kind, v['probe_val_uar'],
                v['probe_val_uar'] - REF['pfb_source_probeval'],
                v['probe_val_uar'] - REF['memory_source_probeval_beta256'],
                k.split('|lr')[1]))
L.append('')
L.append('**源侧结果**：T2（变形卷积模块）0.6503 最高，比最强的单臂（支持记忆 0.6360）高 **+1.4pp**、'
         '比 PFB 高 +3.5pp；T1 0.6351 与支持记忆持平；T4 0.6236 居中；'
         '**T3（token 交互记忆）0.5856 明显低于两族**（其 patch 级 top-k 记忆在本实现下没有超过'
         '全局 cache）。注意：这只是**源侧选择结果**，不是验收结论。')
L.append('')
L.append('## 最近邻、实质差异、是否只是重参数化')
L.append('')
L.append('| 模块 | 最近邻 | 实质差异 | 是否只是已有方法的重参数化 |')
L.append('|---|---|---|---|')
L.append('| T1 | CLIP-Adapter、Tip-Adapter-F | 适配发生在**池化之前**：可学习注意力在 196 个 patch 上'
         '加权后再分类；CLIP-Adapter 只在已池化的 CLS 特征上做残差 MLP | 否（聚合算子不同），'
         '但表达力接近"加权平均池化 + 线性分类"，增益主要来自池化权重 |')
L.append('| T2 | CLIP-Adapter、Tip-Adapter-F | 3×3 卷积在 **14×14 patch 网格**上做空间混合；'
         '全局特征没有空间维度，无法表达 | 否（引入空间邻域算子）；与在 token 上做轻量 CNN 头的'
         '思路同源，但本研究把它作为对照臂而非新机制 |')
L.append('| T3 | Tip-Adapter（全局 cache）、APE、HOSO、2026 token 级 CLIP 适配'
         '（attention-guided test-time prompt tuning） | 记忆检索从整图单向量变成 patch-patch '
         'top-k 相似度；本实现用"每类 patch 原型"近似 support token 集 | **与已有 token 级适配同源**，'
         '本轮只是把支持集换成类原型 + top-k 的廉价变体，未引入新的交互算子 |')
L.append('| T4 | 视频/TSM 类通道划分变体 | 通道划分发生在 **token 维度**（每个 patch 的 768 维切 4 段）'
         '再各自线性后合并 | 否（保留 patch 结构），但本质是分组线性，接近重参数化 |')
L.append('')
L.append('## 阶段状态')
L.append('')
L.append('阶段 2（模块实现 + 源域训练/选择）**完成**：四个模块均已实现、训练并记录参数量/步数/耗时/'
         '显存峰值；最优配置逐条上报。**阶段 3（未参与挑选的两套确认集：crossed 网格 ckplus_prior/'
         'kdef_prior 全 ratio、第二源域 KDEF-source）的评估与 shared-unit paired bootstrap 尚未执行**，'
         '下一轮将按"两族各自最好结果为参照列"的纪律逐格报出，第二编码器因 L/14 token 权重缺失留空。')
L.append('')
(OUT / 'phase2_stage2_report.md').write_text('\n'.join(L), encoding='utf-8')
man = json.loads((OUT / 'MANIFEST.json').read_text(encoding='utf-8'))
man['stage2_training'] = {'grid': {'lr': [0.001, 0.003], 'epochs': 8, 'batch': 128},
                          'runs': tr, 'total_seconds': 179.7,
                          'source_probeval_reference': REF,
                          'best_per_module': {k: v[0] for k, v in best.items()}}
man.setdefault('commands', []).append('python _m1.py (train T1-T4 on source only)')
for p in [OUT / 'phase2_stage2_report.md', OUT / 'phase2_training.json']:
    man.setdefault('output_hashes', {})[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
(OUT / 'MANIFEST.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
print('wrote phase2_stage2_report.md')
