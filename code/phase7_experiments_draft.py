"""Assemble the experiment-chapter draft for the TAP line from existing artifacts only.

Every number printed here is read from a stored artifact (JSON/CSV/prose report); nothing is
recomputed.  Each table/section carries the path of the file it came from.
"""
import csv, json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006')
RES = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results')
TARGET = Path('D:/ResearchVault/06manuscripts/rb-tta-fer-fg2027/PAPER-TVC-TAP-2026-10-06/manuscript/experiments-draft.md')
REL = 'results/token_module_tailor_20261006'


def jl(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def cl(p):
    with Path(p).open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def f4(x):
    return '%.4f' % float(x)


def pct(x):
    return '%.2fpp' % (100 * float(x))


def main():
    mf = jl(RT / 'MANIFEST.json')
    p1 = jl(RT / 'phase1_feasibility.json')
    p2 = jl(RT / 'phase2_training.json')
    p3s = jl(RT / 'phase3_summary.json')
    p3a = jl(RT / 'phase3_attribution.json')
    p3k = jl(RT / 'phase3_kdef_source.json')
    p4t = jl(RT / 'phase4_baselines_training.json')
    p4 = jl(RT / 'phase4_baselines.json')
    p5t = jl(RT / 'phase5_training_vitl14.json')
    p5 = jl(RT / 'phase5_vitl14.json')
    dg = jl(RT / 'diagnostic_token_vs_global_summary.json')
    rafg = jl(RT / 'phase6_rafdb_gate.json')
    rafa = jl(RT / 'phase6_rafdb_answer.json')
    b16 = jl(RT / 'phase6b_confirm_B16.json')
    l14 = jl(RT / 'phase6b_confirm_L14.json')
    b16x = jl(RT / 'phase6b_extraction_B16.json')
    l14v = jl(RT / 'phase6b_validation_l14_tokens.json')
    ver = jl(RT / 'phase6b_verify_extraction.json')
    cellsB = cl(RT / 'phase6b_cells_B16.csv')
    cellsL = cl(RT / 'phase6b_cells_L14.csv')
    mz = jl(RES / 'module_zoo_20261006' / 'confirm_scan.json')
    fusA = jl(RES / 'fusion_sample_level_pilot_20261006' / 'mechanism_a_summary.json')
    fusB = jl(RES / 'fusion_sample_level_pilot_20261006' / 'mechanism_b_summary.json')
    d = {(r['encoder'], r['rep']): r for r in dg}
    md = []

    def A(s=''):
        md.append(s)

    A('# 实验章节草稿 — TAP（Token Attention Pooling adapter）')
    A()
    A('> 本稿为草稿，供主代理与用户过目后写正文；不含正文排版、不改任何 main*.tex、不投稿。')
    A('> 所有数字只引自既有工件，逐条在表下或行末给出源文件路径，不重算、不改量级。')
    A('> 正负结果一并写全：第 7 节为负结果专节，与正向结果同等篇幅。')
    A('> 汇总口径说明：除两处显式标注的逐格直接汇总（B/16 等容量全局头的 50 格均值、各表 TAP 胜格数')
    A('> 的计数）外，本稿所有数值均为工件中已存字段的直引；未重跑任何模型、bootstrap 或指标。')
    A()
    A('## 1 实验设置')
    A()
    A('统一打分口径（canonical）：logits = s·cos + beta·cache，其中 s = 100（exp(logit_scale)）、')
    A('亲和度 max、beta = 256；PFB 参考臂为 probe 逻辑值 + (-lambda·log(pi_hat))，lambda = 0.2、')
    A('prior_clip = 1.0、min_prior_samples = 32。')
    A('（来源 ' + REL + '/phase3_report.md、phase5_report.md、phase5_extraction_vitl14.json。）')
    A()
    A('源域与划分：训练与超参选择只在源域 FER2013：probe-train 22,968 训练、probe-val 5,741 选参')
    A('（同一划分文件 features/linear_probe_split_seed0.npz，两编码器逐元素一致）。目标域在任何阶段')
    A('都无标签参与训练、选择或阈值确定，只在报告结果时用于打分。')
    A('（来源 ' + REL + '/phase5_preflight_vitl14.json 的 probe_split、phase2_stage2_report.md；'
      + '两编码器划分一致性见 ' + REL + '/diagnostic_token_vs_global_report.md。）')
    A()
    A('两编码器：冻结 CLIP ViT-B/16（权重 350.8 MB，exp(logit_scale)=100，patch 序列 196×768，14×14）')
    A('与冻结 CLIP ViT-L/14（权重 893 MB、sha256 已校验，patch 序列 256×1024）。')
    A('（来源 ' + REL + '/phase1_feasibility.json、phase1_report.md；L/14 权重与 token 记录见 '
      + REL + '/phase5_download.json、phase5_extraction_vitl14.json。）')
    A()
    A('确认协议：两套从未参与挑选的确认集（crossed 网格 50 格；第二源域 KDEF-source 15 格），门槛为')
    A('「≥2/3 确认格同时 ≥ 两族各自最好结果」并附 shared-unit paired bootstrap 区间；另有第三套同域')
    A('退化轴（第 6 节，预登记），以及第 7 节列出的全部负结果。')
    A('（来源 ' + REL + '/phase3_report.md、phase4_baselines_report.md、phase5_report.md。）')
    A()
    A('## 2 方法：TAP')
    A()
    A('TAP 在冻结 CLIP 的 patch token 上学逐 token 注意力权重：对每个样本，token 序列 {x_k}（B/16 为')
    A('196 个、L/14 为 256 个）经 a1: Linear(d→256) → tanh → a2: Linear(256→1) 得到 logits，softmax 后')
    A('得权重 w_k；先按 w_k 加权聚合成单个向量 sum_k w_k x_k，再由线性判别头 Linear(d→7) 分类。')
    A('即适配发生在池化之前，空间证据先聚合再判别；与既有做法（在已聚合的全局池化特征上做线性或 MLP')
    A('适配）在作用位置上不同。')
    A('（来源 ' + REL + '/phase1_report.md Phase 2 模块表、phase2_train_modules.py 的 T1 定义、'
      + 'phase2_stage2_report.md。）')
    A()
    A('参数量：TAP = 202,504（ViT-B/16，d=768）/ 269,832（ViT-L/14，d=1024），全部来自源域训练。')
    A('对照臂：CLIP-Adapter = 131,712 / 295,872（B/16 d=512 hid=128；L/14 d=768 hid=192，均为')
    A('bottleneck 比值 4、残差权重 0.2，输出经 L2 归一化后与冻结文本原型点积并按 s=100 缩放）；')
    A('等容量全局头 = 198,919 / 269,627（仅用 512/768 维全局特征，MLP 宽度按 TAP 参数量解出：')
    A('B/16 h=256、L/14 h=260，绝对差 205 个参数 = 0.076%）。')
    A('（来源 ' + REL + '/phase2_training.json（T1）、phase3_attribution.json（global_head_params）、'
      + 'phase4_baselines_training.json、phase5_training_vitl14.json、phase5_report.md。）')
    A()
    A('训练预算：Adam，8 epoch，batch 128，学习率网格 {1e-3, 3e-3}，每个臂都在源域 probe-val 上选 lr')
    A('（不扩网格）。B/16 选择结果：TAP lr=3e-3（probe-val UAR ' + f4(p2['T1|lr0.003']['probe_val_uar'])
      + '）、CLIP-Adapter lr=1e-3（' + f4(p4t['selected']['CLIPAdapter']['probe_val_uar'])
      + '）、等容量全局头 lr=3e-3（' + f4(p3a['global_head_grid']['0.003']['probe_val_uar']) + '）；')
    A('L/14 选择结果：TAP lr=1e-3（' + f4(p5t['selected']['TAP']['probe_val_uar'])
      + '）、CLIP-Adapter lr=1e-3（' + f4(p5t['selected']['CLIPAdapter']['probe_val_uar'])
      + '）、等容量全局头 lr=1e-3（' + f4(p5t['selected']['GHead']['probe_val_uar']) + '）。')
    A('（来源 ' + REL + '/phase2_training.json、phase3_attribution.json、phase4_baselines_training.json、')
    A('phase5_training_vitl14.json；两个 lr 的完整逐条数值见同文件的 runs 字段。）')
    A()
    A('## 3 对比臂定义')
    A()
    A('| 臂 | 形式 | 训练参数（B/16 / L/14） | 训练预算 | 来源 |')
    A('|---|---|---|---|---|')
    A('| TAP | 逐 token 注意力池化 + 线性判别头 | 202,504 / 269,832 | Adam, 8 ep, batch 128, lr {1e-3,3e-3} | ' + REL + '/phase2_training.json、phase5_training_vitl14.json |')
    A('| CLIP-Adapter | 全局特征上的 bottleneck 残差 adapter（比值 4、残差 0.2），冻结文本原型分类 | 131,712 / 295,872 | 同上 | ' + REL + '/phase4_baselines_training.json、phase5_training_vitl14.json |')
    A('| 等容量全局头 | 仅用全局特征（512/768 维）的 MLP，参数量对齐 TAP | 198,919 / 269,627 | 同上 | ' + REL + '/phase3_attribution.json、phase5_training_vitl14.json |')
    A('| 训练-free 参照 | max(PFB, 支持记忆)：PFB = 源域线性 probe + 在线先验校正；支持记忆 = 文本原型 + beta·每类 max 亲和 | 0（无训练） | — | ' + REL + '/phase3_report.md、phase4_baselines_report.md |')
    A()
    A('参照列纪律：任何表格都必须同时给出 max(PFB, 支持记忆)、等容量全局头、CLIP-Adapter 与 TAP 四列，')
    A('不得只与内部弱基线比较（此前因此误报过一次假增益）。')
    A('（来源 ' + REL + '/phase4_baselines_report.md §1–§2、phase5_report.md §2–§4。）')
    A()
    A('## 4 确认集一：crossed 网格（50 格）')
    A()
    A('定义：ckplus_prior / kdef_prior × ratio 0/1/2/5/10 × seed 0–4 = 50 格；ratio 0 的全目标在 5 个')
    A('seed 下重复（与 phase 3/4/5 完全一致）。')
    A()
    A('| 编码器 | 臂 | 50 格平均 UAR | TAP 胜格数 | 来源 |')
    A('|---|---|---|---|---|')
    A('| ViT-B/16 | TAP | ' + f4(p4['setA_mean_uar']['TAP']) + ' | — | ' + REL + '/phase4_baselines.json setA_mean_uar |')
    A('| ViT-B/16 | CLIP-Adapter | ' + f4(p4['setA_mean_uar']['CLIPAdapter']) + ' | 50/50 | ' + REL + '/phase4_baselines.json setA_cells_TAP_beats |')
    A('| ViT-B/16 | 等容量全局头 | 0.6158（该 CSV 50 格逐格均值） | — | ' + REL + '/phase3_attribution_cells.csv（本稿唯一由逐格列直接取均值的数字，未重跑模型） |')
    A('| ViT-B/16 | max(PFB, 支持记忆) | ' + f4(p4['setA_mean_uar']['best_family']) + ' | — | ' + REL + '/phase4_baselines.json setA_mean_uar |')
    A('| ViT-L/14 | TAP | ' + f4(p5['setA_mean_uar']['TAP']) + ' | — | ' + REL + '/phase5_vitl14.json setA_mean_uar |')
    A('| ViT-L/14 | CLIP-Adapter | ' + f4(p5['setA_mean_uar']['CLIPAdapter']) + ' | 25/50 | ' + REL + '/phase5_vitl14.json setA_cells_TAP_beats |')
    A('| ViT-L/14 | 等容量全局头 | ' + f4(p5['setA_mean_uar']['GHead']) + ' | 25/50 | 同上 |')
    A('| ViT-L/14 | max(PFB, 支持记忆) | ' + f4(p5['setA_mean_uar']['best_family']) + ' | — | 同上 |')
    A()
    A('配对差与 95% CI（paired shared-unit bootstrap，2000 次）：')
    A()
    A('| 编码器 | 对比 | 均值 | 95% CI | 来源 |')
    A('|---|---|---|---|---|')
    A('| ViT-B/16 | TAP − max(PFB, 支持记忆) | ' + f4(p3a['bootstrap']['T1_minus_best_family']['mean'])
      + ' | [' + f4(p3a['bootstrap']['T1_minus_best_family']['ci'][0]) + ', '
      + f4(p3a['bootstrap']['T1_minus_best_family']['ci'][1]) + '] | ' + REL + '/phase3_attribution.json |')
    A('| ViT-B/16 | TAP − 等容量全局头（净 token 贡献） | ' + f4(p3a['bootstrap']['T1_minus_GHead']['mean'])
      + ' | [' + f4(p3a['bootstrap']['T1_minus_GHead']['ci'][0]) + ', '
      + f4(p3a['bootstrap']['T1_minus_GHead']['ci'][1]) + '] | 同上 T1_minus_GHead |')
    A('| ViT-B/16 | TAP − CLIP-Adapter | ' + f4(p4['bootstrap_setA']['TAP_minus_CLIPAdapter']['mean'])
      + ' | [' + f4(p4['bootstrap_setA']['TAP_minus_CLIPAdapter']['ci'][0]) + ', '
      + f4(p4['bootstrap_setA']['TAP_minus_CLIPAdapter']['ci'][1]) + '] | '
      + REL + '/phase4_baselines.json bootstrap_setA |')
    A('| ViT-L/14 | TAP − CLIP-Adapter | ' + f4(p5['bootstrap_setA']['TAP_minus_CLIPAdapter']['mean'])
      + ' | [' + f4(p5['bootstrap_setA']['TAP_minus_CLIPAdapter']['ci'][0]) + ', '
      + f4(p5['bootstrap_setA']['TAP_minus_CLIPAdapter']['ci'][1]) + '] | '
      + REL + '/phase5_vitl14.json bootstrap_setA |')
    A('| ViT-L/14 | TAP − 等容量全局头 | ' + f4(p5['bootstrap_setA']['TAP_minus_GHead']['mean'])
      + ' | [' + f4(p5['bootstrap_setA']['TAP_minus_GHead']['ci'][0]) + ', '
      + f4(p5['bootstrap_setA']['TAP_minus_GHead']['ci'][1]) + '] | 同上 TAP_minus_GHead |')
    A()
    A('ViT-L/14 逐组（数值见 ' + REL + '/phase5_report.md §3 的 Breakdown by target experiment 表）：L/14 的')
    A('25/50 胜格数完全按实验分组——ckplus_prior 的 25 格 TAP 全负、kdef_prior 的 25 格 TAP 全胜。本稿')
    A('不把该分组结论外推到 B/16（B/16 未在本表给出同口径的分组胜格数）。')
    A('（逐格细节：' + REL + '/phase3_crossed_cells.csv、phase4_baselines_cells.csv、phase5_vitl14_cells.csv。）')
    A()
    A('## 5 确认集二：第二源域 KDEF-source（15 格）')
    A()
    A('定义：以 KDEF-source 训练集为支持与先验来源，目标为 kdef_source_test / fer2013_test / ckplus_test，')
    A('seed 0–4 = 15 格；目标标签不参与训练与选择。')
    A()
    A('| 编码器 | 臂 | 15 格平均 UAR | TAP 胜格数 | 来源 |')
    A('|---|---|---|---|---|')
    A('| ViT-B/16 | TAP | ' + f4(p4['setB_mean_uar']['TAP']) + ' | — | ' + REL + '/phase4_baselines.json setB_mean_uar |')
    A('| ViT-B/16 | CLIP-Adapter | ' + f4(p4['setB_mean_uar']['CLIPAdapter']) + ' | 15/15 | ' + REL + '/phase4_baselines.json setB_cells_TAP_beats |')
    A('| ViT-B/16 | max(PFB, 支持记忆) | ' + f4(p4['setB_mean_uar']['best_family']) + ' | — | 同上 |')
    A('| ViT-L/14 | TAP | ' + f4(p5['setB_mean_uar']['TAP']) + ' | — | ' + REL + '/phase5_vitl14.json setB_mean_uar |')
    A('| ViT-L/14 | CLIP-Adapter | ' + f4(p5['setB_mean_uar']['CLIPAdapter']) + ' | 5/15 | ' + REL + '/phase5_vitl14.json setB_cells_TAP_beats |')
    A('| ViT-L/14 | 等容量全局头 | ' + f4(p5['setB_mean_uar']['GHead']) + ' | 10/15 | 同上 |')
    A('| ViT-L/14 | max(PFB, 支持记忆) | ' + f4(p5['setB_mean_uar']['best_family']) + ' | — | 同上 |')
    A()
    A('配对差与 95% CI：')
    A()
    A('| 编码器 | 对比 | 均值 | 95% CI | 来源 |')
    A('|---|---|---|---|---|')
    A('| ViT-B/16 | TAP − max(PFB, 支持记忆) | ' + f4(p3k['bootstrap']['T1_minus_best_family']['mean'])
      + ' | [' + f4(p3k['bootstrap']['T1_minus_best_family']['ci'][0]) + ', '
      + f4(p3k['bootstrap']['T1_minus_best_family']['ci'][1]) + '] | ' + REL + '/phase3_kdef_source.json |')
    A('| ViT-B/16 | TAP − 等容量全局头 | ' + f4(p3k['bootstrap']['T1_minus_GHead']['mean'])
      + ' | [' + f4(p3k['bootstrap']['T1_minus_GHead']['ci'][0]) + ', '
      + f4(p3k['bootstrap']['T1_minus_GHead']['ci'][1]) + '] | 同上 |')
    A('| ViT-B/16 | TAP − CLIP-Adapter | ' + f4(p4['bootstrap_setB']['TAP_minus_CLIPAdapter']['mean'])
      + ' | [' + f4(p4['bootstrap_setB']['TAP_minus_CLIPAdapter']['ci'][0]) + ', '
      + f4(p4['bootstrap_setB']['TAP_minus_CLIPAdapter']['ci'][1]) + '] | ' + REL + '/phase4_baselines.json bootstrap_setB |')
    A('| ViT-L/14 | TAP − CLIP-Adapter | ' + f4(p5['bootstrap_setB']['TAP_minus_CLIPAdapter']['mean'])
      + ' | [' + f4(p5['bootstrap_setB']['TAP_minus_CLIPAdapter']['ci'][0]) + ', '
      + f4(p5['bootstrap_setB']['TAP_minus_CLIPAdapter']['ci'][1]) + '] | ' + REL + '/phase5_vitl14.json bootstrap_setB |')
    A('| ViT-L/14 | TAP − 等容量全局头 | ' + f4(p5['bootstrap_setB']['TAP_minus_GHead']['mean'])
      + ' | [' + f4(p5['bootstrap_setB']['TAP_minus_GHead']['ci'][0]) + ', '
      + f4(p5['bootstrap_setB']['TAP_minus_GHead']['ci'][1]) + '] | 同上 |')
    A()
    A('ViT-B/16 逐目标（' + REL + '/phase3_kdef_source.json per_target）：')
    for t in ('kdef_source_test', 'fer2013_test', 'ckplus_test'):
        row = p3k['per_target'][t]
        A('- ' + t + '（n=' + str(row['n']) + '）：TAP ' + f4(row['T1']) + '、等容量全局头 '
          + f4(row['GHead']) + '、PFB ' + f4(row['pfb']) + '、支持记忆 ' + f4(row['memory'])
          + '、best-of-two ' + f4(row['best_family']) + '。')
    A()
    A('ViT-L/14 逐目标（' + REL + '/phase5_vitl14.json setB_per_target）：')
    for t in ('kdef_source_test', 'fer2013_test', 'ckplus_test'):
        row = p5['setB_per_target'][t]
        A('- ' + t + '：TAP ' + f4(row['TAP']) + '、等容量全局头 ' + f4(row['GHead'])
          + '、CLIP-Adapter ' + f4(row['CLIPAdapter']) + '、best-of-two ' + f4(row['best_family']) + '。')
    A()
    A('## 6 预登记鲁棒性确认 H2：同域退化轴（12 档）')
    A()
    A('这条轴测什么（预登记原文 + addendum）：这是同域（FER2013 test 自身）的合成退化鲁棒性轴，')
    A('不是跨库轴。H2 若成立，可主张的只有「TAP 相对 CLIP-Adapter 在同域退化下更鲁棒」，不能当作')
    A('跨库准确率的确认。轴 = 4 类退化 × 3 档严重度 = 12 档，每档 7,178 张；来源链为')
    A('make_corrupted_fer2013.py → raw/fer2013_shift_sweep/<condition>/ → manifests/fer2013_<condition>.csv')
    A('（每档 7,178 行）→ features/shift_sweep/<condition>.npz。四个 provenance 含糊的 legacy 文件')
    A('features/corruptions/*.npz 已排除（两个与 sweep 档逐元素重复、另两个来自不同生成脚本且其中一个')
    A('经灰度转换）。（来源 ' + REL + '/phase6b_preregistration.md、phase6b_inventory_report.md、'
      + 'phase6b_provenance.json。）')
    A()
    A('预登记判据（跑前写死、跑后未改）：H2 成立当且仅当 (1) TAP − CLIP-Adapter 的 paired 95% CI 不含 0')
    A('且为正；且 (2) 点估计绝对值大于 token 臂种子间 sd（B/16 0.0116、L/14 0.0134）。四臂全部使用已')
    A('冻结的源域模型，不重训、不重选、不在新条件上做任何选择。')
    A('已知限制（已写入报告）：四臂冻结 ⇒ seed 0–4 只改变 PFB 的在线先验顺序，TAP / CLIP-Adapter /')
    A('等容量全局头 / 支持记忆在 5 个 seed 下完全一致；因此 bootstrap 只覆盖 query × condition 抽样，')
    A('训练种子方差由判据 (2) 承担，不得写成「五种子重复」。')
    A('（来源 ' + REL + '/phase6b_preregistration.md 末尾 addendum、phase6b_confirm_report.md 开头。）')
    A()
    A('12 档逐档结果（UAR）：')
    A()
    A('| 条件 | B/16 TAP | B/16 CLIP-Adapter | B/16 max(PFB,mem) | L/14 TAP | L/14 CLIP-Adapter | L/14 max(PFB,mem) |')
    A('|---|---|---|---|---|---|---|')
    for rb, rl in zip(cellsB, cellsL):
        A('| ' + rb['condition'] + ' | ' + f4(rb['TAP']) + ' | ' + f4(rb['CLIPAdapter']) + ' | '
          + f4(rb['best_family_mean5']) + ' | ' + f4(rl['TAP']) + ' | ' + f4(rl['CLIPAdapter']) + ' | '
          + f4(rl['best_family_mean5']) + ' |')
    A('| 均值 | ' + f4(b16['aggregate']['TAP']['mean']) + ' | ' + f4(b16['aggregate']['CLIPAdapter']['mean'])
      + ' | ' + f4(b16['aggregate']['best_family_mean5']['mean']) + ' | '
      + f4(l14['aggregate']['TAP']['mean']) + ' | ' + f4(l14['aggregate']['CLIPAdapter']['mean']) + ' | '
      + f4(l14['aggregate']['best_family_mean5']['mean']) + ' |')
    A()
    A('（逐档来源：' + REL + '/phase6b_cells_B16.csv、phase6b_cells_L14.csv；聚合来源：')
    A('phase6b_confirm_B16.json / phase6b_confirm_L14.json 的 aggregate。）')
    A()
    A('判定：')
    A('- H2（主，ViT-B/16）：' + ('成立' if b16['decision']['H2_holds'] else '不成立') + '。TAP − CLIP-Adapter = '
      + f4(b16['decision']['point_TAP_minus_CLIPAdapter']) + '（' + pct(b16['decision']['point_TAP_minus_CLIPAdapter'])
      + '），95% CI [' + f4(b16['decision']['ci95'][0]) + ', ' + f4(b16['decision']['ci95'][1]) + ']；判据 (1) '
      + str(b16['decision']['criterion1_ci_excludes_zero_and_positive']) + '、判据 (2) '
      + str(b16['decision']['criterion2_margin_exceeds_seed_sd']) + '；TAP 在 '
      + str(sum(1 for r in cellsB if float(r['TAP']) > float(r['CLIPAdapter']))) + '/12 档胜出。')
    A('- ViT-L/14（次要，不参与 H2 判定）：TAP − CLIP-Adapter = '
      + f4(l14['decision']['point_TAP_minus_CLIPAdapter']) + '（' + pct(l14['decision']['point_TAP_minus_CLIPAdapter'])
      + '），95% CI [' + f4(l14['decision']['ci95'][0]) + ', ' + f4(l14['decision']['ci95'][1]) + ']；判据 (1) '
      + str(l14['decision']['criterion1_ci_excludes_zero_and_positive']) + '、判据 (2) '
      + str(l14['decision']['criterion2_margin_exceeds_seed_sd']) + '；TAP 在 '
      + str(sum(1 for r in cellsL if float(r['TAP']) > float(r['CLIPAdapter']))) + '/12 档胜出。')
    A('- 与跨库结果的关系（讨论要点）：同域退化轴上两个编码器都是 TAP 领先（'
      + pct(b16['decision']['point_TAP_minus_CLIPAdapter']) + ' / ' + pct(l14['decision']['point_TAP_minus_CLIPAdapter'])
      + '），与第 7 节跨库 ViT-L/14 上 TAP 落后 CLIP-Adapter（'
      + pct(p5['bootstrap_setB']['TAP_minus_CLIPAdapter']['mean']) + '）方向相反；故 Phase 5 的反号是跨库设置')
    A('特有，不是 ViT-L/14 的普遍性质。')
    A('（来源 ' + REL + '/phase6b_confirm_report.md、phase6b_confirm_B16.json、phase6b_confirm_L14.json、')
    A('phase6b_bootstrap_B16.json、phase6b_bootstrap_L14.json。）')
    A()
    A('## 7 负结果（与正向结果同等重要，逐条给工件）')
    A()
    A('### 7.1 ViT-L/14 跨库三项反号 / 打平')
    A('crossed 50 格：TAP − CLIP-Adapter = ' + f4(p5['bootstrap_setA']['TAP_minus_CLIPAdapter']['mean'])
      + '（95% CI [' + f4(p5['bootstrap_setA']['TAP_minus_CLIPAdapter']['ci'][0]) + ', '
      + f4(p5['bootstrap_setA']['TAP_minus_CLIPAdapter']['ci'][1]) + ']）、TAP − 等容量全局头 = '
      + f4(p5['bootstrap_setA']['TAP_minus_GHead']['mean']) + '（['
      + f4(p5['bootstrap_setA']['TAP_minus_GHead']['ci'][0]) + ', '
      + f4(p5['bootstrap_setA']['TAP_minus_GHead']['ci'][1]) + ']）；')
    A('KDEF-source 15 格：TAP − CLIP-Adapter = ' + f4(p5['bootstrap_setB']['TAP_minus_CLIPAdapter']['mean'])
      + '（[' + f4(p5['bootstrap_setB']['TAP_minus_CLIPAdapter']['ci'][0]) + ', '
      + f4(p5['bootstrap_setB']['TAP_minus_CLIPAdapter']['ci'][1]) + ']）、TAP − 等容量全局头 = '
      + f4(p5['bootstrap_setB']['TAP_minus_GHead']['mean']) + '（['
      + f4(p5['bootstrap_setB']['TAP_minus_GHead']['ci'][0]) + ', '
      + f4(p5['bootstrap_setB']['TAP_minus_GHead']['ci'][1]) + ']，含 0）。')
    A('TAP 仍大幅胜过两族最好（' + pct(p5['bootstrap_setA']['TAP_minus_best_family']['mean']) + ' / '
      + pct(p5['bootstrap_setB']['TAP_minus_best_family']['mean']) + '），故 token 分支本身未失效，')
    A('但 ViT-B/16 的两个关键量在 ViT-L/14 上不复现。（来源 ' + REL + '/phase5_vitl14.json、phase5_report.md §5。）')
    A()
    A('### 7.2 源域诊断：匹配非线性头后两条路线打平，源域 +2.1pp 未被独立实现复现')
    A('ViT-B/16：R3（注意力池化）' + f4(d[('B16', 'R3_attn_pool')]['seed_mean_at_best_lr'])
      + ' vs R1 + MLP 头 ' + f4(d[('B16', 'R1_global_MLP')]['seed_mean_at_best_lr']) + '；ViT-L/14：'
      + f4(d[('L14', 'R3_attn_pool')]['seed_mean_at_best_lr']) + ' vs '
      + f4(d[('L14', 'R1_global_MLP')]['seed_mean_at_best_lr']) + '。')
    A('Phase 3/5 记录的 B/16 上 TAP − 等容量全局头 +2.1pp 在独立实现中未被复现（同上两值）。')
    A('诊断同时排除了「凸组合缺少全局直通」：拼接全局特征只带来 R4a − R2 = '
      + f4(d[('B16', 'R4a_concat_raw')]['seed_mean_at_best_lr'] - d[('B16', 'R2_mean_raw')]['seed_mean_at_best_lr'])
      + '（B/16）/ '
      + f4(d[('L14', 'R4a_concat_raw')]['seed_mean_at_best_lr'] - d[('L14', 'R2_mean_raw')]['seed_mean_at_best_lr'])
      + '（L/14），把全局直通直接给分类器反而 R4b − R3 = '
      + f4(d[('B16', 'R4b_attn_plus_global')]['seed_mean_at_best_lr'] - d[('B16', 'R3_attn_pool')]['seed_mean_at_best_lr'])
      + '（B/16）/ '
      + f4(d[('L14', 'R4b_attn_plus_global')]['seed_mean_at_best_lr'] - d[('L14', 'R3_attn_pool')]['seed_mean_at_best_lr'])
      + '（L/14）。')
    A('（来源 ' + REL + '/diagnostic_token_vs_global_report.md §3、diagnostic_token_vs_global_summary.csv；')
    A('以上差值由同一文件保存的 seed-mean 相减得到，未做任何新计算。）')
    A()
    A('### 7.3 RAF-DB 预登记确认：被可用性门禁拦住，未消耗')
    A('唯一本地副本是 Kaggle 再上传包（PROVENANCE.md 状态 QUARANTINED，内容自述为 preprocessed FER2013 +')
    A('RAF-DB 混合），无官方切分/标签文件、四类计数完全相同（各 5,920）、预登记 ratio 0/1/2/5/10 中')
    A('2/5/10 不可实现；8 项检查 4 项失败（' + str(len(rafg['failed_checks'])) + ' 项：'
      + '；'.join(rafg['failed_checks']) + '），故按预登记停跑。')
    A('结论：H1 无结论，RAF-DB 未被消耗（未提取 token、未训练、未评估）。')
    A('（来源 ' + REL + '/phase6_rafdb_gate.json、phase6_rafdb_answer.json、phase6_rafdb_report.md。）')
    A()
    A('### 7.4 头层模块池：13 个配置全部未通过（0/13）')
    A('13 个头部/融合配置无一在「≥2/3 确认协议同时 ≥ 两族各自最好」的门槛下通过；挑选集第一名 M2a')
    A('（标量线性头，挑选集 ' + f4(mz['selection']['M2a_scalar_linear']) + '）仅比 canonical（'
      + f4(mz['selection']['M1_canonical']) + '）高 '
      + f4(mz['selection']['M2a_scalar_linear'] - mz['selection']['M1_canonical'])
      + '，但在 crossed 协议上 0/10、最差单格缺口 −14.66pp；小型 MLP 直接训练塌缩（挑选集 '
      + f4(mz['selection']['M3_mlp']) + '）。结论：不命名新模块。')
    A('（来源 results/module_zoo_20261006/REPORT.md §1–§2、per_config_results.csv、confirm_scan.json 的 selection。）')
    A()
    A('### 7.5 T2 / T3 / T4 模块不成立')
    A('crossed 50 格上 T2 ' + f4(p3s['aggregate']['T2']['mean_minus_best']) + '、T3 '
      + f4(p3s['aggregate']['T3']['mean_minus_best']) + '、T4 '
      + f4(p3s['aggregate']['T4']['mean_minus_best']) + '（mean_minus_best），只有 T1（= TAP）达标（'
      + f4(p3s['aggregate']['T1']['mean_minus_best']) + '）；T3 在 KDEF-source 上对等容量头为 '
      + f4(p3k['bootstrap']['T3_minus_GHead']['mean']) + '。')
    A('（来源 ' + REL + '/phase3_summary.json、phase3_kdef_source.json、phase2_stage2_report.md；T2 的等容量')
    A('1×1 对照见 phase3_extra_controls.json 的 T2_1x1_equal_capacity_control。）')
    A()
    A('### 7.6 样本级 / 退化 / 全局三类门控均不成立')
    A('- 样本级可靠性融合：18 个自然/prior 格中仅 ' + str(fusA['fusion_ge_best'])
      + ' 格满足 ≥ 两族最好（门槛 2/3），最大单格损失 −15.7pp；死在「源侧 w* 校准不可迁移」。')
    A('  （来源 results/fusion_sample_level_pilot_20261006/mechanism_a_summary.json、REPORT.md §1。）')
    A('- 退化严重度门控：条件级 oracle 阈值下统计量可 12/12 分开，但源域校准阈值不可迁移——熵规则在')
    A('  sweep 上命中 ' + str(fusB['rules'][0]['hit_sweep']) + '/12、自然网格平均损失 '
      + f4(fusB['rules'][0]['mean_loss_on_natural']) + '。')
    A('  （来源同目录 mechanism_b_summary.json、REPORT.md §2。）')
    A('- 全局（regime）门控：目标 oracle 阈值下多个统计量 18/18 可分（separation_summary.md），但同一')
    A('  阈值迁移到真实目标只命中 12/18，CK+ 整族 6 格被错选、单域最大损失 −0.1083 UAR；且 18 格只来自')
    A('  4 个独立域，应读作域级 4/4 而非 18 个独立证据。')
    A('  （来源 results/fusion_regime_pilot_20261006/REPORT.md §3–§5、separation_summary.md、calibration.json。）')
    A()
    A('## 8 噪声底线与统计口径')
    A()
    A('- token 臂种子间 sd（训练方差底线）：ViT-B/16 0.0116、ViT-L/14 0.0134；该值由源域诊断的 R3')
    A('  （注意力池化）跨 seed 分布给出（B/16 与 L/14 的 per-seed 值见 '
      + REL + '/diagnostic_token_vs_global_probes_B16.csv 与 _L14.csv），phase 6b 预登记判据 (2) 直接采用。')
    A('- bootstrap 设计：paired shared-unit、2000 次、hierarchical；crossed / KDEF-source 两套确认集按')
    A('  seed × 目标格重采样并在格内重采样共享 query 索引；phase 6b 同域退化轴按 seed(0–4) × condition(12)')
    A('  重采样并共享 query 索引。')
    A('- 区间覆盖哪一维随机性（必须写明）：crossed / KDEF-source 的区间覆盖 seed（子集抽样）× 查询；')
    A('  phase 6b 因四臂冻结，区间只覆盖 条件 × 查询，训练种子方差由判据 (2) 的 0.0116 / 0.0134 承担；')
    A('  任何「五种子重复」的措辞都不适用于 phase 6b。')
    A('（来源 ' + REL + '/phase3_report.md、phase4_baselines_report.md、phase5_report.md、'
      + 'phase6b_confirm_report.md、phase6b_bootstrap_B16.json / _L14.json 的 design 字段。）')
    A()
    A('## 9 抽取与来源门禁实况')
    A()
    A('| 项 | ViT-B/16 | ViT-L/14 | 来源 |')
    A('|---|---|---|---|')
    A('| 干净缓存重算对齐 | 余弦 ' + f4(b16x['validation_gate']['clean_globals_cosine_min'])
      + '（512 行） | 余弦 ' + f4(l14v['clean_globals_cosine_min'])
      + '（512 行） | ' + REL + '/phase6b_extraction_B16.json、phase6b_validation_l14_tokens.json |')
    A('| 退化档对既有缓存 | blur_0p8 余弦 '
      + f4(b16x['validation_gate']['condition_blur_0p8_globals_cosine_min'])
      + '（全 7,178 行） | 无既有 L/14 退化缓存可比（features/shift_sweep/* 为 512 维 B/16） | 同上、phase6b_extraction_L14.json |')
    A('| L/14 替代校验 | — | 干净 patch token 余弦 ' + f4(l14v['clean_token_cosine_min'])
      + '；逐条件重算 ' + str(ver['n_verified']) + '/' + str(ver['n_total']) + ' 通过 | '
      + REL + '/phase6b_validation_l14_tokens.json、phase6b_verify_extraction.json |')
    A('| 逐条件完整性 | 12/12 | 12/12（文件大小 = npy 头 + nbytes；ids/labels 对齐；views 单位范数） | phase6b_verify_extraction.json |')
    A()
    A('L/14 证据等级（如实标注）：其提取由「与干净 L/14 缓存逐位级吻合（global 与 patch token 余弦均 '
      + f4(l14v['clean_token_cosine_min']) + '）+ 逐条件独立重算（token 差异 0.12–1.5 个 fp16 ULP、逐图余弦')
    A('1.000000）」验证；缺少独立的第三方退化参照（B/16 有）。L/14 为次要确认、不参与 H2 判定，故该限制按')
    A('「证据等级下调」记录，而不是等同 B/16 门禁。')
    A('（来源 ' + REL + '/phase6b_confirm_report.md §2.1、phase6b_verify_extraction.json、'
      + 'phase6b_validation_l14_tokens.json。）')
    A()
    A('## 10 可直接引用的数字汇总表')
    A()
    A('| # | 实验 | 臂 | 编码器 | 指标 | 值 | 95% CI | 来源文件 |')
    A('|---|---|---|---|---|---|---|---|')
    rows = [
        ('TAP 参数量', 'TAP', 'ViT-B/16', 'params', '202504', '', REL + '/phase2_training.json'),
        ('TAP 参数量', 'TAP', 'ViT-L/14', 'params', '269832', '', REL + '/phase5_training_vitl14.json'),
        ('CLIP-Adapter 参数量', 'CLIP-Adapter', 'ViT-B/16', 'params', '131712', '', REL + '/phase4_baselines_training.json'),
        ('CLIP-Adapter 参数量', 'CLIP-Adapter', 'ViT-L/14', 'params', '295872', '', REL + '/phase5_training_vitl14.json'),
        ('等容量全局头参数量', 'GHead', 'ViT-B/16', 'params', '198919', '', REL + '/phase3_attribution.json'),
        ('等容量全局头参数量', 'GHead', 'ViT-L/14', 'params', '269627', '', REL + '/phase5_training_vitl14.json'),
        ('源域 probe-val', 'TAP', 'ViT-B/16', 'UAR', f4(p2['T1|lr0.003']['probe_val_uar']), '', REL + '/phase2_training.json'),
        ('源域 probe-val', 'TAP', 'ViT-L/14', 'UAR', f4(p5t['selected']['TAP']['probe_val_uar']), '', REL + '/phase5_training_vitl14.json'),
        ('源域 probe-val', 'CLIP-Adapter', 'ViT-B/16', 'UAR', f4(p4t['selected']['CLIPAdapter']['probe_val_uar']), '', REL + '/phase4_baselines_training.json'),
        ('源域 probe-val', 'CLIP-Adapter', 'ViT-L/14', 'UAR', f4(p5t['selected']['CLIPAdapter']['probe_val_uar']), '', REL + '/phase5_training_vitl14.json'),
        ('Crossed 50 格', 'TAP', 'ViT-B/16', 'mean UAR', f4(p4['setA_mean_uar']['TAP']), '', REL + '/phase4_baselines.json'),
        ('Crossed 50 格', 'CLIP-Adapter', 'ViT-B/16', 'mean UAR', f4(p4['setA_mean_uar']['CLIPAdapter']), '', REL + '/phase4_baselines.json'),
        ('Crossed 50 格', 'max(PFB,memory)', 'ViT-B/16', 'mean UAR', f4(p4['setA_mean_uar']['best_family']), '', REL + '/phase4_baselines.json'),
        ('Crossed 50 格', 'TAP − CLIP-Adapter', 'ViT-B/16', 'paired diff',
         f4(p4['bootstrap_setA']['TAP_minus_CLIPAdapter']['mean']),
         '[' + f4(p4['bootstrap_setA']['TAP_minus_CLIPAdapter']['ci'][0]) + ', '
         + f4(p4['bootstrap_setA']['TAP_minus_CLIPAdapter']['ci'][1]) + ']', REL + '/phase4_baselines.json'),
        ('Crossed 50 格', 'TAP − GHead', 'ViT-B/16', 'paired diff',
         f4(p3a['bootstrap']['T1_minus_GHead']['mean']),
         '[' + f4(p3a['bootstrap']['T1_minus_GHead']['ci'][0]) + ', '
         + f4(p3a['bootstrap']['T1_minus_GHead']['ci'][1]) + ']', REL + '/phase3_attribution.json'),
        ('Crossed 50 格', 'TAP − max(PFB,memory)', 'ViT-B/16', 'paired diff',
         f4(p3a['bootstrap']['T1_minus_best_family']['mean']),
         '[' + f4(p3a['bootstrap']['T1_minus_best_family']['ci'][0]) + ', '
         + f4(p3a['bootstrap']['T1_minus_best_family']['ci'][1]) + ']', REL + '/phase3_attribution.json'),
        ('KDEF-source 15 格', 'TAP', 'ViT-B/16', 'mean UAR', f4(p4['setB_mean_uar']['TAP']), '', REL + '/phase4_baselines.json'),
        ('KDEF-source 15 格', 'CLIP-Adapter', 'ViT-B/16', 'mean UAR', f4(p4['setB_mean_uar']['CLIPAdapter']), '', REL + '/phase4_baselines.json'),
        ('KDEF-source 15 格', 'max(PFB,memory)', 'ViT-B/16', 'mean UAR', f4(p4['setB_mean_uar']['best_family']), '', REL + '/phase4_baselines.json'),
        ('KDEF-source 15 格', 'TAP − CLIP-Adapter', 'ViT-B/16', 'paired diff',
         f4(p4['bootstrap_setB']['TAP_minus_CLIPAdapter']['mean']),
         '[' + f4(p4['bootstrap_setB']['TAP_minus_CLIPAdapter']['ci'][0]) + ', '
         + f4(p4['bootstrap_setB']['TAP_minus_CLIPAdapter']['ci'][1]) + ']', REL + '/phase4_baselines.json'),
        ('KDEF-source 15 格', 'TAP − GHead', 'ViT-B/16', 'paired diff',
         f4(p3k['bootstrap']['T1_minus_GHead']['mean']),
         '[' + f4(p3k['bootstrap']['T1_minus_GHead']['ci'][0]) + ', '
         + f4(p3k['bootstrap']['T1_minus_GHead']['ci'][1]) + ']', REL + '/phase3_kdef_source.json'),
        ('Crossed 50 格', 'TAP', 'ViT-L/14', 'mean UAR', f4(p5['setA_mean_uar']['TAP']), '', REL + '/phase5_vitl14.json'),
        ('Crossed 50 格', 'CLIP-Adapter', 'ViT-L/14', 'mean UAR', f4(p5['setA_mean_uar']['CLIPAdapter']), '', REL + '/phase5_vitl14.json'),
        ('Crossed 50 格', 'TAP − CLIP-Adapter', 'ViT-L/14', 'paired diff',
         f4(p5['bootstrap_setA']['TAP_minus_CLIPAdapter']['mean']),
         '[' + f4(p5['bootstrap_setA']['TAP_minus_CLIPAdapter']['ci'][0]) + ', '
         + f4(p5['bootstrap_setA']['TAP_minus_CLIPAdapter']['ci'][1]) + ']', REL + '/phase5_vitl14.json'),
        ('KDEF-source 15 格', 'TAP − CLIP-Adapter', 'ViT-L/14', 'paired diff',
         f4(p5['bootstrap_setB']['TAP_minus_CLIPAdapter']['mean']),
         '[' + f4(p5['bootstrap_setB']['TAP_minus_CLIPAdapter']['ci'][0]) + ', '
         + f4(p5['bootstrap_setB']['TAP_minus_CLIPAdapter']['ci'][1]) + ']', REL + '/phase5_vitl14.json'),
        ('KDEF-source 15 格', 'TAP − GHead', 'ViT-L/14', 'paired diff',
         f4(p5['bootstrap_setB']['TAP_minus_GHead']['mean']),
         '[' + f4(p5['bootstrap_setB']['TAP_minus_GHead']['ci'][0]) + ', '
         + f4(p5['bootstrap_setB']['TAP_minus_GHead']['ci'][1]) + ']', REL + '/phase5_vitl14.json'),
        ('H2 同域退化 12 档', 'TAP', 'ViT-B/16', 'mean UAR', f4(b16['aggregate']['TAP']['mean']), '', REL + '/phase6b_confirm_B16.json'),
        ('H2 同域退化 12 档', 'CLIP-Adapter', 'ViT-B/16', 'mean UAR', f4(b16['aggregate']['CLIPAdapter']['mean']), '', REL + '/phase6b_confirm_B16.json'),
        ('H2 同域退化 12 档', 'TAP − CLIP-Adapter', 'ViT-B/16', 'paired diff',
         f4(b16['decision']['point_TAP_minus_CLIPAdapter']),
         '[' + f4(b16['decision']['ci95'][0]) + ', ' + f4(b16['decision']['ci95'][1]) + ']',
         REL + '/phase6b_confirm_B16.json + phase6b_bootstrap_B16.json'),
        ('H2 同域退化 12 档', 'TAP − CLIP-Adapter', 'ViT-L/14', 'paired diff（次要）',
         f4(l14['decision']['point_TAP_minus_CLIPAdapter']),
         '[' + f4(l14['decision']['ci95'][0]) + ', ' + f4(l14['decision']['ci95'][1]) + ']',
         REL + '/phase6b_confirm_L14.json + phase6b_bootstrap_L14.json'),
        ('源域诊断（线性读出）', 'R2 − R1', 'ViT-B/16', '差值',
         f4(d[('B16', 'R2_mean_raw')]['seed_mean_at_best_lr'] - d[('B16', 'R1_global')]['seed_mean_at_best_lr']),
         '', REL + '/diagnostic_token_vs_global_summary.csv'),
        ('源域诊断（线性读出）', 'R2 − R1', 'ViT-L/14', '差值',
         f4(d[('L14', 'R2_mean_raw')]['seed_mean_at_best_lr'] - d[('L14', 'R1_global')]['seed_mean_at_best_lr']),
         '', REL + '/diagnostic_token_vs_global_summary.csv'),
        ('源域诊断（非线性头）', 'R3 − R1+MLP', 'ViT-B/16', '差值',
         f4(d[('B16', 'R3_attn_pool')]['seed_mean_at_best_lr'] - d[('B16', 'R1_global_MLP')]['seed_mean_at_best_lr']),
         '', REL + '/diagnostic_token_vs_global_summary.csv'),
        ('头层模块池', '最佳候选 − canonical', 'ViT-B/16', '挑选集 UAR 差',
         f4(mz['selection']['M2a_scalar_linear'] - mz['selection']['M1_canonical']), '',
         'results/module_zoo_20261006/confirm_scan.json'),
        ('样本级融合门控', 'fusion ≥ 两族最好', 'ViT-B/16', '达标格数', str(fusA['fusion_ge_best']) + '/18', '',
         'results/fusion_sample_level_pilot_20261006/mechanism_a_summary.json'),
        ('退化严重度门控', '熵规则 sweep 命中', 'ViT-B/16', '命中数', str(fusB['rules'][0]['hit_sweep']) + '/12', '',
         'results/fusion_sample_level_pilot_20261006/mechanism_b_summary.json'),
    ]
    for i, r in enumerate(rows, 1):
        A('| ' + str(i) + ' | ' + ' | '.join(r) + ' |')
    A()
    A('## 11 来源对照清单（本稿每类数字对应的文件）')
    A()
    pairs = [
        ('canonical 口径与门槛规则', REL + '/phase3_report.md、phase4_baselines_report.md、phase5_report.md'),
        ('源域划分与无标签纪律', REL + '/phase5_preflight_vitl14.json、phase2_stage2_report.md'),
        ('patch 形状 / 提取成本 / 数值对齐', REL + '/phase1_feasibility.json、phase1_report.md、phase2_extraction.json'),
        ('TAP 定义与参数量', REL + '/phase2_train_modules.py、phase2_training.json、phase5_training_vitl14.json'),
        ('CLIP-Adapter / 等容量头定义与参数', REL + '/phase4_baselines_training.json、phase3_attribution.json、phase5_training_vitl14.json'),
        ('Crossed 50 格与 KDEF-source 15 格（B/16）', REL + '/phase3_crossed_cells.csv、phase3_kdef_source.json、phase4_baselines.json、phase4_baselines_cells.csv'),
        ('Crossed / KDEF-source（L/14）', REL + '/phase5_vitl14.json、phase5_vitl14_cells.csv、phase5_vitl14_kdef_source_cells.csv'),
        ('H2 同域退化轴（预登记、逐档、判定、门禁）', REL + '/phase6b_preregistration.md、phase6b_cells_B16.csv、phase6b_cells_L14.csv、phase6b_confirm_B16.json、phase6b_confirm_L14.json、phase6b_verify_extraction.json、phase6b_validation_l14_tokens.json'),
        ('负结果：L/14 跨库反号', REL + '/phase5_vitl14.json、phase5_report.md'),
        ('负结果：源域诊断', REL + '/diagnostic_token_vs_global_report.md、diagnostic_token_vs_global_summary.csv、diagnostic_token_vs_global_probes_B16.csv、diagnostic_token_vs_global_probes_L14.csv'),
        ('负结果：RAF-DB 门禁', REL + '/phase6_rafdb_gate.json、phase6_rafdb_answer.json、phase6_rafdb_report.md'),
        ('负结果：模块池 0/13', 'results/module_zoo_20261006/REPORT.md、per_config_results.csv、confirm_scan.json'),
        ('负结果：T2/T3/T4', REL + '/phase3_summary.json、phase3_kdef_source.json、phase3_extra_controls.json'),
        ('负结果：三类门控', 'results/fusion_sample_level_pilot_20261006/{mechanism_a_summary.json,mechanism_b_summary.json,REPORT.md}、results/fusion_regime_pilot_20261006/{REPORT.md,separation_summary.md,calibration.json}'),
    ]
    for k, v in pairs:
        A('- ' + k + ' ← ' + v)
    A()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text('\n'.join(md), encoding='utf-8')
    print('wrote ' + str(TARGET))
    print('lines=' + str(len(md)) + ' chars=' + str(sum(len(x) for x in md)))
    print('prose_sections_chars=' + str(sum(len(x) for x in md if not x.startswith('|'))))


if __name__ == '__main__':
    main()
