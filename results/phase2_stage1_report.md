# Token 级模块缝合 — Phase 2 阶段 1 报告：全量 patch token 提取

执行：DeepSeek（deepseek-flash），GPU（RTX 3080）。未训练、未评估；未读锁定 test 与旧 2250、未改 main_rev.tex、未投稿。

## 提取结果（fp16，每 512 图一分片，先落盘再合并）

| 数据集 | n | token 形状 | 分片数 | 合并文件 | 合并 sha256 前缀 | 耗时 | 峰值显存 |
|---|---|---|---|---|---|---|---|
| fer2013_train | 28709 | [28709, 196, 768] | 57 | token_cache_vitb16/fer2013_train_tokens_fp16.npz | 05dec03f6867be00 | 213.6 s | 1.70 GB |
| fer2013_test | 7178 | [7178, 196, 768] | 15 | token_cache_vitb16/fer2013_test_tokens_fp16.npz | 1c182eb2fc04e78d | 37.5 s | 1.70 GB |
| ckplus_test | 902 | [902, 196, 768] | 2 | token_cache_vitb16/ckplus_test_tokens_fp16.npz | 0ea35bfb4731efda | 4.8 s | 1.70 GB |
| kdef_test | 2938 | [2938, 196, 768] | 6 | token_cache_vitb16/kdef_test_tokens_fp16.npz | ec9e06fe3523909f | 26.1 s | 1.70 GB |

合计 **11.69 GB**（train 8.45 + test 2.11 + ckplus 0.27 + kdef 0.86 GB），与 Phase 1 估计 11.68 GB 吻合。分片 sha256 全量记录在 phase2_extraction.json（每数据集 2–57 片，逐片列出）。**FER+ 未重复提取**（复用 fer2013_test token + FER+ 标签）。

## 形状与对齐校验（对合并件复核）

| 数据集 | token 形状 | dtype | 网格 | label 与既有 512 维缓存一致 | id 顺序一致 | token 范数均值 |
|---|---|---|---|---|---|---|
| fer2013_train | [28709, 196, 768] | float16 | 14×14 | 是 | 是 | 21.27 |
| fer2013_test | [7178, 196, 768] | float16 | 14×14 | 是 | 是 | 21.35 |
| ckplus_test | [902, 196, 768] | float16 | 14×14 | 是 | 是 | 21.13 |
| kdef_test | [2938, 196, 768] | float16 | 14×14 | 是 | 是 | 21.18 |

Phase 1 已在 200+200 张上确认池化+投影嵌入与缓存 view0 余弦 1.0；本轮复核确认**合并件的样本顺序、标签与既有 512 维缓存逐条一致**，token 形状 196×768、dtype float16、14×14 网格，token 范数均值 ≈ 21（未归一化 transformer 输出，符合预期）。

## 第二编码器（ViT-L/14）权重可得性

**本地不可得**：在 D:/ResearchVault/99system/models/ 下按 *L-14* 检索为空（提取记录字段 vitl14_weights = []）。注意 data/features_vitl14/ 只有 512 维全局特征缓存、**没有 patch token**；因此 token 级第二编码器确认本轮无法进行，**标记留空**，Phase 3 确认集只含 crossed 网格 + 第二源域 KDEF-source 两套，并如实标注该缺口。

## 读取 / 写入清单

- 读：data/manifests/{fer2013_train,fer2013_test,ckplus_test,kdef_test}.csv、models/open_clip/ViT-B-16.pt、data/features/*.npz（仅用于 id/label 对齐复核）。
- 写：data/token_cache_vitb16/<dataset>/shard_XXXX.npz（分片）与 data/token_cache_vitb16/<dataset>_tokens_fp16.npz（合并）；本目录 phase2_extraction.json。
