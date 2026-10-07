# Token 级模块缝合路线 — Phase 1（可行性/成本）报告 + Phase 2 模块表

执行：DeepSeek（deepseek-flash）。Phase 1 用 GPU（RTX 3080）做小样本前向校验，未做全量提取、未训练、未评估。canonical 口径沿用 s=100/max/beta=256/prior_clip=1.0。未读锁定 test 与旧 2250、未改 main_rev.tex、未投稿。

## Phase 1.1 环境与模型

- open_clip 3.3.0、torch 2.8.0+cu128，CUDA 可用，设备 **NVIDIA GeForce RTX 3080（10.74 GB）**。
- 权重 D:/ResearchVault/99system/models/open_clip/ViT-B-16.pt（350.8 MB）在位；exp(logit_scale)=100 与 canonical 口径一致。
- patch token 获取方式：设 model.visual.output_tokens = True，调用 pooled, tokens = model.visual(x)；tokens 已是 **196×768** 的 patch 序列（CLS 已由内部 _pool 剥离，不能再切一刀——这是我第一次写错并已修正的点）。

## Phase 1.2 小样本形状与数值一致性（CK+ 200 + KDEF 200，未全量提取）

| 数据集 | n | token 形状 | dtype | 网格 | 池化嵌入(归一) vs 既有 512 维缓存 view0 余弦 | 吞吐 |
|---|---|---|---|---|---|---|
| ckplus | 200 | [200, 196, 768] | float16 | 14×14 | min 0.9999999 / mean 1.0000000 | 221.7 img/s |
| kdef | 200 | [200, 196, 768] | float16 | 14×14 | min 0.9999998 / mean 1.0000000 | 130.4 img/s |

**结论：patch token 与现有协议完全对齐** —— 池化+投影后的 512 维嵌入与既有缓存 view0 的余弦为 0.9999998~1.0000002（浮点误差级），说明预处理、权重与既有 features/*.npz 是同一套；token 形状 196×768（14×14）与 ViT-B/16 预期一致。校验样本已存 phase1_tokens_ckplus_sample.npy / phase1_tokens_kdef_sample.npy（各前 20 张，fp16）。

## Phase 1.3 原图可得性与成本估计

| 数据集 | 原图清单 | 张数 | 说明 |
|---|---|---|---|
| fer2013_train | 有 | 28709 | probe-train / 校验支持来源 |
| fer2013_test | 有 | 7178 | fer2013_shift 全 ratio；**同时是 FER+ 的图像来源** |
| ckplus_test | 有 | 902 | CK+ 目标 |
| kdef_test | 有 | 2938 | KDEF 目标 |
| FER+ | 有 | 0 | 无独立清单；由 fer2013_test 图像 + FER+ 标签派生，**无需新提取** |

| 项 | 值 |
|---|---|
| 需提取图像总数（train+test+ckplus+kdef，去重 FER+） | **39727** |
| fp16 单图 token 体积 | 0.301 MB（196×768×2 B） |
| fp16 单视图磁盘 | **11.68 GB** |
| fp16 含镜像双视图磁盘 | 23.36 GB |
| 显存需求 | batch 256 时激活约百 MB 级，10.74 GB 绰绰有余；VRAM 不是瓶颈 |
| 实测吞吐 | 130.4 img/s（batch 128，含 IO） |
| 全量提取预计 | **约 5.1 分钟**（单视图；含镜像翻倍） |

**建议**：只落盘**原始视图**（11.7 GB），token 模块按原始视图训练/评估，并在报告中注明与"双视图均值"全局特征的协议差异；如需镜像，按 512 图一批追加落盘。分批方案：每 512 图一个 npz 分片（约 154 MB/片，77 片），先写分片再合并，避免一次性占用内存。

## Phase 1.4 判定

**Phase 1 通过：patch token 可提取、与现有协议数值对齐、成本可接受（11.7 GB / 约 5 分钟 / 10.7 GB 显存足够）。** 不需要放弃该路线。

## Phase 2 模块池（设计表；本轮未训练、未评估）

| 模块 | 机制 | 最近邻 | 与最近邻的实质差异 | 参数量（估） | 训练成本（源域 probe-train） |
|---|---|---|---|---|---|
| **T1 串联** | token 编码器 → attention 池化（196×768 → 768）→ 原型分类 | CLIP-Adapter、Tip-Adapter-F（都在全局 CLS 特征上做线性/MLP 适配） | 适配发生在**池化之前**：用可学习注意力在 196 个 patch 上加权，空间证据先聚合再分类，而不是对已聚合的 CLS 做映射 | 0.6–1.5 M | 分钟级 |
| **T2 变形** | 196 token 重排 14×14 → 1×1（通道混合）+ 3×3（空间混合）小卷积 → 池化 | CLIP-Adapter（MLP）、Tip-Adapter-F（微调 cache） | 3×3 卷积在 **patch 网格**上做邻域混合，引入空间局部性；全局 512 维特征没有空间维度，无法表达"相邻 patch 一致" | 0.3–1.2 M | 分钟级 |
| **T3 交互** | support token ↔ query token cross-attention → token 级支持记忆（patch-patch 相似度）→ 与全局文本分支融合 | Tip-Adapter（全局单点相似度 cache）、APE、HOSO、2026 年 token 级 CLIP 适配（attention-guided test-time prompt tuning） | 记忆检索从"整图一个向量"变成 **196×196 patch 对应**，可表达"支持图的哪块对应查询图的哪块"；需对 support patch 做 top-k 降采样以控制成本 | 0.2–1.0 M（+top-k 检索） | 分钟级，检索开销为主 |
| **T4 通道划分** | 768 维按通道切成子集（如 4×192），各走独立小分支后合并 | 视频/TSM 类通道划分变体（在全局特征上做通道切分） | 划分发生在 **token 维度**，每个 patch 的通道子集各自演化再合并；全局特征上做通道划分无法保留 patch 结构 | 0.1–0.5 M | 分钟级 |

统一纪律（Phase 3 将严格执行）：训练与超参选择**只在源域**（FER2013 probe-train 训练、probe-validation 选择）；确认集只用 crossed 网格、第二源域 KDEF-source、第二编码器 ViT-L/14；门槛 = 至少 2/3 确认格同时 ≥ 两族各自最好结果且 shared-unit paired bootstrap 区间支持。
