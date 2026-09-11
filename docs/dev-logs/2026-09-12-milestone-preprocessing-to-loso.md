# Dev Log 2026-09-12 — 里程碑：预处理全量完成 → LOSO 训练启动

## 时间线回顾

| 阶段 | 内容 | 关键结果 |
|---|---|---|
| Week 00-01 | 仓库 bootstrap、文档体系（Diátaxis + ADR）、方案 C 依赖管理 | 95 文件骨架 |
| Sprint 1 | P0+P1：数据契约/注册表/Trainer/beam 解码/CLI | 107 测试；真实数据 CPU/GPU 验证 |
| 集群验证周 | T0-T3 全通过；网络排坑（p5p1/主机名/rank 注入） | **8 卡 8.7s/epoch（7.3× 扩展）** |
| 训练加固 | 启动检查/NaN 守卫/信号处理/run_metadata/split 层/val loop/AMP 三档 | 124 测试 |
| W&B 修复 | 同 step 多次 log 丢键 → 合并单次调用；offline+sync 方案 | val 曲线进面板 |
| 预处理 Phase 1 | legacy 全流程移植 + sidecar + 黄金等价性 | **4 项产物逐位一致** |
| 预处理 Phase 2 | 自研 runner（文件系统任务队列）+ status/stats | 断点续跑免费 |
| 预处理 Phase 3 | 全 12 被试 × 60 story 集群分发 | **707 pairs / 53 万样本** |
| BrainOmni 集成 | segments + GPU 分块 encode + x_scale 契约 + deepspeed 推理桩 | features (1586,16,8,512) |
| 数据验收 | 跨被试一致性 bit-equal、黄金对账、健康检查 | 全部通过 |
| LOSO 训练 | sub 12 整体留出，12 GPU | 进行中 |

## 架构决策速查（全部有 ADR/设计文档支撑）

1. **Hydra 三层组合**：model/data/paths 分组 + 命令行覆盖 + 插值；每次运行的解析后配置落盘
2. **sidecar provenance**：每个产物 `*.meta.json`（输入 hash + 参数 + commit）→ 幂等续跑、可追溯
3. **文件系统即任务队列**：无 sidecar/不匹配 = 待办 → worker 认领 → 天然断点/崩溃安全
4. **三层分离**：编排（自研 runner）/格式（BIDS-derivatives + sidecar）/计算（预处理模块）
5. **预处理边界**：重预处理继承 BIDS；特征提取进管线；编码缓存落盘（版本化）
6. **尺度契约**：x_scale 把存储尺度映射到模型训练分布（9.508e9 从黄金反推）

## 关键课程（踩坑记录）

1. **旧代码的"魔法值"是验证过的事实**（p5p1、gn11 主机名、黄金 pos 不归一化、
   GPT 上下文 2×context_len）——删除/修改前先 grep 旧代码，以产物为准不以源码注释为准
2. **黄金等价性是管线正确性的机器证明**：4 项逐位一致 + 噪声容差分档
   （纯算术=bit-exact；sin/dot=1e-9；跨版本推理=1e-3）
3. **数据验收的层级**：覆盖率（sidecar）→ 规模（stats）→ 跨被试一致性
   （zstim 与 subject 无关=最强检查）→ 数值健康 → 黄金对账
4. **无 barrier 的 DDP 设计**：隐式同步点（梯度 all_reduce/val all_reduce/
   启动通信检查）足够；对单 rank 崩溃更鲁棒（10 分钟超时 vs 永久卡死）
5. **推理路径依赖审计**：BrainOmni tokenize 无内部归一化 → 输入尺度=训练分布契约；
   deepspeed 只训练用 → 推理桩
6. **网络现实**：本地 push 依赖代理（Clash 7897）；集群直连 GitHub 可作推送通道

## 现状与未完成（截至本日志）

见 docs/guides/09-manual-review-checklist.md 与本文"未完成清单"：

- LOSO 训练结果验收 + decode 评估基线（CRR/CER）——进行中
- 故事泛化实验（命令已给，待跑）
- fMRI cube 全量（0/720，未跑）
- 监控面板（mgmt TUI：进度/GPU/损失/事件/中断）——设计定稿未实现
- 断点续训 CLI 入口（load_checkpoint 已就绪，fit 入口缺）
- 产物版本化 `--tag`（YAGNI 延后）
- BrainOmni 对齐头（recon/models/brainomni 占位）
- worker 错误可见性（`|| true` 吞错，status 兜底）
- 数据集接入手册升级（draft → 实践核验）
