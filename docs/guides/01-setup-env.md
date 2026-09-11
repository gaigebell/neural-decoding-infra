# Guide 01: Set up dev / cluster environment

> **Audience**: Anyone setting up to work on this project.
> **Status**: 2026-09-12 更新——本地与集群都已实测（torch 2.6.0 两地对齐、
> 本地 GPU 可用、集群 cu124 验证）。

---

## 两套环境（都已验证）

| | 开发机（本地，Windows） | 集群（6 节点） |
|---|---|---|
| GPU | RTX 4060 Laptop（单卡） | 12 × PH402（Pascal，无 fp16 硬件） |
| torch | **2.6.0+cu124** | **2.6.0+cu124**（CentOS 7 实测可用，cu118 顾虑不成立） |
| conda env | `ndinf`（py3.10） | `ndinf`（每节点同款） |
| 用途 | 开发、单卡验证、解码、GPT 特征提取 | 训练、DDP、全量预处理 |

本地可以跑 GPU（4060 单卡：训练 smoke、bf16 AMP、解码、GPT-2 特征提取）；
集群负责多卡/多节点与全量数据。

## 本地（Windows）安装

```bash
conda create -n ndinf python=3.10 -y && conda activate ndinf

# CUDA torch（与集群同版本）
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 项目本体（dev + brainomni 推理依赖）
pip install -e ".[dev,brainomni]"

# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest tests/ -q -m "not slow"     # 期望 145+ passed
```

⚠️ torch 版本对齐是刻意决策：cu124 是集群（CentOS 7）实测可用的最高组合，
本地同版本保证 AMP/DDP 行为一致。

## 集群安装（mgmt 与计算节点共享 NFS，装一次）

```bash
cd /home/test/reconstruction/neural-decoding-infra
conda create -n ndinf python=3.10 -y && conda activate ndinf
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,brainomni]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

外部资产（各放各的位置，路径在 `configs/paths/cluster.yaml`）：
- GPT-2：`/home/test/reconstruction/llm/gpt2-chinese-cluecorpussmall`
- BrainOmni：`/home/test/reconstruction/BrainOmni/`（代码+权重，sys.path 链接，不 pip 装）

## 验证四件套（新环境装完必跑）

```bash
python -m pytest tests/ -q -m "not slow"        # ① 代码与依赖
python -m recon.cli.train model=meg_model_a data=fake train.smoke=true paths=cluster  # ② 训练链路
python -m recon.cli.preprocess stage=meg_context subject=1 stories=[1] paths=cluster  # ③ 预处理链路
python -m recon.cli.preprocess_run status       # ④ 数据侧车覆盖率
```

## 网络与环境变量

- W&B：计算节点 `WANDB_MODE=offline`（launch 脚本默认），mgmt 上
  `wandb login` 后 `bash scripts/sync_wandb.sh` 上传
- push：本地依赖代理（Clash 7897）；集群直连 GitHub 可作备用推送通道
- DDP 环境变量协议见 [Guide 05](05-launch-multi-node.md)
