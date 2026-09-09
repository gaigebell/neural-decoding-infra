# Dev Log 2026-08-29 — BrainOmni 预处理阶段 + 黄金对比调查

## 完成的事

1. **BrainOmni 两个阶段**（`recon/preprocessing/brainomni.py`）：
   - `meg_brainomni_segments`：移植新版 legacy `brainomni_process`
     （pick MEG → notch 60 → bandpass 0.1-96 → 256Hz → 每字前 2s 片段
     （零填充）→ 传感器类型归一化仅用于 NaN 检查 → 存 raw 片段 dict
     {x, pos, sensor_type} + word_time_meta）。x 存 float32（legacy 新版
     的 float64 会破坏 conv bias）。
   - `brainomni_encode`：移植 `infer_story`（BrainOmni 模型经
     `paths.brainomni_repo` sys.path 注入加载，冻结 tokenizer，
     GPU encode → (B, n_neurons, seq, dim) features）。
   - deepspeed 推理桩：仅当 deepspeed 缺失时注入 no-op（训练期码本
     同步才会用到它）。
2. **依赖方案定稿**：`[brainomni]` extra = einops/einx/
   vector-quantize-pytorch/torch-complex（代码不 pip 装，文件夹链接
   + sys.path）；`optional.py` 的 extra 探测改用 einx。
3. **黄金对比调查**（E 盘 seg4 产物）：
   - `sensor_type`：**逐位一致** ✓
   - `pos`：**逐位一致** ✓ —— 发现黄金 pos = **原始 fif 坐标**（未做
     `normalize_pos`；源码有、产物没有 → 黄金出自更早版本；BrainOmni
     在原始坐标上训练）。我们的移植复刻黄金行为。
   - `x`：形状一致，**量级不同**——E 盘 fif 读出物理 T（std ~1.2e-13，
     mne 加载时已应用 cal=1.33e-10）；黄金 ±0.5。两个黄金目录本身尺度
     也矛盾（±215 vs ±0.5）→ 黄金来自不同缩放的 fif 副本；legacy 代码
     用相对路径 `./preprocessed_data` → 大概率是集群副本。
   - BrainOmni tokenize 推理路径**无输入归一化** → 尺度必须匹配模型
     训练分布，否则编码无效。
4. **遗留问题**：集群 fif 量级待确认（±0.5 → 自然对齐；1e-13 →
   需加可配置缩放参数）。

## 测试

- 单元：归一化辅助 4 个；集成：segments 生成、黄金 pos/sensor_type
  逐位断言、tiny-ckpt GPU 编码（4 段切片；全 story 需集群大显存）。
- 145 快测 + 3 brainomni 慢测全过。
