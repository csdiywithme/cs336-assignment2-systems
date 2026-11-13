# 双 B300 上的 FSDP2 copy-engine all-gather

固定 symmetric memory staging 后，把 all-gather 从 SM 切到 copy engine（CE），前向 all-gather 的 GPU 活跃时长中位数从 **447.093 降到 277.811 µs**，完整训练步中位数从 **515.899 降到 512.257 ms（0.71%）**。在捕获的训练步中，两条路径的前向第 01–31 层和反向 all-gather 都在计算流就绪前完成。两条路径都与通信重叠的 164 对 GEMM 耗时中位差为 **+0.448 µs**。

实验使用同节点两张双向 NVLink 互通的 B300、课程 XL Transformer（3,406,809,600 参数）、FP32、global batch 4、序列长 512 和 AdamW。各组固定初始权重、输入、按层分片、默认预取和 reshard 配置，关闭激活重计算和 TF32。软件栈为 PyTorch 2.12.1+cu132、CUDA 13.2、NCCL 2.29.7、Nsight Systems 2026.5.1。与默认 FSDP2 对拍，两张卡的 loss、重建的 291 个梯度张量及 AdamW 更新后的 291 个参数张量最大绝对误差均为 0。

## 三组正式对照

每组 3 轮、每轮 12 个无 profiler 的完整训练步，执行顺序轮换。每步取两张卡耗时的最大值。峰值显存取两张卡 PyTorch allocator 的最大 allocated 值。通信路径来自单独捕获的两个训练步。

| 配置 | 实际 all-gather 路径（每 rank） | 整步中位数 (ms) | P95 (ms) | 峰值 allocated (GiB) |
| --- | --- | ---: | ---: | ---: |
| 默认 FSDP2 | SM × 130 | 516.783 | 524.811 | 34.077 |
| symmetric memory，默认 CTA 策略 | CE × 130 | 511.948 | 519.236 | 34.079 |
| symmetric memory + zero-CTA | CE × 130 | 511.392 | 517.694 | 34.079 |

NCCL 2.29.7 会在支持 CE 的 Blackwell NVLink 拓扑上，为满足注册和元素数条件的 symmetric all-gather 自动选择 CE。典型 XL block 每 rank 输入 52,431,360 个 FP32 元素，高于默认阈值 8,388,608。默认配置到 symmetric memory 的 4.835 ms 变化包含 staging 和通信路径两项变化。两个 symmetric memory 配置实际都走 CE。 [NCCL 路径选择源码](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L2725-L2779)

## 固定 staging 的路径对照

在另一对同型号、同软件栈的 NVLink B300 上，两组均使用 symmetric memory 并设 `NCCL_SYM_CE_THRESHOLD=1073741824`。trace 确认默认策略走 SM、显式 zero-CTA 走 CE。每组 24 个无 profiler 的训练步，两轮顺序互换。

| 指标 | SM | Copy engine |
| --- | ---: | ---: |
| 完整训练步中位数 (ms) | 515.899 | 512.257 |
| 完整训练步 P95 (ms) | 519.296 | 520.074 |
| 前向 all-gather GPU 活跃时长中位数 (µs) | 447.093 | 277.811 |
| 反向 all-gather GPU 活跃时长中位数 (µs) | 389.252 | 277.811 |
| 峰值 allocated (GiB) | 34.079 | 34.079 |

- GPU 活跃时长统计实际 SM kernel 或 GPU copy 的执行，CE 的同步及等待另计。896 个配对前向 GEMM 的耗时中位差为 −0.001 µs。两条路径都与 all-gather 重叠的 164 对为 +0.448 µs。此次测量未观察到 GEMM 或训练步 P95 改善。
- CE 仍有 root 和首层暴露等待。rank 0 的第二个捕获步分别等了 2.655 ms 和 1.593 ms。对应的 rank 1 API 分别晚发 2.647 ms 和 2.902 ms，rank 0 的 GPU copy 在对端发起后 0.102 ms 和 0.068 ms 开始，实际执行 0.139 ms 和 0.277 ms。两个样本的时间顺序支持跨 rank 发起时间差参与等待，主机调度原因及链路饱和程度尚未量化。

## 数据与复现

| 文件 | 内容 |
| --- | --- |
| [三组训练步与数值对拍](timing.json) | 按轮、按步和按 rank 的计时，显存峰值与对拍汇总 |
| [三组时间线统计](trace.json) | 每 rank 的通信路径、各训练阶段的暴露等待与 GEMM 匹配 |
| [固定 staging 的路径对照](route-control.json) | 两轮训练计时、路径验证、分阶段时间线统计和跨 rank 发起时间差 |
| [小模型验收](smoke.json) | 从 SQLite 重新确认每 rank 9 次 CE all-gather 与 GPU P2P copy 的对应关系，以及数值对拍 |

这些 JSON 由原始运行记录复核后导出。源文件哈希对应采集时的版本。发布版的 launcher 增加了路径对照入口，报告脚本更新了验收逻辑，模型构造和 benchmark 的计时逻辑一致。原始 Nsight 文件和云端日志保存在本地归档。使用公开脚本可重新采集并分析。 [实验代码和复现命令](../../experiments/fsdp2_ce/)
