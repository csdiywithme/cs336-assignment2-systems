# CS336 Assignment 2：已有实验与复现报告

## 阅读说明与完成边界

本报告整理截至 2026-09-14 已取得的真实实验记录。文档、汇总和图表在本地生成，不启动 Modal GPU 或 CPU 容器。它是“已有证据完成版”，不是“全部实验通过”或“可无缺项提交”的声明。待补项目见最后的 GPU 任务清单。

数据集中共有 448 条去重的实验结果记录，其中 431 条状态为 ok。实验记录数不等于单元测试数；历史失败和 OOM 单独保留。最新实验开始时间（UTC）：2026-09-14T09:40:55.765216+00:00。Flash 最后两格、10B compiled 及 tuned 候选的结果与所需环境/源码归档已只读补齐；证据包不是全部远程文件及编译缓存的备份。

最重要的实际缺项是：编译模型的完整训练步对照、小/中模型独立阶段计时、其余模型与通信时间线、单 block 显存释放诊断，以及最终修订版的 GPU 回归验证。MemoryViz 两张主图已完成，Flash 的 240 组计时已齐，无须重新运行。缺项不通过理论数值或合成截图填补。

## 实验平台与统一口径

- GPU：Modal NVIDIA B200；单卡或同一任务内 2/4/6 卡；不是 B300。精确设备信息和环境由每次运行的 environment.json、nvidia-smi.txt 记录。
- 软件：PyTorch 2.11.0 + CUDA 12.8、Triton 3.6.0；Nsight CLI 2026.5.1。不同运行以自身 pip-freeze 和源文件 SHA256 为准。
- 基线：本作业随附的 staff cs336-basics Transformer 与 AdamW，不改动用户 Assignment 1 的实现或训练结果。
- 常规默认：vocab=10000、batch=4、context=512、随机输入、seed=2026。FP32 全模型关闭 TF32；BF16 autocast 保留 FP32 主参数及状态。
- forward 指前向；backward 模式指前向+交叉熵+反向；train 指前向+交叉熵+反向+AdamW。CUDA-event 分阶段计时另列，不能把组合时间写成反向单独耗时。
- 常规模型测试每配置 10 个测量样本；5 次预热为主对照，0/1/2 次用于冷启动敏感性。Flash 使用 triton.testing.do_bench，warmup=100 ms、rep=300 ms，不虚构未保存的标准差。
- 多卡表采用逐步最慢 rank 再求均值；all-reduce 的 MB 为十进制。所有具体 case 参数、命令、超时及种子以附录和原始记录为准。

## 1. 五模型基准与混合精度

基础矩阵的配置均已尝试，10B 的完整 AdamW 训练存在真实 OOM；OOM 不等于脚本未运行。10B 名称只是 handout 的型号标签，实际参数量以记录为准。五种模型的 compiled 前向与前向+反向也已测量，其中 10B 使用已保存返回值补充证据。

![model_timings](figures/model_timings.svg)

图 1：稳态 wall-clock 均值，对数纵轴。前向+反向曲线包含 loss，不包含优化器。所有 10 个样本及均值、标准差见数据表。

BF16 对大模型的加速更明显，但不能据此断言训练显存减半：参数、梯度和 Adam 状态仍可能为 FP32；autocast 的权重转换缓存也会增加短上下文前向的活跃内存。小/中模型早期记录没有直接的 CUDA-event 阶段计时，报告保留组合测量而不以两个独立实验的均值相减伪装成直接测量。

## 2. 显存与 checkpoint

xl 已运行 context=128/2048、FP32/BF16、forward/train 的 8 个配置。能完成的配置记录 allocated/reserved 及峰值；OOM 配置记录异常与失败时内存，不把失败点内存当成可成功完成训练的峰值。五份已有 pickle 已经通过只读存储接口取回，两张 Active Memory Timeline 使用官方 PyTorch v2.11.0 MemoryViz 在本地浏览器打开并直接截图，没有重新运行 GPU。

![memoryviz_forward](memory/forward-memoryviz.png)

显存图 A：xl / context 128 / FP32 / forward。原始快照含两个测量步和随后一个记录阶段信息的额外步，因此图上出现重复的上升、释放形态；横轴是分配事件序列，不能当作毫秒时间轴。

![memoryviz_train](memory/train-memoryviz.png)

显存图 B：相同配置的完整训练步。图中保留了 MemoryViz 的选择项和 Detail 数量，未裁剪或改绘。精确峰值由 CUDA 内存统计给出：前向 18.05 GiB、训练 51.41 GiB。仅凭形态不能严格确定每一个阶段边界，需要结合阶段数值与分配栈。具体大分配归因见 handout 答案。

![checkpoint_memory](figures/checkpoint_memory.svg)

图 2：横轴为 checkpoint 分段数。32 段是每层一个 checkpoint，16 段是两层一个，64 段按 attention/FFN 半层切分；1 段运行 OOM。1 MiB 级差别不足以证明半层和整层存在稳定的显存优劣。

单层 TransformerBlock 已用 saved_tensors_hooks 对保存的独立 storage 去重统计，并排除参数/缓冲区别名；该统计可支持显存归因推理，但并非 handout 要求的 Nsight 分配事件截图。算子名称是保存张量的 producer/module 标签，也不能当作 CUDA kernel 名。

## 3. Attention 与 FlashAttention

普通 PyTorch / compiled attention 的 batch=8 规定矩阵共 40 个配置已有结果，规定范围内没有 OOM；没有为满足叙述而虚构“最小 OOM”。Flash 的三实现扩展矩阵共 240 个配置，均已有前向、反向、端到端三项计时。此前缺归档的两个必做 Triton 配置已通过已结束调用的返回值及只读存储文件补齐，没有再开 GPU。

![flash_scaling](figures/flash_scaling.svg)

图 3：FP32、D=64 的端到端前向+反向切片。完整 FP32/BF16、D=16/32/64/128、S=128..65536 表在附录。全 Triton backward 属于额外实现；必做版本是 Triton forward + PyTorch compiled backward。Triton 的 tf32x3 执行与全模型 baseline 关闭 TF32 的设置不同，必须结合数值验证解释速度。

## 4. 通信、DDP 与 FSDP

同节点 2/4/6 ranks、Gloo/NCCL、1/10/100/1000 MB 的 24 个通信配置已测量。Gloo 使用 CPU 张量，但当时的历史作业仍分配了 GPU，不能将历史成本标成零；此次整理不重跑这些任务。

![allreduce_gloo](figures/allreduce_gloo.svg)

![allreduce_nccl](figures/allreduce_nccl.svg)

图 4-5：每轮最慢 rank 的 all-reduce 时间。CPU 配额固定，跨作业节点并非保证同一物理机器；这些差异限制了带宽扩展结论。bus bandwidth 是按通信算法流量推导的指标，不是直接读出的 NVLink 线速。

![distributed_steps](figures/distributed_steps.svg)

图 6：两卡 xl 模型的训练步时间，含实际同步及优化器更新。所有原始 rank 样本和初始化/优化器前后显存见附录。overlap 的 exposed_sync 只是反向结束后未隐藏的等待，不能代表总通信时间，更不能代替时间线证明通信重叠。

## 5. 完整 8B 候选的真实结果

两张 B200、34 层、d_model=4096、d_ff=11008、32 heads、vocab=151936、context=32768、全局 batch=2。使用自编 Triton attention、block checkpoint、分块投影交叉熵、重叠 DDP、优化器状态分片及自编 fused AdamW。

初次完整候选稳态均值为 9.7024 s/step（3 次），调整 tile 后为 6.7601 s/step（5 次），本次样本均值降低约 30.3%。两组都保留冷启动与预热记录。计时来自本项目同步后的全步 wall-clock，不是已通过课程官方排行榜验证的成绩；不同计时协议不能直接等同。

小规模数值检查和核心测试已有通过记录，但不宣称完整 8B 输出/梯度逐项与官方参考模型比较过；最终候选在课程误差容限、适配器及最终源版本上的验证仍属于交付前检查。由于预算暂停，不再为进一步调优申请 GPU。

## 6. Profiling 故障与证据边界

早期 Nsight 导出包含 CUDA API、NVTX 和内存事件，但缺少 CUPTI_ACTIVITY_KIND_KERNEL；分析日志明确报 no such table。运行返回 ok 或 trace_created=true 只说明目标程序及导出结束，不足以说明捕获有效 GPU kernel。

最后一次 cuda-sw 软件追踪探针的完整 SQLite 和 nsys-rep 已通过只读存储接口取回，包含 3637 个真实 GPU kernel，总执行时间 29.847779 ms。修复本地解析器对 autograd 工作线程的归因后，measurement 全部 kernel 均有匹配，backward 正确累计为 16.380591 ms，而不是旧解析结果的 0.001312 ms。此修复只重算已有数据，没有使用 GPU。

该 small/context256 配置可复用为六项中的一项；其余五个模型 profile、两份 DDP 重叠 trace、一份 FSDP trace 仍缺有效 GPU 证据。现有探针的完整问题回答、归因规则和诊断限制见补充 Nsight 分析附录。统计汇总不是 GUI 截图，不冒充已完成所有 profiling 交付。

## 7. 本地复现与交付文件

本地重建命令见 output/README.md；三个报告/汇总/打包脚本均不调用 Modal 或 CUDA。原始文件只读，结果保留来源；成本只计有依据的运行期估价，不是账单。下列附录给出逐题解答、全量数值、过程和待补任务。


---

# 补充 · 既有 Nsight 探针分析


本节没有新增 GPU 实验。已停止的 `20260914T093845Z-profile-probe-cuda-software-trace` 任务的
SQLite、`.nsys-rep`、结果和环境记录恢复到本地后，只在本机执行 SQLite 分析。
因此 small、context=256 这一组已有可用 CUDA kernel 证据，可以复用为规定六组中的一组，剩下五组不因此自动完成。
本文件不含、也不冒充 Nsight GUI 截图。

## 来源与复现

- 原始 SQLite：[profile.sqlite](../runs/cloud/20260914T093845Z-profile-probe-cuda-software-trace/20260914T101417Z-storage-api/000-nsys-small-s256/profile.sqlite)。
- 原始 Nsight report：[profile.nsys-rep](../runs/cloud/20260914T093845Z-profile-probe-cuda-software-trace/20260914T101417Z-storage-api/000-nsys-small-s256/profile.nsys-rep)。
- 原始计时：[result.json](../runs/cloud/20260914T093845Z-profile-probe-cuda-software-trace/20260914T101417Z-storage-api/000-nsys-small-s256/result.json)。
- 配对无 profiler 结果：[plain/result.json](../runs/cloud/20260914T092252Z-profiles/20260914T093412Z-rpc/000-profile-small-s256-plain/result.json)。
- 修复后派生统计：[nsys_summary.json](profiling/small-s256/nsys_summary.json)；原始事件时序导出：[nsys_timeline.json](profiling/small-s256/nsys_timeline.json)。
- 分析器：[nsight_summary.py](../cs336_systems/nsight_summary.py)；CPU 测试：[test_nsight_summary.py](../tests/test_nsight_summary.py)。

配置为 small（12 层、d=768、dff=3072、12 heads）、batch=4、context=256、vocab=10000、FP32、
staff 模型和 AdamW、B200、5 次预热、捕获一个完整训练步。配对 plain 运行捕获 5 个测量步；
两者配置相同但采集时间/进程不同，不是同一步开关 profiler 的随机交叉实验。

本地复现命令（在 Assign2 根目录执行；不访问云服务）：

```bash
PYTHONPATH=. ../cs336-assignment1-basics/.venv/bin/python -m cs336_systems.nsight_summary \
  runs/cloud/20260914T093845Z-profile-probe-cuda-software-trace/20260914T101417Z-storage-api/000-nsys-small-s256/profile.sqlite \
  --output-dir output/profiling/small-s256
PYTHONPATH=cs336-basics:. ../cs336-assignment1-basics/.venv/bin/python -m pytest -q tests/test_nsight_summary.py
```

SQLite 用 `mode=ro` 打开，原始摘要、时间线、数据库和报告未被改写；派生结果放在独立 `output/profiling` 下，
并保存源文件 SHA256。两个合成 SQLite 回归测试通过（0.06 s），覆盖跨线程 backward、不同进程复用 correlationId、
GPU 实际执行晚于 CPU NVTX 结束、缺少 GPU kernel 表时明确报证据不足，以及原始数据库哈希不变。

## 为什么旧摘要漏掉几乎全部 backward

Nsight 中 `correlationId` 只在进程内有意义。旧实现既仅按 correlationId 合并 CUDA API，
又要求 kernel 的 launch 线程和顶层 NVTX range 的线程完全相同：后者会漏掉主线程 `loss.backward()`
期间由 autograd worker 启动的 kernel；前者在多进程 DDP 时可能把不同 rank 的 API 串在一起。

该数据库中 Python 进程 encoded globalPid 为 `281475664576512`（真实 pid=41），
主线程 globalTid 为 `281475664576553`，autograd 线程为 `281475664576570`。
对两者清除低 24 位都还原到同一 globalPid；不能只取低位 pid 而丢掉 host/VM 编码。
修复后的关联键为 `(globalPid, correlationId)`；顶层 `measurement/forward/loss/backward/optimizer`
按同进程的 launch 时刻归属，attention 的细粒度范围仍要求同线程，以避免不必要的跨线程归因。
此规则适用于当前串行测量循环；不声称能准确分离任意同进程并发模型工作。

审计结果：

| 检查 | 结果 |
|---|---:|
| GPU kernel 事件 | 3637 |
| launch API：cudaLaunchKernel / cuLaunchKernelEx / cuLaunchKernel | 3310 / 326 / 1 |
| kernel 无法关联 launch | 0 |
| 同进程关联键歧义 | 0 |
| measurement 内 kernel | 3637 |
| measurement 内未归属阶段 | 0 |
| 多阶段重复归属 | 0 |
| GPU 设备 / stream | device 0 / stream 7 |

3637 个 launch 与 3637 个 kernel 一一对应，forward+loss+backward+optimizer 共计 3637 项，
累计 GPU kernel 时间为 29.847779 ms。旧线程约束将 backward 记为约 0.0013 ms，
修复后为 16.380591 ms；旧结果作为历史证据保留，不继续用于题目回答。

软件跟踪诊断明确包含 `CUDA legacy/software instrumented trace was collected.`。
同时仍有统一内存计数器不可用、调度信息缺失以及“Not all ... events might have been collected”警告，
这些诊断在派生 JSON 全部保留。本次内部计数完全闭合，足以对已捕获训练步做以下分析；
它并不证明 profiler 捕获了进程生命周期的所有事件，也不据此分析 unified-memory 计数器。

## nsys_profile(a)：前向总耗时与 Python 计时

| 口径 | Forward ms | Backward ms | AdamW ms | 完整训练步 ms |
|---|---:|---:|---:|---:|
| Nsight CPU NVTX 区间 | 26.734338 | 50.521075 | 27.805890 | 107.347421 |
| Nsight kernel 累计时长 | 8.713439 | 16.380591 | 4.714646 | 29.847779 |
| profiled CUDA event | 27.079840 | 50.454655 | 27.862495 | 不以阶段相加替代 wall-clock |
| plain CUDA event 均值 | 10.061715 | 18.426873 | 7.660237 | 不以阶段相加替代 wall-clock |
| 同步 Python wall-clock | 无独立前向区间记录 | 无独立反向区间记录 | 无独立更新区间记录 | profiled 107.144371 / plain 36.911612 |

前向 GPU 首尾 kernel 的经过时间是 26.925727 ms，接近同次 CUDA event 的 27.079840 ms，
而累计 kernel 仅 8.713439 ms，差额包含调度间隙，不能把 kernel 总和当作端到端延迟。
同次完整训练的 Nsight measurement 107.347 ms 与同步 Python 107.144 ms 接近，但相比 plain 36.912 ms 慢约 2.90 倍，
说明 profiler 扰动明显；当前没有单独 Python forward 区间，所以不编造该口径的直接前向对照。

## nsys_profile(b)：最耗时 kernel 与调用次数

定义两个原始完整 kernel 名称，以下表用 K1/K2 缩写，避免丢失实际名称：

```text
K1 = cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_tnn_align1_bias_f32_relu
K2 = cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_nnn_align1_bias_f32_relu
```

| 阶段范围 | 累计最耗时 kernel | 调用数 | kernel 累计 ms | 占阶段 GPU kernel 时间 |
|---|---|---:|---:|---:|
| 单次 forward | K1 | 24 | 2.744523 | 31.498% |
| forward+loss+backward | K2 | 72 | 4.670557 | 18.583% |
| 完整训练步 | K2 | 72 | 4.670557 | 15.648% |

因此前向最耗时的 kernel 和前反向整体最耗时的 kernel 不同。这里报告的是单次捕获训练步的累计排名，
不是单个 kernel 的最大一次延迟。名称中的 `bias_f32_relu` 是库 kernel 符号的一部分，
不能仅据符号名字声称模型额外执行了语义上的 bias/ReLU。

## nsys_profile(c)：非矩阵乘法也占可见时间

前向一个 `elementwise_kernel ... BinaryFunctor ... MulFunctor<float>` 类型启动 146 次，共 0.465151 ms（5.338%）；
`vectorized_elementwise_kernel ... CUDAFunctor_add<float>` 72 次，共 0.171647 ms（1.970%）；
`CatArrayBatchedCopy` 24 次，共 0.165919 ms（1.904%）。
softmax 的逐元素除法 12 次、0.133343 ms，最大值归约 12 次、0.117310 ms，也各占可辨认比例。
这些是按完整 demangledName 分组的真实结果；省略号只用于文内缩写，完整名称、次数、毫秒和比例均可在 JSON 中复核。
不能把一个通用乘法 kernel 的所有调用都只归属到某个特定模型算子。

## nsys_profile(d)：GEMM 时间占比如何改变

采用 kernel 名称包含 `gemm/gemv` 的明确规则分类（当前匹配的是实际 SGEMM kernel），
不用 FLOP 占比推测 GPU 时间。前向 GEMM 共 6.705967 ms，占 8.713439 ms 的 **76.961%**；
加入 loss/backward 后为 18.647003/25.133133 ms，即 **74.193%**；加入 AdamW 后为
18.647003/29.847779 ms，即 **62.474%**。

AdamW 的 1776 个 kernel 全部被分类为非 GEMM，累计 4.714646 ms，约占全步 GPU kernel 时间的 15.796%；
其中标量乘法类 666 次、1.944142 ms，加法类 444 次、1.158771 ms。
因此优化器增加了逐元素、填充、平方根等内存访问和启动工作，拉低矩阵乘法占比。
“前向占比”取自同一训练步中的前向部分，不是额外 `inference_mode()` 的推理配置。

## nsys_profile(e)：softmax 与 QK/AV 矩阵乘法

12 层的 forward attention NVTX 范围汇总如下。

| 操作范围 | kernel 数 | 范围内 GPU kernel ms | 其中纯 GEMM ms |
|---|---:|---:|---:|
| QK 分数计算（含缩放） | 24 | 0.253024 | 0.204896 |
| softmax（max/sub/exp/sum/div） | 60 | 0.466300 | 0 |
| AV（含布局复制） | 24 | 0.235614 | 0.179616 |

softmax 时间是两个纯 GEMM 合计 0.384512 ms 的 **1.213 倍**；若比较包含缩放/布局复制的完整 QK/AV 范围，
则是 0.488638 ms 的 **0.954 倍**。必须说清比较边界，不能把两个结果混为一谈。
每层两个 attention GEMM 共 `4*batch*heads*S^2*Dh = 805306368` FLOPs（Dh=64），
softmax 约是每个 score 的常数次归约/逐元素操作；即使粗略将 max/sub/exp/sum/div 各算一次，
也只有 GEMM 的 `5/(4*64) ≈ 1.95%` 操作量。exp/归约与乘加不是同成本硬件操作，
且 softmax 要多次读写同一矩阵并启动五个 kernel，所以其低算术操作数不意味着低延迟。

## 未完成项

可以复用本组，不再为“验证 cuda-sw 是否有 kernel”或重复 small-s256 开 GPU。
剩余必需模型配置为 small 1024/4096、xl 256/512/1024；它们已有旧硬件跟踪文件，但未获得有效 GPU kernel 证据。
DDP 两张通信重叠截图和 FSDP 预取计时/截图仍缺。GPU 补跑必须等待新的预算许可；
本组可在已有 `.nsys-rep` 上本地打开 GUI 截图，无需重新租 GPU。

## 补充审计：单个 xl block 的既有 Nsight allocation trace

已在本地只读检查
[saved-block profile.sqlite](../runs/cloud/20260914T092252Z-profiles/20260914T102148Z-storage-api/012-profile-saved-block-xl-s2048/profile.sqlite)，
派生证据保存在 [allocation_summary.json](profiling/saved-block/allocation_summary.json)。
本项没有重新运行 GPU，原 SQLite 未改写，也没有制作或声称完成 Nsight GUI 截图。

该文件虽没有 GPU kernel 表，但仍包含可用的 CUDA allocation 与 PyTorch NVTX 证据：
1357 条内存事件中有 35 次动态 Device 分配和 1322 条 Device Static 分配，**动态 Device 释放事件为 0**。
整个 trace 中动态 Device 累计申请 18,518,116,352 bytes；这个累计数不是峰值，也不是保存给 backward 的残差量。
measurement 的 CPU NVTX 长度为 285.513460 ms。

按 `(globalPid, correlationId)` 关联分配 API，再定位同线程最内层 `aten::` 范围，可以确认
前向 attention 的 `bmm`、缩放 `div`、`exp`、概率归一化 `div` 都触发过 2 GiB 分配；
反向的 `bmm`、`div`、`neg`、`mul` 也有 2 GiB 分配。完整操作名、seq/op_id、地址、大小和嵌套 NVTX 保存在 JSON。
在整个 measurement 中按最内层操作分组的前五个**累计申请量**为：

| 操作 | 申请次数 | 累计申请 bytes |
|---|---:|---:|
| aten::div | 3 | 6,442,450,944 |
| aten::bmm | 12 | 4,702,078,976 |
| aten::mul | 4 | 2,315,255,808 |
| aten::exp | 1 | 2,147,483,648 |
| aten::neg | 1 | 2,147,483,648 |

这些是 CUDA 缓存分配器向驱动申请的新 segment，不等于张量创建/销毁：张量释放回 PyTorch 缓存后，
可能没有对应 `cudaFree`，新张量也可能复用既有 segment。因此不能把上表替换为 handout(f) 的
“saved-for-backward 操作占比”，也不能从全为分配、没有释放的曲线反推梯度大小。
已有 saved-tensor hooks 的 6,818,955,264 bytes 残差、419,450,880 bytes 参数梯度和
83,886,080 bytes 输入梯度仍是有效的独立统计，但不是从这张 Nsight 时间线净下降推导出来的。

结论：现有 `.nsys-rep` 可用于本地 GUI 的部分 allocation/NVTX 截图证据；严格完成(f)中
“根据释放与净变化反推梯度”的要求，现有 trace 不足。若后续获得预算，需要的最小新增 GPU 工作是
**一个 xl 单 block、batch=4、context=2048 的内存诊断**，可在新进程禁用 CUDA caching allocator，
显式标记 forward/backward，并继续使用 saved-tensor hooks 区分短期临时值与 residual；
应先确认记录同时包含分配和释放，再做截图/分析，不必重跑完整 xl 模型或整套内存矩阵。
这仅列为待批准方案，本次未执行。


---

# 附录 A · Handout 逐题解答


本文件已完成**现有证据能够支持的文字解答与理论推导**，不是“Assign2 全部实验完成”的声明。
原始题目以仓库内 [26.1.3 handout](../cs336_assignment2_systems.pdf) 为准；完整实验表见
[EXPERIMENT_TABLES.md](tables/EXPERIMENT_TABLES.md)，逐次记录见
[all_attempts.json](data/all_attempts.json)，每条记录的 `source` 指向不可变的原始 JSON。
本次收尾只读恢复已结束实验的既有产物并在本地分析，不启动 Modal GPU/CPU 容器或其他付费云端计算。

需要明确保留的缺口是：五模型 compiled 完整训练步；small/medium 的独立阶段计时；
其余 5 组单卡模型的有效 CUDA kernel profiling（small-s256 已恢复）；普通/overlap DDP 的两张时间线；FSDP 预取的计时和截图；
Nsight 内存生命周期、净变化推导及截图证据（两张 memory_viz 图已由已有快照在本地完成）；
最终候选的统一严格数值检查和独立 NCCL 数值回归。建议的原始 Gloo suite 五轮已在本地完成，
不能把 CPU/Gloo 结果说成 CUDA/NCCL 通过。OOM 是已完成实验的结果，不能填成 0 ms，也不算漏跑。

## 实验约定

- 使用 handout 随附的 `cs336-basics` 作为模型及普通 AdamW 基线，不改动 Assign1。
  因而本报告不能声称是在用户 Assign1 实现上获得相同速度；这是用于固定数学行为和方便复现的基线选择。
- 模型统一 vocab=10000、全局 batch=4、context=512，除非题目明确指定其他值。
- 标称模型大小不是实测参数量；例如 xl 实际有 3,406,809,600 个参数。
- FP32 全模型关闭 TF32；BF16 使用 autocast，保留 FP32 主参数、梯度和优化器状态。
- 普通单独注意力对照使用矩阵乘法、`torch.softmax` 和矩阵乘法，不调用 SDPA/FlashAttention 库。
  它的 softmax 是一个 PyTorch 算子；完整 staff 模型的 softmax 则由基本算子组合，两者不混为同一基线。
- `backward` 模式的总时间指前向、loss 和反向；反向单独时间另列 CUDA event 列。
  `train` 还包括 AdamW。`forward` 不含 loss，也不关闭 autograd，以便分析题目要求的前向残差。
- 常规计时使用同步后的 `perf_counter`，每步末尾同步；Flash 矩阵按题目使用 `triton.testing.do_bench`。
- profiler 扰动下的时间与无 profiler 时间分开报告。OOM、timeout、脚本错误分别标记。
- 分布式步长汇总采用“每次迭代先取最慢 rank，再对迭代求平均”，不用某个 rank 的均值冒充全局时间。
  统计标准差是样本标准差；只有 10 次短跑、没有独立重复运行时，不宣称微小差值显著。
- B200 实测环境为 compute capability 10.0、148 SM、约 178 GiB 可见显存、PyTorch 2.11.0+cu128、Triton 3.6.0。
  所有性能数值只适用于本次硬件/驱动/软件与输入；它们不是模型训练质量或收敛性实验。

## 1. Benchmarking / mixed precision / profiling

### benchmarking_script

(a) 入口为 [benchmark.py](../benchmark.py)，支持模型大小、任意层数/宽度/头数、batch、context、精度、编译、预热与重复次数，
以及 forward/backward/train 三种模式。使用随机固定种子数据，不下载语料。正式矩阵覆盖五种模型、
5/0/1/2 次预热、每配置 10 次测量；BF16 与 FP32 对照使用 5 次预热。

(b) 下表为 5 次预热、10 次测量的 FP32 同步 wall-clock 均值 ± 标准差，单位 ms。
注意第二列之后的 `F+B` 包含 loss；独立 backward/AdamW 只能引用 event 记录，不能用不同运行的均值相减伪造实测。

| 模型 | Forward | F+B | F+B+AdamW |
|---|---:|---:|---:|
| small | 16.639 ± 0.157 | 49.373 ± 0.318 | 57.321 ± 2.477 |
| medium | 47.321 ± 0.082 | 139.904 ± 0.249 | 170.295 ± 41.395 |
| large | 104.262 ± 0.224 | 311.508 ± 0.125 | 345.264 ± 0.080 |
| xl | 292.682 ± 0.109 | 861.903 ± 0.112 | 941.142 ± 0.124 |
| 10B | 941.318 ± 0.129 | 2810.166 ± 0.075 | OOM |

large/xl/10B 的单独 backward event 均值分别为 206.808/568.639/1867.789 ms；
large/xl 的 AdamW event 均值为 32.021/79.298 ms。small/medium 原始基线运行早于阶段 event 插桩，
这两种模型独立阶段计时仍缺。多数前向/前反向数据标准差很小，但 medium 完整训练有一次 288.105 ms 的样本，
中位数为 157.222 ms，不能删掉这个离群值后声称全部稳定；仅凭现有数据无法确定其根因。

(c) small 前向不预热为 31.298 ± 46.670 ms，1/2 次预热分别为 16.539 ± 0.079 和 16.517 ± 0.028 ms；
medium 不预热为 64.426 ± 54.160 ms，而 5 次预热为 47.321 ± 0.082 ms。
不预热时，第一次实际调用可能包含 cuBLAS 算法选择、缓存分配、Python/算子初始化或编译，均值和方差都会升高。
一次预热足以消除部分初始化，但首次 backward、Adam 状态懒初始化、不同计算图和频率爬升未必都完成。
这些是对机制的解释，不是从缺失 GPU trace 中得到的逐项归因。证据来自
[small/medium 原始记录目录](../runs/cloud/20260914T084942Z-models-small-medium/20260914T085455Z-rpc)
和 [large/xl/10B 原始记录目录](../runs/cloud/20260914T085158Z-models-large-xl-10b/20260914T091555Z-rpc)。

### mixed_precision_accumulation

实测四种累加依次为 10.0001335144、9.953125、10.0021362305、10.0021362305。
第二种在每次加法后都舍入至 FP16，因此累积误差显著；后两种只在 0.01 的表示上发生 FP16 舍入，
之后用 FP32 累加，显式转回 FP32 不能恢复已经丢失的有效位。所有输出及执行代码保存于 precision 原始记录。

### benchmarking_mixed_precision

(a) ToyModel 采用题目指定的无 bias Linear -> ReLU -> LayerNorm -> Linear 结构，实测如下。

| 对象 | FP16 autocast 下的 dtype |
|---|---|
| context 内模型参数 | FP32 |
| fc1 输出 | FP16 |
| LayerNorm 输出 | FP32 |
| logits | FP16 |
| 交叉熵 loss | FP32 |
| 模型参数梯度 | FP32 |

(b) FP16 autocast 下主参数与参数梯度保持 FP32，Linear 输出和 logits 为 FP16，LayerNorm 与交叉熵结果为 FP32。
LayerNorm 的均值、方差、平方和及倒平方根对溢出和舍入敏感；BF16 虽然扩大量程，但只有 7 位尾数，
因此仍适合用 FP32 进行统计量归约，不能把“不会像 FP16 那样容易溢出”等同于“无需高精度归约”。

(c) BF16 的 forward/F+B 均值（ms）依次是：small 7.230/25.214，medium 17.263/51.363，
large 44.591/110.595，xl 49.970/155.557，10B 118.142/404.701；对应 FP32 值见上一表。
前向加速约 2.30/2.74/2.34/5.86/7.97 倍，整体上大模型更能从低精度矩阵乘法中获益，但并非严格单调；
xl 完整训练从 941.142 降到 234.963 ms，而 AdamW 阶段仍约 79.3 ms，说明未降精度的更新逐渐成为更大的时间占比。
这不是 optimizer kernel 时间占比的 Nsight 结论，而是独立 CUDA event 阶段数据。精度小实验的原始记录见
[precision/result.json](../runs/cloud/20260914T084725Z-probe-leaf-fix/20260914T085436Z-rpc/002-precision/result.json)。

### nsys_profile

选择 small 的 256/1024/4096，以及 xl 的 256/512/1024 三种 context；4096/1024 分别是本次实测
普通 FP32 全训练能容纳的最大二次幂长度，下一档已单独运行并记录 OOM。
每个配置有配对的无 profiler 数据及 Nsight 文件，预热不在捕获区间。旧硬件跟踪归档只有 CPU API/NVTX、
GPU memory 等数据；最新 small-s256 `cuda-sw` 探针的既有 SQLite 恢复后，已核验 3637 个 CUDA kernel，
无需重跑即可完成这组分析，其余五组仍缺有效 GPU kernel 证据。完整结果和修复方法见
[NSIGHT_EXISTING_ANALYSIS.md](NSIGHT_EXISTING_ANALYSIS.md)，以下数值仅指 small-s256 单次捕获训练步。

- (a) 同步 Python wall-clock 包括主机调度和等待，CUDA event 是两个 stream 事件之间的经过时间，kernel 时长和还须处理并发。
  本组前向 kernel 累计 8.713439 ms，GPU 首尾跨度 26.925727 ms，同次 event 27.079840 ms；
  完整 measurement NVTX 107.347421 ms 接近同步 Python 107.144371 ms，但无 profiler 全步只有 36.911612 ms，
  工具带来约 2.90 倍扰动。没有独立 Python forward 区间，不能伪造该口径的直接前向对照。
- (b) 前向最多时间的 kernel 为 `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_tnn_align1_bias_f32_relu`，
  24 次、2.744523 ms；F+B 的首位改为同族 `128x64x16 ... nnn` kernel，72 次、4.670557 ms，完整名称见附录。
- (c) 前向逐元素 MulFunctor 类型 146 次、0.465151 ms（5.338%），加法类型 72 次、0.171647 ms，
  CatArrayBatchedCopy 24 次、0.165919 ms；softmax 归约/除法也占可见时间，完整分组均保存于派生 JSON。
- (d) 按真实 SGEMM kernel 名称分类，前向 GEMM 占 76.961%，F+loss+B 为 74.193%，完整 AdamW 训练为 62.474%。
  AdamW 增加 1776 个非 GEMM kernel、4.714646 ms，拉低 GEMM 占比；这些是 GPU kernel 口径，不是 wall-clock 比例。
- (e) softmax 为 0.466300 ms，QK/AV 纯 GEMM 合计 0.384512 ms，前者约为后者 1.213 倍；
  若包含 QK 缩放和 AV 布局复制，两个范围合计 0.488638 ms，softmax 比值为 0.954。
  理论两次矩阵乘法共约 4*B*H*S^2*Dh FLOPs，softmax 是 O(B*H*S^2) 归约/逐元素工作，
  算术量小并未转化为同等低延迟，原因包括内存流量、归约与 kernel 启动。

旧摘要按同线程过滤，把 autograd worker 的 backward 错漏为约 0.0013 ms；修复为同进程阶段归属后是
16.380591 ms。3637 个 kernel 均能按进程+correlationId 关联 launch，阶段无漏归属、无重复；
Nsight 仍保留事件可能不全、统一内存计数器不可用等诊断，详见附录，不能声明进程整个生命周期无任何丢事件。
本组可复用为六组之一，剩余五组模型与多卡时间线依旧待补，不因探针成功自动完成。

### memory_profiling

(a) 脚本支持 memory history；五个成功配置的 pickle 已通过只读存储接口取回，本地使用官方 PyTorch v2.11.0
MemoryViz 得到 [forward 图](memory/forward-memoryviz.png) 和 [完整训练图](memory/train-memoryviz.png)。
前向时间线反复逐层增长后释放，训练时间线峰值更高，并在反向和更新过程中呈现复杂的分配/释放形态；
每份记录包含两次计时步骤和一次额外的阶段/峰值记录步骤，所以出现三个循环，而不是三个新 GPU 实验。
图的横轴是分配事件序列，不是毫秒；仅凭峰形不能严格定位全部阶段边界，需结合阶段统计和调用栈。
原始 pickle、未经改绘的截图、低 Detail 图、查看页面与 SHA256 均位于 [memory 目录](memory)。

(b)(c) 以下为一次单独记录步骤的 `peak_allocated_bytes`，单位 GiB（除以 1024^3）；不是 reserved。

| context | FP32 forward | FP32 完整训练 | BF16 forward | BF16 完整训练 |
|---|---:|---:|---:|---:|
| 128 | 18.053 | 51.415 | 22.440 | 51.405 |
| 2048 | OOM | OOM | 163.664 | OOM |

内存区分 allocated（活跃张量）和 reserved（缓存分配器持有的总量），两者不是同一个峰值。
RMSNorm 主参数、Adam m/v、主梯度保持 FP32，因而 BF16 并不会将总训练显存简单减半；
autocast 的参数转换缓存甚至可能让短上下文的前向峰值增加。
ctx128 的 BF16 训练峰值只比 FP32 少约 0.010 GiB，实测并未减半；ctx2048 的 BF16 前向能运行但完整训练仍 OOM。
为 fit 而加入 checkpoint 后属于另一项优化实验，不能替换本表的原始 OOM。

(d) 默认 xl 残差流大小为 4*512*2560*4 / 1024^2 = 20 MiB；context=128/2048 时分别为 5/80 MiB。

(e) ctx128 的前向 MemoryViz 已查看全量 5407 entries，并把 Detail 降到 540/5407（约10%）。
界面可见的最大单块分配为 100 MiB（104857600 bytes），而不是 ctx2048 的 2 GiB 注意力矩阵。
这个大小与 FFN 权重 10240*2560*4 完全相符，且大于 embedding/head 各 97.656 MiB；
但界面明确提示该块没有 frames，因为记录在 warmup 后才开始，因此“属于 FFN 权重”是依据形状和大小的推断，
不是从已缺失的调用栈直接证实的结论。截图和快照如实保留这一限制，不编造 stack trace。

(f) 已有补充的 `saved_tensors_hooks` accounting：单独 xl block（2048）保存的独立非参数 storage 为
6,818,955,264 bytes；去重且排除了参数/缓冲区的 view。下表为 hook 捕获的“所属模块:producer”分组，
不是 Nsight GPU kernel 或内存操作的排名，也不替代题目要求的 NVTX 内存截图。

| 独立 storage 的 producer 分组 | bytes | 占保存量 |
|---|---:|---:|
| attn:ViewBackward0 | 2,231,369,728 | 32.723% |
| attn:ExpBackward0 | 2,147,483,648 | 31.493% |
| ffn:ViewBackward0 | 671,088,640 | 9.842% |
| ffn:SigmoidBackward0 | 335,544,320 | 4.921% |
| ffn:MulBackward0 | 335,544,320 | 4.921% |

第五名与 ffn.w2:ViewBackward0 同值。最大的两个矩阵各为 4*32*2048*2048*4 = 2 GiB，
来自注意力的指数/概率路径，而非一个 80 MiB 残差流；View 名称表示分组来源，不说明 view 本身新分配了所有字节。
单 block 参数梯度理论大小为 (4*d^2 + 3*d*dff + 2*d)*4 = 419,450,880 bytes，
输入梯度另占 83,886,080 bytes。快照下降量需同时考虑释放的残差、产生的参数/输入梯度和仍存活的输出，不能只看净下降。
hook 实际统计的参数梯度、输入梯度分别与上式相符；两者合计 503,336,960 bytes = 480.020 MiB。
用实测块前后活跃显存直接减法还会包含最终输出和被复用的输入；没有配套 Nsight allocation 生命周期对照时，
不能把这个解析相符结果冒充题目要求的“从时间线净变化反推”的验证。
旧 Nsight SQLite 已取回并离线审计：共 1357 条 GPU memory 事件，其中 35 条动态分配、1322 条静态事件，
释放事件为零。可将部分分配与 bmm/div/exp 的 NVTX 区间关联，但它们是缓存分配器申请的 segment，
不是全部张量的逻辑生命周期，不能据此推导 backward 中 residual 的释放或梯度净增量。
严格补齐本题需一个有界的单 GPU、单 xl block 诊断：关闭缓存分配器并显式标记前后向，
取得真实 allocation/free 事件后再生成截图和生命周期分析。此诊断会改变分配器行为，不能替换原始性能基线；
无需重跑完整 xl 内存矩阵。详见 [既有 trace 审计](NSIGHT_EXISTING_ANALYSIS.md) 与 GPU 待办 G6。
原始证据：[内存矩阵](../runs/cloud/20260914T085600Z-memory-and-checkpoint/20260914T091525Z-rpc)、
[修复后单块统计](../runs/cloud/20260914T091830Z-optimized-validation-fp32-oracle/20260914T093348Z-rpc/001-saved-block-xl-s2048/result.json)。

## 2. gradient_checkpointing

### (a) 不限制重算次数的激活内存最优策略

采用不平衡的“前缀 checkpoint”：第 n 个 block 的输入通过 checkpoint 重算前 n-1 个 block，
而每一个嵌套 checkpoint 都接收同一个原始输入 x。这样 checkpoint 输入引用共享同一 storage，
不会保留 N 个不同的中间激活；使用 non-reentrant 方式按需重算残差，任一时刻只保留常数个 block 的活跃计算数据。
忽略题目指定的每 checkpoint 管理开销、模型参数及其梯度后，峰值激活内存为 O(1) 个 block，
前缀长度累计为 N+(N-1)+...+1，所以总前向计算为 O(N^2)，正常反向为 O(N)。
二分递归 checkpoint 是 O(log N) 激活内存 / O(N log N) 重算的折中，但不是忽略计算代价时的最小激活内存。

```python
def prefix(x, n):
    if n == 1:
        return blocks[0](x)
    h = checkpoint(lambda original: prefix(original, n - 1),
                   x, use_reentrant=False)
    return blocks[n - 1](h)
```

完整实现见 `cs336_systems/checkpointing.py`。8 层 CPU 回归测试对输入及每个参数的梯度与不 checkpoint 版本逐项完全一致。

### (b) 只允许一次重算

设每个 checkpoint 的输入占 C，每个 block 内部残差占 R，每个 checkpoint 包含 k 层，
静态上界近似为 (N/k)*C+k*R，连续最优解为 k=sqrt(N*C/R)。本配置 C=80 MiB，R 远大于 C，
所以应测试按单层划分，而不是简单地固定 sqrt(N) 层；实测比较 2 层、1 层及半层（attention/FFN 分开）三档。
两层峰值 47,434,947,584 bytes，单层 41,034,394,624 bytes，半层 41,035,443,200 bytes；
后两者只差 1 MiB，实际处于几乎同一峰值平台，按整层 checkpoint 更简单。
这里测量包含模型/梯度等总显存，静态式只描述激活上界，不应要求二者数值完全一致。

## 3. attention / torch_compile / FlashAttention

### pytorch_attention / torch_compile

`pytorch_attention(a)` 与 `torch_compile(a)`：普通与 compiled 注意力矩阵均覆盖 batch=8、D=16/32/64/128 和 S=256/1024/4096/8192/16384，
每配置 100 次 forward、100 次固定图 backward。此次 B200 的规定矩阵没有 OOM，报告不会虚构一个规定范围内的 OOM。
保存的概率矩阵以 O(B*S^2) 增长；例如 B=8、S=16384，仅一个 FP32 S*S 矩阵就占 8 GiB，
反向临时 score-gradient、概率和输出梯度会增加峰值。FlashAttention 通过分块重算把持久保存量变为 O(B*S*D)。
如需展示实际最小 OOM，必须明确标为规定网格之外的扩展边界实验，不能和题目原始网格混淆。

以规定网格最大项 B=8、S=16384、D=128 为例，Q/K/V/O 四个张量各 64 MiB，dO 另 64 MiB；
加一个 8192 MiB 概率矩阵得到约 8512 MiB 的张量工作集，实测 backward 前 allocated 为 8,933,998,592 bytes
（约 8520.125 MiB），还包含运行时额外分配。反向峰值还会出现多个平方临时张量；不能将 backward 前 allocated 当作反向峰值。
该项 eager forward/backward 为 25.755/47.417 ms，compiled 为 21.881/40.886 ms；
编译能融合部分逐元素与归约操作，但没有从根本上消除保存概率矩阵的平方复杂度。
小输入上调度和 launch 开销占比高，compiled 不保证每一项都快，完整 40 组对照见实验表。
原始证据：[attention matrix](../runs/cloud/20260914T085159Z-attention-matrix)。

`torch_compile(b)`：下面均为 FP32、batch=4、context=512、5 次预热和 10 次测量，单位 ms。

| 模型 | eager forward | compiled forward | eager F+B | compiled F+B | compiled F+B+AdamW |
|---|---:|---:|---:|---:|---|
| small | 16.639 | 13.169 | 49.373 | 37.898 | 未运行 |
| medium | 47.321 | 39.220 | 139.904 | 113.319 | 未运行 |
| large | 104.262 | 86.922 | 311.508 | 261.239 | 未运行 |
| xl | 292.682 | 268.682 | 861.903 | 785.778 | 未运行 |
| 10B | 941.318 | 893.501 | 2810.166 | 2665.385 | 未运行 |

小模型 forward 加速约 1.26 倍，10B 约 1.05 倍，说明这组输入下编译收益随模型变化，不能照搬为常数。
五种模型都缺 compiled 完整训练步；eager 的 10B train OOM 不能当作 compiled train 的实际结果。
10B compiled 数据由已有的
[完成状态快照](../runs/status/20260914T093858Z-models-compile-10b-20260914T094503Z.json)
与随后只读取回的 [27 文件归档](../runs/cloud/20260914T093858Z-models-compile-10b/20260914T104956Z-storage-api) 交叉核对；
该归档含结果、环境和源码，文件大小/SHA256 以及环境记录的源码哈希已核验，没有重跑 GPU。

### flash_forward / flash_backward

`flash_forward(a)`：PyTorch 版本采用 Q tile=32、K/V tile=64 的在线 softmax。每行维护 m、l、acc：
新最大值 m' = max(m,max(scores))，旧累加乘 exp(m-m')，新概率为 exp(scores-m')；
最终 O=acc/l，L=m+log(l)。保存 Q、K、V、O、L，且 L 恰好一个 (batch, sequence) 张量。
`flash_forward(b)`：Triton 版本由 query tile 和 batch/head 两维 grid 分配 program，使用 block pointers，FP32 归约和累加。
`flash_forward(c)`：因果标志可省略，默认 False，保存在 ctx 中；Triton 因果分支为编译期参数，
对遮挡位置使用 -1e6 并跳过完全遮挡的后续 tile。额外处理边界，支持非 tile 整倍数长度。

`flash_backward`：反向重算 S=QK^T/sqrt(D)、P=exp(S-L)，先计算 Delta=sum(O*dO,-1)，然后
dV=P^T*dO，dP=dO*V^T，dS=P*(dP-Delta)，dQ=dS*K/sqrt(D)，dK=dS^T*Q/sqrt(D)。
必做版本用 PyTorch 并编译反向；额外实现的 Triton 反向分别遍历 dQ 和 dK/dV，避免原子加法和跨 program 同步。
必做反向仍会重建完整 P/dS，所以“前向少保存”不等于“整个训练峰值已经线性”；全 Triton 反向才进一步避免平方临时矩阵。
浮点归约顺序不同，不承诺逐 bit 一致。官方 6 个 attention 测试在 GPU 环境通过，使用课程规定 `rtol=atol=1e-2`；
附加验证每轮包含 32 个配置、128 次张量比较，涵盖 FP32/BF16、因果/非因果及 n=63，
但使用单独容限（BF16 atol=rtol=0.03，FP32 atol=0.001/rtol=0.03），
不能称为对官方误差要求更严格的完整验收。最终 tuned 模型仍需统一官方标准检查。
实现见 [attention.py](../cs336_systems/attention.py)、[triton_attention.py](../cs336_systems/triton_attention.py)；
课程测试证据见 [attention junit](../runs/cloud/20260914T084725Z-probe-leaf-fix/20260914T085436Z-rpc/000-attention-tests/junit.xml)。

### flash_benchmarking

完整网格为 batch=1、causal=True、S=128..65536 的 10 个二次幂、D=16/32/64/128、FP32/BF16。
分别比较普通 PyTorch、Triton forward+compiled backward、Triton 全前后向；每个配置记录 forward、
backward-only、end-to-end 三项。前两类是题目必要比较，第三类为 leaderboard 优化扩展。
Triton FP32 dot 使用 tf32x3，保留 FP32 输入与近 FP32 精度，和关闭 TF32 的普通 GEMM 不是相同硬件执行模式。

Flash 网格现已核验 240/240 组成功：eager 80/80、必做 Triton+compiled backward 80/80、额外全 Triton 80/80。
最初 238 项有原实验目录归档；此前缺少的必做版本 BF16、S=65536、D=64/128 两项已通过读取既有完成任务的
[状态快照](../runs/status/20260914T092409Z-flash-triton-compiled-backward-20260914T100623Z.json) 确认并取得完整数值，
没有重新运行 GPU。随后又只读取回这两项结果/日志与最终 summary 的
[9 个文件](../runs/cloud/20260914T092409Z-flash-triton-compiled-backward/20260914T105002Z-storage-api)，
与此前该 run 的源码/环境共同保存；现在 240 项均有本地实验结果文件，不在 GPU 补跑清单内。
例如 S=16384、D=128、BF16 的 eager、必做版本、全 Triton 的 F+B 分别为 3.364、7.448、1.590 ms：
必做 forward 的确更快（0.479 对 1.938 ms），但其 FP32 重算反向使整体反而更慢。
全 Triton 在这个 BF16 输入上约快 2.12 倍；而 FP32 同形状全 Triton 为 11.139 ms，
并不优于必做版本 9.260 ms，展示 tile、精度和反向实现都影响收益。
S=65536、D=128、BF16 的完整 Triton F+B 为 20.724 ms，eager 为 52.235 ms；
这些是微基准而非端到端模型加速率。原始目录：
[eager](../runs/cloud/20260914T091848Z-flash-eager)、
[必做版本](../runs/cloud/20260914T092409Z-flash-triton-compiled-backward)、
[完整 Triton](../runs/cloud/20260914T092847Z-flash-triton-full)。

## 4. Distributed / optimizer sharding / FSDP

### distributed_communication_single_node

同节点 2/4/6 卡，分别测 CPU Gloo 与 GPU NCCL，tensor 为 FP32，采用十进制 MB/GB。
5 次预热、20 次测量，reset 和 barrier 放在计时外；每轮按最慢 rank 聚合，另保留所有 rank 原始样本。
总 CPU 配额固定为 8 physical cores，因此 Gloo 扩展时也受到 CPU/内存带宽影响。
NCCL bus bandwidth 按 2*(N-1)/N*bytes/time 计算，它不是直接读取的 NVLink 线速。
部分 Modal 节点限制 nvidia-smi topo 查询；拓扑命令失败被保留，不把它当作 NVLink 缺失的证据。

24 项设置都有结果。NCCL 的 1 GB all-reduce 在 2/4/6 卡时分别为 1.868/2.489/2.656 ms，
折算 bus bandwidth 为 535.2/602.6/627.6 GB/s；进程增多增加 collective 代价，但大消息更充分利用带宽。
1 MB 时 NCCL 约 0.075-0.091 ms，固定开销更明显；Gloo 的 1 GB 分别为 1472.320/1730.584/1764.047 ms，
明显受 CPU 通路限制。2/4/6 卡为不同 Modal 分配，不能保证物理节点完全相同，因此细小差异不归因于单一拓扑因素。
完整表和每 rank 样本见实验表及 `all_attempts.json` 中 `kind=allreduce`。

### naive_ddp / minimal_ddp_flat_benchmarking / ddp_overlap_individual_parameters

初始化广播参数和缓冲区；普通 DDP 在 backward 后逐参数 all-reduce 并除以 world size。
flat DDP 按 dtype/device 拼接梯度，减少小 collective 的启动开销，但增加打包/解包流量和临时内存。
overlap DDP 使用 post-accumulate-grad hook，梯度就绪立即发起异步 all-reduce，optimizer 前统一等待。
同一个 tied Parameter 只注册一次 hook；冻结参数不注册梯度 hook，但仍参与初始参数一致化。
计时中的 exposed_sync 仅指未被计算遮蔽的末尾等待，不能用它声称全部通信量为零；Nsight 时间线另外验证重叠。

`naive_ddp_benchmarking`、`minimal_ddp_flat_benchmarking` 和 `ddp_overlap_individual_parameters_benchmarking(a)`：
全局 batch=4（各 rank 2）、xl、context=512、FP32、staff AdamW、两张同节点 B200，5 次预热和 10 次测量。
下表总时间及尾部同步时间都由逐次最慢 rank 聚合；尾部比例为两列均值之比。

| 实现 | 全步均值 ms | 梯度尾部同步 ms | 尾部同步/全步 |
|---|---:|---:|---:|
| 逐参数 naive DDP | 590.083 | 40.194 | 6.812% |
| flat DDP | 589.047 | 39.317 | 6.675% |
| overlap DDP | 568.190 | 1.816 | 0.320% |
| overlap + optimizer sharding | 557.964 | 1.839 | 0.330% |
| FSDP + AdamW | 545.001 | 0.430 | 0.079% |

flat 与 naive 只差约 0.18%，打包开销和大张量传输使减少调用数未转化为明显加速；
overlap 相比 naive 全步时间减少约 3.71%，末尾等待也减少，但这仍不是 GPU 时间线的直接重叠证明。
对 naive/flat，同步区覆盖 post-backward 梯度阶段（flat 含打包/解包）；不是纯 NCCL kernel 耗时。
FSDP 行的尾部等待也不包含前反向内所有 gather/reduce-scatter，所以不能横向当作各方法总通信量。

`ddp_overlap_individual_parameters_benchmarking(b)`：两张有效 Nsight 截图仍缺。
已有带 profiler 的两组 wall-clock 约 5 秒，存在明显工具扰动，不能拿它们与上表无 profiler 约 0.57 秒的值直接比较。
官方 distributed 测试归档为 8 passed，建议的 5 轮稳定性测试未完整做完；并发正确性不能仅由一次通过保证。
原始证据：[双卡基准](../runs/cloud/20260914T085559Z-communication-and-xl/20260914T091516Z-rpc)、
[distributed junit](../runs/cloud/20260914T084950Z-distributed-tests/20260914T085228Z-rpc/000-distributed-repeat-0/junit.xml)。

### optimizer_state_sharding / optimizer_state_sharding_accounting

按完整参数确定 owner，使用累计元素数贪心负载均衡；各 rank 只给自己的参数建立 optimizer state，
所有梯度仍由 DDP 平均，owner 更新后广播参数。支持参数组、运行时 add_param_group、学习率修改、
完整模型 zero_grad 和同 rank/world-size 的状态恢复；不会隐式把不同 world-size 的 checkpoint 重新分片。
以 FP32、P 个参数、N 个 rank 计，未分片的参数/梯度/m/v 为 16P bytes，分片优化器后为 8P+8P/N。
完整参数分配有不可分割的大张量，可能不完全平均；报告按真实 owner 分配与测量解释。

`optimizer_state_sharding_accounting(a)`：xl 的 P=3,406,809,600，FP32 参数和梯度各 12.691 GiB；
Adam m/v 总计 25.383 GiB，双卡分片后各 rank 约 12.691 GiB。因此持久张量由约 50.766 GiB 降为 38.074 GiB，
还要另加激活、loss、临时值、非参数 buffer 及 allocator 对齐。下表为 rank0 的代表步骤，单位 GiB；
`累计峰值` 是自该步 reset 后的最高 allocated，`当前活跃` 是观测点的 allocated，两者均保留，不能混名。

| 设置 | 初始化当前/累计峰值 | AdamW 前当前/累计峰值 | AdamW 后当前/累计峰值 |
|---|---:|---:|---:|
| overlap + 普通 AdamW | 12.817 / 12.817 | 50.947 / 52.088 | 50.947 / 52.088 |
| overlap + 分片 AdamW | 12.817 / 12.817 | 38.255 / 39.396 | 38.255 / 39.396 |

普通/分片的测量峰值相差约 12.692 GiB，与省下半份 Adam m/v 的理论量相符。
这些点在 5 次预热之后，Adam 状态已经初始化，所以不会在这一步 optimizer 前后再增加一整份状态；
不能拿这张表描述首次 Adam step 的懒分配瞬态。原始记录仍保留每步、每 rank 和 reserved 数据。

`(b)` 在相同 overlap DDP 基础上，全步从 568.190 降到 557.964 ms（约 1.80%），
本次测量说明减少更新工作可抵消更新后广播的代价；只有一组短跑，不能推广为所有硬件或张量划分都更快。

`(c)` 与 ZeRO-1 的关键差别是这里保留全量 all-reduce 后又广播更新参数，而 ZeRO 可以先将梯度 reduce 到 owner，
再 all-gather 更新参数，避免先把完整约简梯度复制给不负责更新的 rank。比较内存时还必须对齐主参数和混合精度格式。
按同精度、理想大张量 ring 计，标准 DDP 每 rank 通信为 2*(N-1)/N*S；这里 AR 后再分发更新参数约为
3*(N-1)/N*S，而 reduce-to-owner + all-gather 参数的 ZeRO-1 可约为 2*(N-1)/N*S。
这里是逻辑流量模型；逐参数 broadcast 的实际 collective 实现、启动次数和负载不均会改变实测开销。
实现：[sharded_optimizer.py](../cs336_systems/sharded_optimizer.py)；源数据为双卡基准的 `010/011` 两项。

### fsdp / fsdp_accounting

仅分片 Linear/Embedding 权重，norm 等小参数复制；FP32 主 shard 包含必要 padding，优化器只持有本地 shard。
前向逐层 all-gather，当前层结束后才开始两层之后的预取；按 SwiGLU 的实际 w1,w3,w2 执行顺序安排。
反向线性层重新 gather 权重用于 dX，完整 dW reduce-scatter 到 owner；embedding backward 只需 indices 和 dO，
无需为无用的权重再做 gather。使用后释放完整权重；复制的 norm 梯度另外 all-reduce。
compute_dtype 在通信前转换，而 master/optimizer 保持 FP32。自定义 autograd 返回 shard 形状梯度，
不在 autograd 正使用 Parameter 时反复修改其 storage。
理想 FP32 FSDP 的持久参数/梯度/optimizer 为 16P/N bytes，相比仅 optimizer 分片节省 8P*(1-1/N)；
实际峰值还包含当前/预取完整权重、reduce-scatter 输入、激活和 allocator 缓存。

`fsdp_accounting(a)`：双卡 xl 理论上在 optimizer 分片基础上再节省 4P bytes = 12.691 GiB，
持久张量由约 38.074 GiB 降为 25.383 GiB（忽略不分片的 norm 和 padding）。
实测 FSDP 记录步骤峰值 33.051 GiB，对照 optimizer 分片 39.396 GiB 少 6.346 GiB，
没有达到完整理论差值，因为峰值不是只有持久张量，且不同阶段的峰值和 gather 临时值不相同。
初始化时最终活跃仅约 6.408 GiB，但历史初始化峰值较大，因为构造期间先有未分片模型；不能把最终活跃值当作整个构造过程的峰值。

`fsdp_accounting(b)`：无 profiler 全训练步为 545.001 ms；它说明端到端可运行，不能回答每层 gather 是否及时完成。
需要对每层 `fsdp_all_gather` 完成与 `linear_forward` 启动的关系、`weight_wait` 暴露时长给出有效 GPU 时间线和截图，
当前归档的 CUDA kernel 事件缺失，因此该项明确未完成，不把“有预取代码”当作“预取总能赶上”的证据。
实现：[fsdp.py](../cs336_systems/fsdp.py)；源数据为双卡基准 `012-xl-fsdp-adamw/result.json`。

## 5. Parallelism calculations

统一记 Dff 为 FFN 中间维，B 为全局 token 数，C 为每设备 FLOP/s，W 为每设备 egress bytes/s。
本节按题目简化模型，所有通信元素 2 bytes，每次乘加 2 FLOPs，忽略逐元素计算与启动延迟。
AG 和 RS 各需 (N-1)/N*S/W，AR 需 2*(N-1)/N*S/W。

### alternate_ring_all_reduce

替代算法每一步传递完整 S-byte 原始张量，共 N-1 步，时间为 (N-1)*S/W；
相比 RS+AG 的 2*(N-1)/N*S/W，在 N>2 时多传 N/2 倍的数据。

### data_parallel_calcs

(a) 每设备 backward 有 6 个大小相当的 GEMM（dZ、dX 的两个贡献、三个权重梯度），
各自为 2*(B/N)*D*Dff FLOPs，总 FLOPs = 12*B*D*Dff/N。

(b) 3 个权重梯度共 3*D*Dff 元素，即 6*D*Dff bytes，
AR 时间 = 2*(N-1)/N * 6*D*Dff/W = 12*(N-1)*D*Dff/(N*W)。

(c) 要求通信不慢于 backward 计算，即
12*(N-1)*D*Dff/(N*W) <= 12*B*D*Dff/(N*C)，消去公共项得到 N <= 1+B*W/C。
增大 batch 能提升每设备通信所对应的计算量；D、Dff 同时出现在两边，所以在简化模型中消去。

### fsdp_calcs

(a) forward 有 3 个 GEMM，FLOPs = 6*B*D*Dff/N；backward 有 6 个 GEMM，FLOPs = 12*B*D*Dff/N，
FSDP 改变权重的持有方式，不改变取得完整权重后本地 batch 的矩阵乘法工作量。

(b) forward 的三个权重总大小为 6*D*Dff bytes，AG 时间 = 6*(N-1)*D*Dff/(N*W)。
backward 重新 AG 加梯度 RS，两个 collective 的大小相同，所以通信时间 = 12*(N-1)*D*Dff/(N*W)。

(c) 前向要求 6*(N-1)*D*Dff/(N*W) <= 6*B*D*Dff/(N*C)；
后向将不等式两边都乘 2，因此两者的 compute-bound 条件均为 N <= 1+B*W/C。
此结论依赖题目的完全重叠和理想带宽假设；实际上细粒度层的依赖、延迟和预取距离会进一步限制伸缩。

### tp_calcs（Tensor Parallel）

(a) 令 i 为 TP rank，W1_i/W2_i 为 (D,Dff/N)，W3_i 为 (Dff/N,D)；
所有设备收到相同的 dY，保存共同输入 X 和本地 X1_i、X2_i、Z_i。反向方程为：

```text
dZ_i  = dY @ W3_i.T                       # (B, Dff/N)
dX2_i = dZ_i * f(X1_i)                    # (B, Dff/N)
dX1_i = dZ_i * f'(X1_i) * X2_i            # (B, Dff/N)
dW3_i = Z_i.T @ dY                       # (Dff/N, D)
dW2_i = X.T @ dX2_i                      # (D, Dff/N)
dW1_i = X.T @ dX1_i                      # (D, Dff/N)
dX_partial_i = dX1_i @ W1_i.T + dX2_i @ W2_i.T
dX = all_reduce_sum({dX_partial_i}_{i=0..N-1})   # (B, D)
```

每个权重梯度对应自己的特征 shard，不需要在 TP 轴再约简；上游的同一个 dY 不是 N 份独立 loss，
不能再对它做一次 sum 而重复乘 N。两个 dX 局部贡献先本地相加，再一次 AR。

(b) 前向 3 个 GEMM 的局部中间维都是 Dff/N，共 6*B*D*Dff/N FLOPs；
反向 6 个 GEMM 共 12*B*D*Dff/N FLOPs，忽略非矩阵乘法。

(c) 前向输出局部和、反向 dX 局部和均为 (B,D)，2*B*D bytes；因此前反向均需
AR 时间 = 2*(N-1)/N * 2*B*D/W = 4*(N-1)*B*D/(N*W)。

(d) 用各自 FLOPs/C 与通信时间比较，前向 N <= 1+(3/2)*Dff*W/C，
反向 N <= 1+3*Dff*W/C；同样通信量下 backward 计算约为 forward 的两倍，因此允许更大的 N。
把同一 dX 的两个贡献分别 all-reduce 会白白多一倍通信。

### fsdp_tp_calcs

(a) 记 F=N_FSDP、T=N_TP，forward FLOPs = 6*B*D*Dff/(F*T)，因为三个 GEMM 同时缩小 batch 到 B/F、
中间宽度到 Dff/T。

(b) 每个 TP shard 的三个权重共 6*D*Dff/T bytes，
FSDP 轴 AG 时间 A = 6*(F-1)*D*Dff/(F*T*W)；每个 FSDP shard 的输出有 2*B*D/F bytes，
TP 轴 AR 时间 Bc = 4*(T-1)*B*D/(F*T*W)。两轴按题设可同时通信，因此通信时间=max(A,Bc)。

(c) 令计算时间 t=6*B*D*Dff/(F*T*C)，A/t=(F-1)/a，Bc/t=(T-1)/b，
其中 a=B*W/C、b=(3/2)*Dff*W/C；分别不大于 1 要求 F<=1+a、T<=1+b，
因此 N=F*T <= (1+a)*(1+b)，忽略整数约束时同时取上界可达到。
展开为 N <= (1+B*W/C)*(1+3*Dff*W/(2*C))；实际还需可整除维度及整数设备数。

(d) 不能重叠时通信时间=A+Bc，条件为 (F-1)/a+(T-1)/b<=1，且 F,T>=1。
令 L=1+1/a+1/b。若内部最优解 F*=a*L/2、T*=b*L/2 均至少为 1，
则 N<=a*b*L^2/4；若 b<a/(a+1)，最优在 T=1，N<=a+1；
若 a<b/(b+1)，最优在 F=1，N<=b+1。大 a,b 时内部上界近似 a*b/4，
但直接写此近似会丢掉有限 N 和单轴最优的边界情况。
推导为：最优点用满通信预算，T=b*(L-F/a)，于是
N(F)=b*L*F-(b/a)*F^2 是开口向下的二次函数；导数为零得到上面的 F*，
若 F* 或 T* 小于 1，就沿可行线段取相应端点。a、b 已显式给出 B、Dff、C、W 的表达式，
没有额外未知参数。有限设备数约束只会降低这一连续上界。

## 6. leaderboard

只测本地提交候选，不向公开 leaderboard 或课程平台发送内容。配置严格为 34 层、d=4096、dff=11008、
32 头、vocab=151936、context=32768、全局 batch=2、两张 B200、BF16 计算及 FP32 主参数。
路径保留 logits-producing forward，额外 training_loss 用等价分块投影交叉熵避免完整大 logits；
按 block checkpoint、自编 Triton 前后向、重叠 DDP、优化器状态分片、自编融合 AdamW。
先比较小模型输出/梯度、FP32 分块损失代数与 BF16 误差，再运行完整配置；冷缓存从新的独立目录开始，
wall-clock 全步包含 forward/loss/backward/通信/AdamW/owner broadcast，单个完整尝试限制 590 秒。
数值容限、失败尝试、首次编译时间、每一步原始数据和未达成事项都必须在最终报告中写明。

本次已有两个成功完整配置候选。默认 tile 候选 2 次预热、3 次测量，最慢 rank 全步样本为
9831.028/9692.502/9583.723 ms，均值 9702.418 ms；调大 tile 的 tuned 候选为 Q64/K128、反向 tile64，
2 次预热、5 次测量的样本如下（每次先取两 rank 的最大值）：

| 测量步 | 完整训练步 ms |
|---|---:|
| 1 | 6799.804107 |
| 2 | 6778.172392 |
| 3 | 6732.982269 |
| 4 | 6759.455822 |
| 5 | 6730.328561 |
| 均值 ± 样本标准差 | 6760.148630 ± 29.686447 |

现有最佳候选均值为 **6.760149 s/step**，相对默认候选约快 1.44 倍。
rank0 首次冷缓存预热步为 16.780 s，第二次为 6.828 s；完整 case 含初始化/编译/预热/测量墙钟为 76.173 s，
远低于 600 s 的限定。本条是本地候选实验结果，不是课程评测或公开榜单验收结果；没有向外提交。
数值来自已有 [tuned 完成状态快照](../runs/status/20260914T093842Z-leaderboard-tuned-20260914T094503Z.json)，
并与随后只读取回的 [27 个非缓存文件](../runs/cloud/20260914T093842Z-leaderboard-tuned/20260914T105045Z-storage-api) 交叉核对。
环境、源码、全部 rank 样本及日志已归档并核验 SHA256；可重新生成的编译缓存未下载，不把它宣称为全远程文件备份。

正确性证据和限制：tuned 小模型输出最大绝对差为 0.01953125，最大参数梯度绝对差为 0.000244140625；
该辅助测试的输出容限是 atol=0.025、rtol=0.03，梯度容限是 atol=0.001、rtol=0.1。
最大绝对差大于 0.01 本身不能判定 `atol=rtol=0.01` 是否通过，因为允许误差随参考值变化；
反过来，宽容限测试通过也不能证明正式验收通过，最终候选的统一官方容限检查仍待做。

分块交叉熵的 FP32 loss/dX/dW 最大差为 1.907e-6/4.433e-7/3.278e-7，
BF16 loss 差为 0.00575447、dX/dW 最大差为 0.00439453/0.00325394；
输入被分块后 GEMM 的 M 维改变，可能选择不同归约次序，故将 FP32 代数检查和 BF16 舍入检查分开。
自编融合 AdamW 与 staff AdamW 相同梯度迭代 10 步后参数最大差为 2.384e-7。
这些检查支持继续使用候选进行实验，但不替代完整 BF16 因果模型/梯度的正式测试，
也不把训练中 loss 数值变化解释为模型在真实语料上收敛。
原始数值和检查配置见 [tuned validation](../runs/status/20260914T093124Z-tuned-validation-20260914T093546Z.json)。

## 7. 证据完整性与交付边界

- 实验过程、配置来源、错误与修复记录见 [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) 和 [WORK_LOG.md](../WORK_LOG.md)。
  最初的 non-leaf checker 失败、优化器验证中 BF16 严格代数 oracle 失败、saved hook 返回值错误等历史记录保留，
  不把这些失败计入成功结果。原始 OOM 也保留其 trace，而非替换为预计可运行的较小输入。
- “6 个 attention 测试 + 8 个 distributed 测试”是历史不同执行批次的通过记录，不等于最终源码做过一次全套 GPU 验收。
  CPU 测试中 CUDA skip 不是 GPU pass；补充 checkpoint/optimizer CPU 测试也不是完整多 GPU race 验证。
- 当前源码的原始 Gloo 分布式 suite 已在本机五个独立进程中各通过 8 项（共 40 次测试执行），无失败或跳过，
  源码前后 23 个文件哈希一致。见 [五轮 JUnit、日志与 manifest](../runs/local/final-gloo-stability-localhost)。
  首次受限沙箱不能访问本机 TCP 端口的环境失败另行保留，不混入这五轮结果；没有租用 GPU。
- 模型/注意力表中 `mean_ms`、原始样本、OOM 和错误状态来自相应配置；profile 脚本外层成功、内部 summary 缺失时，
  不将其计作有效 GPU profiling。截图必须是真实工具界面，不能用绘制示意图冒充 Nsight/memory_viz。
- 所有 GPU 后续任务及验收产物单列在 [GPU_REMAINING_TASKS.md](../GPU_REMAINING_TASKS.md)。
  Flash 240 项、通信 24 项、已有模型/内存/checkpoint/DDP 计时不整套重跑；先利用现有快照，必要补跑须另获预算许可。
- 本文的理论题均已给出答案；未完成实验的小问明示缺口。报告可按“现有实验阶段性完成、保留不足”交付，
  不能在文首或总结改写为“所有 handout deliverable 已齐”。


---

# 附录 B · 完整数值表


时间单位为 ms；± 为样本标准差（ddof=1），不是置信区间。GiB=2³⁰ bytes；通信 MB=10⁶ bytes。E 编号指向 evidence_index.json，R 编号见文末。n=1 时脚本保存的 SD=0 仅是占位，不能推断稳定性。结果 ok 不等于 Nsight GPU kernel 已捕获，也不表示所有附件已在本地。本次整理不启动 GPU/云端计算。

## 0. 覆盖概况

| 矩阵 | 计划格子 | ok | OOM | 失败 | 缺失 |
| --- | --- | --- | --- | --- | --- |
| models | 75 | 70 | 5 | 0 | 0 |
| models_compile_including_missing_train | 15 | 10 | 0 | 0 | 5 |
| attention | 40 | 40 | 0 | 0 | 0 |
| flash | 240 | 240 | 0 | 0 | 0 |
| communication | 24 | 24 | 0 | 0 | 0 |
| distributed_xl | 5 | 5 | 0 | 0 | 0 |
| memory | 8 | 5 | 3 | 0 | 0 |
| checkpoint | 7 | 6 | 1 | 0 | 0 |
| distributed_repeat_recommended | 5 | 5 | 0 | 0 | 0 |

建议五轮稳定性采用 final-gloo-stability 下本地 CPU/Gloo JUnit，未新增 GPU 实验，也不等于 NCCL 数值验证。历史云端单轮结果仍保留。

## 1. 五模型基准、混合精度与预热

默认 B=4、S=512、词表 10000。F 保留 autograd；F+CE+B 是前向/损失/反向合计，不能叫纯 backward。完整训练再含 staff AdamW。BF16 为 autocast，参数仍 FP32。成功配置各保存 10 次原始计时。

### small

| 精度 | 预热 | 计时范围 | 均值 ± SD | 实存 n | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 5 | F | 16.639 ± 0.157 | 10 | ok | E007/R04 |
| fp32 | 5 | F+CE+B | 49.373 ± 0.318 | 10 | ok | E008/R04 |
| fp32 | 5 | F+CE+B+AdamW | 57.321 ± 2.477 | 10 | ok | E009/R04 |
| bf16 | 5 | F | 7.230 ± 0.023 | 10 | ok | E010/R04 |
| bf16 | 5 | F+CE+B | 25.214 ± 1.695 | 10 | ok | E011/R04 |
| bf16 | 5 | F+CE+B+AdamW | 31.858 ± 1.827 | 10 | ok | E012/R04 |
| fp32 | 0 | F | 31.298 ± 46.670 | 10 | ok | E013/R04 |
| fp32 | 0 | F+CE+B | 83.473 ± 110.491 | 10 | ok | E014/R04 |
| fp32 | 0 | F+CE+B+AdamW | 92.676 ± 114.457 | 10 | ok | E015/R04 |
| fp32 | 1 | F | 16.539 ± 0.079 | 10 | ok | E016/R04 |
| fp32 | 1 | F+CE+B | 48.782 ± 0.269 | 10 | ok | E017/R04 |
| fp32 | 1 | F+CE+B+AdamW | 56.788 ± 0.895 | 10 | ok | E018/R04 |
| fp32 | 2 | F | 16.517 ± 0.028 | 10 | ok | E019/R04 |
| fp32 | 2 | F+CE+B | 48.622 ± 0.220 | 10 | ok | E020/R04 |
| fp32 | 2 | F+CE+B+AdamW | 56.993 ± 1.553 | 10 | ok | E021/R04 |

### medium

| 精度 | 预热 | 计时范围 | 均值 ± SD | 实存 n | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 5 | F | 47.321 ± 0.082 | 10 | ok | E022/R04 |
| fp32 | 5 | F+CE+B | 139.904 ± 0.249 | 10 | ok | E023/R04 |
| fp32 | 5 | F+CE+B+AdamW | 170.295 ± 41.395 | 10 | ok | E024/R04 |
| bf16 | 5 | F | 17.263 ± 0.132 | 10 | ok | E025/R04 |
| bf16 | 5 | F+CE+B | 51.363 ± 0.678 | 10 | ok | E026/R04 |
| bf16 | 5 | F+CE+B+AdamW | 68.062 ± 0.823 | 10 | ok | E027/R04 |
| fp32 | 0 | F | 64.426 ± 54.160 | 10 | ok | E028/R04 |
| fp32 | 0 | F+CE+B | 173.742 ± 106.951 | 10 | ok | E029/R04 |
| fp32 | 0 | F+CE+B+AdamW | 199.185 ± 133.775 | 10 | ok | E030/R04 |
| fp32 | 1 | F | 47.248 ± 0.129 | 10 | ok | E031/R04 |
| fp32 | 1 | F+CE+B | 139.933 ± 0.321 | 10 | ok | E032/R04 |
| fp32 | 1 | F+CE+B+AdamW | 158.197 ± 4.464 | 10 | ok | E033/R04 |
| fp32 | 2 | F | 47.365 ± 0.369 | 10 | ok | E034/R04 |
| fp32 | 2 | F+CE+B | 139.819 ± 0.258 | 10 | ok | E035/R04 |
| fp32 | 2 | F+CE+B+AdamW | 156.692 ± 0.159 | 10 | ok | E036/R04 |

### large

| 精度 | 预热 | 计时范围 | 均值 ± SD | 实存 n | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 5 | F | 104.262 ± 0.224 | 10 | ok | E038/R06 |
| fp32 | 5 | F+CE+B | 311.508 ± 0.125 | 10 | ok | E039/R06 |
| fp32 | 5 | F+CE+B+AdamW | 345.264 ± 0.080 | 10 | ok | E040/R06 |
| bf16 | 5 | F | 44.591 ± 0.072 | 10 | ok | E041/R06 |
| bf16 | 5 | F+CE+B | 110.595 ± 0.352 | 10 | ok | E042/R06 |
| bf16 | 5 | F+CE+B+AdamW | 150.556 ± 0.538 | 10 | ok | E043/R06 |
| fp32 | 0 | F | 140.349 ± 114.569 | 10 | ok | E044/R06 |
| fp32 | 0 | F+CE+B | 380.214 ± 217.536 | 10 | ok | E045/R06 |
| fp32 | 0 | F+CE+B+AdamW | 450.220 ± 285.339 | 10 | ok | E046/R06 |
| fp32 | 1 | F | 103.992 ± 0.054 | 10 | ok | E047/R06 |
| fp32 | 1 | F+CE+B | 311.386 ± 0.100 | 10 | ok | E048/R06 |
| fp32 | 1 | F+CE+B+AdamW | 345.395 ± 0.320 | 10 | ok | E049/R06 |
| fp32 | 2 | F | 104.111 ± 0.118 | 10 | ok | E050/R06 |
| fp32 | 2 | F+CE+B | 311.411 ± 0.073 | 10 | ok | E051/R06 |
| fp32 | 2 | F+CE+B+AdamW | 345.660 ± 1.927 | 10 | ok | E052/R06 |

### xl

| 精度 | 预热 | 计时范围 | 均值 ± SD | 实存 n | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 5 | F | 292.682 ± 0.109 | 10 | ok | E053/R06 |
| fp32 | 5 | F+CE+B | 861.903 ± 0.112 | 10 | ok | E054/R06 |
| fp32 | 5 | F+CE+B+AdamW | 941.142 ± 0.124 | 10 | ok | E055/R06 |
| bf16 | 5 | F | 49.970 ± 0.046 | 10 | ok | E056/R06 |
| bf16 | 5 | F+CE+B | 155.557 ± 0.089 | 10 | ok | E057/R06 |
| bf16 | 5 | F+CE+B+AdamW | 234.963 ± 0.202 | 10 | ok | E058/R06 |
| fp32 | 0 | F | 323.585 ± 97.550 | 10 | ok | E059/R06 |
| fp32 | 0 | F+CE+B | 935.201 ± 231.831 | 10 | ok | E060/R06 |
| fp32 | 0 | F+CE+B+AdamW | 1014.685 ± 232.459 | 10 | ok | E061/R06 |
| fp32 | 1 | F | 292.721 ± 0.195 | 10 | ok | E062/R06 |
| fp32 | 1 | F+CE+B | 861.900 ± 0.082 | 10 | ok | E063/R06 |
| fp32 | 1 | F+CE+B+AdamW | 941.124 ± 0.326 | 10 | ok | E064/R06 |
| fp32 | 2 | F | 292.798 ± 0.212 | 10 | ok | E065/R06 |
| fp32 | 2 | F+CE+B | 861.961 ± 0.214 | 10 | ok | E066/R06 |
| fp32 | 2 | F+CE+B+AdamW | 941.126 ± 0.222 | 10 | ok | E067/R06 |

### 10B

| 精度 | 预热 | 计时范围 | 均值 ± SD | 实存 n | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 5 | F | 941.318 ± 0.129 | 10 | ok | E068/R06 |
| fp32 | 5 | F+CE+B | 2810.166 ± 0.075 | 10 | ok | E069/R06 |
| fp32 | 5 | F+CE+B+AdamW | — | — | oom | E070/R06 |
| bf16 | 5 | F | 118.142 ± 0.149 | 10 | ok | E071/R06 |
| bf16 | 5 | F+CE+B | 404.701 ± 1.191 | 10 | ok | E072/R06 |
| bf16 | 5 | F+CE+B+AdamW | — | — | oom | E073/R06 |
| fp32 | 0 | F | 968.797 ± 84.202 | 10 | ok | E074/R06 |
| fp32 | 0 | F+CE+B | 2872.096 ± 195.182 | 10 | ok | E075/R06 |
| fp32 | 0 | F+CE+B+AdamW | — | — | oom | E076/R06 |
| fp32 | 1 | F | 962.881 ± 65.824 | 10 | ok | E077/R06 |
| fp32 | 1 | F+CE+B | 2810.342 ± 0.980 | 10 | ok | E078/R06 |
| fp32 | 1 | F+CE+B+AdamW | — | — | oom | E079/R06 |
| fp32 | 2 | F | 942.264 ± 0.428 | 10 | ok | E080/R06 |
| fp32 | 2 | F+CE+B | 2809.047 ± 1.017 | 10 | ok | E081/R06 |
| fp32 | 2 | F+CE+B+AdamW | — | — | oom | E082/R06 |

## 2. torch.compile 模型对照

FP32、预热 5、每个成功配置 10 次计时。compile 作用于 model。此前计划遗漏五种完整 train 配置，现显式列出，不能用 F+CE+B 代替完整训练。

| 模型 | 范围 | compiled 均值 ± SD | eager/compiled | n | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| small | F | 13.169 ± 0.012 | 1.263 | 10 | ok | E243/R15 |
| small | F+CE+B | 37.898 ± 0.149 | 1.303 | 10 | ok | E244/R15 |
| small | F+CE+B+AdamW | — | — | — | 缺失 | 无本地证据 |
| medium | F | 39.220 ± 0.122 | 1.207 | 10 | ok | E245/R15 |
| medium | F+CE+B | 113.319 ± 0.206 | 1.235 | 10 | ok | E246/R15 |
| medium | F+CE+B+AdamW | — | — | — | 缺失 | 无本地证据 |
| large | F | 86.922 ± 0.283 | 1.199 | 10 | ok | E441/R24 |
| large | F+CE+B | 261.239 ± 0.301 | 1.192 | 10 | ok | E442/R24 |
| large | F+CE+B+AdamW | — | — | — | 缺失 | 无本地证据 |
| xl | F | 268.682 ± 0.302 | 1.089 | 10 | ok | E443/R24 |
| xl | F+CE+B | 785.778 ± 0.395 | 1.097 | 10 | ok | E444/R24 |
| xl | F+CE+B+AdamW | — | — | — | 缺失 | 无本地证据 |
| 10B | F | 893.501 ± 0.140 | 1.054 | 10 | ok | E447/R27 |
| 10B | F+CE+B | 2665.385 ± 0.434 | 1.054 | 10 | ok | E448/R27 |
| 10B | F+CE+B+AdamW | — | — | — | 缺失 | 无本地证据 |

## 3. CUDA event 分阶段计时

只列原始记录实际保存的事件时间；该口径不同于同步 wall-clock。小/中模型首次 eager 矩阵未加入埋点，不能倒算纯 backward 或 AdamW。

| 配置 | F | CE | B | AdamW | 证据 |
| --- | --- | --- | --- | --- | --- |
| large/fp32/w5/eager/F | 102.960 ± 0.105 | — | — | — | E038/R06 |
| large/fp32/w5/eager/F+CE+B | 102.868 ± 0.051 | 0.069 ± 0.000 | 206.808 ± 0.032 | — | E039/R06 |
| large/fp32/w5/eager/F+CE+B+AdamW | 102.797 ± 0.047 | 0.069 ± 0.000 | 206.816 ± 0.033 | 32.021 ± 0.076 | E040/R06 |
| large/bf16/w5/eager/F | 40.664 ± 0.064 | — | — | — | E041/R06 |
| large/bf16/w5/eager/F+CE+B | 40.371 ± 0.179 | 0.128 ± 0.010 | 66.648 ± 0.219 | — | E042/R06 |
| large/bf16/w5/eager/F+CE+B+AdamW | 41.385 ± 0.213 | 0.136 ± 0.010 | 67.813 ± 0.299 | 37.324 ± 0.090 | E043/R06 |
| large/fp32/w0/eager/F | 138.861 ± 113.725 | — | — | — | E044/R06 |
| large/fp32/w0/eager/F+CE+B | 132.366 ± 93.282 | 3.992 ± 12.406 | 241.999 ± 111.331 | — | E045/R06 |
| large/fp32/w0/eager/F+CE+B+AdamW | 160.922 ± 143.534 | 6.424 ± 20.100 | 246.602 ± 122.123 | 32.546 ± 2.367 | E046/R06 |
| large/fp32/w1/eager/F | 102.841 ± 0.054 | — | — | — | E047/R06 |
| large/fp32/w1/eager/F+CE+B | 102.814 ± 0.067 | 0.069 ± 0.000 | 206.841 ± 0.028 | — | E048/R06 |
| large/fp32/w1/eager/F+CE+B+AdamW | 102.851 ± 0.124 | 0.069 ± 0.001 | 206.878 ± 0.103 | 31.881 ± 0.108 | E049/R06 |
| large/fp32/w2/eager/F | 102.896 ± 0.043 | — | — | — | E050/R06 |
| large/fp32/w2/eager/F+CE+B | 102.822 ± 0.042 | 0.069 ± 0.000 | 206.859 ± 0.023 | — | E051/R06 |
| large/fp32/w2/eager/F+CE+B+AdamW | 102.727 ± 0.063 | 0.068 ± 0.001 | 206.775 ± 0.062 | 32.163 ± 1.423 | E052/R06 |
| xl/fp32/w5/eager/F | 291.647 ± 0.036 | — | — | — | E053/R06 |
| xl/fp32/w5/eager/F+CE+B | 291.718 ± 0.042 | 0.067 ± 0.001 | 568.639 ± 0.067 | — | E054/R06 |
| xl/fp32/w5/eager/F+CE+B+AdamW | 291.708 ± 0.022 | 0.067 ± 0.001 | 568.614 ± 0.063 | 79.298 ± 0.100 | E055/R06 |
| xl/bf16/w5/eager/F | 48.952 ± 0.029 | — | — | — | E056/R06 |
| xl/bf16/w5/eager/F+CE+B | 48.975 ± 0.050 | 0.122 ± 0.000 | 105.025 ± 0.047 | — | E057/R06 |
| xl/bf16/w5/eager/F+CE+B+AdamW | 48.943 ± 0.090 | 0.122 ± 0.000 | 104.937 ± 0.074 | 79.335 ± 0.100 | E058/R06 |
| xl/fp32/w0/eager/F | 322.548 ± 97.507 | — | — | — | E059/R06 |
| xl/fp32/w0/eager/F+CE+B | 319.893 ± 89.119 | 1.058 ± 3.135 | 612.643 ± 139.066 | — | E060/R06 |
| xl/fp32/w0/eager/F+CE+B+AdamW | 316.996 ± 79.887 | 1.049 ± 3.107 | 602.869 ± 108.449 | 92.110 ± 40.376 | E061/R06 |
| xl/fp32/w1/eager/F | 291.676 ± 0.038 | — | — | — | E062/R06 |
| xl/fp32/w1/eager/F+CE+B | 291.724 ± 0.037 | 0.067 ± 0.001 | 568.685 ± 0.072 | — | E063/R06 |
| xl/fp32/w1/eager/F+CE+B+AdamW | 291.715 ± 0.063 | 0.067 ± 0.001 | 568.566 ± 0.091 | 79.303 ± 0.194 | E064/R06 |
| xl/fp32/w2/eager/F | 291.710 ± 0.069 | — | — | — | E065/R06 |
| xl/fp32/w2/eager/F+CE+B | 291.688 ± 0.044 | 0.067 ± 0.000 | 568.660 ± 0.070 | — | E066/R06 |
| xl/fp32/w2/eager/F+CE+B+AdamW | 291.691 ± 0.026 | 0.067 ± 0.001 | 568.628 ± 0.126 | 79.315 ± 0.123 | E067/R06 |
| 10B/fp32/w5/eager/F | 939.575 ± 0.018 | — | — | — | E068/R06 |
| 10B/fp32/w5/eager/F+CE+B | 939.781 ± 0.046 | 0.066 ± 0.001 | 1867.789 ± 0.047 | — | E069/R06 |
| 10B/bf16/w5/eager/F | 116.255 ± 0.048 | — | — | — | E071/R06 |
| 10B/bf16/w5/eager/F+CE+B | 116.894 ± 0.629 | 0.123 ± 0.001 | 285.111 ± 1.019 | — | E072/R06 |
| 10B/fp32/w0/eager/F | 966.461 ± 84.384 | — | — | — | E074/R06 |
| 10B/fp32/w0/eager/F+CE+B | 968.746 ± 91.616 | 1.125 ± 3.346 | 1899.139 ± 99.578 | — | E075/R06 |
| 10B/fp32/w1/eager/F | 960.718 ± 65.888 | — | — | — | E077/R06 |
| 10B/fp32/w1/eager/F+CE+B | 939.822 ± 0.186 | 0.066 ± 0.001 | 1867.203 ± 0.817 | — | E078/R06 |
| 10B/fp32/w2/eager/F | 939.866 ± 0.107 | — | — | — | E080/R06 |
| 10B/fp32/w2/eager/F+CE+B | 939.593 ± 0.181 | 0.067 ± 0.000 | 1866.488 ± 0.843 | — | E081/R06 |
| small/fp32/w5/compile/F | 12.698 ± 0.013 | — | — | — | E243/R15 |
| small/fp32/w5/compile/F+CE+B | 12.873 ± 0.065 | 0.069 ± 0.000 | 24.148 ± 0.008 | — | E244/R15 |
| medium/fp32/w5/compile/F | 38.172 ± 0.090 | — | — | — | E245/R15 |
| medium/fp32/w5/compile/F+CE+B | 38.214 ± 0.090 | 0.069 ± 0.001 | 73.571 ± 0.013 | — | E246/R15 |
| large/fp32/w5/compile/F | 86.099 ± 0.132 | — | — | — | E441/R24 |
| large/fp32/w5/compile/F+CE+B | 86.374 ± 0.143 | 0.069 ± 0.000 | 173.070 ± 0.081 | — | E442/R24 |
| xl/fp32/w5/compile/F | 267.874 ± 0.203 | — | — | — | E443/R24 |
| xl/fp32/w5/compile/F+CE+B | 268.012 ± 0.124 | 0.066 ± 0.001 | 516.423 ± 0.311 | — | E444/R24 |
| 10B/fp32/w5/compile/F | 892.033 ± 0.107 | — | — | — | E447/R27 |
| 10B/fp32/w5/compile/F+CE+B | 892.049 ± 0.187 | 0.066 ± 0.000 | 1771.058 ± 0.040 | — | E448/R27 |

## 4. 精度累加与 autocast

| 累加器 | 增量 | 结果 | 证据 |
| --- | --- | --- | --- |
| torch.float32 | torch.float32 | 10.000133514404297 | E005/R02 |
| torch.float16 | torch.float16 | 9.953125 | E005/R02 |
| torch.float32 | torch.float16 | 10.00213623046875 | E005/R02 |
| float32 | explicit_cast | 10.00213623046875 | E005/R02 |

| 张量/操作 | 实际 dtype | 证据 |
| --- | --- | --- |
| fc1 | torch.float16 | E005/R02 |
| layernorm | torch.float32 | E005/R02 |
| logits | torch.float16 | E005/R02 |
| loss | torch.float32 | E005/R02 |
| 参数 fc1.weight | torch.float32 | E005/R02 |
| 参数 ln.weight | torch.float32 | E005/R02 |
| 参数 ln.bias | torch.float32 | E005/R02 |
| 参数 fc2.weight | torch.float32 | E005/R02 |
| 梯度 fc1.weight | torch.float32 | E005/R02 |
| 梯度 ln.weight | torch.float32 | E005/R02 |
| 梯度 ln.bias | torch.float32 | E005/R02 |
| 梯度 fc2.weight | torch.float32 | E005/R02 |

## 5. 普通与 compiled attention：40 格

B=8、非 causal、FP32；F/B/F+B 各预热 5，再计时 100 次。B 是固定前向图的 backward-only。B 前显存是 allocated，不是整步 peak。

| 实现/S/D | F | B | F+B | B 前 GiB | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| eager/256/16 | 0.076 ± 0.007 | 0.243 ± 0.069 | 0.356 ± 0.063 | 0.010 | ok | E083/R07 |
| eager/256/32 | 0.074 ± 0.006 | 0.242 ± 0.066 | 0.343 ± 0.061 | 0.011 | ok | E084/R07 |
| eager/256/64 | 0.080 ± 0.006 | 0.246 ± 0.051 | 0.358 ± 0.063 | 0.012 | ok | E085/R07 |
| eager/256/128 | 0.080 ± 0.007 | 0.250 ± 0.061 | 0.350 ± 0.048 | 0.015 | ok | E086/R07 |
| eager/1024/16 | 0.091 ± 0.004 | 0.231 ± 0.046 | 0.324 ± 0.061 | 0.042 | ok | E087/R07 |
| eager/1024/32 | 0.094 ± 0.003 | 0.231 ± 0.041 | 0.313 ± 0.054 | 0.044 | ok | E088/R07 |
| eager/1024/64 | 0.117 ± 0.003 | 0.249 ± 0.036 | 0.323 ± 0.039 | 0.049 | ok | E089/R07 |
| eager/1024/128 | 0.154 ± 0.003 | 0.326 ± 0.034 | 0.412 ± 0.017 | 0.059 | ok | E090/R07 |
| eager/4096/16 | 0.989 ± 0.003 | 1.601 ± 0.036 | 2.483 ± 0.009 | 0.518 | ok | E091/R07 |
| eager/4096/32 | 1.075 ± 0.003 | 1.683 ± 0.033 | 2.658 ± 0.009 | 0.527 | ok | E092/R07 |
| eager/4096/64 | 1.348 ± 0.003 | 2.237 ± 0.029 | 3.460 ± 0.010 | 0.547 | ok | E093/R07 |
| eager/4096/128 | 1.889 ± 0.003 | 3.325 ± 0.025 | 5.091 ± 0.009 | 0.586 | ok | E094/R07 |
| eager/8192/16 | 4.154 ± 0.005 | 5.767 ± 0.027 | 9.798 ± 0.015 | 2.027 | ok | E095/R07 |
| eager/8192/32 | 4.496 ± 0.005 | 6.117 ± 0.026 | 10.487 ± 0.009 | 2.047 | ok | E096/R07 |
| eager/8192/64 | 5.546 ± 0.005 | 8.215 ± 0.058 | 13.606 ± 0.233 | 2.086 | ok | E097/R07 |
| eager/8192/128 | 7.827 ± 0.006 | 12.989 ± 0.029 | 20.666 ± 0.021 | 2.164 | ok | E098/R07 |
| eager/16384/16 | 12.662 ± 0.019 | 21.674 ± 0.135 | 34.295 ± 0.179 | 8.047 | ok | E099/R07 |
| eager/16384/32 | 14.124 ± 0.006 | 23.273 ± 0.182 | 37.301 ± 0.319 | 8.086 | ok | E100/R07 |
| eager/16384/64 | 18.420 ± 0.004 | 32.072 ± 0.025 | 50.352 ± 0.012 | 8.164 | ok | E101/R07 |
| eager/16384/128 | 25.755 ± 0.005 | 47.417 ± 0.028 | 73.036 ± 0.018 | 8.320 | ok | E102/R07 |
| compiled/256/16 | 0.135 ± 0.010 | 0.220 ± 0.012 | 0.419 ± 0.060 | 0.010 | ok | E103/R07 |
| compiled/256/32 | 0.136 ± 0.010 | 0.255 ± 0.067 | 0.454 ± 0.077 | 0.011 | ok | E104/R07 |
| compiled/256/64 | 0.133 ± 0.009 | 0.230 ± 0.042 | 0.429 ± 0.079 | 0.012 | ok | E105/R07 |
| compiled/256/128 | 0.137 ± 0.009 | 0.235 ± 0.046 | 0.423 ± 0.060 | 0.015 | ok | E106/R07 |
| compiled/1024/16 | 0.136 ± 0.009 | 0.268 ± 0.072 | 0.473 ± 0.065 | 0.042 | ok | E107/R07 |
| compiled/1024/32 | 0.134 ± 0.007 | 0.250 ± 0.067 | 0.415 ± 0.046 | 0.044 | ok | E108/R07 |
| compiled/1024/64 | 0.142 ± 0.008 | 0.262 ± 0.036 | 0.433 ± 0.053 | 0.049 | ok | E109/R07 |
| compiled/1024/128 | 0.179 ± 0.005 | 0.345 ± 0.055 | 0.504 ± 0.040 | 0.059 | ok | E110/R07 |
| compiled/4096/16 | 0.703 ± 0.004 | 1.162 ± 0.045 | 1.784 ± 0.023 | 0.518 | ok | E111/R07 |
| compiled/4096/32 | 0.790 ± 0.005 | 1.253 ± 0.056 | 1.957 ± 0.026 | 0.527 | ok | E112/R07 |
| compiled/4096/64 | 1.054 ± 0.005 | 1.807 ± 0.061 | 2.761 ± 0.035 | 0.547 | ok | E113/R07 |
| compiled/4096/128 | 1.597 ± 0.007 | 2.921 ± 0.057 | 4.370 ± 0.018 | 0.586 | ok | E114/R07 |
| compiled/8192/16 | 2.416 ± 0.003 | 4.196 ± 0.041 | 6.453 ± 0.015 | 2.027 | ok | E115/R07 |
| compiled/8192/32 | 2.758 ± 0.003 | 4.570 ± 0.040 | 7.163 ± 0.015 | 2.047 | ok | E116/R07 |
| compiled/8192/64 | 3.782 ± 0.004 | 6.569 ± 0.037 | 10.191 ± 0.017 | 2.086 | ok | E117/R07 |
| compiled/8192/128 | 6.071 ± 0.004 | 11.403 ± 0.039 | 17.304 ± 0.013 | 2.164 | ok | E118/R07 |
| compiled/16384/16 | 8.954 ± 0.083 | 15.160 ± 0.134 | 24.114 ± 0.235 | 8.047 | ok | E119/R07 |
| compiled/16384/32 | 10.308 ± 0.056 | 16.683 ± 0.180 | 26.974 ± 0.241 | 8.086 | ok | E120/R07 |
| compiled/16384/64 | 14.520 ± 0.084 | 25.548 ± 0.038 | 39.908 ± 0.014 | 8.164 | ok | E121/R07 |
| compiled/16384/128 | 21.881 ± 0.007 | 40.886 ± 0.035 | 62.594 ± 0.018 | 8.320 | ok | E122/R07 |

## 6. FlashAttention：240 格

B=1、causal。eager=普通 PyTorch；triton=手写前向+compiled PyTorch 反向；triton-full=手写前向和反向。调用 do_bench(warmup=100ms,rep=300ms,return_mode=mean)，内部 n/原始样本/SD 未保存，配置 steps=10 未用于此分支。下列为聚合均值；不是 10 个独立样本。

### eager / fp32

| S | D | F | B | F+B | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 16 | 0.122 | 0.249 | 0.455 | ok | E163/R14 |
| 128 | 32 | 0.141 | 0.255 | 0.448 | ok | E164/R14 |
| 128 | 64 | 0.126 | 0.230 | 0.424 | ok | E165/R14 |
| 128 | 128 | 0.123 | 0.226 | 0.413 | ok | E166/R14 |
| 256 | 16 | 0.122 | 0.231 | 0.423 | ok | E167/R14 |
| 256 | 32 | 0.140 | 0.256 | 0.468 | ok | E168/R14 |
| 256 | 64 | 0.116 | 0.234 | 0.437 | ok | E169/R14 |
| 256 | 128 | 0.120 | 0.245 | 0.456 | ok | E170/R14 |
| 512 | 16 | 0.131 | 0.250 | 0.473 | ok | E171/R14 |
| 512 | 32 | 0.148 | 0.256 | 0.474 | ok | E172/R14 |
| 512 | 64 | 0.128 | 0.250 | 0.451 | ok | E173/R14 |
| 512 | 128 | 0.136 | 0.244 | 0.447 | ok | E174/R14 |
| 1024 | 16 | 0.136 | 0.248 | 0.449 | ok | E175/R14 |
| 1024 | 32 | 0.120 | 0.232 | 0.422 | ok | E176/R14 |
| 1024 | 64 | 0.118 | 0.222 | 0.410 | ok | E177/R14 |
| 1024 | 128 | 0.118 | 0.223 | 0.402 | ok | E178/R14 |
| 2048 | 16 | 0.131 | 0.242 | 0.447 | ok | E179/R14 |
| 2048 | 32 | 0.121 | 0.216 | 0.408 | ok | E180/R14 |
| 2048 | 64 | 0.120 | 0.218 | 0.410 | ok | E181/R14 |
| 2048 | 128 | 0.136 | 0.216 | 0.414 | ok | E182/R14 |
| 4096 | 16 | 0.313 | 0.405 | 0.718 | ok | E183/R14 |
| 4096 | 32 | 0.331 | 0.394 | 0.718 | ok | E184/R14 |
| 4096 | 64 | 0.351 | 0.450 | 0.790 | ok | E185/R14 |
| 4096 | 128 | 0.391 | 0.574 | 0.944 | ok | E186/R14 |
| 8192 | 16 | 1.033 | 1.181 | 2.210 | ok | E187/R14 |
| 8192 | 32 | 1.086 | 1.185 | 2.267 | ok | E188/R14 |
| 8192 | 64 | 1.162 | 1.388 | 2.540 | ok | E189/R14 |
| 8192 | 128 | 1.398 | 1.861 | 3.257 | ok | E190/R14 |
| 16384 | 16 | 3.179 | 3.941 | 7.106 | ok | E191/R14 |
| 16384 | 32 | 3.348 | 3.986 | 7.323 | ok | E192/R14 |
| 16384 | 64 | 3.764 | 4.802 | 8.558 | ok | E193/R14 |
| 16384 | 128 | 4.838 | 6.945 | 11.775 | ok | E194/R14 |
| 32768 | 16 | 12.589 | 16.039 | 28.622 | ok | E195/R14 |
| 32768 | 32 | 12.933 | 15.950 | 28.879 | ok | E196/R14 |
| 32768 | 64 | 14.569 | 19.064 | 33.633 | ok | E197/R14 |
| 32768 | 128 | 18.825 | 27.608 | 46.426 | ok | E198/R14 |
| 65536 | 16 | 49.400 | 58.315 | 107.726 | ok | E199/R14 |
| 65536 | 32 | 51.013 | 59.945 | 111.051 | ok | E200/R14 |
| 65536 | 64 | 58.370 | 74.388 | 132.744 | ok | E201/R14 |
| 65536 | 128 | 74.238 | 110.898 | 185.127 | ok | E202/R14 |

### eager / bf16

| S | D | F | B | F+B | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 16 | 0.133 | 0.226 | 0.405 | ok | E203/R14 |
| 128 | 32 | 0.113 | 0.218 | 0.384 | ok | E204/R14 |
| 128 | 64 | 0.123 | 0.238 | 0.407 | ok | E205/R14 |
| 128 | 128 | 0.121 | 0.215 | 0.398 | ok | E206/R14 |
| 256 | 16 | 0.114 | 0.216 | 0.385 | ok | E207/R14 |
| 256 | 32 | 0.125 | 0.226 | 0.418 | ok | E208/R14 |
| 256 | 64 | 0.133 | 0.224 | 0.407 | ok | E209/R14 |
| 256 | 128 | 0.133 | 0.226 | 0.430 | ok | E210/R14 |
| 512 | 16 | 0.113 | 0.211 | 0.400 | ok | E211/R14 |
| 512 | 32 | 0.113 | 0.221 | 0.398 | ok | E212/R14 |
| 512 | 64 | 0.114 | 0.220 | 0.387 | ok | E213/R14 |
| 512 | 128 | 0.123 | 0.227 | 0.424 | ok | E214/R14 |
| 1024 | 16 | 0.133 | 0.229 | 0.424 | ok | E215/R14 |
| 1024 | 32 | 0.124 | 0.230 | 0.435 | ok | E216/R14 |
| 1024 | 64 | 0.134 | 0.241 | 0.447 | ok | E217/R14 |
| 1024 | 128 | 0.118 | 0.224 | 0.401 | ok | E218/R14 |
| 2048 | 16 | 0.122 | 0.230 | 0.421 | ok | E219/R14 |
| 2048 | 32 | 0.131 | 0.244 | 0.439 | ok | E220/R14 |
| 2048 | 64 | 0.144 | 0.259 | 0.476 | ok | E221/R14 |
| 2048 | 128 | 0.127 | 0.239 | 0.434 | ok | E222/R14 |
| 4096 | 16 | 0.172 | 0.213 | 0.454 | ok | E223/R14 |
| 4096 | 32 | 0.172 | 0.241 | 0.448 | ok | E224/R14 |
| 4096 | 64 | 0.171 | 0.237 | 0.463 | ok | E225/R14 |
| 4096 | 128 | 0.172 | 0.235 | 0.436 | ok | E226/R14 |
| 8192 | 16 | 0.643 | 0.406 | 1.038 | ok | E227/R14 |
| 8192 | 32 | 0.640 | 0.399 | 1.030 | ok | E228/R14 |
| 8192 | 64 | 0.643 | 0.405 | 1.038 | ok | E229/R14 |
| 8192 | 128 | 0.646 | 0.408 | 1.045 | ok | E230/R14 |
| 16384 | 16 | 1.919 | 1.403 | 3.307 | ok | E231/R14 |
| 16384 | 32 | 1.917 | 1.404 | 3.305 | ok | E232/R14 |
| 16384 | 64 | 1.923 | 1.416 | 3.327 | ok | E233/R14 |
| 16384 | 128 | 1.938 | 1.443 | 3.364 | ok | E234/R14 |
| 32768 | 16 | 7.371 | 5.409 | 12.763 | ok | E235/R14 |
| 32768 | 32 | 7.369 | 5.411 | 12.758 | ok | E236/R14 |
| 32768 | 64 | 7.371 | 5.449 | 12.813 | ok | E237/R14 |
| 32768 | 128 | 7.455 | 5.577 | 12.973 | ok | E238/R14 |
| 65536 | 16 | 28.967 | 22.439 | 51.361 | ok | E239/R14 |
| 65536 | 32 | 28.930 | 22.392 | 51.263 | ok | E240/R14 |
| 65536 | 64 | 28.806 | 22.402 | 51.192 | ok | E241/R14 |
| 65536 | 128 | 29.296 | 23.090 | 52.235 | ok | E242/R14 |

### triton / fp32

| S | D | F | B | F+B | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 16 | 0.010 | 0.146 | 0.191 | ok | E272/R20 |
| 128 | 32 | 0.010 | 0.128 | 0.183 | ok | E273/R20 |
| 128 | 64 | 0.011 | 0.113 | 0.177 | ok | E274/R20 |
| 128 | 128 | 0.016 | 0.125 | 0.173 | ok | E275/R20 |
| 256 | 16 | 0.012 | 0.136 | 0.184 | ok | E276/R20 |
| 256 | 32 | 0.014 | 0.114 | 0.157 | ok | E277/R20 |
| 256 | 64 | 0.015 | 0.133 | 0.184 | ok | E278/R20 |
| 256 | 128 | 0.025 | 0.108 | 0.186 | ok | E279/R20 |
| 512 | 16 | 0.017 | 0.122 | 0.173 | ok | E280/R20 |
| 512 | 32 | 0.023 | 0.116 | 0.176 | ok | E281/R20 |
| 512 | 64 | 0.024 | 0.117 | 0.165 | ok | E282/R20 |
| 512 | 128 | 0.039 | 0.131 | 0.164 | ok | E283/R20 |
| 1024 | 16 | 0.027 | 0.126 | 0.182 | ok | E284/R20 |
| 1024 | 32 | 0.037 | 0.114 | 0.165 | ok | E285/R20 |
| 1024 | 64 | 0.039 | 0.122 | 0.177 | ok | E286/R20 |
| 1024 | 128 | 0.072 | 0.131 | 0.192 | ok | E287/R20 |
| 2048 | 16 | 0.049 | 0.153 | 0.199 | ok | E288/R20 |
| 2048 | 32 | 0.066 | 0.168 | 0.230 | ok | E289/R20 |
| 2048 | 64 | 0.072 | 0.168 | 0.238 | ok | E290/R20 |
| 2048 | 128 | 0.135 | 0.199 | 0.335 | ok | E291/R20 |
| 4096 | 16 | 0.091 | 0.356 | 0.446 | ok | E292/R20 |
| 4096 | 32 | 0.127 | 0.396 | 0.522 | ok | E293/R20 |
| 4096 | 64 | 0.137 | 0.441 | 0.577 | ok | E294/R20 |
| 4096 | 128 | 0.260 | 0.591 | 0.850 | ok | E295/R20 |
| 8192 | 16 | 0.244 | 0.929 | 1.170 | ok | E296/R20 |
| 8192 | 32 | 0.352 | 1.041 | 1.390 | ok | E297/R20 |
| 8192 | 64 | 0.345 | 1.297 | 1.643 | ok | E298/R20 |
| 8192 | 128 | 0.731 | 1.872 | 2.603 | ok | E299/R20 |
| 16384 | 16 | 0.670 | 2.929 | 3.596 | ok | E300/R20 |
| 16384 | 32 | 1.006 | 3.261 | 4.261 | ok | E301/R20 |
| 16384 | 64 | 0.981 | 4.313 | 5.286 | ok | E302/R20 |
| 16384 | 128 | 2.308 | 6.959 | 9.260 | ok | E303/R20 |
| 32768 | 16 | 2.228 | 11.966 | 14.188 | ok | E304/R20 |
| 32768 | 32 | 3.406 | 12.676 | 16.070 | ok | E305/R20 |
| 32768 | 64 | 3.247 | 16.782 | 20.026 | ok | E306/R20 |
| 32768 | 128 | 8.007 | 27.353 | 35.372 | ok | E307/R20 |
| 65536 | 16 | 8.153 | 43.529 | 51.686 | ok | E308/R20 |
| 65536 | 32 | 12.617 | 46.760 | 59.383 | ok | E309/R20 |
| 65536 | 64 | 11.880 | 65.346 | 77.227 | ok | E310/R20 |
| 65536 | 128 | 29.913 | 109.895 | 139.785 | ok | E311/R20 |

### triton / bf16

| S | D | F | B | F+B | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 16 | 0.008 | 0.162 | 0.247 | ok | E312/R20 |
| 128 | 32 | 0.008 | 0.165 | 0.243 | ok | E313/R20 |
| 128 | 64 | 0.008 | 0.152 | 0.214 | ok | E314/R20 |
| 128 | 128 | 0.010 | 0.152 | 0.234 | ok | E315/R20 |
| 256 | 16 | 0.008 | 0.167 | 0.209 | ok | E316/R20 |
| 256 | 32 | 0.009 | 0.158 | 0.206 | ok | E317/R20 |
| 256 | 64 | 0.010 | 0.140 | 0.213 | ok | E318/R20 |
| 256 | 128 | 0.011 | 0.137 | 0.212 | ok | E319/R20 |
| 512 | 16 | 0.010 | 0.155 | 0.191 | ok | E320/R20 |
| 512 | 32 | 0.012 | 0.143 | 0.212 | ok | E321/R20 |
| 512 | 64 | 0.013 | 0.170 | 0.208 | ok | E322/R20 |
| 512 | 128 | 0.016 | 0.142 | 0.189 | ok | E323/R20 |
| 1024 | 16 | 0.016 | 0.166 | 0.224 | ok | E324/R20 |
| 1024 | 32 | 0.016 | 0.157 | 0.228 | ok | E325/R20 |
| 1024 | 64 | 0.019 | 0.144 | 0.192 | ok | E326/R20 |
| 1024 | 128 | 0.023 | 0.143 | 0.206 | ok | E327/R20 |
| 2048 | 16 | 0.025 | 0.163 | 0.220 | ok | E328/R20 |
| 2048 | 32 | 0.027 | 0.179 | 0.230 | ok | E329/R20 |
| 2048 | 64 | 0.031 | 0.186 | 0.224 | ok | E330/R20 |
| 2048 | 128 | 0.039 | 0.215 | 0.252 | ok | E331/R20 |
| 4096 | 16 | 0.044 | 0.372 | 0.414 | ok | E332/R20 |
| 4096 | 32 | 0.047 | 0.407 | 0.451 | ok | E333/R20 |
| 4096 | 64 | 0.057 | 0.455 | 0.511 | ok | E334/R20 |
| 4096 | 128 | 0.073 | 0.605 | 0.676 | ok | E335/R20 |
| 8192 | 16 | 0.094 | 0.946 | 1.036 | ok | E336/R20 |
| 8192 | 32 | 0.109 | 1.058 | 1.164 | ok | E337/R20 |
| 8192 | 64 | 0.128 | 1.317 | 1.442 | ok | E338/R20 |
| 8192 | 128 | 0.170 | 1.888 | 2.054 | ok | E339/R20 |
| 16384 | 16 | 0.263 | 2.944 | 3.205 | ok | E340/R20 |
| 16384 | 32 | 0.313 | 3.274 | 3.585 | ok | E341/R20 |
| 16384 | 64 | 0.354 | 4.326 | 4.678 | ok | E342/R20 |
| 16384 | 128 | 0.479 | 6.978 | 7.448 | ok | E343/R20 |
| 32768 | 16 | 0.711 | 11.978 | 12.687 | ok | E344/R20 |
| 32768 | 32 | 0.859 | 12.691 | 13.548 | ok | E345/R20 |
| 32768 | 64 | 0.981 | 16.795 | 17.771 | ok | E346/R20 |
| 32768 | 128 | 1.572 | 27.380 | 28.947 | ok | E347/R20 |
| 65536 | 16 | 2.485 | 43.555 | 46.045 | ok | E348/R20 |
| 65536 | 32 | 3.034 | 46.727 | 49.769 | ok | E349/R20 |
| 65536 | 64 | 3.459 | 65.337 | 68.783 | ok | E350/R20 |
| 65536 | 128 | 5.673 | 109.885 | 115.536 | ok | E351/R20 |

### triton-full / fp32

| S | D | F | B | F+B | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 16 | 0.008 | 0.063 | 0.092 | ok | E360/R22 |
| 128 | 32 | 0.010 | 0.059 | 0.097 | ok | E361/R22 |
| 128 | 64 | 0.011 | 0.053 | 0.069 | ok | E362/R22 |
| 128 | 128 | 0.016 | 0.064 | 0.074 | ok | E363/R22 |
| 256 | 16 | 0.012 | 0.062 | 0.086 | ok | E364/R22 |
| 256 | 32 | 0.014 | 0.058 | 0.094 | ok | E365/R22 |
| 256 | 64 | 0.016 | 0.066 | 0.095 | ok | E366/R22 |
| 256 | 128 | 0.025 | 0.107 | 0.124 | ok | E367/R22 |
| 512 | 16 | 0.017 | 0.061 | 0.071 | ok | E368/R22 |
| 512 | 32 | 0.023 | 0.080 | 0.099 | ok | E369/R22 |
| 512 | 64 | 0.025 | 0.080 | 0.101 | ok | E370/R22 |
| 512 | 128 | 0.039 | 0.193 | 0.228 | ok | E371/R22 |
| 1024 | 16 | 0.027 | 0.103 | 0.124 | ok | E372/R22 |
| 1024 | 32 | 0.037 | 0.145 | 0.179 | ok | E373/R22 |
| 1024 | 64 | 0.039 | 0.147 | 0.182 | ok | E374/R22 |
| 1024 | 128 | 0.072 | 0.368 | 0.431 | ok | E375/R22 |
| 2048 | 16 | 0.049 | 0.189 | 0.231 | ok | E376/R22 |
| 2048 | 32 | 0.066 | 0.273 | 0.334 | ok | E377/R22 |
| 2048 | 64 | 0.072 | 0.278 | 0.343 | ok | E378/R22 |
| 2048 | 128 | 0.134 | 0.714 | 0.838 | ok | E379/R22 |
| 4096 | 16 | 0.092 | 0.363 | 0.449 | ok | E380/R22 |
| 4096 | 32 | 0.127 | 0.583 | 0.708 | ok | E381/R22 |
| 4096 | 64 | 0.138 | 0.543 | 0.673 | ok | E382/R22 |
| 4096 | 128 | 0.262 | 1.419 | 1.670 | ok | E383/R22 |
| 8192 | 16 | 0.244 | 0.827 | 1.068 | ok | E384/R22 |
| 8192 | 32 | 0.353 | 1.308 | 1.657 | ok | E385/R22 |
| 8192 | 64 | 0.345 | 1.320 | 1.660 | ok | E386/R22 |
| 8192 | 128 | 0.730 | 3.461 | 4.177 | ok | E387/R22 |
| 16384 | 16 | 0.666 | 2.003 | 2.664 | ok | E388/R22 |
| 16384 | 32 | 1.000 | 3.383 | 4.357 | ok | E389/R22 |
| 16384 | 64 | 0.975 | 3.245 | 4.207 | ok | E390/R22 |
| 16384 | 128 | 2.299 | 8.850 | 11.139 | ok | E391/R22 |
| 32768 | 16 | 2.231 | 7.139 | 9.363 | ok | E392/R22 |
| 32768 | 32 | 3.415 | 11.942 | 15.368 | ok | E393/R22 |
| 32768 | 64 | 3.252 | 11.934 | 15.159 | ok | E394/R22 |
| 32768 | 128 | 8.017 | 32.808 | 40.807 | ok | E395/R22 |
| 65536 | 16 | 8.160 | 27.135 | 35.185 | ok | E396/R22 |
| 65536 | 32 | 12.629 | 44.186 | 56.802 | ok | E397/R22 |
| 65536 | 64 | 11.891 | 45.391 | 57.295 | ok | E398/R22 |
| 65536 | 128 | 29.929 | 125.687 | 155.848 | ok | E399/R22 |

### triton-full / bf16

| S | D | F | B | F+B | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 16 | 0.008 | 0.038 | 0.063 | ok | E400/R22 |
| 128 | 32 | 0.008 | 0.059 | 0.080 | ok | E401/R22 |
| 128 | 64 | 0.008 | 0.048 | 0.083 | ok | E402/R22 |
| 128 | 128 | 0.009 | 0.039 | 0.065 | ok | E403/R22 |
| 256 | 16 | 0.008 | 0.042 | 0.058 | ok | E404/R22 |
| 256 | 32 | 0.009 | 0.057 | 0.082 | ok | E405/R22 |
| 256 | 64 | 0.010 | 0.055 | 0.089 | ok | E406/R22 |
| 256 | 128 | 0.011 | 0.053 | 0.062 | ok | E407/R22 |
| 512 | 16 | 0.010 | 0.059 | 0.090 | ok | E408/R22 |
| 512 | 32 | 0.012 | 0.042 | 0.083 | ok | E409/R22 |
| 512 | 64 | 0.013 | 0.063 | 0.094 | ok | E410/R22 |
| 512 | 128 | 0.015 | 0.053 | 0.076 | ok | E411/R22 |
| 1024 | 16 | 0.016 | 0.054 | 0.080 | ok | E412/R22 |
| 1024 | 32 | 0.016 | 0.051 | 0.069 | ok | E413/R22 |
| 1024 | 64 | 0.018 | 0.048 | 0.086 | ok | E414/R22 |
| 1024 | 128 | 0.023 | 0.054 | 0.082 | ok | E415/R22 |
| 2048 | 16 | 0.025 | 0.056 | 0.084 | ok | E416/R22 |
| 2048 | 32 | 0.027 | 0.067 | 0.094 | ok | E417/R22 |
| 2048 | 64 | 0.031 | 0.064 | 0.095 | ok | E418/R22 |
| 2048 | 128 | 0.039 | 0.093 | 0.128 | ok | E419/R22 |
| 4096 | 16 | 0.044 | 0.097 | 0.138 | ok | E420/R22 |
| 4096 | 32 | 0.047 | 0.121 | 0.166 | ok | E421/R22 |
| 4096 | 64 | 0.057 | 0.114 | 0.167 | ok | E422/R22 |
| 4096 | 128 | 0.073 | 0.173 | 0.242 | ok | E423/R22 |
| 8192 | 16 | 0.094 | 0.218 | 0.309 | ok | E424/R22 |
| 8192 | 32 | 0.109 | 0.292 | 0.399 | ok | E425/R22 |
| 8192 | 64 | 0.129 | 0.274 | 0.400 | ok | E426/R22 |
| 8192 | 128 | 0.170 | 0.437 | 0.602 | ok | E427/R22 |
| 16384 | 16 | 0.264 | 0.616 | 0.876 | ok | E428/R22 |
| 16384 | 32 | 0.313 | 0.830 | 1.140 | ok | E429/R22 |
| 16384 | 64 | 0.355 | 0.845 | 1.195 | ok | E430/R22 |
| 16384 | 128 | 0.476 | 1.119 | 1.590 | ok | E431/R22 |
| 32768 | 16 | 0.712 | 1.788 | 2.496 | ok | E432/R22 |
| 32768 | 32 | 0.861 | 2.307 | 3.163 | ok | E433/R22 |
| 32768 | 64 | 0.983 | 2.493 | 3.469 | ok | E434/R22 |
| 32768 | 128 | 1.574 | 4.057 | 5.626 | ok | E435/R22 |
| 65536 | 16 | 2.488 | 6.217 | 8.702 | ok | E436/R22 |
| 65536 | 32 | 3.037 | 8.320 | 11.350 | ok | E437/R22 |
| 65536 | 64 | 3.462 | 8.768 | 12.225 | ok | E438/R22 |
| 65536 | 128 | 5.679 | 15.048 | 20.724 | ok | E439/R22 |

## 7. XL 显存快照实验

预热 2、计时 2，再额外执行一个记录步骤；峰值是该额外步骤 reset 后的 peak allocated，reserved 是缓存保留量。OOM 不填伪造峰值；远端生成不等于本地持有快照。

| S/精度 | 范围 | 耗时 | 峰值 GiB | 保留峰值 GiB | 状态 | 本地快照 | 证据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128/fp32 | F | 90.350 ± 1.102 | 18.053 | 18.195 | ok | 有 | E136/R09 |
| 128/fp32 | F+CE+B+AdamW | 345.924 ± 0.541 | 51.415 | 57.295 | ok | 有 | E137/R09 |
| 128/bf16 | F | 41.729 ± 0.824 | 22.440 | 22.662 | ok | 有 | E138/R09 |
| 128/bf16 | F+CE+B+AdamW | 201.130 ± 0.561 | 51.405 | 58.270 | ok | 有 | E139/R09 |
| 2048/fp32 | F | — | — | — | oom | 无 | E140/R09 |
| 2048/fp32 | F+CE+B+AdamW | — | — | — | oom | 无 | E141/R09 |
| 2048/bf16 | F | 355.094 ± 0.394 | 163.664 | 164.693 | ok | 有 | E142/R09 |
| 2048/bf16 | F+CE+B+AdamW | — | — | — | oom | 无 | E143/R09 |

## 8. Activation checkpoint

XL、B=4、S=2048、FP32、32 层；F+CE+B 不含 AdamW，预热 2、计时 3。64 段表示每层 attention/FFN 分开重算。

| 段数 | 每段 | 耗时 | 峰值 GiB | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- |
| 1 | 32 层 | — | — | oom | E144/R09 |
| 2 | 16 层 | 5151.431 ± 0.454 | 127.624 | ok | E145/R09 |
| 4 | 8 层 | 5136.645 ± 0.690 | 79.946 | ok | E146/R09 |
| 8 | 4 层 | 5106.333 ± 0.452 | 56.101 | ok | E147/R09 |
| 16 | 2 层 | 5046.192 ± 0.361 | 44.177 | ok | E148/R09 |
| 32 | 1 层 | 4927.029 ± 0.966 | 38.216 | ok | E149/R09 |
| 64 | 半层 | 4864.743 ± 0.065 | 38.217 | ok | E150/R09 |

## 9. 单 block 保存张量归因

XL block、B=4、S=2048、D=2560；按 storage 去重并排除参数/缓冲区。这是 autograd 保存存储，不是 kernel 时间。

证据 E161/R12；全部 records 留存于 JSON。

| 模块 | 保存来源 | bytes | GiB | 占比 |
| --- | --- | --- | --- | --- |
| attn | ViewBackward0 | 2231369728 | 2.078 | 32.72% |
| attn | ExpBackward0 | 2147483648 | 2.000 | 31.49% |
| ffn | ViewBackward0 | 671088640 | 0.625 | 9.84% |
| ffn | SigmoidBackward0 | 335544320 | 0.312 | 4.92% |
| ffn | MulBackward0 | 335544320 | 0.312 | 4.92% |
| ffn.w2 | ViewBackward0 | 335544320 | 0.312 | 4.92% |
| ln1 | leaf | 83886080 | 0.078 | 1.23% |
| ln1 | MulBackward0 | 83886080 | 0.078 | 1.23% |
| attn.q_proj | ViewBackward0 | 83886080 | 0.078 | 1.23% |
| attn | ReshapeAliasBackward0 | 83886080 | 0.078 | 1.23% |
| attn | UnsafeViewBackward0 | 83886080 | 0.078 | 1.23% |
| attn.output_proj | ViewBackward0 | 83886080 | 0.078 | 1.23% |
| ln2 | AddBackward0 | 83886080 | 0.078 | 1.23% |
| ln2 | MulBackward0 | 83886080 | 0.078 | 1.23% |
| ffn.w1 | ViewBackward0 | 83886080 | 0.078 | 1.23% |
| attn | leaf | 6291456 | 0.006 | 0.09% |
| attn | SumBackward1 | 1048576 | 0.001 | 0.02% |
| ln1 | RsqrtBackward0 | 32768 | 0.000 | 0.00% |
| ln2 | RsqrtBackward0 | 32768 | 0.000 | 0.00% |

| 原始统计字段 | 值 |
| --- | --- |
| unique_saved_bytes | 6818955264 |
| before_forward_bytes | 508186624 |
| after_forward_bytes | 7335661568 |
| after_backward_bytes | 1112449024 |
| peak_bytes | 14345501696 |
| parameter_gradient_bytes | 419450880 |
| input_gradient_bytes | 83886080 |

## 10. All-reduce：24 格

每配置预热 5、计时 20；每迭代先取全部 rank 最大值，再求均值/SD。barrier 和 tensor.fill 不计时。Gloo 用 CPU 张量；NCCL 用 GPU 张量。总线带宽=2(N-1)/N×数据量/耗时，不是实测 NVLink 峰值。

| rank | 后端 | MB | 耗时 | 总线 GB/s | n | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | gloo | 1 | 3.644 ± 0.720 | 0.274 | 20 | E123/R08 |
| 2 | gloo | 10 | 17.449 ± 2.576 | 0.573 | 20 | E124/R08 |
| 2 | gloo | 100 | 160.730 ± 29.573 | 0.622 | 20 | E125/R08 |
| 2 | gloo | 1000 | 1472.320 ± 132.222 | 0.679 | 20 | E126/R08 |
| 2 | nccl | 1 | 0.085 ± 0.063 | 11.733 | 20 | E127/R08 |
| 2 | nccl | 10 | 0.108 ± 0.052 | 92.767 | 20 | E128/R08 |
| 2 | nccl | 100 | 0.310 ± 0.076 | 322.111 | 20 | E129/R08 |
| 2 | nccl | 1000 | 1.868 ± 0.039 | 535.217 | 20 | E130/R08 |
| 4 | gloo | 1 | 12.131 ± 1.901 | 0.124 | 20 | E247/R16 |
| 4 | gloo | 10 | 28.278 ± 3.837 | 0.530 | 20 | E248/R16 |
| 4 | gloo | 100 | 195.217 ± 13.676 | 0.768 | 20 | E249/R16 |
| 4 | gloo | 1000 | 1730.584 ± 170.627 | 0.867 | 20 | E250/R16 |
| 4 | nccl | 1 | 0.091 ± 0.071 | 16.531 | 20 | E251/R16 |
| 4 | nccl | 10 | 0.119 ± 0.040 | 125.961 | 20 | E252/R16 |
| 4 | nccl | 100 | 0.347 ± 0.095 | 432.735 | 20 | E253/R16 |
| 4 | nccl | 1000 | 2.489 ± 0.034 | 602.631 | 20 | E254/R16 |
| 6 | gloo | 1 | 14.841 ± 2.519 | 0.112 | 20 | E352/R21 |
| 6 | gloo | 10 | 24.461 ± 3.673 | 0.681 | 20 | E353/R21 |
| 6 | gloo | 100 | 195.656 ± 55.376 | 0.852 | 20 | E354/R21 |
| 6 | gloo | 1000 | 1764.047 ± 74.559 | 0.945 | 20 | E355/R21 |
| 6 | nccl | 1 | 0.075 ± 0.030 | 22.100 | 20 | E356/R21 |
| 6 | nccl | 10 | 0.129 ± 0.024 | 129.106 | 20 | E357/R21 |
| 6 | nccl | 100 | 0.359 ± 0.050 | 464.039 | 20 | E358/R21 |
| 6 | nccl | 1000 | 2.656 ± 0.023 | 627.561 | 20 | E359/R21 |

## 11. 双 B200 XL 分布式

全局 B=4（每 rank 2）、S=512、FP32，预热 5、计时 10。完整步含 F/CE/B/同步/AdamW（以及 owner broadcast）。每迭代取最大 rank。暴露尾部等待不是总通信耗时，也不能直接推算重叠率。

| 策略 | 优化器 | 完整步 | 尾部等待 | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- |
| naive | adamw | 590.083 ± 1.526 | 40.194 ± 0.940 | ok | E131/R08 |
| flat | adamw | 589.047 ± 0.875 | 39.317 ± 0.576 | ok | E132/R08 |
| overlap | adamw | 568.190 ± 1.483 | 1.816 ± 0.317 | ok | E133/R08 |
| overlap | sharded | 557.964 ± 5.163 | 1.839 ± 0.795 | ok | E134/R08 |
| fsdp | adamw | 545.001 ± 1.768 | 0.430 ± 0.072 | ok | E135/R08 |

| 策略 | 优化器 | 初始化 GiB | 更新前 GiB | 更新后 GiB | 步峰值 GiB | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| naive | adamw | 12.817 | 50.947 | 50.947 | 52.089 | E131/R08 |
| flat | adamw | 12.817 | 50.946 | 50.946 | 63.638 | E132/R08 |
| overlap | adamw | 12.817 | 50.947 | 50.947 | 52.089 | E133/R08 |
| overlap | sharded | 12.817 | 38.255 | 38.255 | 39.397 | E134/R08 |
| fsdp | adamw | 6.408 | 25.564 | 25.565 | 33.068 | E135/R08 |

## 12. Nsight 与上下文容量探测

Nsight wall-clock 有采集开销，不替代普通基准。trace_created 只说明 SQLite 生成，不保证有 GPU kernel。早期 HES 采集缺 kernel；取回后已确认最后 small/S256 cuda-sw 探针有效。只认这一个 case，不把其他尚无有效 kernel 证据的 profile 一并算完成。

| case | wall-clock | 结果 | trace_created | 本地 SQLite | 有效 kernels | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| nsys-small-s256 | 74.163 ± 0.000 | ok | — | 无 | 未验证 | E006/R03 |
| nsys-small-s256 | — | failed | False | 无 | 未验证 | E152/R10 |
| fit-small-4096 | 758.724 ± 0.000 | ok | — | 无 | 未验证 | E153/R10 |
| fit-small-8192 | — | oom | — | 无 | 未验证 | E154/R10 |
| fit-small-16384 | — | oom | — | 无 | 未验证 | E155/R10 |
| fit-xl-1024 | 1782.476 ± 0.000 | ok | — | 无 | 未验证 | E156/R10 |
| fit-xl-2048 | — | oom | — | 无 | 未验证 | E157/R10 |
| fit-xl-4096 | — | oom | — | 无 | 未验证 | E158/R10 |
| nsys-small-s256 | 105.862 ± 0.000 | ok | True | 无 | 未验证 | E162/R13 |
| profile-small-s256-plain | 36.912 ± 0.175 | ok | — | 无 | 未验证 | E256/R18 |
| profile-small-s256-nsys | 56.085 ± 0.000 | ok | True | 无 | 未验证 | E257/R18 |
| profile-small-s1024-plain | 114.085 ± 0.192 | ok | — | 无 | 未验证 | E258/R18 |
| profile-small-s1024-nsys | 115.159 ± 0.000 | ok | True | 无 | 未验证 | E259/R18 |
| profile-small-s4096-plain | 764.620 ± 1.119 | ok | — | 无 | 未验证 | E260/R18 |
| profile-small-s4096-nsys | 765.564 ± 0.000 | ok | True | 无 | 未验证 | E261/R18 |
| profile-xl-s256-plain | 532.849 ± 0.463 | ok | — | 无 | 未验证 | E262/R18 |
| profile-xl-s256-nsys | 534.803 ± 0.000 | ok | True | 无 | 未验证 | E263/R18 |
| profile-xl-s512-plain | 937.970 ± 0.587 | ok | — | 无 | 未验证 | E264/R18 |
| profile-xl-s512-nsys | 939.794 ± 0.000 | ok | True | 无 | 未验证 | E265/R18 |
| profile-xl-s1024-plain | 1788.954 ± 0.131 | ok | — | 无 | 未验证 | E266/R18 |
| profile-xl-s1024-nsys | 1789.437 ± 0.000 | ok | True | 无 | 未验证 | E267/R18 |
| profile-saved-block-xl-s2048 | — | ok | True | 有 | 未验证 | E268/R18 |
| profile-xl-naive | 5392.876 ± 0.000 | ok | True | 无 | 未验证 | E269/R19 |
| profile-xl-overlap | 5001.367 ± 0.000 | ok | True | 无 | 未验证 | E270/R19 |
| profile-xl-fsdp | 4998.872 ± 0.000 | ok | True | 无 | 未验证 | E271/R19 |
| nsys-small-s256 | 107.144 ± 0.000 | ok | True | 有 | 3637 | E446/R26 |

### 有效 kernel 统计：nsys-small-s256（E446/R26）

按 (globalPid, correlationId) 关联 CUDA launch；顶层 phase 跨 autograd worker 线程，attention 细分 range 只关联同线程。串行训练步假设下，无未匹配/歧义 launch，measurement 内 phase 无漏分/重复。GPU 时间是 kernel 时长之和，不是 wall-clock，也不是跨 range 首尾 span。矩阵乘法分类依据名称含 gemm/gemv，是启发式而非 FLOPs 测量。

| 统计 | 实测值 |
| --- | --- |
| kernel 总数 | 3637 |
| kernel 时间总和 ms | 29.847779 |
| 未匹配 launch | 0 |
| 歧义 launch | 0 |
| measurement_kernel_count | 3637 |
| measurement_unassigned_count | 0 |
| multiply_assigned_phase_kernel_count | 0 |

| range | kernels | CPU range ms | GPU kernel ms | matmul ms | matmul 比例 |
| --- | --- | --- | --- | --- | --- |
| attention_mask | 24 | 0.796 | 0.126333 | 0.000000 | 0.000% |
| attention_scores_matmul | 24 | 1.321 | 0.253024 | 0.204896 | 80.979% |
| attention_softmax | 60 | 1.485 | 0.466300 | 0.000000 | 0.000% |
| attention_values_matmul | 24 | 1.035 | 0.235614 | 0.179616 | 76.233% |
| backward | 1227 | 50.521 | 16.380591 | 11.941036 | 72.897% |
| forward | 632 | 26.734 | 8.713439 | 6.705967 | 76.961% |
| gradient_sync_wait | 0 | 0.000 | 0.000000 | 0.000000 | — |
| loss | 2 | 0.090 | 0.039103 | 0.000000 | 0.000% |
| measurement | 3637 | 107.347 | 29.847779 | 18.647003 | 62.474% |
| optimizer | 1776 | 27.806 | 4.714646 | 0.000000 | 0.000% |
| forward_loss_backward | 1861 | — | 25.133133 | 18.647003 | 74.193% |

| 耗时前 10 kernel（完整名称见 JSON） | 调用数 | GPU ms | 占总 kernel 时间 |
| --- | --- | --- | --- |
| cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_nnn_align1_bias_f32_relu | 72 | 4.670557 | 15.648% |
| cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_tnn_align1_bias_f32_relu | 24 | 2.744523 | 9.195% |
| cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_ntn_align1_bias_f32_relu | 24 | 2.051347 | 6.873% |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>,… | 716 | 2.041070 | 6.838% |
| cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x128x16_1x1x1_3_tnn_align1_bias_f32_relu | 49 | 2.032528 | 6.810% |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3) | 673 | 1.746256 | 5.851% |
| cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_nnn_align1_bias_f32_relu | 36 | 1.709493 | 5.727% |
| cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_tnn_align1_bias_f32_relu | 12 | 1.544404 | 5.174% |
| cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_ntn_align1_bias_f32_relu | 48 | 1.412118 | 4.731% |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBa… | 181 | 1.081076 | 3.622% |

分析来源：output/profiling/small-s256/nsys_summary.json。此记录不能替代其他 5 个上下文/模型配置、双卡通信重叠、FSDP 预取或逐配置内存截图。

## 13. 正确性与历史测试

通过仅覆盖当时源码与测试容差，最终 GPU 源码重验另列待跑。原课程 distributed tests 硬编码 Gloo；在 GPU 机器选择 CUDA tensor 也不会自动换成 NCCL。历史失败保留；测试次数不与不同配置个数混淆。

| GPU 验证 | 通过 | 跳过 | 失败 | 异常 | 结果 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| attention-tests | 6 | 0 | 0 | 0 | ok | E001/R01 |
| extended-attention | 0 配置检查 | — | — | — | failed | E002/R01 |
| attention-tests | 6 | 0 | 0 | 0 | ok | E003/R02 |
| extended-attention | 32 配置检查 | — | — | — | ok | E004/R02 |
| distributed-repeat-0 | 8 | 0 | 0 | 0 | ok | E037/R05 |

| 本地 XML（相对 runs/local） | 通过 | 跳过 | 失败 | 异常 | 秒 |
| --- | --- | --- | --- | --- | --- |
| extended-tests.xml | 2 | 0 | 0 | 0 | 0.757 |
| final-gloo-stability/repeat-1.xml | 0 | 0 | 8 | 0 | 9.559 |
| final-gloo-stability-localhost/repeat-1.xml | 8 | 0 | 0 | 0 | 12.561 |
| final-gloo-stability-localhost/repeat-2.xml | 8 | 0 | 0 | 0 | 12.560 |
| final-gloo-stability-localhost/repeat-3.xml | 8 | 0 | 0 | 0 | 15.816 |
| final-gloo-stability-localhost/repeat-4.xml | 8 | 0 | 0 | 0 | 12.060 |
| final-gloo-stability-localhost/repeat-5.xml | 8 | 0 | 0 | 0 | 12.298 |
| fsdp-prefetch-optimizer-load.xml | 10 | 4 | 0 | 0 | 11.558 |
| initial/cs336-assignment2-first-tests.xml | 2 | 4 | 8 | 0 | 10.438 |
| initial/cs336-assignment2-unrestricted-tests.xml | 10 | 4 | 0 | 0 | 13.737 |
| report-final-cpu-tests.xml | 14 | 4 | 0 | 0 | 12.856 |

### 扩展 attention：32 配置 / 128 张量比较（E004/R02）

2 精度 × 4 形状 × 2 causal × 2 实现 = 32 配置；每配置比较 O、dQ、dK、dV。FP32 atol=1e-3；BF16 atol=0.03；两者 rtol=0.03。不能把此扩展测试的宽容差冒充课程全部官方容差。

| 实现 | dtype | N/D/causal | O 最大误差 | dQ 最大误差 | dK 最大误差 | dV 最大误差 |
| --- | --- | --- | --- | --- | --- | --- |
| Triton | float32 | 32/16/False | 3.5762787e-07 | 3.8743019e-07 | 3.5762787e-07 | 3.5762787e-07 |
| TritonFull | float32 | 32/16/False | 3.5762787e-07 | 5.9604645e-07 | 1.1920929e-06 | 3.5762787e-07 |
| Triton | float32 | 32/16/True | 3.5762787e-07 | 4.7683716e-07 | 6.5565109e-07 | 4.7683716e-07 |
| TritonFull | float32 | 32/16/True | 3.5762787e-07 | 5.9604645e-07 | 8.3446503e-07 | 1.1920929e-06 |
| Triton | float32 | 128/64/False | 8.9406967e-07 | 4.7683716e-07 | 2.9802322e-07 | 3.5762787e-07 |
| TritonFull | float32 | 128/64/False | 8.9406967e-07 | 8.9406967e-07 | 7.7486038e-07 | 8.3446503e-07 |
| Triton | float32 | 128/64/True | 9.5367432e-07 | 1.4433367e-06 | 1.6689301e-06 | 1.9073486e-06 |
| TritonFull | float32 | 128/64/True | 9.5367432e-07 | 1.4433364e-06 | 1.9073486e-06 | 1.9073486e-06 |
| Triton | float32 | 256/128/False | 7.4505806e-07 | 5.9604645e-07 | 7.4505806e-07 | 3.8743019e-07 |
| TritonFull | float32 | 256/128/False | 7.4505806e-07 | 1.4901161e-06 | 1.6093254e-06 | 7.1525574e-07 |
| Triton | float32 | 256/128/True | 1.3709068e-06 | 2.1457672e-06 | 3.3378601e-06 | 1.6689301e-06 |
| TritonFull | float32 | 256/128/True | 1.3709068e-06 | 3.0617787e-06 | 3.3378601e-06 | 3.8146973e-06 |
| Triton | float32 | 63/32/False | 5.364418e-07 | 4.7683716e-07 | 7.1525574e-07 | 7.7486038e-07 |
| TritonFull | float32 | 63/32/False | 5.364418e-07 | 8.3446503e-07 | 9.5367432e-07 | 7.1525574e-07 |
| Triton | float32 | 63/32/True | 5.9604645e-07 | 7.8076158e-07 | 9.983778e-07 | 9.5367432e-07 |
| TritonFull | float32 | 63/32/True | 5.9604645e-07 | 7.1525574e-07 | 7.7486038e-07 | 1.1920929e-06 |
| Triton | bfloat16 | 32/16/False | 0.0035938025 | 0.0027356148 | 0.0056738853 | 0.0029381514 |
| TritonFull | bfloat16 | 32/16/False | 0.0035938025 | 0.0052903891 | 0.0099511147 | 0.0031245947 |
| Triton | bfloat16 | 32/16/True | 0.0089302063 | 0.0093176588 | 0.0082417428 | 0.0073385239 |
| TritonFull | bfloat16 | 32/16/True | 0.0089302063 | 0.0094397292 | 0.0091793984 | 0.0082864761 |
| Triton | bfloat16 | 128/64/False | 0.0020393133 | 0.0021257997 | 0.0020567775 | 0.0018876195 |
| TritonFull | bfloat16 | 128/64/False | 0.0020393133 | 0.0027580261 | 0.0026833415 | 0.0034151077 |
| Triton | bfloat16 | 128/64/True | 0.0074849129 | 0.0081107318 | 0.0093326569 | 0.0068154335 |
| TritonFull | bfloat16 | 128/64/True | 0.0074849129 | 0.0081107318 | 0.0093326569 | 0.0089478493 |
| Triton | bfloat16 | 256/128/False | 0.001439184 | 0.0021341443 | 0.0019589067 | 0.0017980337 |
| TritonFull | bfloat16 | 256/128/False | 0.001439184 | 0.004137218 | 0.0058609247 | 0.0023134947 |
| Triton | bfloat16 | 256/128/True | 0.0065236092 | 0.0088639855 | 0.0084702969 | 0.014790535 |
| TritonFull | bfloat16 | 256/128/True | 0.0065236092 | 0.010335922 | 0.0091814995 | 0.014790535 |
| Triton | bfloat16 | 63/32/False | 0.0024687052 | 0.004160881 | 0.0040476322 | 0.0029249191 |
| TritonFull | bfloat16 | 63/32/False | 0.0024687052 | 0.004160881 | 0.0046956539 | 0.0037475824 |
| Triton | bfloat16 | 63/32/True | 0.0070888996 | 0.0057815015 | 0.0057191849 | 0.0076982975 |
| TritonFull | bfloat16 | 63/32/True | 0.0070888996 | 0.0077683926 | 0.0082210302 | 0.01135397 |

### optimized-validation（E159/R11，failed）

此历史尝试在断言处失败，没有 measurements 汇总；异常完整保留在原始 JSON，末行见历史异常表。不能将未生成的数据补成 0。

### optimized-validation（E160/R12，ok）

| 检查项（最大绝对误差，除非另标） | 值 |
| --- | --- |
| model_max_output_error | 0.01953125 |
| loss_reference | 28.68728065 |
| loss_chunked | 28.68152618 |
| chunked_dx_max_error | 0.00439453125 |
| chunked_dw_max_error | 0.003253936768 |
| fp32_loss_error | 1.907348633e-06 |
| fp32_dx_error | 4.433095455e-07 |
| fp32_dw_error | 3.278255463e-07 |
| adamw_10_step_max_error | 2.384185791e-07 |
| gradient/token_embeddings.weight | 2.760946518e-05 |
| gradient/layers.0.attn.q_proj.weight | 7.629394531e-05 |
| gradient/layers.0.attn.k_proj.weight | 7.629394531e-05 |
| gradient/layers.0.attn.v_proj.weight | 0.0001220703125 |
| gradient/layers.0.attn.output_proj.weight | 9.155273438e-05 |
| gradient/layers.0.ffn.w1.weight | 0.0001068115234 |
| gradient/layers.0.ffn.w2.weight | 9.155273438e-05 |
| gradient/layers.0.ffn.w3.weight | 9.155273438e-05 |
| gradient/layers.0.ln1.weight | 7.982668467e-05 |
| gradient/layers.0.ln2.weight | 0.000108756125 |
| gradient/layers.1.attn.q_proj.weight | 6.103515625e-05 |
| gradient/layers.1.attn.k_proj.weight | 6.866455078e-05 |
| gradient/layers.1.attn.v_proj.weight | 7.629394531e-05 |
| gradient/layers.1.attn.output_proj.weight | 7.629394531e-05 |
| gradient/layers.1.ffn.w1.weight | 9.155273438e-05 |
| gradient/layers.1.ffn.w2.weight | 9.155273438e-05 |
| gradient/layers.1.ffn.w3.weight | 9.155273438e-05 |
| gradient/layers.1.ln1.weight | 8.269166574e-05 |
| gradient/layers.1.ln2.weight | 6.094807759e-05 |
| gradient/ln_final.weight | 7.891654968e-05 |
| gradient/lm_head.weight | 0.000244140625 |

### tuned-validation（E440/R23，ok）

| 检查项（最大绝对误差，除非另标） | 值 |
| --- | --- |
| model_max_output_error | 0.01953125 |
| loss_reference | 28.68728065 |
| loss_chunked | 28.68152618 |
| chunked_dx_max_error | 0.00439453125 |
| chunked_dw_max_error | 0.003253936768 |
| fp32_loss_error | 1.907348633e-06 |
| fp32_dx_error | 4.433095455e-07 |
| fp32_dw_error | 3.278255463e-07 |
| adamw_10_step_max_error | 2.384185791e-07 |
| gradient/token_embeddings.weight | 2.760946518e-05 |
| gradient/layers.0.attn.q_proj.weight | 7.629394531e-05 |
| gradient/layers.0.attn.k_proj.weight | 7.629394531e-05 |
| gradient/layers.0.attn.v_proj.weight | 0.0001220703125 |
| gradient/layers.0.attn.output_proj.weight | 9.155273438e-05 |
| gradient/layers.0.ffn.w1.weight | 0.0001068115234 |
| gradient/layers.0.ffn.w2.weight | 9.155273438e-05 |
| gradient/layers.0.ffn.w3.weight | 9.155273438e-05 |
| gradient/layers.0.ln1.weight | 7.982668467e-05 |
| gradient/layers.0.ln2.weight | 0.000108756125 |
| gradient/layers.1.attn.q_proj.weight | 6.103515625e-05 |
| gradient/layers.1.attn.k_proj.weight | 6.866455078e-05 |
| gradient/layers.1.attn.v_proj.weight | 7.629394531e-05 |
| gradient/layers.1.attn.output_proj.weight | 7.629394531e-05 |
| gradient/layers.1.ffn.w1.weight | 9.155273438e-05 |
| gradient/layers.1.ffn.w2.weight | 9.155273438e-05 |
| gradient/layers.1.ffn.w3.weight | 9.155273438e-05 |
| gradient/layers.1.ln1.weight | 8.269166574e-05 |
| gradient/layers.1.ln2.weight | 6.094807759e-05 |
| gradient/ln_final.weight | 7.891654968e-05 |
| gradient/lm_head.weight | 0.000244140625 |

## 14. 8B / 32K 完整训练步

34 层、D=4096、FFN=11008、32 heads、词表 151936、S=32768、全局 B=2；双 B200，BF16 compute+FP32 参数。每步包含 F/分块 CE/B/梯度 all-reduce/AdamW/owner broadcast。每次取两 rank 最大同步 wall-clock；新缓存路径验证冷启动。均值是同一 run 稳态样本，不是多次独立冷启动分布。

| 版本 | 预热 | n | 完整步 ms | rank 冷启动总秒 | 测量步峰值 GiB | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| first | 2 | 3 | 9702.418 ± 123.950 | 54.294 | 102.253 | E255/R17 |
| tuned | 2 | 5 | 6760.149 ± 29.686 | 65.781 | 102.253 | E445/R25 |

| 版本 | 实际计时样本 ms | 证据 |
| --- | --- | --- |
| first | 9831.027827, 9692.502016, 9583.723316 | E255/R17 |
| tuned | 6799.804107, 6778.172392, 6732.982269, 6759.455822, 6730.328561 | E445/R25 |

## 15. 样本与附件审计

逐个带 samples_ms 的对象重算均值、中位数、样本标准差并核对 n。do_bench 仅有聚合均值，不伪造原始样本或 SD。

| 审计项 | 结果 |
| --- | --- |
| 原始统计对象 | 635 |
| 不一致 | 0 |
| do_bench 配置 | 240 |
| memory_snapshots | 5 |
| nsight_sqlite | 2 |
| nsight_reports | 2 |
| verified_gpu_profile_cases | 1 |
| memoryviz_main_screenshots | 2 |

| 历史异常配置 | 结果状态 | 外层进程状态 | 异常末行 | 证据 |
| --- | --- | --- | --- | --- |
| extended-attention | failed | failed | AttributeError: 'NoneType' object has no attribute 'float' | E002/R01 |
| 10B-fp32-train-w5 | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 216.00 MiB. GPU 0 has a total capacity of 178.35 GiB of which 55.88 MiB is free. Process 1 has 178.28 GiB memory in us | E070/R06 |
| 10B-bf16-train-w5 | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 216.00 MiB. GPU 0 has a total capacity of 178.35 GiB of which 197.88 MiB is free. Process 1 has 178.14 GiB memory in u | E073/R06 |
| 10B-fp32-train-w0 | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 216.00 MiB. GPU 0 has a total capacity of 178.35 GiB of which 55.88 MiB is free. Process 1 has 178.28 GiB memory in us | E076/R06 |
| 10B-fp32-train-w1 | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 216.00 MiB. GPU 0 has a total capacity of 178.35 GiB of which 55.88 MiB is free. Process 1 has 178.28 GiB memory in us | E079/R06 |
| 10B-fp32-train-w2 | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 216.00 MiB. GPU 0 has a total capacity of 178.35 GiB of which 55.88 MiB is free. Process 1 has 178.28 GiB memory in us | E082/R06 |
| memory-xl-s2048-fp32-forward | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 178.35 GiB of which 15.88 MiB is free. Process 1 has 178.32 GiB memory in use. | E140/R09 |
| memory-xl-s2048-fp32-train | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 178.35 GiB of which 15.88 MiB is free. Process 1 has 178.32 GiB memory in use. | E141/R09 |
| memory-xl-s2048-bf16-train | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 178.35 GiB of which 847.88 MiB is free. Process 1 has 177.51 GiB memory in use | E143/R09 |
| checkpoint-1 | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 178.35 GiB of which 69.88 MiB is free. Process 1 has 178.27 GiB memory in use. | E144/R09 |
| saved-block-xl-s2048 | failed | failed | AttributeError: 'str' object has no attribute 'size' | E151/R09 |
| nsys-small-s256 | failed | failed | TypeError: annotate_attention.<locals>.attention() got an unexpected keyword argument 'K'. Did you mean 'k'? | E152/R10 |
| fit-small-8192 | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 GiB. GPU 0 has a total capacity of 178.35 GiB of which 8.37 GiB is free. Process 1 has 169.96 GiB memory in use. | E154/R10 |
| fit-small-16384 | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 GiB. GPU 0 has a total capacity of 178.35 GiB of which 31.56 GiB is free. Process 1 has 146.77 GiB memory in use | E155/R10 |
| fit-xl-2048 | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 178.35 GiB of which 2.01 GiB is free. Process 1 has 176.32 GiB memory in use.  | E157/R10 |
| fit-xl-4096 | oom | ok | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.00 GiB. GPU 0 has a total capacity of 178.35 GiB of which 3.49 GiB is free. Process 1 has 174.84 GiB memory in use.  | E158/R10 |
| optimized-validation | failed | failed | Relative difference: 0.00020063335501264182 (up to 0.0001 allowed) | E159/R11 |

OOM 由 worker 捕获后正常退出，因此外层 ok 与结果 oom 并不矛盾；归一化数据分别保留两种状态。

## 16. 运行登记与费用：不完整小计，不是账单

按 dispatch 的 GPU 数纠正早期默认单卡计价。沿用实验记录费率 B200 $0.001736/GPU/s，8 核 CPU+64 GiB RAM $0.00024688/s。仅 summary.seconds×费率，缺 summary、启动/导入/镜像、存储、CPU 附件任务均不在内。

| R | run_id | GPU | 已有/请求 case | 运行秒 | 估价 USD | summary |
| --- | --- | --- | --- | --- | --- | --- |
| R01 | 20260914T082016Z-probe | 1 | 2/2 | 39.791 | 0.07890 | 有 |
| R02 | 20260914T084725Z-probe-leaf-fix | 1 | 3/3 | 55.622 | 0.11029 | 有 |
| R03 | 20260914T084933Z-profile-probe-nsys | 1 | 1/1 | 12.667 | 0.02512 | 有 |
| R04 | 20260914T084942Z-models-small-medium | 1 | 30/30 | 174.187 | 0.34539 | 有 |
| R05 | 20260914T084950Z-distributed-tests | 2 | 1/1 | 51.139 | 0.19018 | 有 |
| R06 | 20260914T085158Z-models-large-xl-10b | 1 | 45/45 | 840.838 | 1.66728 | 有 |
| R07 | 20260914T085159Z-attention-matrix | 1 | 40/40 | 640.682 | 1.27040 | 有 |
| R08 | 20260914T085559Z-communication-and-xl | 2 | 13/13 | 264.120 | 0.98223 | 有 |
| R09 | 20260914T085600Z-memory-and-checkpoint | 1 | 16/16 | 278.066 | 0.55137 | 有 |
| R10 | 20260914T085757Z-nsys-probe-and-fit-nvtx-filter-fix | 1 | 7/7 | 51.748 | 0.10261 | 有 |
| R11 | 20260914T091103Z-optimized-validation | 1 | 1/1 | 11.531 | 0.02286 | 有 |
| R12 | 20260914T091830Z-optimized-validation-fp32-oracle | 1 | 2/2 | 16.373 | 0.03246 | 有 |
| R13 | 20260914T091834Z-profile-probe-keyword-fix | 1 | 1/1 | 39.327 | 0.07798 | 有 |
| R14 | 20260914T091848Z-flash-eager | 1 | 80/80 | 838.928 | 1.66349 | 有 |
| R15 | 20260914T091931Z-models-compile-small-medium | 1 | 4/4 | 286.723 | 0.56854 | 有 |
| R16 | 20260914T091932Z-communication | 4 | 8/8 | 158.358 | 1.13874 | 有 |
| R17 | 20260914T092247Z-leaderboard-first-full | 2 | 1/1 | 65.402 | 0.24322 | 有 |
| R18 | 20260914T092252Z-profiles | 1 | 13/13 | 281.522 | 0.55822 | 有 |
| R19 | 20260914T092406Z-distributed-profiles | 2 | 3/3 | 110.417 | 0.41063 | 有 |
| R20 | 20260914T092409Z-flash-triton-compiled-backward | 1 | 80/80 | 992.882 | 1.96877 | 有 |
| R21 | 20260914T092847Z-communication | 6 | 8/8 | 197.108 | 2.10174 | 有 |
| R22 | 20260914T092847Z-flash-triton-full | 1 | 80/80 | 680.668 | 1.34968 | 有 |
| R23 | 20260914T093124Z-tuned-validation | 1 | 1/1 | 20.518 | 0.04069 | 有 |
| R24 | 20260914T093129Z-models-compile-large-xl | 1 | 4/4 | 258.430 | 0.51244 | 有 |
| R25 | 20260914T093842Z-leaderboard-tuned | 2 | 1/1 | 82.873 | 0.30819 | 有 |
| R26 | 20260914T093845Z-profile-probe-cuda-software-trace | 1 | 1/1 | 36.275 | 0.07193 | 有 |
| R27 | 20260914T093858Z-models-compile-10b | 1 | 2/2 | 232.659 | 0.46134 | 有 |

已知 summary 的运行费率估价小计 $16.85470，覆盖 27/27 个已登记 GPU job。GPU job 的 summary 已齐；其他计费组成仍未计入，不能称总花费或账单。

完整来源路径、JSON pointer、配置、源码 SHA256、下载附件可用性见 evidence_index.json 和 run_registry.json。原始 runs 文件未修改。数值均可从 all_attempts.json 重算。


---

# 附录 C · 实验步骤与过程


## 记录范围与预算状态

本记录整理 2026-09-14 的既有执行过程，包含 Flash 完成记录、内存/Nsight 文件，以及最后恢复的 10B compile、Flash 两项和 tuned leaderboard 的结果/日志/源码环境归档。时间戳中的 `Z` 是 UTC，北京时间加 8 小时。本日志编写读取本地代码、原始 PDF、请求记录、日志和 JSON。主任务通过既有调用/Volume 的控制面 API 恢复结果，在本机修复分析器、运行 CPU 回归并生成截图，没有启动 GPU、云端 CPU、下载 RPC 或调参任务。

**预算暂停继续生效：没有用户新的明确预算授权，不启动任何 GPU 或 CPU 云端任务。** 包括“只读”的远程结果 RPC、云端 GUI 截图和容器式文件提取；它们虽然可能不使用 GPU，仍会启动计费计算。保留已有 Volume，不能擅自删除实验数据。历史检查中相关 Modal 应用已经 stopped、任务数为零，这只说明当时没有运行计算，不代表零存储费或账单结清。

剩余工作及逐项验收条件见 [GPU_REMAINING_TASKS.md](../GPU_REMAINING_TASKS.md)。本日志不是已完成作业的声明，也不是费用发票。

## 基线、原始要求与来源

- 原题是 `cs336_assignment2_systems.pdf`，Spring 2026，PDF 版本 26.1.3；starter/test 版本 26.1.4。请求记录保存的 PDF SHA256 是 `f5667e3b6fac23c3a6d4ad4a999445b96aab3fb2246904d13ec86b79320b1a84`。
- 模型基线使用本作业自带的 staff `cs336-basics/cs336_basics/model.py`，优化器使用其 `optimizer.py` 中的 AdamW。没有替换成用户 Assignment 1 实现，用户 Assignment 1 目录未在本任务中修改。
- 实验实现放在 `cs336_systems/`。FlashAttention、DDP、优化器分片和 FSDP 通过 `tests/adapters.py` 接入原始测试；额外回归在 `tests/test_systems_extended.py`。原始测试用例与新增回归分开记录。
- 题目第 3-4 页要求五种模型、5 次 warmup、10 次计时及 warmup 为 0/1/2 的比较；第 6 页要求两种模型乘三种 context 的 Nsight 分析；第 9-10 页要求 memory_viz 与 Nsight 内存证据；第 17 页的 `torch_compile(b)` 要求 forward 和含 optimizer 的完整训练步；第 28 页的 mandatory Flash 是 eager 与 Triton forward + compiled PyTorch backward；第 32 页要求 2/4/6 rank 通信；第 36 页要求 DDP 两张 trace，并**建议**重复测试约 5 次。
- 第 28 页另列的 full Triton backward 是可选扩展。现有 flash 总计划 240 个配置包含这 80 个额外配置；课程必需的 eager + partially Triton 矩阵是 160 个配置。

## 环境与固定参数

以 `runs/cloud/20260914T091848Z-flash-eager/20260914T094025Z-rpc/environment.json` 及相邻 `pip-freeze.txt`、`nvidia-smi.txt`、`topology.txt`、`nvlink-status.txt` 为具体来源；不同作业有各自的环境记录，不能据此假设所有作业分配到了同一台物理主机。

| 项目 | 记录值 / 设置 |
| --- | --- |
| GPU | NVIDIA B200，compute capability 10.0；样例设备记录为 182631 MB、148 SM |
| GPU 数量 | 普通矩阵 1；分布式 XL 与 leaderboard 2；通信分别 2、4、6 |
| 系统 / Python | Linux gVisor x86_64；Python 3.13.3，GCC 12.2.0 |
| PyTorch / CUDA / Triton | `2.11.0+cu128` / `12.8` / `3.6.0` |
| Modal SDK | 请求 provenance 记录 `1.5.5` |
| Nsight Systems | 已安装 CLI `2026.5.1.161`；安装成功不等于采集有效 |
| 依赖示例 | einops 0.8.2、einx 0.4.3、jaxtyping 0.3.9、numpy 2.4.4、pytest 9.0.2；完整版本以每个 run 的 freeze 为准 |
| 容器资源 | 8 CPU，64 GiB 主存；`OMP_NUM_THREADS=8`，`TOKENIZERS_PARALLELISM=false` |
| Python 路径 | `/workspace/cs336-basics:/workspace` |
| 模型默认 | batch 4，context 512，vocab 10000，随机初始化和随机 token，AdamW lr `1e-3` |
| 模型精度 | FP32 禁用 `torch.backends.cuda.matmul.allow_tf32`；BF16 使用 autocast，参数仍为 FP32 |
| Triton FP32 dot | 显式 `input_precision="tf32x3"`；与 full-model FP32 设定的差异必须披露 |

| 模型标签 | d_model | d_ff | 层数 | heads |
| --- | ---: | ---: | ---: | ---: |
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 2560 | 10240 | 32 | 32 |
| 10B | 4608 | 12288 | 50 | 36 |

“10B”是题目配置名称，不把它当成实测参数数量；参数数量读取每条记录中的 `measurements.parameters`。

随机性按实际代码记录：模型和 attention benchmark 使用 `torch.manual_seed(2026)`；分布式模型 worker 也设 2026，不能声称各 rank 使用不同随机种子；attention 梯度验证使用 0；优化模型验证使用 7；leaderboard 权重使用 2026、各 rank 数据使用 `2026 + rank`。all-reduce 输入是每轮重置为 `rank + 1` 的确定性张量。单个 saved-block 记录的实现未显式设种子；不能补写一个不存在的历史种子。

## 执行与证据保存流程

1. 本地实现后先运行原始测试与少量回归，记录 JUnit；本机没有 CUDA 的测试被明确 skipped。
2. `modal_assignment2.py` 从 `experiment_plans.make_plan` 生成明确配置，切片形成一次有界请求；保存 `runs/dispatch/<run_id>/request.json` 和 `dispatch.json`。每个 run 有唯一 ID、call ID、GPU 数、作业总超时以及逐文件 SHA256。
3. 容器记录硬件、版本、完整 case 配置，并将**当时上传的** `cs336_systems` 复制到 `/data/<run_id>/source/`。每个 case 启动新的 Python 子进程；同一 run 内不同 case 不复用 Python 模型对象，但编译缓存状态仍可能影响首次编译，因此不把整个 job 时间当成 steady-state 时间。
4. 子进程的完整 argv 写入每个 `status.json.command`。日志同时保留 stdout/stderr。每完成一项更新 `progress.json` 并 commit Volume，正常结束时写 `summary.json` 和 artifact inventory。
5. 每 case 的超时由父进程监督；超时先向整个子进程组发 SIGTERM，10 秒后未退出才 SIGKILL。云任务 `retries=0`，禁止无限重试。历史常规 job 上限 1800 秒；两次 leaderboard 是 720 秒，6 卡通信为 900 秒。
6. 常规 Volume 下载因代理 TLS reset 不稳定，后来使用只读 CPU RPC 取回文本。`*-rpc` 目录仅包含小于 500000 字节的 `.json/.log/.xml/.txt/.py`；默认不会带回 `.pickle/.sqlite/.nsys-rep/.png`。因此“已下载文本快照”不能写成“二进制证据已下载完整”。
7. 下载目录使用不可变时间戳。`download_manifest.json` 保存内容哈希；直接下载流程还会校验文件大小。保留失败下载和历史失败，不覆盖成最新成功结果。
8. 预算暂停后改用已有调用返回值和既有 Volume 的存储 API 恢复文件，不启动新的 GPU 或 CPU 容器。五个 memory pickle 与一个有效 Nsight probe 的 SQLite/report 已恢复到 `*-storage-api` 目录，各自的 `storage_download_manifest.json` 记录文件校验信息；本地可视化的副本与 viewer 来源另记在 `output/memory/memory_view_manifest.json`。
9. 最后三批文本归档也已完成：10B compile 为 `20260914T104956Z-storage-api`（27 文件，124248 bytes）；Flash 最后两项及 summary 为 `20260914T105002Z-storage-api`（9 文件，189824 bytes，前 78 项/source/environment 使用原有快照）；tuned leaderboard 为 `20260914T105045Z-storage-api`（27 个非缓存文件，161806 bytes，可再生成的 cache 排除）。共 63 文件通过长度/SHA256 核对；10B compile 与 tuned 两个 run 各 13 个源码文件匹配 environment provenance。E1/E3 所需结果、日志、源码环境归档已完成。Flash 的 80 组 result/status/stdout、13 个源码和 environment 均齐，但最终 `artifact_inventory.json` 未在指定下载范围，因此自动 full-inventory 标志保持 false；这不是缺少 case 测量或日志，也不需要新实验。

证据优先级：原始 `result.json`/`status.json`/JUnit 与 trace 内容 > job `summary.json` > 派生 CSV/Markdown > 目录名。特别注意外层 `status="ok"` 只表示子进程正常退出；内层 `result.status="oom"` 是 OOM 观察，不是成功时间。无 `summary.json` 也不能仅凭应用停止推出每个 case 已完成。

逐次复现应先匹配该 run 的 `source/` 与 `request.json.provenance.sha256`，再使用其完整配置和相同依赖。当前本地源码可能已修改，不能把历史结果自动归属到当前修订版；staff 代码和 tests 的哈希同样在 provenance 中。调试修正前后的结果应保留为不同 run。

## 历史命令的精确程度

请求记录没有保存最外层终端命令行的逐字文本，因此不虚构当时 shell 的参数顺序。`dispatch.json` 可证明 plan、GPU 数、作业上限；`request.json` 可证明提交的 case 和 provenance；`status.json.command` 是实际执行的子进程 argv。下面的命令按这些 argv 恢复，**只用于说明历史过程，预算暂停期间不执行**。

模型完整训练，来源 `20260914T084942Z-models-small-medium/.../002-small-fp32-train-w5/status.json`：

```sh
/usr/local/bin/python -m cs336_systems.experiments --config '{"name": "small-fp32-train-w5", "kind": "model", "size": "small", "dtype": "fp32", "mode": "train", "warmup": 5, "steps": 10, "timeout": 500}' --output /data/20260914T084942Z-models-small-medium/002-small-fp32-train-w5
```

两卡 NCCL，来源 `20260914T085559Z-communication-and-xl/.../004-nccl-2ranks-1MB/status.json`：

```sh
/usr/local/bin/python -m cs336_systems.experiments --config '{"name": "nccl-2ranks-1MB", "kind": "allreduce", "world": 2, "backend": "nccl", "bytes": 1000000, "steps": 20, "warmup": 5, "timeout": 240}' --output /data/20260914T085559Z-communication-and-xl/004-nccl-2ranks-1MB
```

历史分布式测试，来源 `20260914T084950Z-distributed-tests/.../000-distributed-repeat-0/status.json`：

```sh
/usr/local/bin/python -m pytest -q --tb=short tests/test_ddp.py tests/test_fsdp.py tests/test_sharded_optimizer.py --junitxml=/data/20260914T084950Z-distributed-tests/000-distributed-repeat-0/junit.xml
```

最后一次 CUDA software tracing 探针，来源 `runs/status/20260914T093845Z-profile-probe-cuda-software-trace-20260914T094503Z.json`：

```sh
nsys profile --trace=cuda-sw,nvtx,osrt,cublas --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --cuda-memory-usage=true --pytorch=autograd-nvtx --export=sqlite -o /data/20260914T093845Z-profile-probe-cuda-software-trace/000-nsys-small-s256/profile /usr/local/bin/python -m cs336_systems.experiments --config '{"name": "nsys-small-s256", "kind": "model", "size": "small", "seq": 256, "mode": "train", "dtype": "fp32", "warmup": 5, "steps": 1, "nsys": true, "timeout": 300}' --output /data/20260914T093845Z-profile-probe-cuda-software-trace/000-nsys-small-s256
```

要查看任一现存 case 的精确命令，可在本地执行 `jq '.command' <具体status.json>`；要查看返回快照的命令，使用 `jq '.result.cases[].command' <具体runs/status文件>`。这些读取不会运行命令。不要把命令打印结果自动 pipe 到 shell。

## 计时和内存口径

| 类别 | warmup / 正式测量 | 实际测量内容 |
| --- | --- | --- |
| 普通模型 | FP32 w=5/0/1/2，BF16 w=5；每项 n=10 | `perf_counter` 两侧 synchronize；保存 samples、mean、sample std、median |
| 模型 `forward` | 同上 | forward-only，不算 loss；autograd 开启，符合残差内存观测目的 |
| 模型 `backward` | 同上 | **forward + loss + backward 的总耗时**；不能把它直接命名为纯 backward 时间 |
| 模型 `train` | 同上 | forward + FP32 cross entropy + backward + AdamW；每步 zero_grad |
| 模型独立阶段 | 较晚 harness 添加 CUDA events | `cuda_event_stages.forward/loss/backward/optimizer`；GPU events 与同步 Python 总时间分别展示 |
| compiled 模型 | w=5，n=10 | 只对模型 `torch.compile(model)`；历史 case 仅 forward/backward，缺 train |
| 普通 attention | w=5，n=100 | batch 8，FP32，非 causal；forward/backward 分别同步计时；保存 backward 前内存 |
| Flash | `do_bench(warmup=100, rep=300, return_mode="mean")` | 单卡 B200，batch 1，causal；分别 forward、backward、end-to-end；实际 warmup/rep 单位为毫秒 |
| all-reduce | w=5，n=20 | 每 rank 计时，操作前 barrier，CUDA 同步；每迭代取各 rank 最大时间 |
| 分布式 XL | w=5，n=10 | 两 rank，各保存 step、exposed_sync 与 optimizer 前后 memory |
| memory | w=2，n=2 | XL，context 128/2048，FP32/BF16，forward/train；成功项 dump pickle |
| checkpoint | w=2，n=3 | XL context 2048，FP32，backward 模式；segments=1/2/4/8/16/32/64 |
| profile | w=5，trace n=1，配对 plain n=5 | NVTX `measurement` 范围过滤 warmup；trace 有自身开销，不能作为常规性能基线 |

Flash 请求里虽然有 `steps=10,warmup=5`，实际 `triton_bench=true` 分支使用上表的毫秒式 `do_bench`，仅保存 mean，不应声称保存了十个独立 Flash 样本或标准差。

普通模型计时后另跑一遍记录内存阶段并 reset peak；`memory` 字段与正式计时 samples 不是同一个样本集合。`allocated`、`reserved`、peak allocated 和 peak reserved 分开报告。内存快照记录 warmup 之后的历史。OOM case 保存 traceback 和已能取得的部分信息，不用相邻配置外推缺失时间。

all-reduce 的 MB 是十进制，1MB=`1_000_000` bytes，1GB=`1_000_000_000` bytes；GiB/MiB 内存表使用 1024 进制。带宽采用 `2*(world-1)/world * bytes / latency`。Gloo case 使用 CPU 张量，NCCL 使用 CUDA 张量，尽管它们被打包在同一多 GPU 容器内。DDP 的 `exposed_sync` 只表示 backward 完成后的显式等待，不是所有通信总量；发生重叠时不能解释为通信消失。

## 历史运行登记表

完整路径是 `runs/dispatch/<run_id>/`；已取回内容位于 `runs/cloud/<run_id>/<snapshot>/`，`runs/status/` 还保留返回快照。最后三批原来只有返回值的结果/日志/源码环境已经恢复到 storage-api 快照；读取时按 run/case 去重合并，不把不同快照当成重复测量。下面的 `ok/OOM/failed` 优先读取内层实验状态。

| run_id | GPU | 已有结果与用途 |
| --- | ---: | --- |
| `20260914T082016Z-probe` | 1 | attention 原测试 6 passed；extended checker 失败，保留日志 |
| `20260914T084725Z-probe-leaf-fix` | 1 | 3 项完成：6 个原 attention 测试、扩展梯度检查、precision |
| `20260914T084933Z-profile-probe-nsys` | 1 | 1 个 profile 进程完成；不能据此确认 kernel trace 有效 |
| `20260914T084942Z-models-small-medium` | 1 | 30 个配置成功，各 10 samples；均没有独立阶段 event 数据 |
| `20260914T084950Z-distributed-tests` | 2 | 实际请求只有 repeat-0，8 tests passed；不是五轮 |
| `20260914T085158Z-models-large-xl-10b` | 1 | 45 个配置：40 成功、5 个 10B train OOM；40 个成功项有阶段 events |
| `20260914T085159Z-attention-matrix` | 1 | 40 个 eager/compiled attention 配置完成 |
| `20260914T085559Z-communication-and-xl` | 2 | 8 个通信配置与 5 个 XL 分布式策略完成 |
| `20260914T085600Z-memory-and-checkpoint` | 1 | 16 项：11 成功、4 OOM、saved-block 1 失败；五个成功 memory pickle 已用存储 API 恢复，memory_viz 图片已生成 |
| `20260914T085757Z-nsys-probe-and-fit-nvtx-filter-fix` | 1 | probe 失败；fit 2 成功、4 OOM，建立最大可用 context 边界 |
| `20260914T091103Z-optimized-validation` | 1 | 最新 `20260914T102152Z` 返回快照确认历史验证失败：chunked loss 与旧 reference 差 0.005754；11.531143 秒，不将后续成功倒填给它 |
| `20260914T091830Z-optimized-validation-fp32-oracle` | 1 | optimized-validation、saved-block 均成功 |
| `20260914T091834Z-profile-probe-keyword-fix` | 1 | probe 计算和 trace 文件生成完成；kernel 完整性仍未通过证据验收 |
| `20260914T091848Z-flash-eager` | 1 | 80/80 成功，均有 forward/backward/end-to-end mean |
| `20260914T091931Z-models-compile-small-medium` | 1 | 4/4 成功，各 10 samples；只含 forward/backward |
| `20260914T091932Z-communication` | 4 | 8/8 成功，各 rank 20 samples |
| `20260914T092247Z-leaderboard-first-full` | 2 | 完整大配置运行成功；未提交 leaderboard |
| `20260914T092252Z-profiles` | 1 | 13 个进程完成：6 plain、6 model trace、1 saved-block trace；有效截图仍缺 |
| `20260914T092406Z-distributed-profiles` | 2 | naive/overlap/FSDP 三项进程完成；有效 GPU trace 截图仍缺 |
| `20260914T092409Z-flash-triton-compiled-backward` | 1 | 80/80 成功，各有三项 mean；最后两项及 summary 已恢复到 `20260914T105002Z-storage-api`，与旧 78 项/source/environment 合并归档完成 |
| `20260914T092847Z-communication` | 6 | 8/8 成功，各 rank 20 samples；job 上限 900 秒 |
| `20260914T092847Z-flash-triton-full` | 1 | 可选 full Triton 80/80 成功，均有三项 mean |
| `20260914T093124Z-tuned-validation` | 1 | 1 项完成；这是优化方案验证，不能替代整套最终版测试 |
| `20260914T093129Z-models-compile-large-xl` | 1 | 4/4 成功，各 10 samples |
| `20260914T093842Z-leaderboard-tuned` | 2 | 返回快照确认完成；`20260914T105045Z-storage-api` 已恢复 27 个非缓存文件，结果/日志/源码环境归档完成 |
| `20260914T093845Z-profile-probe-cuda-software-trace` | 1 | SQLite/report 已恢复；本地修正归因后 3637 个 kernel 与 launch 全部匹配，small seq256 的五个子问题已有分析，可复用 |
| `20260914T093858Z-models-compile-10b` | 1 | 两个 case 成功，各 10 samples；`20260914T104956Z-storage-api` 已恢复 27 文件，结果/日志/源码环境归档完成 |

## 失败、修复和证据缺口

| 现象 | 已记录原因 / 处理 | 不能做的推断 |
| --- | --- | --- |
| 本机最初 distributed 测试未能建 localhost socket | 受限环境在执行分布式代码前失败；之后有放宽该限制的历史 JUnit | 不把环境失败计为 DDP 算法失败 |
| extended attention checker non-leaf 错误 | FP32 reference alias；改为 detach 后建立 leaf copies，随后 rerun 成功 | 早期失败不是 kernel 数值失败证据 |
| 上传时 `.pyc` 变化 | 并发导入改变 bytecode；上传过滤 `__pycache__`/`*.pyc` | 无测量的启动中止不算完成实验 |
| NVTX capture 初始不可靠 | 修正 capture filter，后续增加 register-only 环境设置 | 进程退出零不保证 capture 范围与 CUDA kernel 完整 |
| annotated attention unexpected keyword `K` | staff 模型使用大写 `Q/K/V` 关键字；wrapper 参数已对齐 | 该 harness 失败不能用于模型性能比较 |
| saved-block 收到字符串 | forward hook 返回了 `stack.pop()` 的字符串，替换了模型输出；修为显式返回 None | 原失败条目保留；使用 `...fp32-oracle` 的成功结果 |
| 优化模型 reference 核对 | `091103` 已恢复的历史结果显示 scalar 比较失败：28.687281 vs 28.681526，absolute diff 0.005754 超出 1e-4；后续 `fp32-oracle` job 另存修正后的成功证据 | 失败不能删除或改为成功；修复前后的 reference 与源码不同 |
| 旧 Nsight SQLite 缺少可用 kernel 证据 | `runs/nsight-schema.json` 记录不完整 CUPTI events 被丢弃；存在 NVTX/运行时事件仍不足 | 不能编造最耗时 kernel、调用次数、softmax/GEMM 比例或 overlap 截图 |
| 最新 `cuda-sw` probe 与本地归因修复 | 已恢复有效 kernel；旧解析器只匹配主线程、且未按进程限定 correlationId，遗漏 autograd worker 的 backward。改按 `(globalPid, correlationId)` 关联，并对顶层阶段使用同进程 launch 时刻归因 | 不再使用旧约 0.0013ms 的 backward 结果；修正统计不等于重新测量，也不证明其他 profile 有效 |
| Volume 下载 TLS reset | 保存失败快照，历史使用 CPU RPC 取回小文本 | CPU RPC 不是免费本地操作，预算暂停时不自动重试 |

明确的 OOM 共 13 个历史记录：五个 10B train（FP32 w=5/0/1/2、BF16 w=5）；memory XL context 2048 的 FP32 forward、FP32 train、BF16 train；checkpoint segments=1；fit small context 8192/16384、XL context 2048/4096。它们是实际资源边界，应在对应表内报告 OOM，无需仅为了把表填成数字而重跑。成功的 fit 边界为 small 4096、XL 1024。

`runs/status/20260914T092409Z-flash-triton-compiled-backward-20260914T100623Z.json` 通过既有调用返回值确认最后两项 mandatory Triton BF16、context 65536、dim 64/128 均成功，并有三项 mean。之后这两项的 result/status/stdout/partial 和完整 summary 已恢复到 `runs/cloud/20260914T092409Z-flash-triton-compiled-backward/20260914T105002Z-storage-api/`。现在 **Flash 240/240 已有成功测量和本地 case 归档证据**，其中课程必需矩阵 160/160、可选 full Triton 80/80；旧快照 238 项只是历史下载范围，新旧快照合并后已补齐，不安排任何 Flash 重跑。

small/medium 的 30 个历史基线虽然都有总耗时，但没有 `cuda_event_stages`。需要独立 backward/optimizer 的表格单元仍为空；不能把 forward+backward 总时间当作 backward，或把不同运行的均值相减写成直接测量。5 个 compiled-train 配置不在旧计划内，题目第 17 页仍要求它们。

成功 memory case 共 5 个：context 128 的两精度乘两模式，加 context 2048 的 BF16 forward。五个 pickle 已下载到 `runs/cloud/20260914T085600Z-memory-and-checkpoint/20260914T100931Z-storage-api/`，并使用官方 PyTorch v2.11.0 MemoryViz 在本地 HTML 中展示。题目要求的两张图已保存为 `output/memory/forward-memoryviz.png` 和 `output/memory/train-memoryviz.png`，均为 XL、seq128、FP32；另有 detail10/detail9 图片观察大分配。每条轨迹包含 **2 个正式计时步 + 1 个额外内存记录步，共 3 步**，已排除最初两次 warmup。横轴是 allocation/free 事件序列，不能标成毫秒或用间距直接推导阶段耗时。来源、文件大小和 SHA256 见 `output/memory/memory_view_manifest.json`。saved-tensor hooks 得到的数字仍是辅助证据，不能代替第 10 页要求的 Nsight 内存截图。

`runs/cloud/20260914T093845Z-profile-probe-cuda-software-trace/20260914T101417Z-storage-api/` 中的有效 SQLite/report 已在本机完成统计。3637 个 launch 与 3637 个 kernel 一一匹配，measurement 内没有未归属或重复归属阶段，累计 GPU kernel 时间 29.847779ms，backward 为 16.380591ms。这里是 kernel 累计时间，不能当作同步墙钟时间；同一 profile 的完整 Python 时间约 107.144ms，配对 plain 为 36.912ms，显示 profiler 有明显扰动。完整来源、修复原因、五个子问题和限制见 [output/NSIGHT_EXISTING_ANALYSIS.md](NSIGHT_EXISTING_ANALYSIS.md)。该分析没有冒充 Nsight GUI 截图；若需此组截图，可本地打开现有 `.nsys-rep`，无需 GPU。

六组模型 profile 中 small seq256 已有有效证据，因此仅剩 small seq1024/4096、XL seq256/512/1024 五组。DDP naive/overlap 两组和 FSDP 一组的有效 trace 仍缺。单 block allocation 的 SQLite/report 已恢复至 `runs/cloud/20260914T092252Z-profiles/20260914T102148Z-storage-api/012-profile-saved-block-xl-s2048/` 并完成只读审计：1357 条内存事件中有 35 次动态 Device 分配、1322 条 Static 分配，**动态释放为 0**。累计申请 18,518,116,352 bytes 是 CUDA 缓存分配器申请 segment 的总量，不是 peak 或 residual；分配 top-5 也不能代替 saved-for-backward top-5。

该旧文件可用于本地 GUI 的部分 allocation/NVTX 截图，但不足以按题目要求从 residual 释放及净变化反推梯度大小。已有 hooks 的 residual 6,818,955,264 bytes、参数梯度 419,450,880 bytes、输入梯度 83,886,080 bytes 继续作为独立证据，不能声称来自该曲线推导。严格补齐 memory(f) 需要 G6：一个有界的单 B200、单 XL block 内存诊断，在新进程禁用 caching allocator 并显式标记 forward/backward，以观察分配和释放；无需重跑完整 XL 或整个内存矩阵。派生审计见 `output/profiling/saved-block/allocation_summary.json` 和 `output/NSIGHT_EXISTING_ANALYSIS.md`，本次没有补采。

## 测试记录与修订版限制

- `runs/local/initial/cs336-assignment2-unrestricted-tests.xml`：14 collected，10 passed、4 CUDA skipped，13.737 秒。
- `runs/local/fsdp-prefetch-optimizer-load.xml`：14 collected，10 passed、4 CUDA skipped，11.558 秒。
- `runs/local/extended-tests.xml`：2 passed，0 skipped，0.757 秒。
- B200 原 attention suite：第一次 6 passed，修复 checker 后再次 6 passed；各自 JUnit 保留。
- 历史两卡 distributed 原 suite：`distributed-repeat-0` 的 JUnit 为 8 passed，0 skipped，45.183 秒。这条旧 run 只含一轮；后续五轮本地 Gloo 证据另列如下。
- 后续源代码核查确认上述原测试固定 `backend="gloo"`（DDP 第 40 行、FSDP 第 120/207 行、sharded optimizer 第 31 行）。因此“两卡容器中通过”不能解释为 NCCL 数值回归通过。原测试保留，另列小模型 NCCL 输出/同步梯度/更新与 reference 比较；已有 NCCL benchmark 只是运行/计时证据。
- 最新本地回归 `runs/local/report-final-cpu-tests.xml`：18 collected，14 passed、4 CUDA skipped，12.856 秒（约 12.86 秒），已包含两个新增 Nsight parser 测试；这些 skipped 不算 GPU/NCCL 验证。较早一次是 16 collected、12 passed、4 skipped、15.638 秒，已被本次重跑结果更新，不再作为该 XML 的当前计数。
- 本地 Nsight parser 的两个合成 SQLite 测试先前单独运行 passed，0.06 秒，之后已包含在上述 14 passed 中，不能再加一次。覆盖跨线程 backward、进程内 correlationId、GPU launch/执行时间边界、缺少 kernel 时明确报错及原数据库哈希不变，详见 `output/NSIGHT_EXISTING_ANALYSIS.md`。
- 本地课程分布式稳定性已完成：`runs/local/final-gloo-stability-localhost/` 的五个独立进程各 8 passed、0 failure/error/skip，共 40 次用例执行（8 个用例重复五轮，不是 40 个独立测试）。逐轮 wall time 为 14.0135/13.5914/16.8382/13.0814/13.3180 秒，总计 70.8426 秒；`repeat-1..5.xml/.log`、`results.json` 和 `manifest.json` 保存完整命令、JUnit 与 23 个源码文件前后哈希不变的证据，指纹 `f1cdc2a4de5d952e26a3f8def0ff0c2ad3d9aea63daf82196f9300156c404d37`。这些是 CPU/Gloo，不是 CUDA/NCCL。
- 前一次受限尝试 `runs/local/final-gloo-stability/` 在首轮出现 8 个失败，原因是 localhost:12390 的 `Operation not permitted`，在进程组创建阶段发生；停止后保留原日志。随后仅允许本地 loopback 的独立目录重跑通过。环境失败不改写为算法失败，也不覆盖或混入五轮成功结果。

测试报告与 source manifest 绑定。最后代码变更后的整套 GPU/NCCL 测试尚无当前修订版证据；不把 earlier run、优化验证或本机 CPU 回归当成最终 GPU 全套回归。当前 tuned Flash/optimized-model 的输出、loss、梯度还须按 `rtol=atol=1e-2` 严格核对：现有 model-output rtol=0.03、model-gradient rtol=0.1 等较松验证不能替代它，FP32 chunked algebra 的较严小测试也不覆盖全部 tuned 路径。V1 单列原始 CUDA、NCCL 小模型回归和 tuned 严格检查；V2 的原始 Gloo 五轮已在本地完成，NCCL 五轮是另外的预算可选项。本日志编写未执行测试；上述新增 CPU 检查由主任务完成并保存证据。

## 费用与存储口径

本地核对的 `output/data/coverage_snapshot.json` 记录 448 条云端实验记录，其中 431 ok、13 OOM、4 failed；27 个已 dispatch 的 run 都有完成摘要（包括返回快照）。审计 635 个带原始 samples 的统计对象，发现 0 个统计不一致、0 个跨 rank 不一致；240 个 do_bench case 均有有限值。431 是实验记录数，不是单元测试数，也不是最终版本已验证用例数；本地五轮 Gloo 的 40 次测试执行另记，不混入这 448 个云端 case。此前 09:44 UTC 的 438 条/$14.021608，以及 10:15 UTC 的 447 条/$16.831832，都是恢复更多既有结果之前的历史快照。

最新 `output/data/cost_ledger.json` 给出 **$16.854697 的部分 runtime-only 估计**，覆盖全部 27 个已知 run；最后恢复的 `20260914T091103Z-optimized-validation` 是历史失败验证，11.531143 秒，对小计增加 $0.022864873。估计公式使用历史记录的 `GPU_count*0.001736 + 8*0.0000131 + 64*0.00000222` 美元/秒；早期多卡 raw estimate 曾按单卡导入默认值计算，派生 ledger 用 dispatch 的 GPU 数修正，原始文件不改写。该价格是历史估计输入，不是本次核验的当前报价。

虽然已知实验 run 的摘要完整，这个小计仍不含 image build/冷启动、存储、CPU 提取或 GUI 容器，以及其他账单项目。ledger 保留逐 run 来源与 `is_complete_total=false,is_invoice=false`。不能据此承诺“总共只花了 $16.85”、剩余额度或任一任务的确定费用；没有新账单证据时保持不确定。Volume 保留数据可能持续产生存储费，但删除会损失复现证据，需用户另外决定。

## 此轮本地整理结果

已把执行环境、配置、实际命令来源、计时口径、修复过程、失败/OOM 和证据不足处集中记录，并将所有仍可能需要 GPU 的事项单列。主任务恢复了 Flash 完成结果、五个内存 pickle 和有效 small-seq256 Nsight 文件；本地生成两张主要 memory_viz 图、两张细节图，修正 Nsight 统计并完成该组五个子问题。没有因此启动云容器、执行 GPU、刷新外部价格或提交作业。compiled train、阶段 baseline、其余五组 model profile 与三组 distributed profile 等缺口继续列在清单；恢复预算后逐项选择，不自动启动整个旧矩阵。


---

# 附录 D · 待补 GPU 任务


## 当前约束

**预算已暂停。没有用户新的明确授权，不运行任何 GPU 或 CPU 云端任务。** 这份清单是可审核的工作队列，不是启动指令，不自动执行计划，也不承诺费用。CPU RPC 读取、云端 Nsight GUI、远程文件分块下载同样可能启动计费容器，须遵守暂停。

证据包含 Flash 完成快照、五个已恢复 memory pickle、`20260914T101417Z-storage-api` 中有效 small-seq256 Nsight 文件及其本地分析。当前代码的后续本地修订，必须与最终验证时的源码哈希匹配。所有未来实验保存唯一 run ID、完整 config/argv、source SHA256、环境、原始 samples、stdout/stderr、超时及 OOM；不覆盖历史结果。

优先级：P0 先处理已有证据，P1 题目必需的缺口，P2 最终修订版验证与稳定性，P3 可选优化。下列时间是**单次执行的停止上限**，含编译的实验可能来不及完成；不是预计耗时、计费上限或价格承诺。达到上限后保留 partial/timeout 并停止，无自动重试。

## 无需重跑的范围

| 范围 | 已完成证据 | 后续处理 |
| --- | --- | --- |
| 2/4/6 rank 通信 | Gloo/NCCL × 1/10/100/1000 MB × 3 world size，共 24 个配置；各 rank 都有 20 samples | 直接复用，**不再申请 4/6 卡重跑** |
| 两卡 XL 策略 | naive、flat、overlap、overlap+sharded optimizer、FSDP，各 rank 10 samples | 性能数字复用；截图证据单独补 |
| 模型 baseline 总耗时 | 75 项已尝试：70 成功、5 个 10B train OOM | 保留 OOM；仅补真正缺失的独立阶段证据 |
| 普通 attention | eager/compiled 共 40 个配置完成 | 直接使用已有数据 |
| Flash eager | 80/80 完成，各三项 mean | 不重跑 |
| Flash mandatory Triton | 80/80 完成，各三项 mean；最后两项和 summary 已补入本地归档 | 不重跑；旧快照 78 项与新快照合并后已完整 |
| Flash full Triton | 可选扩展 80/80 完成，各三项 mean | 不重跑，不以此替代 mandatory compiled backward |
| memory_viz 图片 | 五个 pickle 已恢复，官方 PyTorch v2.11.0 viewer 的两张主要图和两张 detail 图已保存 | 已完成，0 GPU |
| small seq256 Nsight | 已恢复并验证 3637 个 kernel，五个子问题已有分析 | 不重跑 probe；若需 GUI 截图，只用本地 report |
| 原始 Gloo 稳定性 | 本地五个独立进程，各 8 passed、0 failure/error/skip，23 个源码哈希不变 | 已完成，0 GPU；不算 NCCL 验证 |
| 内存/checkpoint OOM | 已有真实 OOM 与成功配置；small 最大已测可容 context 4096、XL 1024 | 不为消除 OOM 标记改变基线或反复试跑 |

## P0：既有证据恢复结果

| ID | 具体任务与配置 | 验收 | GPU / 时间边界 / 依赖 |
| --- | --- | --- | --- |
| E1（已完成） | `20260914T092409Z-flash-triton-compiled-backward/20260914T105002Z-storage-api` 已恢复最后两项及 summary；前 78 项/source/environment 原快照已有 | 9 个新增文件、189824 bytes，经长度/SHA256 验证；合并后 80/80 mandatory Triton 归档完整，全 Flash 240/240 | 0 GPU，仅既有存储 API；无待下载/待重跑项 |
| E2（已完成） | `20260914T093845Z-profile-probe-cuda-software-trace/20260914T101417Z-storage-api` 中 SQLite/report 已恢复并本地分析 | 3637 个 kernel 与 launch 全部匹配，阶段无遗漏/重复，累计 29.847779ms；修复后的 backward 为 16.380591ms，五个子问题见 `output/NSIGHT_EXISTING_ANALYSIS.md` | 0 GPU；有效 probe 已复用为六组之一，不开新探针。此分析不是 GUI 截图；需要时本地打开既有 report |
| E3（已完成） | 10B compile 在 `20260914T093858Z-models-compile-10b/20260914T104956Z-storage-api`；tuned 在 `20260914T093842Z-leaderboard-tuned/20260914T105045Z-storage-api` | 分别 27 文件/124248 bytes、27 非缓存文件/161806 bytes；长度/SHA256 全通过，每 run 13 个 source 文件匹配 environment provenance，tuned 可再生 cache 排除 | 0 GPU，仅既有存储 API；结果/日志/源码环境归档完成，不重跑这三项成功 case |
| E4（已完成） | memory run 五个 pickle 已恢复到 `20260914T100931Z-storage-api` 并用官方 PyTorch v2.11.0 MemoryViz 本地展示 | `output/memory/forward-memoryviz.png`、`train-memoryviz.png`，另有 detail10/detail9 图；来源和 SHA256 记入 `memory_view_manifest.json` | 0 GPU；不再生成内存实验，不再等待 pickle。截图为 XL seq128 FP32，排除 warmup 后有 2 个计时步+1 个额外记录步；横轴是事件序列，不是毫秒 |
| E5（审计已完成，证据部分可用） | `20260914T092252Z-profiles/20260914T102148Z-storage-api/012-profile-saved-block-xl-s2048` 的 allocation SQLite/report 已恢复并只读分析 | 35 次动态 Device 分配、0 次动态释放；有 allocation/NVTX 证据，不能从缓存 segment 曲线反推 residual 释放和梯度净增量 | 0 GPU 审计完成；可本地取得部分截图。严格补齐 memory(f) 仍需 G6 单 block 诊断，预算暂停时不执行 |

E4 已完成题目第 9 页的两张图片交付，没有 GPU 缺口。已有 hooks 表或自行画的概念图不能冒充 memory_viz/Nsight 截图。第 10 页的单 block Nsight allocation 截图仍按 E5 单独检查。

E1 的“完成”指所需测量、日志、源码环境齐备：Flash 80 组 result/status/stdout 全部本地存在；最终 `artifact_inventory.json` 未在指定下载范围，full-inventory 标志仍为 false。该元数据例外不影响 80 组证据覆盖，也不是待跑或待用户操作项。

## P1：题目所需、尚无完整证据的 GPU 项

### G1. 五个 compiled full-train 配置

- 缺口：题目第 17 页要求 `torch_compile` 后 forward、backward、optimizer 的完整训练步比较。旧 `models-compile` 只生成 forward/backward，后者包含 forward+loss+backward，仍不含 optimizer。
- 配置：`kind=model`，size=`small/medium/large/xl/10B`，`dtype=fp32`，`mode=train`，`compile=true`，batch=4、seq=512、vocab=10000，seed=2026，warmup=5、steps=10；沿用 staff AdamW、lr=1e-3 与 FP32/TF32 设置。`torch.compile(model)` 的边界须写清，不宣称 optimizer 本身被编译。
- 验收：五项各有十个完整训练步 samples、mean/std、阶段 timing 和 peak memory，或真实 OOM/timeout；与相同 eager 基线比较。10B eager train 已 OOM，因此 compiled train 也允许以实际 OOM 作结果，不能替换为较小 batch 后称同配置完成。
- 依赖：先在本地加入明确 `mode=train` case 并检查配置；冻结待验证源代码。旧计划不会自动包含这些项。
- GPU：1×B200。每 case 上限 600 秒，分批 job 每批不超过 1800 秒；不并发发起五批，也不自动重试。
- 优先级：P1。

### G2. small / medium 的独立阶段 baseline

- 缺口：这两种模型 30 个历史配置各有总耗时，却全都缺 `cuda_event_stages`。不能直接用 `mode=backward` 总耗时回答纯 backward 时间，也不能把独立运行之差冒充直接测量。
- 最小主要配置：size=`small/medium` × dtype=`fp32/bf16`，`mode=train`、`compile=false`，batch=4、seq=512、vocab=10000，warmup=5、steps=10、seed=2026，共 4 项。每次完整训练同时记录 forward/loss/backward/optimizer，复用已有 forward-only 基线。
- warmup 矩阵的完整补齐：若报告要给 w=0/1/2 的独立阶段数字，再加两种 size × 三个 warmup 的 FP32 train，共 6 项。原始 w=0/1/2 的总时间数据已存在；仅用总时间讨论 warmup 时不需要这 6 项。
- 验收：每个要求的阶段有 10 个 samples、mean/std，分别标注同步 Python 总时间与 GPU event 时间；FP32/BF16 使用相同模型和数据形状。若需要纯 backward Python 墙钟时间，应先在本地定义测量边界，不用差分假装实测。
- 依赖：本地 stage instrumentation 可读且标签正确；与旧 baseline 的 provenance/定义一致。可与 G1 在同一获批单卡会话依次执行，保留独立 case。
- GPU：1×B200。每 case 上限 300 秒；只跑获批子集，建议先 4 个 w5 主配置，job 上限 1800 秒。
- 优先级：P1 主要四项；补完整 w0/1/2 阶段表按报告需求决定。

### G3. 剩余五个有效模型 Nsight profile

- 缺口：题目六组中 **small seq256 已完成有效 kernel 分析**。剩余 small 的 seq=1024/4096，以及 XL 的 seq=256/512/1024，共五个配置；旧硬件 trace 尚无可用 GPU kernel 证据。这两个最大 context 的下一档 OOM 已有证据，无需重新找边界。
- 配置：`kind=model,mode=train,dtype=fp32,nsys=true`，batch=4、vocab=10000，seed=2026，warmup=5、steps=1；保留 forward/loss/backward/optimizer 和 attention score/softmax/value matmul NVTX 标签。配对 plain timing 已存在。
- 验收：每组 trace 有 measurement 内有效 CUDA kernel 时间及调用次数；能回答最大累计耗时 kernel、forward 与 forward+backward 差异、非 GEMM 时间、optimizer 后 GEMM 占比和 attention softmax/GEMM 比较；截图来自真实 Nsight GUI。
- 依赖：E2 已验证 software tracing 和本地解析流程，不再等待探针。复用该采集配置和修正后的跨线程/跨进程归因；开始前仍需新预算授权及冻结源码。
- GPU：采集时 1×B200，读取/分析时 0。每个正式 profile 上限 300 秒，单批 job 不超过 1800 秒；不为 small-seq256 或相同探针重开 GPU。
- 优先级：P1，预算暂停中。

### G4. 两个 DDP 通信重叠 profile

- 配置：1 node × 2 B200，XL、global batch=4（每 rank=2）、seq=512、vocab=10000、FP32，staff AdamW；strategy=`naive` 与 `overlap` 各一项，warmup=5、steps=1、nsys=true，seed=2026。
- 验收：两张真实时间线能同时看到 CUDA compute、NCCL 通信、backward NVTX；naive 与 overlap 的差异清晰。只展示 CPU API/NVTX 或 `exposed_sync` 数字不算截图完成。
- 依赖：先检查 `20260914T092406Z-distributed-profiles` 已有两条 trace；若不满足验收，复用 E2 已验证的 software tracing 办法补采。rank 各自设备及子进程采集必须有效，解析按进程限定 correlationId，不能把不同 rank 串在一起。
- GPU：采集 2×B200，分析 0。每 profile 上限 300 秒，job 上限 900 秒，无自动重试。
- 优先级：P1。已有 naive/overlap 的十步 benchmark 不重跑。

### G5. 一个 FSDP all-gather profile

- 配置：1 node × 2 B200，XL、global batch=4、seq=512、FP32、strategy=`fsdp`、optimizer=`adamw`、warmup=5、steps=1、nsys=true，seed=2026。
- 验收：真实时间线包含权重 all-gather、compute 和后向相关通信；能识别题目要求的 all-gather 与计算时间关系，注明是否有 prefetch/重叠，不能从 API 入队时间直接推断 GPU 完成时间。
- 依赖：先检查既有 `profile-xl-fsdp`；确需重新采集时复用 E2 已验证办法，并冻结当前 FSDP 源码。单卡 probe 有效不自动证明多进程采集完整。
- GPU：采集 2×B200，分析 0。case 上限 300 秒，job 上限 600 秒。
- 优先级：P1。已有 FSDP benchmark、峰值显存和 optimizer 数据复用。

### G6. 一个单 TransformerBlock 内存诊断

- 缺口：E5 已确认旧文件有真实分配/NVTX，但动态释放事件为 0；它只显示 CUDA caching allocator 申请的新 segment，不能严格完成第 10 页 memory(f) 的释放/净变化梯度推导。现有 allocation top-5 与 hooks residual top-5 不可混用。
- 配置：XL 的一个 TransformerBlock，batch=4、seq=2048、d_model=2560、d_ff=10240、heads=32、FP32；`kind=saved_block,nsys=true`，单次 forward/backward，CUDA memory usage 与 autograd NVTX 开启。新进程禁用 CUDA caching allocator，显式标记 forward/backward，并继续用 saved-tensor hooks 区分临时张量和 residual。现有 saved-block 未显式设 seed；新诊断明确设 seed=2026 并记录，不回写旧 provenance。
- 验收：先确认 trace 同时有实际分配和释放，再在真实 Nsight 截图上关联 forward 残差、backward 释放和梯度生成；给出 top-5 操作、比例和净变化推导，与既有 storage 去重 hooks 数字交叉核对。不能以累计申请量充当峰值。禁用缓存后的诊断时间不与普通 benchmark 直接比较。
- 依赖：本地先准备独立的无缓存诊断配置与阶段标签；E5 审计已完成，补采必须等待新预算授权。只补一个 block，不重跑完整 XL 或整个内存/checkpoint 矩阵。
- GPU：1×B200；case 上限 300 秒，job 上限 600 秒，无自动重试。
- 优先级：P1，严格补齐 memory(f) 的剩余证据任务。

## P2：最终修订版验证与建议稳定性

### V1. 当前修订版的 GPU/NCCL 正确性验证

- 原因：已有 CUDA attention 和分布式测试对应历史源码。并且原始 `test_ddp.py:40`、`test_fsdp.py:120/207`、`test_sharded_optimizer.py:31` 固定 `backend="gloo"`；即使在 GPU 容器执行，也不能算作 NCCL 数值正确性验证。已有 NCCL 性能数据不替代输出/梯度/更新的 reference 比较。
- 最新本机检查：`runs/local/report-final-cpu-tests.xml` 为 18 collected、14 passed、4 CUDA skipped，12.856 秒，包含两个 Nsight parser 回归。这是 CPU 验证，四项 CUDA skipped 仍须按本项在 GPU 上确认。
- 原始测试：冻结源码与 adapters，保留课程 tests 不改 backend 或放宽容差；原 attention tests 仍待在 1×B200 验证。原 Gloo distributed tests 已在本地 CPU 完成五轮，每轮 8 passed，单列结果并沿用原始容差，不再列为 GPU 待办。
- NCCL 小模型回归：新增独立测试明确 `backend="nccl"`、world=2、每 rank 一张 CUDA 卡。使用相同初始权重和可复核的全局 batch 切分，对 naive/flat/overlap DDP、sharded optimizer、FSDP 的输出、同步梯度、AdamW 更新后参数与未分布式 reference 比较；FSDP 先重建完整参数/梯度。采用小模型（例如 2 层、d_model=64、d_ff=128、4 heads、batch=4、seq=16、vocab=128）、seed=2026、3 个更新步；覆盖实际实现涉及的 FP32/BF16 与 tied/frozen 参数，容差在执行前固定并记录。**这是数值回归，不重跑 XL 或通信性能矩阵。**
- tuned 实现严格核对：在 1×B200 对当前实际启用的 tuned Flash 和 optimized-model 路径新增/收紧独立检查；Flash 比较输出 O 及 dQ/dK/dV，优化小模型比较 logits、loss、全部参数梯度，统一要求 `torch.testing.assert_close(..., rtol=1e-2, atol=1e-2)`，不再沿用现有 output rtol=0.03、model-gradient rtol=0.1 等较松通过结论。输入覆盖 FP32/BF16、causal/noncausal 及非整块长度；使用一致输入、上游梯度和 reference，不通过放宽容差消除失败。
- 验收：原始 Gloo、原始 CUDA、独立 NCCL 与 tuned 严格检查分别保存 JUnit、完整 config、实际 backend/device、最大误差和 source hash；每 rank 的 reference 比较都须完成，必要 CUDA/NCCL tests 不可 skipped。旧测试或现有较松 validation 不算这些新增检查已通过。失败先本地诊断，不开循环重试。
- GPU/时间边界：原 Gloo 回归 0 GPU、每轮上限 240 秒；原 attention suite 1×B200、上限 600 秒；tuned 严格检查 1×B200、上限 600 秒；独立小模型 NCCL 回归 2×B200、上限 600 秒。可在获批会话内依次运行，单个云 job 不超过 1800 秒，不自动重试。
- 优先级：P2，最终版本交付的验证缺口；不是又一轮性能矩阵。

### V2. 原始 CPU/Gloo 五轮已完成；NCCL 重复为可选

- 题目第 36 页建议重复 DDP tests，例如 5 次。现已在 `runs/local/final-gloo-stability-localhost/` 完成五个独立 CPU/Gloo 进程，DDP/FSDP/sharded-optimizer 原 suite 各轮均 8 passed、0 failure/error/skip，共 40 次执行；23 个源码文件哈希前后不变。逐轮 wall time 为 14.0135/13.5914/16.8382/13.0814/13.3180 秒，总 70.8426 秒，JUnit/日志/命令与 manifest 全部保留。
- 原受限目录 `runs/local/final-gloo-stability/` 的首轮 8 个失败是 localhost 套接字 EPERM 的环境失败，已独立保留，不混入五轮成功结果，也不据此声称算法失败。
- 可选扩展：若另外要求 NCCL 稳定性，在相同冻结修订版上对 V1 的独立小模型 NCCL 回归做共 5 轮；V1 成功一轮可算第一轮，再补 4 轮。Gloo 与 NCCL、不同修订版的轮数不能混算；课程 Gloo 五轮完成不自动完成此扩展。
- 可选 NCCL 验收/边界：实际使用 CUDA/NCCL，每轮 JUnit 无 failure/error/skip，并保留 runtime 和 manifest；2×B200，每轮上限 600 秒，分批 job 不超过 1800 秒。发现挂起/超时即停止，额外四轮须纳入新的明确预算授权。
- 优先级：原始课程建议已完成；额外 GPU/NCCL 五轮仅为 P2 可选，不默认运行。V1 的一次严格 CUDA/NCCL 验证仍单独保留。

## P3：可选事项，当前不继续投入

进一步 leaderboard 调优、更多 tile 搜索、额外 full Triton backward 调参、更大模型/更多卡实验都不是补齐上述题目缺口的前置条件。两次 leaderboard 已有成功记录，最新一次的非缓存归档也已完成。不默认安排新实验，不提交 leaderboard，不发布作业。

## 最小恢复顺序

1. 继续完成本地报告、表格、推导和缺口说明；冻结本地代码。E1-E4 的恢复/验收已完成；E5 审计已完成、旧 trace 可用于部分截图，严格补齐交给 G6。涉及计费容器时仍等待新的授权。
2. 明确选择 G1 和 G2 的批准范围；G3-G5 复用已验证的采集办法。G6 只做一个单 block 无缓存内存诊断。这些 GPU 项都等待新预算授权；Flash 与 memory_viz 没有待跑项。
3. 报告中所有真实 OOM 保留；通信 24 配置、已完成 attention/Flash 数据不重复计算。
4. 最后执行获批的当前修订版 V1。V2 的原始 CPU/Gloo 五轮已完成；只有用户额外选择时才做 NCCL 重复。采集和数值 benchmark 分开报告，随后仅在本地整理截图/PDF与提交包。

此清单没有配置新的定时任务、后台 watcher、GPU job 或 CPU artifact job。任何恢复操作都逐项有界，用户可直接选择 ID；没有总费用承诺。
