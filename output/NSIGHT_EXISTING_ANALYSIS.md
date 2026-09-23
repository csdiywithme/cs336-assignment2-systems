# 既有 Nsight 数据的本地修复与分析

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
