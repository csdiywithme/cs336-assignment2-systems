# Assign2 handout 答案与证据索引

本文件已完成**现有证据能够支持的文字解答与理论推导**，不是“Assign2 全部实验完成”的声明。
原始题目以仓库内 [26.1.3 handout](cs336_assignment2_systems.pdf) 为准；完整实验表见
[EXPERIMENT_TABLES.md](output/tables/EXPERIMENT_TABLES.md)，逐次记录见
[all_attempts.json](output/data/all_attempts.json)，每条记录的 `source` 指向不可变的原始 JSON。
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

(a) 入口为 [benchmark.py](benchmark.py)，支持模型大小、任意层数/宽度/头数、batch、context、精度、编译、预热与重复次数，
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
[small/medium 原始记录目录](runs/cloud/20260914T084942Z-models-small-medium/20260914T085455Z-rpc)
和 [large/xl/10B 原始记录目录](runs/cloud/20260914T085158Z-models-large-xl-10b/20260914T091555Z-rpc)。

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
[precision/result.json](runs/cloud/20260914T084725Z-probe-leaf-fix/20260914T085436Z-rpc/002-precision/result.json)。

### nsys_profile

选择 small 的 256/1024/4096，以及 xl 的 256/512/1024 三种 context；4096/1024 分别是本次实测
普通 FP32 全训练能容纳的最大二次幂长度，下一档已单独运行并记录 OOM。
每个配置有配对的无 profiler 数据及 Nsight 文件，预热不在捕获区间。旧硬件跟踪归档只有 CPU API/NVTX、
GPU memory 等数据；最新 small-s256 `cuda-sw` 探针的既有 SQLite 恢复后，已核验 3637 个 CUDA kernel，
无需重跑即可完成这组分析，其余五组仍缺有效 GPU kernel 证据。完整结果和修复方法见
[NSIGHT_EXISTING_ANALYSIS.md](output/NSIGHT_EXISTING_ANALYSIS.md)，以下数值仅指 small-s256 单次捕获训练步。

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
MemoryViz 得到 [forward 图](output/memory/forward-memoryviz.png) 和 [完整训练图](output/memory/train-memoryviz.png)。
前向时间线反复逐层增长后释放，训练时间线峰值更高，并在反向和更新过程中呈现复杂的分配/释放形态；
每份记录包含两次计时步骤和一次额外的阶段/峰值记录步骤，所以出现三个循环，而不是三个新 GPU 实验。
图的横轴是分配事件序列，不是毫秒；仅凭峰形不能严格定位全部阶段边界，需结合阶段统计和调用栈。
原始 pickle、未经改绘的截图、低 Detail 图、查看页面与 SHA256 均位于 [memory 目录](output/memory)。

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
无需重跑完整 xl 内存矩阵。详见 [既有 trace 审计](output/NSIGHT_EXISTING_ANALYSIS.md) 与 GPU 待办 G6。
原始证据：[内存矩阵](runs/cloud/20260914T085600Z-memory-and-checkpoint/20260914T091525Z-rpc)、
[修复后单块统计](runs/cloud/20260914T091830Z-optimized-validation-fp32-oracle/20260914T093348Z-rpc/001-saved-block-xl-s2048/result.json)。

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
原始证据：[attention matrix](runs/cloud/20260914T085159Z-attention-matrix)。

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
[完成状态快照](runs/status/20260914T093858Z-models-compile-10b-20260914T094503Z.json)
与随后只读取回的 [27 文件归档](runs/cloud/20260914T093858Z-models-compile-10b/20260914T104956Z-storage-api) 交叉核对；
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
实现见 [attention.py](cs336_systems/attention.py)、[triton_attention.py](cs336_systems/triton_attention.py)；
课程测试证据见 [attention junit](runs/cloud/20260914T084725Z-probe-leaf-fix/20260914T085436Z-rpc/000-attention-tests/junit.xml)。

### flash_benchmarking

完整网格为 batch=1、causal=True、S=128..65536 的 10 个二次幂、D=16/32/64/128、FP32/BF16。
分别比较普通 PyTorch、Triton forward+compiled backward、Triton 全前后向；每个配置记录 forward、
backward-only、end-to-end 三项。前两类是题目必要比较，第三类为 leaderboard 优化扩展。
Triton FP32 dot 使用 tf32x3，保留 FP32 输入与近 FP32 精度，和关闭 TF32 的普通 GEMM 不是相同硬件执行模式。

Flash 网格现已核验 240/240 组成功：eager 80/80、必做 Triton+compiled backward 80/80、额外全 Triton 80/80。
最初 238 项有原实验目录归档；此前缺少的必做版本 BF16、S=65536、D=64/128 两项已通过读取既有完成任务的
[状态快照](runs/status/20260914T092409Z-flash-triton-compiled-backward-20260914T100623Z.json) 确认并取得完整数值，
没有重新运行 GPU。随后又只读取回这两项结果/日志与最终 summary 的
[9 个文件](runs/cloud/20260914T092409Z-flash-triton-compiled-backward/20260914T105002Z-storage-api)，
与此前该 run 的源码/环境共同保存；现在 240 项均有本地实验结果文件，不在 GPU 补跑清单内。
例如 S=16384、D=128、BF16 的 eager、必做版本、全 Triton 的 F+B 分别为 3.364、7.448、1.590 ms：
必做 forward 的确更快（0.479 对 1.938 ms），但其 FP32 重算反向使整体反而更慢。
全 Triton 在这个 BF16 输入上约快 2.12 倍；而 FP32 同形状全 Triton 为 11.139 ms，
并不优于必做版本 9.260 ms，展示 tile、精度和反向实现都影响收益。
S=65536、D=128、BF16 的完整 Triton F+B 为 20.724 ms，eager 为 52.235 ms；
这些是微基准而非端到端模型加速率。原始目录：
[eager](runs/cloud/20260914T091848Z-flash-eager)、
[必做版本](runs/cloud/20260914T092409Z-flash-triton-compiled-backward)、
[完整 Triton](runs/cloud/20260914T092847Z-flash-triton-full)。

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
原始证据：[双卡基准](runs/cloud/20260914T085559Z-communication-and-xl/20260914T091516Z-rpc)、
[distributed junit](runs/cloud/20260914T084950Z-distributed-tests/20260914T085228Z-rpc/000-distributed-repeat-0/junit.xml)。

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
实现：[sharded_optimizer.py](cs336_systems/sharded_optimizer.py)；源数据为双卡基准的 `010/011` 两项。

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
实现：[fsdp.py](cs336_systems/fsdp.py)；源数据为双卡基准 `012-xl-fsdp-adamw/result.json`。

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
数值来自已有 [tuned 完成状态快照](runs/status/20260914T093842Z-leaderboard-tuned-20260914T094503Z.json)，
并与随后只读取回的 [27 个非缓存文件](runs/cloud/20260914T093842Z-leaderboard-tuned/20260914T105045Z-storage-api) 交叉核对。
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
原始数值和检查配置见 [tuned validation](runs/status/20260914T093124Z-tuned-validation-20260914T093546Z.json)。

## 7. 证据完整性与交付边界

- 实验过程、配置来源、错误与修复记录见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) 和 [WORK_LOG.md](WORK_LOG.md)。
  最初的 non-leaf checker 失败、优化器验证中 BF16 严格代数 oracle 失败、saved hook 返回值错误等历史记录保留，
  不把这些失败计入成功结果。原始 OOM 也保留其 trace，而非替换为预计可运行的较小输入。
- “6 个 attention 测试 + 8 个 distributed 测试”是历史不同执行批次的通过记录，不等于最终源码做过一次全套 GPU 验收。
  CPU 测试中 CUDA skip 不是 GPU pass；补充 checkpoint/optimizer CPU 测试也不是完整多 GPU race 验证。
- 当前源码的原始 Gloo 分布式 suite 已在本机五个独立进程中各通过 8 项（共 40 次测试执行），无失败或跳过，
  源码前后 23 个文件哈希一致。见 [五轮 JUnit、日志与 manifest](runs/local/final-gloo-stability-localhost)。
  首次受限沙箱不能访问本机 TCP 端口的环境失败另行保留，不混入这五轮结果；没有租用 GPU。
- 模型/注意力表中 `mean_ms`、原始样本、OOM 和错误状态来自相应配置；profile 脚本外层成功、内部 summary 缺失时，
  不将其计作有效 GPU profiling。截图必须是真实工具界面，不能用绘制示意图冒充 Nsight/memory_viz。
- 所有 GPU 后续任务及验收产物单列在 [GPU_REMAINING_TASKS.md](GPU_REMAINING_TASKS.md)。
  Flash 240 项、通信 24 项、已有模型/内存/checkpoint/DDP 计时不整套重跑；先利用现有快照，必要补跑须另获预算许可。
- 本文的理论题均已给出答案；未完成实验的小问明示缺口。报告可按“现有实验阶段性完成、保留不足”交付，
  不能在文首或总结改写为“所有 handout deliverable 已齐”。
