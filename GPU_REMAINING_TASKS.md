# 剩余实验、取证与 GPU 清单

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
