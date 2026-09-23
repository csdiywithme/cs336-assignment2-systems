# Assignment 2 implementation and experiment journal

最新补充见文末的归档与本地五轮 Gloo 稳定性记录。较早段落按时间保留，
其 438 条记录、$14.02 以及六组 profile pending 等是当时状态；最新为 448 条记录、
27/27 个已知摘要的 $16.854697 部分估计，且模型 profile 剩五组，memory_viz、
E1/E3 归档和课程 CPU/Gloo 五轮稳定性已完成；CUDA/NCCL 严格验证仍未完成。

## Budget pause — 2026-09-14 17:44 Asia/Shanghai

The user reported that funds are nearly exhausted. This supersedes the earlier
open-ended authorization to use Modal: do not launch more cloud GPU or CPU jobs,
profiling retries, tuning runs, or artifact-rendering containers without explicit
new budget approval. Local analysis/reporting and control-plane reads of existing
results remain available. Preserve all existing evidence; do not delete cloud
Volumes to reduce storage fees without the user's approval.

Modal's application listing was checked after this message. All listed
`cs336-assignment2-20260914` and `cs336-assignment2-artifacts` applications were
already stopped, with zero tasks; no cancellation was necessary. This confirms
no running compute for these applications, not zero storage fees or an invoice.
The local ledger at 09:44 UTC contains 22 downloaded completed-job summaries and
a $14.0216 runtime-only estimate. It excludes startup/build/storage, CPU artifact
jobs, and jobs whose summaries have not been downloaded. It is not total billed
spend. Local result coverage is 438 records: 422 successful, 13 OOM, and 3 preserved
historical failures. Successful records are experiments, not 422 unit tests.

The implementation and most experiments are present, but the final report/PDF,
verified profiler screenshots, and final submission packaging are not complete.
Do not label missing profiler evidence as measured or claim the assignment fully
complete. Finish what can be supported by existing data locally; report the gaps.

## Scope and authority — 2026-09-14

The user requested completion of Assignment 2 with detailed experimental steps,
process records, raw data, and answers to the handout. Modal B200 or B300 GPUs are
authorized. B200 is the primary measurement platform because the handout specifies
it. No course/leaderboard submission or publication is authorized.

- Handout: Spring 2026, PDF 26.1.3; starter package/test revision 26.1.4.
- Source baseline: the staff `cs336-basics` bundled with this assignment.
- Assignment 1 is left untouched. Its existing Modal connection was verified as
  workspace `csdiywithme`; credentials are never copied into experiment artifacts.
- Existing changes preserved: `.gitignore`, deleted `AGENTS.md`, deleted `CLAUDE.md`.
- No tests or GPU experiments have passed yet at this initial checkpoint.

## Evidence conventions

Each cloud job has a unique run ID, exact command, source SHA256 manifest,
environment record, unedited stdout/stderr, exit status, timeout, and raw samples.
Failures/OOMs are retained. No omitted case is labelled measured. Cold compilation,
warmup, steady-state compute, profiler overhead, and communication are distinguished.
Modal jobs use bounded timeouts, no automatic retries, and persistent artifact
storage. Raw records are authoritative; report tables are generated from them.

## Work sequence

1. Implement tiled PyTorch and Triton attention with recomputation backward.
2. Implement naive, flat, overlapping DDP; optimizer sharding; FSDP.
3. Run original tests and targeted gradient/sharding regression tests.
4. Run bounded single-B200 environment and kernel probes.
5. Run handout benchmark matrices, memory/checkpointing experiments, Nsight traces.
6. Run 2/4/6-GPU communication and two-GPU DDP/optimizer/FSDP experiments.
7. Write all theoretical answers; generate experiment tables and figures.
8. Attempt the full leaderboard configuration, verify, render writeup, package.

## 2026-09-14: initial inspection

The systems module was empty. Original tests require two FlashAttention classes,
overlapping DDP, sharded optimizer, FSDP and parameter reconstruction. Tests also
cover tied weights/frozen parameters, mixed-precision FSDP, and synchronized norms.
The handout's Triton adapter name is misspelled in one place; actual adapter names
in `tests/adapters.py` govern. Mandatory backward may use PyTorch + torch.compile;
the adapter docstring's reference to mandatory Triton backward is stale.

## 2026-09-14: correctness and first cloud jobs

- Original local suite: 10 passed, 4 CUDA-only skipped (13.74 s). An earlier
  restricted attempt failed at localhost socket creation before distributed code
  ran. Both JUnit files are preserved under `runs/local/initial`.
- `20260914T082016Z-probe`: original attention suite passed all 6 tests on B200;
  the extended checker failed because its FP32 reference aliased the original
  tensor, making a later clone non-leaf. Fixed checker with detached leaf copies;
  this was a harness failure, not evidence of kernel failure. Runtime-only cost
  estimate: $0.0789, not a billing invoice.
- Corrected checker rerun: `20260914T084725Z-probe-leaf-fix`.
- Source uploads initially included `__pycache__`; concurrent imports changed a
  `.pyc` during mounting, aborting one model job before launch. Mount filters now
  exclude generated bytecode. No GPU measurements were produced by that attempt.
- Model baseline uses staff model and staff Assignment 1 AdamW. Forward-only
  measurements omit loss; backward/train modes include FP32 cross entropy. Autograd
  remains enabled for forward residual-memory accounting. TF32 is disabled for
  the full-model FP32 baseline. Triton FP32 dots use tf32x3 for accurate tensor-core
  products; this distinction is disclosed in comparisons.
- Volume block downloads encountered a proxy TLS reset. The incomplete download
  is preserved, and a read-only CPU RPC fallback successfully recovered 20 text
  artifacts. This fallback does not launch GPUs. Batch reads avoid Modal's app
  creation rate limit observed when creating several read-only apps concurrently.
- Nsight Systems CLI 2026.5.1.161 installed from NVIDIA's official devtools repo.
  CUDA/NVTX/memory trace probe submitted; successful installation alone is not
  treated as successful profiling.

## 2026-09-14：预算暂停后的本地证据审计与复现文档

此阶段只读取本地源码、原始 PDF、dispatch/request、已下载的 cloud JSON 与
`runs/status` 快照，未调用远程实验/API、未启动 GPU 或 CPU 云容器、未重跑测试。
按原题与实际测量字段新增 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) 和
[GPU_REMAINING_TASKS.md](GPU_REMAINING_TASKS.md)，记录执行环境、真实命令来源、
随机种子、warmup、计时边界、失败修复、source provenance、费用口径及逐项验收。

- 模型基线是 75 项已尝试、70 项成功、5 项 10B train OOM；不能因外层进程
  `status=ok` 写成 75 个成功测量。small/medium 的 30 个结果都有十个总时间样本，
  但没有独立阶段 CUDA events；large/XL/10B 成功的 40 项有阶段数据。
- 原 `models-compile` 的十项 forward/backward 都有成功证据：8 项已下载文本，
  10B 两项在 `20260914T094503Z` 返回快照中各有十个样本。第 17 页要求的五个
  compiled full-train case 不在旧计划内，仍缺；full backward 不能代替 optimizer。
- Flash 已下载 238/240 项，全部内层 ok，且每项都有 forward/backward/end-to-end
  mean：eager 80、mandatory Triton 78、可选 full Triton 80。mandatory Triton 的
  BF16 seq65536 dim64/128 未下载、无 final summary，需先恢复结果，不断言没跑完。
- 2/4/6 rank 的 Gloo/NCCL 通信 24/24 项已完成，每个 rank 都有 20 samples；
  五个两卡 XL 策略每个 rank 都有 10 samples。它们不在重跑清单。
- `distributed-repeat-0` 的历史 JUnit 为 8 passed，实际请求仅一轮。五轮稳定性是
  题目建议；最终修订版 GPU/NCCL 验证与可选五轮重复单独列出。
- 最新 `cuda-sw` Nsight probe 的返回快照确认计算完成、trace_created=true，
  但尚未检查其 SQLite 的 GPU kernel 数据。先取回检验，再决定是否需要补六个
  model profile、两个 DDP profile、一个 FSDP profile；不能以文件存在宣称有效。
- memory_viz 两张截图优先复用既有成功生成的 pickle，不需要重跑 GPU；Nsight
  saved-block 也先检查已有 allocation trace，只有证据不足才补采。
- 费用沿用 09:44:46 UTC 的部分记录：22 个 downloaded job summary 的 runtime-only
  估计 $14.021608，不是总账单，不包含未下载 summary、启动/构建/存储及 CPU
  artifact jobs。不刷新价格、不承诺后续费用、不删除云存储。

预算暂停仍优先于早先开放式授权：没有用户新的明确同意，不启动任何 GPU 或 CPU
云端作业，也不把远程只读 RPC 当作免费本地操作。可选 leaderboard/调参继续暂停。

### 同日 10:06 UTC：Flash 完成证据已恢复，无需补跑

主任务通过既有调用的控制面 API 读取已完成的返回值，没有启动新的 GPU 或 CPU
云端任务。新增 `runs/status/20260914T092409Z-flash-triton-compiled-backward-20260914T100623Z.json`
确认 mandatory Triton 80/80 全部内层 ok，并且每项都有 forward/backward/end-to-end
mean，包括原来 cloud 快照没有覆盖的 BF16 seq65536 dim64/128。因此全 Flash
矩阵已测 240/240（必需 160 + 可选 80），上述 238/240 是较早下载范围的历史状态。
复现日志和剩余清单已更新，删除 Flash 条件重跑项；后续只有完整 artifacts 归档，
不需要重新计算。原 09:44 UTC 的部分费用估计仍保留其历史日期，不能当作最新总账单。

## 2026-09-14：内存图片、有效 Nsight 证据与本地统计修复

主任务通过既有 Volume 的存储 API 恢复文件，全程没有新 GPU 或 CPU 云端容器。
五个 memory pickle 保存于
`runs/cloud/20260914T085600Z-memory-and-checkpoint/20260914T100931Z-storage-api/`，
在本地 HTML 中使用官方 PyTorch v2.11.0 MemoryViz 展示。两张主要截图为
`output/memory/forward-memoryviz.png` 和 `output/memory/train-memoryviz.png`，
另有 detail10/detail9 图。它们均为 XL seq128 FP32，排除两次 warmup 后包含
2 个正式计时步和 1 个额外内存记录步；横轴是分配/释放事件序列，不能标为毫秒。
E4 已完成，不再需要 GPU 或等待 pickle；文件来源和哈希已保存。

最新 CUDA software trace probe 的 SQLite 与 `.nsys-rep` 恢复至
`runs/cloud/20260914T093845Z-profile-probe-cuda-software-trace/20260914T101417Z-storage-api/`。
本地修复 parser：以 `(globalPid, correlationId)` 关联 API/kernel，顶层 NVTX 阶段
允许同进程 autograd worker 的 launch，避免旧同线程条件遗漏 backward。3637 个 kernel
与 launch 全部匹配，阶段零遗漏/重复，kernel 累计 29.847779ms，backward 为
16.380591ms；旧约 0.0013ms 是解析缺陷，不是新的性能提升。原始库保持只读。
两项合成 SQLite 回归测试 passed，完整五个子问题和限制记录在
[output/NSIGHT_EXISTING_ANALYSIS.md](output/NSIGHT_EXISTING_ANALYSIS.md)。它不是 GUI 截图，
如需该组 GUI 图只需本地打开 report。E2 完成，small seq256 可复用，剩余 model profile
为 small seq1024/4096、XL seq256/512/1024，共五组；DDP 两组、FSDP 一组仍缺。
E5 的既有单 block allocation trace 仍优先取证，不能未经检查就增加 GPU 任务。

最新本地 `runs/local/report-final-cpu-tests.xml` 已重跑更新为 18 collected、14 passed、
4 CUDA skipped，12.856 秒（约 12.86 秒），其中已包含两个 Nsight parser 测试。
较早一轮的 16 collected、12 passed、4 skipped、15.638 秒仅作为历史过程，
不再是该 XML 的当前结果；单独 parser 测试也不重复计入总数。
这些 CPU 结果不替代当前修订版 GPU/NCCL 回归。
`output/data/coverage_snapshot.json` 在 10:15:19 UTC 统计 447 条实验记录：
431 ok、13 OOM、3 failed；635 个样本统计对象零不一致、跨 rank 零不一致。
最新费用 ledger 覆盖 26/27 个已知 run，runtime-only 小计 $16.831832；
`20260914T091103Z-optimized-validation` 缺少完成摘要、费用未知且未计入。
小计仍不包含构建/冷启动/存储/CPU artifact 等费用，不是完整花费或账单。

复现日志和 GPU 清单同步缩减：Flash 240/240 与通信 24/24 不重跑，memory_viz 完成，
有效单卡 Nsight probe 不再补跑；余下配置继续受预算暂停约束，没有自动提交或调参。

### 同日 10:26 UTC：最后一个历史失败摘要恢复

`runs/status/20260914T091103Z-optimized-validation-20260914T102152Z.json` 确认 R11
是历史 chunked/reference scalar 验证失败，absolute difference 0.005754 超出 1e-4；
耗时 11.531143 秒，runtime-only 估计 $0.022864873。后续 fp32-oracle 成功另存，
不能覆盖此失败。重新生成的本地 coverage 为 448 项：431 ok、13 OOM、4 failed；
635 个统计对象依然零不一致。27/27 个已知实验摘要均已取得，费用小计更新为
$16.85469719214138；仍不包含启动/构建/存储/CPU artifact 等费用，不是完整账单。

另外，单 block 的 allocation SQLite/report 已恢复至
`runs/cloud/20260914T092252Z-profiles/20260914T102148Z-storage-api/012-profile-saved-block-xl-s2048/`。
E5 的只读分析已完成：1357 条内存事件中 35 次动态分配、1322 条 Static 分配，
动态释放为 0。旧 trace 可作为部分 allocation/NVTX 截图，但缓存 allocator 的
segment 申请曲线不足以推导 residual 释放与梯度净增量。审计保存于
`output/profiling/saved-block/allocation_summary.json`，说明并入
`output/NSIGHT_EXISTING_ANALYSIS.md`。G6 因此收敛为一个有界单 B200、单 XL block
的无缓存内存诊断，显式标记 forward/backward；不重跑完整 XL 或内存矩阵，仍需新预算授权。
恢复和解析均没有新增 GPU 或 CPU 云任务。

### 最终验证清单修正：Gloo 不等于 NCCL

源代码核查确认原始 DDP/FSDP/sharded-optimizer tests 固定 `backend="gloo"`；
历史两卡容器中的 8 passed 不能据此标记 NCCL 数值正确性通过。原始 tests 保持不变，
V1 另列 2×B200 的小模型 NCCL 输出/同步梯度/AdamW 更新与 reference 回归，
只验证数值，不重跑 XL 或通信性能矩阵；原始 Gloo tests 及其五轮建议可在本地 CPU 完成。

V1 还明确增加当前 tuned Flash 的 O/dQ/dK/dV，以及 optimized-model 的 logits/loss/
参数梯度严格检查，`rtol=atol=1e-2`；现有 output rtol=0.03、model-gradient rtol=0.1
的较松验证不算满足。原始 CUDA、独立 NCCL、tuned 严格检查分别保存 backend/device、
容差、JUnit 和 source hash；每部分上限 600 秒，云 job 上限 1800 秒，无自动重试。
这次只更新文档，未运行这些 GPU 检查，预算暂停继续生效。

### 最后归档完成：E1/E3 没有用户待办

通过已有存储 API 恢复三批文件，没有启动 GPU/CPU 云端容器：

- 10B compile：`runs/cloud/20260914T093858Z-models-compile-10b/20260914T104956Z-storage-api`，
  27 文件、124248 bytes。
- Flash 最后两项及 summary：
  `runs/cloud/20260914T092409Z-flash-triton-compiled-backward/20260914T105002Z-storage-api`，
  9 文件、189824 bytes；前 78 项/source/environment 原快照已有。
- tuned leaderboard：`runs/cloud/20260914T093842Z-leaderboard-tuned/20260914T105045Z-storage-api`，
  27 个非缓存文件、161806 bytes；可再生成的编译 cache 不归档。

63 个文件全部通过长度与 SHA256 检查，10B compile 与 tuned 各 13 个源码文件
均匹配 environment provenance。E1/E3 的结果、日志、源码环境归档完成；新旧
cloud 快照合并后 Flash case 归档也达 240/240，较早 238/240 下载状态不再是当前缺口。
最终核对确认 mandatory Triton 的 80 组 result/status/stdout、13 个源码与 environment
均齐；只有最终 `artifact_inventory.json` 未在指定下载范围，full-inventory 标志仍 false。
这不是测量或日志缺失，不把它重新列为用户必须处理的实验任务。

### 本地五轮 CPU/Gloo 稳定性完成

`runs/local/final-gloo-stability-localhost/` 完成五个独立 pytest 进程，运行原始
DDP/FSDP/sharded-optimizer suite，各轮均 8 passed、0 failure/error/skip，
共 40 次测试执行。逐轮 wall time 为 14.0135、13.5914、16.8382、13.0814、
13.3180 秒，总 70.8426 秒；对应 JUnit 时间为 12.561、12.560、15.816、
12.060、12.298 秒，两种时间边界分别保留。

`repeat-1..5.xml/.log`、`results.json` 与 `manifest.json` 保存命令和原始结果，
23 个 `cs336_systems/tests` 源码文件的前后哈希不变，汇总指纹为
`f1cdc2a4de5d952e26a3f8def0ff0c2ad3d9aea63daf82196f9300156c404d37`。
原受限尝试 `runs/local/final-gloo-stability/` 首轮 8 个失败由 localhost:12390
套接字 EPERM 引起，停在进程组建立阶段；原日志完整保留。随后仅放开本地 loopback
执行上述新目录的五轮，无云调用、无 GPU、无新的 CPU 云端容器。

V2 的课程 CPU/Gloo 稳定性建议已完成。这 40 次执行不是 40 个独立用例，
不加入云端 448 个实验 case，也不能算作 CUDA/NCCL 验证。V1 的原 CUDA、
独立小模型 NCCL 和 tuned 路径 `rtol=atol=1e-2` 验证继续等待新预算授权；
NCCL 五轮额外稳定性仅是可选扩展，不默认运行。三份复现/剩余清单至此冻结供 PDF 生成。
