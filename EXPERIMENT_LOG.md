# Assignment 2 实验与复现记录

## 记录范围与预算状态

本记录整理 2026-09-14 的既有执行过程，包含 Flash 完成记录、内存/Nsight 文件，以及最后恢复的 10B compile、Flash 两项和 tuned leaderboard 的结果/日志/源码环境归档。时间戳中的 `Z` 是 UTC，北京时间加 8 小时。本日志编写读取本地代码、原始 PDF、请求记录、日志和 JSON。主任务通过既有调用/Volume 的控制面 API 恢复结果，在本机修复分析器、运行 CPU 回归并生成截图，没有启动 GPU、云端 CPU、下载 RPC 或调参任务。

**预算暂停继续生效：没有用户新的明确预算授权，不启动任何 GPU 或 CPU 云端任务。** 包括“只读”的远程结果 RPC、云端 GUI 截图和容器式文件提取；它们虽然可能不使用 GPU，仍会启动计费计算。保留已有 Volume，不能擅自删除实验数据。历史检查中相关 Modal 应用已经 stopped、任务数为零，这只说明当时没有运行计算，不代表零存储费或账单结清。

剩余工作及逐项验收条件见 [GPU_REMAINING_TASKS.md](GPU_REMAINING_TASKS.md)。本日志不是已完成作业的声明，也不是费用发票。

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

`runs/cloud/20260914T093845Z-profile-probe-cuda-software-trace/20260914T101417Z-storage-api/` 中的有效 SQLite/report 已在本机完成统计。3637 个 launch 与 3637 个 kernel 一一匹配，measurement 内没有未归属或重复归属阶段，累计 GPU kernel 时间 29.847779ms，backward 为 16.380591ms。这里是 kernel 累计时间，不能当作同步墙钟时间；同一 profile 的完整 Python 时间约 107.144ms，配对 plain 为 36.912ms，显示 profiler 有明显扰动。完整来源、修复原因、五个子问题和限制见 [output/NSIGHT_EXISTING_ANALYSIS.md](output/NSIGHT_EXISTING_ANALYSIS.md)。该分析没有冒充 Nsight GUI 截图；若需此组截图，可本地打开现有 `.nsys-rep`，无需 GPU。

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
