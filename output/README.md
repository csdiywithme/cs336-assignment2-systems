# Assignment 2 已有证据交付目录

这是一份可追溯的阶段性交付：已经产生的数据、实现、解答和实验过程完成整理；GPU 缺项单独列出，不宣称全部课程要求已完成。2026-09-14 起云端预算暂停，未经新的明确预算授权，不运行任何 Modal GPU/CPU 作业。

## 从哪里开始

| 文件 | 内容 |
| --- | --- |
| `pdf/assignment2_existing_evidence_report.pdf` | 合并报告：摘要、六张统计图、逐题答案、全量数值表、过程和 GPU 待办 |
| `FULL_REPORT.md` | 同内容的可搜索、可编辑文本合并版 |
| `REPORT.md` | 较短的结论、方法与证据边界说明 |
| `../HANDOUT_ANSWERS.md` | 按原始 handout 题号组织的回答与未完成说明 |
| `tables/EXPERIMENT_TABLES.md` | 全矩阵结果、均值/标准差和来源标识 |
| `../EXPERIMENT_LOG.md` | 实验步骤、参数、命令、调试过程、计时口径和复现说明 |
| `../GPU_REMAINING_TASKS.md` | 所有剩余 GPU 任务及应优先取回既有结果的项目 |
| `data/all_attempts.json` | 去重结果；保留原始样本、配置、结果状态和来源 |
| `data/cost_ledger.json` | 可证实运行期的费用估计，不是总账单 |
| `figures/*.svg` | 从真实数值绘制的矢量统计图，不是 Nsight/MemoryViz 截图 |
| `memory/*.png` | 已有 pickle 在官方 MemoryViz 中显示的真实浏览器截图 |
| `memory/*.html` | 官方 PyTorch v2.11.0 查看页面；只读加载同目录 snapshots 下的已有数据 |
| `NSIGHT_EXISTING_ANALYSIS.md` | 已恢复 small/context256 kernel 分析与已有单层内存 trace 的局限 |
| `profiling/` | 对已有 SQLite 的本地重新归因结果与原始文件哈希 |
| `VERIFICATION.md` | 数据审计、CPU 测试、逐页 PDF 检查与验证边界 |
| `bundles/assignment2_code.zip` | 当前代码与说明；不包含 Assignment 1 用户工作区 |
| `bundles/assignment2_existing_evidence.zip` | 代码、当前本地原始记录、报告、原始 handout 的合并证据包 |
| `bundles/bundle_checksums.json` | 两个 ZIP 的 SHA256 与大小 |

原始证据保留于仓库 `runs/cloud`、`runs/status`、`runs/local`；归档文件带独立清单与 SHA256。证据包只涵盖当前本地文件，不假装包含尚未取回的远程大文件。

## 本地重新生成

从 `cs336-assignment2-systems` 目录执行。以下三个脚本只读取/整理本地文件，不启动 CUDA 或 Modal。

```sh
../cs336-assignment1-basics/.venv/bin/python scripts/summarize_results.py
/Users/simida/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_report.py
../cs336-assignment1-basics/.venv/bin/python scripts/package_artifacts.py
```

报告生成需要 ReportLab 和中文 TrueType 字体。当前构建使用 macOS 的 `STHeiti Light.ttc`，字体嵌入 PDF；换到其他机器时可修改 `scripts/build_report.py` 的 `register_font()` 使用当地安装的中文字体。没有安装字体或 ReportLab 时，先准备本地依赖，而不是启动云端容器。

## 如何理解结果

- `status=ok` 指某次实验完成，不等于它满足 handout 的所有验收项。Nsight 程序成功返回也可能没有捕获 GPU kernel。
- `backward` 模式的顶层总时间是前向+loss+反向，不是单独反向时间；`cuda_event_stages` 才是直接阶段计时。
- OOM 是实测结果；失败时的内存不是成功完成该配置的峰值。历史 harness 错误与修正后结果共同保留。
- 多卡均值按每轮最慢 rank 汇总；不要把 rank 0 值、各 rank 均值的最大值、每轮最慢值的均值混用。
- 全 Triton backward 的 80 组是扩展；必做 Flash 比较矩阵是 PyTorch 80 组与 Triton-forward/compiled-backward 80 组。
- 8B 的 6.76 秒是本项目候选的实测全步均值，不代表课程官方验证或排行榜成绩。
- 文件中保留了历史 Modal 命令，仅为复现记录；预算暂停期间不要执行它们。也不需要为整理已有快照开 GPU。

已有显存图的浏览方式：运行 `scripts/build_memory_views.py`，然后在本机用 `python -m http.server 8842 --bind 127.0.0.1 --directory output` 启动临时服务，访问 `http://127.0.0.1:8842/memory/000-memory-xl-s128-fp32-forward.html` 等页面。图的 JavaScript 来自官方 PyTorch v2.11.0 release，快照由本地服务读取；不需要上传数据或运行 CUDA。查看结束关闭本地服务即可。

Nsight 的 `.nsys-rep` 可用 [NVIDIA 官方 macOS Host 查看器](https://developer.nvidia.com/nsight-systems/get-started) 在没有 NVIDIA GPU 的本机打开。当前机器未安装该 GUI，本次完成的是 SQLite 离线分析，不冒充已取得所有 Nsight GUI 截图；单层内存题缺少释放事件，仍须按 GPU 待办补充有效采集。
