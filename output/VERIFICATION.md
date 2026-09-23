# 交付验证与证据边界

本次工作只读恢复已经结束的实验产物，并在本地整理、分析、测试和生成文件；未新增 Modal GPU/CPU 计算任务，未提交课程作业或 leaderboard。

## 数据与来源

- 448 条实验记录：431 ok、13 OOM、4 历史失败；不把记录数等同于最终代码通过的测试数。
- Flash 240/240 配置均已有结果；通信 24/24 配置和原始各 rank samples 保留。
- 635 个带样本的统计对象重算审计无异常；跨 rank 聚合一致，Flash 的三项计时均为有限正值。
- 最后恢复的 63 个 Flash/10B compiled/tuned 文件均通过长度与 SHA256 检查。10B 与 tuned 的 13 个实验源码文件分别匹配各自 environment provenance。tuned 可再生缓存未取回；证据包不宣称是整个远端 Volume 的镜像。
- 五份内存 pickle 已恢复；两张主要 MemoryViz 截图和两张细节截图来自官方查看器的真实界面，未改绘。
- 有效 small/context256 Nsight trace 的 3637 个 kernels 均关联到本进程的 CUDA launch，阶段无漏归属或重复归属。只验证了这一组，不把缺 kernel 的旧 trace 算成成功 GPU profile。
- 数据明细及生成文件哈希见 `data/coverage_snapshot.json`、`data/sample_audit.json`、`data/generated_manifest.json`。费用文件只是有依据的运行期估价小计，不是账单。

## 本地测试

完整本地 suite：18 collected、14 passed、4 CUDA skipped，12.856 秒。JUnit：`../runs/local/report-final-cpu-tests.xml`。其中包含两个 Nsight 解析器测试，不能在总数之外重复相加。

原始 DDP/FSDP/sharded-optimizer suite 固定使用 Gloo。本机五个独立进程中每轮 8 passed，零失败、错误或跳过，共 40 次测试执行，墙钟合计 70.8426 秒。23 个源码文件在运行前后哈希一致；JUnit、stdout 和源码清单位于 `../runs/local/final-gloo-stability-localhost/`。

最初受限沙箱无法连接本机 TCP 端口，产生 8 项环境失败；该记录独立保存在 `../runs/local/final-gloo-stability/`。获准使用本机 loopback 后才执行上述五轮，不删除或混淆失败记录。

这些结果只证明本地 CPU/Gloo 路径；不证明 CUDA/NCCL 或 tuned 候选的严格数值验收通过。剩余验证要求见 `../GPU_REMAINING_TASKS.md` 的 V1，额外 NCCL 稳定性重复属于可选项。

报告、取证与打包脚本均已通过本地语法检查。查看显存快照使用的临时本地 HTTP 服务已关闭。

## PDF 版式核验

最终 PDF 共 59 页，全部用 Poppler 以 100 dpi 渲染。已人工检查所有页面的联系表，并单独放大检查代码、密集数值表和 GPU 清单；自动检查没有发现超出页面的文字或失效的本地 Markdown 链接。

检查中修正了代码块缩进。最终版本重渲染后只有第 9、16 页的 PNG 发生变化，两页均重新逐页检查；其他 57 页的 PNG 与已经检查的版本字节完全一致。最终 SHA256 为 `a4da82521b1d247dc95ac61e2a65f950cb4eb5481428c7cbc3c04ee21ea620ad`。自动检查和人工核验记录分开保存在 `verification/`，不把机器检查冒充人工阅图。

临时渲染图不是交付证据的替代品，可以使用 `scripts/verify_report.py` 从 PDF 重新生成。文档中的原始截图仍完整保存在 `memory/`。
