"""Build the evidence-only Chinese report locally. Never contacts Modal or uses CUDA.

Run summarize_results.py first. Uses ReportLab and the bundled Python runtime.
The generated charts are aggregate data plots, NOT profiler screenshots.
"""
from __future__ import annotations

import datetime
import html
import json
import math
import os
from pathlib import Path
import re
import statistics

from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String, Circle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table,
    TableStyle, PageBreak, KeepTogether, Preformatted, CondPageBreak,
    Image,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
INK = colors.HexColor("#142E41")
MUTED = colors.HexColor("#566B7A")
TEAL = colors.HexColor("#087F8C")
BLUE = colors.HexColor("#3769AA")
ORANGE = colors.HexColor("#BC633C")
PALE = colors.HexColor("#EEF4F7")
COLORS = [TEAL, BLUE, ORANGE, colors.HexColor("#9067A7")]
FONT = "A2Heiti"


def register_font():
    font = Path("/System/Library/Fonts/STHeiti Light.ttc")
    if not font.exists():
        raise RuntimeError("Report font unavailable: install a Chinese TrueType font and update register_font()")
    pdfmetrics.registerFont(TTFont(FONT, str(font)))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT, italic=FONT, boldItalic=FONT)


def text(d, x, y, s, size=8, color=MUTED, anchor="start"):
    d.add(String(x, y, str(s), fontName=FONT, fontSize=size, fillColor=color, textAnchor=anchor))


def line_chart(title, xs, series, y_label, labels=None, logarithmic=True, width=490, height=290):
    d = Drawing(width, height)
    left, bottom, top, right = 52, 64, height - 54, width - 14
    text(d, 0, height - 16, title, 12, INK)
    vals = [v for _, ys, _ in series for v in ys if v is not None and v > 0]
    if not vals:
        text(d, left, top / 2, "没有可用的实测数据")
        return d
    if logarithmic:
        lo, hi = math.floor(math.log10(min(vals))), math.ceil(math.log10(max(vals)))
        if hi == lo:
            hi += 1
        ticks = [10 ** k for k in range(lo, hi + 1)]
        ycoord = lambda y: bottom + (math.log10(y) - lo) / (hi - lo) * (top - bottom)
    else:
        maximum = max(vals) * 1.1
        ticks = [maximum * i / 4 for i in range(5)]
        ycoord = lambda y: bottom + y / maximum * (top - bottom)
    xcoord = lambda i: left + i * (right - left) / max(1, len(xs) - 1)
    for tick in ticks:
        yy = ycoord(tick)
        d.add(Line(left, yy, right, yy, strokeColor=colors.HexColor("#DCE5E9"), strokeWidth=.5))
        text(d, left - 7, yy - 3, f"{tick:g}", 8, anchor="end")
    d.add(Line(left, bottom, left, top, strokeColor=MUTED, strokeWidth=.5))
    for i, x in enumerate(labels or xs):
        text(d, xcoord(i), bottom - 16, x, 8, anchor="middle")
    text(d, left, top + 12, y_label + (" (log10)" if logarithmic else ""), 8)
    for j, (name, ys, color) in enumerate(series):
        points = []
        for i, y in enumerate(ys):
            if y is None or y <= 0:
                if len(points) >= 4:
                    d.add(PolyLine(points, strokeColor=color, strokeWidth=1.4))
                points = []
                continue
            xx, yy = xcoord(i), ycoord(y)
            points.extend([xx, yy])
            d.add(Circle(xx, yy, 2.2, fillColor=color, strokeColor=color))
        if len(points) >= 4:
            d.add(PolyLine(points, strokeColor=color, strokeWidth=1.4))
        lx, ly = 10 + (j % 2) * 244, 16 + (j // 2) * 13
        d.add(Line(lx, ly + 3, lx + 18, ly + 3, strokeColor=color, strokeWidth=2))
        text(d, lx + 23, ly, name, 8)
    return d


def bar_chart(title, labels, values, unit, height=265, width=490, color=TEAL):
    d = Drawing(width, height)
    text(d, 0, height - 16, title, 12, INK)
    left, right, bottom, top = 47, width - 12, 50, height - 48
    vmax = max(v for v in values if v is not None) * 1.16
    for i in range(5):
        v = vmax * i / 4
        y = bottom + i / 4 * (top - bottom)
        d.add(Line(left, y, right, y, strokeColor=colors.HexColor("#DCE5E9"), strokeWidth=.5))
        text(d, left - 6, y - 3, f"{v:.0f}", 8, anchor="end")
    text(d, left, top + 12, unit, 8)
    cell = (right - left) / len(labels)
    for i, (label, value) in enumerate(zip(labels, values)):
        center = left + cell * (i + .5)
        for k, part in enumerate(str(label).split("\n")):
            text(d, center, bottom - 14 - 11 * k, part, 8, anchor="middle")
        if value is None:
            text(d, center, bottom + 8, "OOM", 9, ORANGE, "middle")
            continue
        h = value / vmax * (top - bottom)
        d.add(Rect(center - cell * .3, bottom, cell * .6, h, fillColor=color, strokeColor=None))
        text(d, center, bottom + h + 7, f"{value:.2f}", 8, INK, "middle")
    return d


def select(rows, **criteria):
    candidates = [r for r in rows if all(r.get("config", {}).get(k) == v for k, v in criteria.items())]
    return sorted(candidates, key=lambda r: (r.get("run_id", ""), r.get("source", "")))[-1] if candidates else None


def mean(r, field=None):
    if not r or r.get("status") != "ok":
        return None
    m = r.get("measurements", {})
    return (m.get(field, {}) if field else m).get("mean_ms")


def figures(rows):
    result = {}
    sizes = ["small", "medium", "large", "xl", "10B"]
    series = []
    for j, (dtype, mode) in enumerate([("fp32", "forward"), ("bf16", "forward"),
                                      ("fp32", "backward"), ("bf16", "backward")]):
        ys = [mean(select(rows, name=f"{size}-{dtype}-{mode}-w5")) for size in sizes]
        series.append((f"{dtype.upper()} / " + ("前向" if mode == "forward" else "前向+loss+反向"), ys, COLORS[j]))
    result["model_timings"] = line_chart("五种模型的稳态时间", sizes, series, "ms / step")
    segments = [1, 2, 4, 8, 16, 32, 64]
    values = []
    for n in segments:
        r = select(rows, name=f"checkpoint-{n}")
        values.append(r["measurements"]["memory"]["peak_allocated_bytes"] / 2 ** 30 if r and r["status"] == "ok" else None)
    result["checkpoint_memory"] = bar_chart("Checkpoint 峰值显存：xl / batch 4 / context 2048", segments, values, "GiB allocated")
    steps = []
    for strategy, opt in [("naive", "adamw"), ("flat", "adamw"), ("overlap", "adamw"), ("overlap", "sharded"), ("fsdp", "adamw")]:
        r = select(rows, name=f"xl-{strategy}-{opt}")
        ranks = r["measurements"]["ranks"] if r else []
        steps.append(statistics.mean(max(rank["step"]["samples_ms"][i] for rank in ranks)
                                     for i in range(10)) if ranks else None)
    result["distributed_steps"] = bar_chart("双 B200 / xl：完整训练步时间", ["naive", "flat", "overlap", "overlap\n+sharded", "FSDP"], steps, "ms / step")
    for backend in ("gloo", "nccl"):
        series = []
        for j, n in enumerate([2, 4, 6]):
            ys = []
            for mb in [1, 10, 100, 1000]:
                r = select(rows, kind="allreduce", world=n, backend=backend, bytes=mb * 1000000)
                ys.append(mean(r, "rank_max"))
            series.append((f"{n} ranks", ys, COLORS[j]))
        result[f"allreduce_{backend}"] = line_chart(f"All-reduce / {backend.upper()} / 最慢 rank", [1, 10, 100, 1000], series, "ms", ["1 MB", "10 MB", "100 MB", "1 GB"])
    seqs = [2 ** i for i in range(7, 17)]
    series = []
    for j, impl in enumerate(["eager", "triton", "triton-full"]):
        ys = [mean(select(rows, kind="attention", implementation=impl, dtype="fp32", dim=64, seq=s, batch=1), "end_to_end") for s in seqs]
        series.append((impl, ys, COLORS[j]))
    result["flash_scaling"] = line_chart("Causal attention：FP32 / batch 1 / D=64", seqs, series, "前向+反向 ms", [str(s) for s in seqs], height=300)
    folder = OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for name, drawing in result.items():
        renderSVG.drawToFile(drawing, str(folder / (name + ".svg")))
    for name in ("forward", "train"):
        screenshot = OUT / "memory" / (name + "-memoryviz.png")
        if screenshot.exists():
            from PIL import Image as PILImage
            with PILImage.open(screenshot) as im:
                w, h = im.size
            result[f"memoryviz_{name}"] = Image(str(screenshot), width=340, height=340 * h / w)
    return result


def overview(rows):
    tuned = select(rows, kind="leaderboard", tuned=True)
    leader = mean(tuned, "rank_max_step")
    initial = mean(select(rows, name="leaderboard-8b-32768"), "rank_max_step")
    improvement = (1 - leader / initial) * 100 if leader and initial else None
    newest = max((r.get("started_utc", "") for r in rows), default="unknown")
    valid = sum(r.get("status") == "ok" for r in rows)
    return f"""# CS336 Assignment 2：已有实验与复现报告

## 阅读说明与完成边界

本报告整理截至 2026-09-14 已取得的真实实验记录。文档、汇总和图表在本地生成，不启动 Modal GPU 或 CPU 容器。它是“已有证据完成版”，不是“全部实验通过”或“可无缺项提交”的声明。待补项目见最后的 GPU 任务清单。

数据集中共有 {len(rows)} 条去重的实验结果记录，其中 {valid} 条状态为 ok。实验记录数不等于单元测试数；历史失败和 OOM 单独保留。最新实验开始时间（UTC）：{newest}。Flash 最后两格、10B compiled 及 tuned 候选的结果与所需环境/源码归档已只读补齐；证据包不是全部远程文件及编译缓存的备份。

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

[[FIG:model_timings]]

图 1：稳态 wall-clock 均值，对数纵轴。前向+反向曲线包含 loss，不包含优化器。所有 10 个样本及均值、标准差见数据表。

BF16 对大模型的加速更明显，但不能据此断言训练显存减半：参数、梯度和 Adam 状态仍可能为 FP32；autocast 的权重转换缓存也会增加短上下文前向的活跃内存。小/中模型早期记录没有直接的 CUDA-event 阶段计时，报告保留组合测量而不以两个独立实验的均值相减伪装成直接测量。

## 2. 显存与 checkpoint

xl 已运行 context=128/2048、FP32/BF16、forward/train 的 8 个配置。能完成的配置记录 allocated/reserved 及峰值；OOM 配置记录异常与失败时内存，不把失败点内存当成可成功完成训练的峰值。五份已有 pickle 已经通过只读存储接口取回，两张 Active Memory Timeline 使用官方 PyTorch v2.11.0 MemoryViz 在本地浏览器打开并直接截图，没有重新运行 GPU。

[[FIG:memoryviz_forward]]

显存图 A：xl / context 128 / FP32 / forward。原始快照含两个测量步和随后一个记录阶段信息的额外步，因此图上出现重复的上升、释放形态；横轴是分配事件序列，不能当作毫秒时间轴。

[[FIG:memoryviz_train]]

显存图 B：相同配置的完整训练步。图中保留了 MemoryViz 的选择项和 Detail 数量，未裁剪或改绘。精确峰值由 CUDA 内存统计给出：前向 18.05 GiB、训练 51.41 GiB。仅凭形态不能严格确定每一个阶段边界，需要结合阶段数值与分配栈。具体大分配归因见 handout 答案。

[[FIG:checkpoint_memory]]

图 2：横轴为 checkpoint 分段数。32 段是每层一个 checkpoint，16 段是两层一个，64 段按 attention/FFN 半层切分；1 段运行 OOM。1 MiB 级差别不足以证明半层和整层存在稳定的显存优劣。

单层 TransformerBlock 已用 saved_tensors_hooks 对保存的独立 storage 去重统计，并排除参数/缓冲区别名；该统计可支持显存归因推理，但并非 handout 要求的 Nsight 分配事件截图。算子名称是保存张量的 producer/module 标签，也不能当作 CUDA kernel 名。

## 3. Attention 与 FlashAttention

普通 PyTorch / compiled attention 的 batch=8 规定矩阵共 40 个配置已有结果，规定范围内没有 OOM；没有为满足叙述而虚构“最小 OOM”。Flash 的三实现扩展矩阵共 240 个配置，均已有前向、反向、端到端三项计时。此前缺归档的两个必做 Triton 配置已通过已结束调用的返回值及只读存储文件补齐，没有再开 GPU。

[[FIG:flash_scaling]]

图 3：FP32、D=64 的端到端前向+反向切片。完整 FP32/BF16、D=16/32/64/128、S=128..65536 表在附录。全 Triton backward 属于额外实现；必做版本是 Triton forward + PyTorch compiled backward。Triton 的 tf32x3 执行与全模型 baseline 关闭 TF32 的设置不同，必须结合数值验证解释速度。

## 4. 通信、DDP 与 FSDP

同节点 2/4/6 ranks、Gloo/NCCL、1/10/100/1000 MB 的 24 个通信配置已测量。Gloo 使用 CPU 张量，但当时的历史作业仍分配了 GPU，不能将历史成本标成零；此次整理不重跑这些任务。

[[FIG:allreduce_gloo]]

[[FIG:allreduce_nccl]]

图 4-5：每轮最慢 rank 的 all-reduce 时间。CPU 配额固定，跨作业节点并非保证同一物理机器；这些差异限制了带宽扩展结论。bus bandwidth 是按通信算法流量推导的指标，不是直接读出的 NVLink 线速。

[[FIG:distributed_steps]]

图 6：两卡 xl 模型的训练步时间，含实际同步及优化器更新。所有原始 rank 样本和初始化/优化器前后显存见附录。overlap 的 exposed_sync 只是反向结束后未隐藏的等待，不能代表总通信时间，更不能代替时间线证明通信重叠。

## 5. 完整 8B 候选的真实结果

两张 B200、34 层、d_model=4096、d_ff=11008、32 heads、vocab=151936、context=32768、全局 batch=2。使用自编 Triton attention、block checkpoint、分块投影交叉熵、重叠 DDP、优化器状态分片及自编 fused AdamW。

初次完整候选稳态均值为 {initial / 1000:.4f} s/step（3 次），调整 tile 后为 {leader / 1000:.4f} s/step（5 次），本次样本均值降低约 {improvement:.1f}%。两组都保留冷启动与预热记录。计时来自本项目同步后的全步 wall-clock，不是已通过课程官方排行榜验证的成绩；不同计时协议不能直接等同。

小规模数值检查和核心测试已有通过记录，但不宣称完整 8B 输出/梯度逐项与官方参考模型比较过；最终候选在课程误差容限、适配器及最终源版本上的验证仍属于交付前检查。由于预算暂停，不再为进一步调优申请 GPU。

## 6. Profiling 故障与证据边界

早期 Nsight 导出包含 CUDA API、NVTX 和内存事件，但缺少 CUPTI_ACTIVITY_KIND_KERNEL；分析日志明确报 no such table。运行返回 ok 或 trace_created=true 只说明目标程序及导出结束，不足以说明捕获有效 GPU kernel。

最后一次 cuda-sw 软件追踪探针的完整 SQLite 和 nsys-rep 已通过只读存储接口取回，包含 3637 个真实 GPU kernel，总执行时间 29.847779 ms。修复本地解析器对 autograd 工作线程的归因后，measurement 全部 kernel 均有匹配，backward 正确累计为 16.380591 ms，而不是旧解析结果的 0.001312 ms。此修复只重算已有数据，没有使用 GPU。

该 small/context256 配置可复用为六项中的一项；其余五个模型 profile、两份 DDP 重叠 trace、一份 FSDP trace 仍缺有效 GPU 证据。现有探针的完整问题回答、归因规则和诊断限制见补充 Nsight 分析附录。统计汇总不是 GUI 截图，不冒充已完成所有 profiling 交付。

## 7. 本地复现与交付文件

本地重建命令见 output/README.md；三个报告/汇总/打包脚本均不调用 Modal 或 CUDA。原始文件只读，结果保留来源；成本只计有依据的运行期估价，不是账单。下列附录给出逐题解答、全量数值、过程和待补任务。
"""


def inline(s):
    s = s.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", lambda m: "^" + m[0].translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")), s)
    s = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", s)
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r'<font color="#087F8C">\1</font>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s


def styles():
    common = dict(fontName=FONT, textColor=INK, wordWrap="CJK")
    return {
        "body": ParagraphStyle("body", fontSize=9, leading=14.5, spaceAfter=6, **common),
        "h1": ParagraphStyle("h1", fontSize=21, leading=29, spaceBefore=8, spaceAfter=16, **common),
        "h2": ParagraphStyle("h2", fontSize=14, leading=20, spaceBefore=15, spaceAfter=9, keepWithNext=True, **common),
        "h3": ParagraphStyle("h3", fontSize=11, leading=16, spaceBefore=11, spaceAfter=6, keepWithNext=True, **common),
        "h4": ParagraphStyle("h4", fontSize=9.5, leading=15, spaceBefore=8, spaceAfter=5, keepWithNext=True, **common),
        "cell": ParagraphStyle("cell", fontSize=7.0, leading=10, **common),
        "head": ParagraphStyle("head", fontSize=7.2, leading=10.5, textColor=colors.white, fontName=FONT, wordWrap="CJK"),
        "code": ParagraphStyle("code", fontName=FONT, fontSize=7.2, leading=10.5, textColor=MUTED, wordWrap="CJK", backColor=PALE, borderPadding=6, spaceAfter=8),
    }


def markdown_flow(source, st, figs, width=490):
    lines = source.splitlines()
    story, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("[[FIG:"):
            name = line.removeprefix("[[FIG:").removesuffix("]]")
            height = getattr(figs[name], "height", getattr(figs[name], "drawHeight", 0))
            story.extend([CondPageBreak(height + 35), figs[name], Spacer(1, 7)])
            i += 1
        elif line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_line = lines[i].replace("\t", "    ")
                indent = code_line[:len(code_line) - len(code_line.lstrip())]
                available = width - 20 - pdfmetrics.stringWidth(indent, FONT, 7.2)
                wrapped = simpleSplit(code_line.lstrip(), FONT, 7.2, available) or [""]
                block.extend(indent + part for part in wrapped)
                i += 1
            story.append(Paragraph("<br/>".join(html.escape(x).replace("  ", "&#160;&#160;") for x in block), st["code"]))
            i += 1
        elif line.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [s.strip() for s in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells):
                    rows.append(cells)
                i += 1
            columns = max(map(len, rows))
            weights = []
            for j in range(columns):
                size = max((len(re.sub(r"[`*]", "", row[j])) if j < len(row) else 0 for row in rows), default=1)
                weights.append(min(3.5, max(1.0, math.sqrt(size / 10))))
            widths = [width * w / sum(weights) for w in weights]
            cells = [[Paragraph(inline(row[j]) if j < len(row) else "", st["head"] if k == 0 else st["cell"])
                      for j in range(columns)] for k, row in enumerate(rows)]
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, 0), 5), ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                ("LINEBELOW", (0, 0), (-1, 0), .5, TEAL),
            ]))
            story.extend([table, Spacer(1, 9)])
        elif line.startswith("#"):
            m = re.match(r"^(#{1,6})\s+(.*)", line)
            if m:
                depth = min(4, len(m[1]))
                story.append(Paragraph(inline(m[2]), st[f"h{depth}"]))
            else:
                story.append(Paragraph(inline(line), st["body"]))
            i += 1
        elif line in ("---", "***"):
            story.append(Spacer(1, 8))
            i += 1
        else:
            parts = [line]
            i += 1
            if not re.match(r"^([-*+] |\d+[.)] )", line):
                while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|\[\[FIG:|[-*+] |\d+[.)] )", lines[i].strip()):
                    parts.append(lines[i].strip())
                    i += 1
            story.append(Paragraph(inline(" ".join(parts)), st["body"]))
    return story


class Report(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("h1", "h2"):
            label = flowable.getPlainText()
            key = f"section-{self.seq.nextf('section')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(label[:100], key, 0 if flowable.style.name == "h1" else 1, False)


def page_frame(canvas, doc):
    canvas.saveState()
    width, height = doc.pagesize
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(1.0)
    canvas.line(52, height - 39, width - 52, height - 39)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(52, height - 31, "CS336 / SYSTEMS / 已有证据报告")
    canvas.drawRightString(width - 52, height - 31, "2026-09-14 · 云端预算暂停")
    canvas.drawString(52, 27, "真实记录优先 · 缺项显式保留 · 非完整提交声明")
    canvas.drawRightString(width - 52, 27, str(doc.page))
    canvas.restoreState()


def main():
    register_font()
    rows = json.loads((OUT / "data/all_attempts.json").read_text())
    figs = figures(rows)
    report = overview(rows)
    def markdown_images(source):
        def replace(match):
            name = match[1]
            path = ("memory/" + name.removeprefix("memoryviz_") + "-memoryviz.png"
                    if name.startswith("memoryviz_") else "figures/" + name + ".svg")
            return f"![{name}]({path})"
        return re.sub(r"\[\[FIG:([^]]+)\]\]", replace, source)
    (OUT / "REPORT.md").write_text(markdown_images(report), encoding="utf-8")
    sections = [report]
    for label, path in [
        ("补充 · 既有 Nsight 探针分析", OUT / "NSIGHT_EXISTING_ANALYSIS.md"),
        ("附录 A · Handout 逐题解答", ROOT / "HANDOUT_ANSWERS.md"),
        ("附录 B · 完整数值表", OUT / "tables/EXPERIMENT_TABLES.md"),
        ("附录 C · 实验步骤与过程", ROOT / "EXPERIMENT_LOG.md"),
        ("附录 D · 待补 GPU 任务", ROOT / "GPU_REMAINING_TASKS.md"),
    ]:
        if not path.exists():
            raise RuntimeError(f"Required appendix not ready: {path}")
        content = path.read_text()
        content = re.sub(r"^#\s+.*\n", "", content, count=1)
        def relocate(match):
            target = match[2]
            if target.startswith(("http:", "https:", "mailto:", "#", "/")):
                return match[0]
            relative = os.path.relpath((path.parent / target).resolve(), OUT)
            return f"[{match[1]}]({relative})"
        pieces = re.split(r"(```[\s\S]*?```)", content)
        content = "".join(part if part.startswith("```") else re.sub(r"\[([^]]+)\]\(([^)]+)\)", relocate, part)
                          for part in pieces)
        sections.append(f"# {label}\n\n" + content)
    combined = "\n\n---\n\n".join(sections)
    (OUT / "FULL_REPORT.md").write_text(markdown_images(combined), encoding="utf-8")
    st = styles()
    story = []
    for i, section in enumerate(sections):
        if i:
            story.append(PageBreak())
        story.extend(markdown_flow(section, st, figs))
    pdf_dir = OUT / "pdf"
    pdf_dir.mkdir(exist_ok=True)
    target = pdf_dir / "assignment2_existing_evidence_report.pdf"
    doc = Report(str(target), pagesize=(595.28, 841.89), leftMargin=52, rightMargin=53.28,
                 topMargin=58, bottomMargin=47, title="CS336 Assignment 2：已有实验与复现报告",
                 author="CS336 Assignment 2 local evidence archive", allowSplitting=True)
    frame = Frame(52, 47, 490, 736.89, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=page_frame)])
    doc.build(story)
    print(f"Generated {target}")
    print(f"Figures: {len(figs)}; sections: {len(sections)}; records: {len(rows)}")


if __name__ == "__main__":
    main()
