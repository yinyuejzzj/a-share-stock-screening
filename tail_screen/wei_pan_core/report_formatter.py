"""wei_pan_core.report_formatter · 报告格式化（Markdown + HTML）。

对照 checklist M9.1 ~ M9.3。
接受 orchestrator 返回的结构化dict，输出带风险头/尾的完整报告。
"""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import RISK_DISCLAIMER


def _fmt_pct(v, suffix: str = "%") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    try:
        return f"{float(v)*100:+.2f}{suffix}" if suffix == "%" else f"{float(v):+.2f}"
    except Exception:
        return str(v)


def format_markdown(result: Dict[str, Any]) -> str:
    """生成完整 Markdown 报告。"""
    lines: List[str] = []

    # 风险头
    lines.append(f"> ⚠️ **风险声明**：{RISK_DISCLAIMER}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 标题
    mode = result.get("mode", "?")
    K = result.get("K", 1.0)
    date = result.get("date", "?")
    run_time = result.get("run_time", "?")
    elapsed = result.get("elapsed_sec", 0)
    lines.append(f"# 尾盘选股报告 · 模式{mode}（K={K:.2f}）")
    lines.append(f"")
    lines.append(f"- 任务日期：**{date}**（最新快照的交易日期需另行核对）")
    lines.append(f"- 运行时间：{run_time}（耗时 {elapsed}s）")
    lines.append(f"- 最终动作：**{result.get('final_action', 'N/A')}**")
    lines.append("")

    # Step1 外盘闸门
    gate = result.get("gate") or {}
    if gate:
        lines.append("## 1. 外盘情绪闸门")
        lines.append("")
        lines.append(f"- 闸门动作：**{gate.get('gate_action', 'N/A').upper()}**")
        lines.append(f"- 加权涨跌幅：{_fmt_pct(gate.get('weighted_pct'))}")
        lines.append(f"- 系数因子：{gate.get('global_tightness_factor', 1.0):.2f}")
        lines.append(f"- 数据不完整：{'是 ⚠️' if gate.get('data_incomplete') else '否'}")
        if gate.get("groups_detail"):
            lines.append("")
            lines.append("| 组 | 原始权重 | 归一权重 | 涨跌幅 | 状态 |")
            lines.append("|---|---|---|---|---|")
            for g in gate["groups_detail"]:
                rw = f"{g.get('renorm_w', 0):.3f}" if g.get("renorm_w") is not None else "—"
                lines.append(
                    f"| {g.get('group_id','')} | {g.get('orig_w',0):.2f} | {rw} | "
                    f"{_fmt_pct(g.get('group_pct'))} | {'✓' if g.get('success') else '✗'} |"
                )
        lines.append("")

    # Step2 大盘状态
    regime = result.get("regime") or {}
    if regime:
        lines.append("## 2. 大盘状态判定")
        lines.append("")
        lines.append(f"- 市场状态：**{regime.get('regime', 'N/A').upper()}**")
        lines.append(f"- 牛市命中：{regime.get('bull_hits', 0)} / 弱市命中：{regime.get('weak_hits', 0)}（阈值≥{regime.get('hit_threshold', 2)}）")
        lines.append(f"- ML权重：{regime.get('ml_weight', 0):.0%}")
        if regime.get("signals"):
            sig = regime["signals"]
            lines.append(f"- HS300 5日收益：{_fmt_pct(sig.get('hs300_5d_return'))}")
            lines.append(f"- 涨停数：{sig.get('limit_up_count', 'N/A')} / 跌停数：{sig.get('limit_down_count', 'N/A')}")
            lines.append(f"- 北向资金：{sig.get('northbound_flow_yuan', 'N/A')}")
        lines.append("")

    # Steps 3-7
    steps = result.get("steps") or []
    if steps:
        lines.append("## 3. 执行步骤")
        lines.append("")
        for s in steps:
            ok = "✓" if s.get("ok") else "✗"
            detail = " ".join(f"{k}={v}" for k, v in s.items() if k not in ("step", "ok"))
            lines.append(f"- {ok} **{s.get('step','')}** {detail}")
        lines.append("")

    # 硬门槛报告
    fr = result.get("filter_report") or {}
    if fr:
        lines.append("## 4. 硬门槛过滤")
        lines.append("")
        lines.append(f"- 输入：{fr.get('total_input', 0)} → 通过：{fr.get('passed', 0)}")
        dropped = fr.get("dropped_by_cond") or {}
        if dropped:
            lines.append("- 各条件剔除数：")
            for cond, cnt in sorted(dropped.items(), key=lambda x: -x[1]):
                lines.append(f"  - {cond}: {cnt}")
        thresholds = fr.get("thresholds") or {}
        if thresholds:
            lines.extend(["", "### 本次有效阈值", "", "| 参数 | 生效值 |", "|---|---|"])
            for key, value in thresholds.items():
                encoded = json.dumps(value, ensure_ascii=False)
                lines.append(f"| {key} | {encoded} |")
        lines.append("")

    # ML重排
    ml = result.get("ml_meta") or {}
    if ml:
        lines.append("## 5. ML辅助重排")
        lines.append("")
        lines.append(f"- 市场状态：{ml.get('regime', 'N/A')}")
        lines.append(f"- ML启用：{'是' if ml.get('ml_used') else '否（纯规则排序）'}")
        if ml.get("ml_used"):
            lines.append(f"- 成功预测：{ml.get('n_predicted', 0)} / 兜底：{ml.get('n_fallback', 0)}")
            lines.append(f"- 权重：规则{ml.get('rule_weight', 0):.0%} + ML{ml.get('ml_weight', 0):.0%}")
        lines.append("")

    # Top5 结果
    top5 = result.get("final_top5")
    if top5 is not None and len(top5) > 0:
        lines.append("## 6. 最终选股结果（Top5）")
        lines.append("")
        lines.append("| # | 代码 | 名称 | 买入价 | 仓位 | 股数 | 资金 | 卖出策略 | 目标价 | 卖出时间 | 最终分 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for i, (_, row) in enumerate(top5.iterrows()):
            lines.append(
                f"| {i+1} | {row.get('code','')} | {row.get('name','')} | "
                f"{row.get('close',0):.2f} | {row.get('position_weight',0):.0%} | "
                f"{row.get('position_shares',0)} | {row.get('position_capital',0):.0f} | "
                f"{row.get('sell_strategy','')} | {row.get('sell_target_price',0):.2f} | "
                f"{row.get('sell_time','')} | {row.get('final_score', row.get('rule_total_100',0)):.1f} |"
            )
        lines.append("")
        # 选股理由
        lines.append("### 选股理由")
        lines.append("")
        for i, (_, row) in enumerate(top5.iterrows()):
            reasons = row.get("rule_why") or row.get("filter_reasons") or ""
            if isinstance(reasons, list):
                reasons = "；".join(reasons[:5])
            lines.append(f"**{i+1}. {row.get('code','')} {row.get('name','')}**（{row.get('sell_detail','')}）")
            lines.append(f"  - {reasons}")
            lines.append("")
    else:
        lines.append("## 6. 最终选股结果")
        lines.append("")
        action = result.get("final_action", "")
        if action == "HOLD_EMPTY":
            lines.append("**今日空仓** — 外盘暴跌≤-2%，不选股。")
        elif action == "NO_PASS":
            lines.append("**无票通过硬门槛** — 今日不选股。")
        else:
            lines.append(f"动作：{action}")
        lines.append("")

    # 错误
    errors = result.get("errors") or []
    if errors:
        lines.append("## 错误日志")
        lines.append("")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    # 风险尾
    lines.append("---")
    lines.append(f"> ⚠️ **风险声明**：{RISK_DISCLAIMER}")

    return "\n".join(lines)


def format_html(result: Dict[str, Any]) -> str:
    """生成完整 HTML 报告（简化版，内嵌CSS）。"""
    md = format_markdown(result)
    # 简单Markdown→HTML转换（不依赖外部库）
    html_lines = ['<!DOCTYPE html>', '<html lang="zh-CN">', '<head>',
                  '<meta charset="UTF-8">',
                  '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
                  '<title>尾盘选股报告</title>',
                  '<style>',
                  'body{font-family:"Microsoft YaHei",sans-serif;max-width:960px;margin:20px auto;padding:20px;color:#333;line-height:1.6}',
                  'table{border-collapse:collapse;width:100%;margin:10px 0}',
                  'th,td{border:1px solid #ddd;padding:6px 10px;text-align:center}',
                  'th{background:#f5f5f5}',
                  'tr:nth-child(even){background:#fafafa}',
                  'blockquote{border-left:4px solid #e74c3c;background:#fdf2f2;padding:10px 16px;margin:10px 0;color:#c0392b;font-size:14px}',
                  'h1{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:10px}',
                  'h2{color:#2980b9;margin-top:24px}',
                  'code{background:#f0f0f0;padding:2px 4px;border-radius:3px}',
                  '</style>', '</head>', '<body>']

    # 极简MD→HTML
    in_table = False
    for line in md.split("\n"):
        stripped = line.strip()
        if stripped.startswith(">"):
            html_lines.append(f"<blockquote>{stripped.lstrip('>').strip()}</blockquote>")
        elif stripped.startswith("# "):
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("### "):
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("---"):
            html_lines.append("<hr>")
        elif stripped.startswith("|"):
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            # 跳过分隔行 |---|---|
            if stripped.replace("|", "").replace("-", "").strip():
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                tag = "th" if not any("**" in c for c in cells) and cells == [c.replace("**","") for c in cells] else "td"
                # 简化：第一行th，其余td
                html_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False
            if stripped:
                # 处理**bold**
                import re
                html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)
                html_lines.append(f"<p>{html}</p>")
    if in_table:
        html_lines.append("</table>")

    html_lines.extend(["</body>", "</html>"])
    return "\n".join(html_lines)
