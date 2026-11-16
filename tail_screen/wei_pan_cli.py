"""尾盘选股 CLI 入口。

用法:
    python wei_pan_cli.py                    # Auto模式（按当前时间判定A/B/C）
    python wei_pan_cli.py --mode B          # 指定模式B
    python wei_pan_cli.py --profile strict --mode A
    python wei_pan_cli.py --no-ml           # 禁用ML重排（纯规则）
    python wei_pan_cli.py --capital 200000  # 指定总资金20万
    python wei_pan_cli.py --codes 600596,000858   # 限定代码（测试用）

输出:
    Markdown报告 → 04_输出结果/04尾盘选股结果/尾盘选股_YYYYMMDD_模式X.md
    HTML报告   → 04_输出结果/04尾盘选股结果/尾盘选股_YYYYMMDD_模式X.html
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
import json
from datetime import datetime, timezone, timedelta

# 确保项目根目录在path中
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wei_pan_core.orchestrator import run_wei_pan
from wei_pan_core.report_formatter import format_markdown, format_html
from wei_pan_core.config import PATHS, RISK_DISCLAIMER, SCREEN_PROFILES, select_profile
from wei_pan_core.hard_filter import _apply_thresholds_for_mode


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="A股尾盘规则筛选工具：参数档位与时间窗口分开选择",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
⚠️ 风险声明：{RISK_DISCLAIMER}

示例:
  python wei_pan_cli.py                      # 自动模式
  python wei_pan_cli.py --mode B             # 14:45-14:52标准模式
  python wei_pan_cli.py --mode A --no-ml     # 14:30严格模式，纯规则不用ML
""".strip(),
    )
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "A", "B", "C"],
                        help="买入时点模式：A=14:30(最严K=1.20) B=14:45(默认K=1.00) C=14:52(最松K=0.85)")
    parser.add_argument("--date", type=str, default=None,
                        help="运行日期(YYYY-MM-DD)，只接受上海时区今天；不是历史回测")
    parser.add_argument("--profile", choices=list(SCREEN_PROFILES), default="standard",
                        help="基准参数档位：strict强限制/standard标准/relaxed弱限制")
    parser.add_argument("--show-config", action="store_true",
                        help="打印档位参数与中性市场闸门下的有效阈值后退出，不联网")
    parser.add_argument("--capital", type=float, default=100000.0,
                        help="总资金(元)，默认10万")
    parser.add_argument("--no-ml", action="store_true", default=True,
                        help="兼容旧参数；公开入口固定为纯规则排序")
    parser.add_argument("--codes", type=str, default=None,
                        help="限定股票代码(逗号分隔，测试用，如 600596,000858)")
    parser.add_argument("--format", type=str, default="both",
                        choices=["md", "html", "both"],
                        help="输出格式：md=Markdown, html=HTML, both=两者")
    parser.add_argument("--verbose", action="store_true",
                        help="详细日志输出")

    args = parser.parse_args()
    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    if args.date is not None and args.date != today:
        parser.error("--date只接受上海时区今天。该入口使用最新行情，不能回填历史日期。")
    args.date = today
    if args.capital <= 0:
        parser.error("--capital必须大于0")
    baseline = select_profile(args.profile)
    if args.show_config:
        modes = "ABC" if args.mode == "auto" else [args.mode]
        print(json.dumps({
            "profile": args.profile, "baseline": baseline,
            "effective_with_neutral_gate": {m: _apply_thresholds_for_mode(m, 1., 0) for m in modes},
            "date_policy": "最新快照；运行日期不保证就是行情交易日期；不是历史回测",
        }, ensure_ascii=False, indent=2))
        return
    print("提示：使用最新行情快照；休市或接口延迟时，行情可能属于上一交易日。")

    # 日志配置
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_dir = PATHS.get("log_dir", "logs/weipan")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"weipan_{datetime.now().strftime('%Y%m%d')}.log")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")]
                  + ([logging.StreamHandler()] if args.verbose else []),
    )
    logger = logging.getLogger("weipan_cli")

    # 代码限定
    code_mask = None
    if args.codes:
        code_mask = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]

    logger.info(f"启动尾盘选股：mode={args.mode} date={args.date} capital={args.capital} ml={not args.no_ml}")

    # 运行主流程
    result = run_wei_pan(
        mode=args.mode,
        date=args.date,
        total_capital=args.capital,
        enable_ml=not args.no_ml,
        code_mask=code_mask,
    )
    result["profile"] = args.profile
    result["profile_baseline"] = baseline

    # 生成报告
    md_report = format_markdown(result)
    html_report = format_html(result)
    config_note = "\n\n## 筛选参数档位\n\n```json\n" + json.dumps(
        {"profile": args.profile, "baseline": baseline}, ensure_ascii=False, indent=2
    ) + "\n```\n\n运行日期仅为任务日期，最新行情快照并不等同于历史时点数据。\n"
    md_report += config_note
    from html import escape
    html_report = html_report.replace("</body>", "<pre>" + escape(config_note) + "</pre></body>")

    # 输出文件
    ymd = (args.date or datetime.now().strftime("%Y-%m-%d")).replace("-", "")
    mode_str = result.get("mode", "X")
    result_dir = PATHS.get("result_base", ".")
    os.makedirs(result_dir, exist_ok=True)

    outputs = []
    if args.format in ("md", "both"):
        md_path = os.path.join(result_dir, f"尾盘选股_{ymd}_{args.profile}_模式{mode_str}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        outputs.append(md_path)
        logger.info(f"Markdown报告已保存: {md_path}")

    if args.format in ("html", "both"):
        html_path = os.path.join(result_dir, f"尾盘选股_{ymd}_{args.profile}_模式{mode_str}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_report)
        outputs.append(html_path)
        logger.info(f"HTML报告已保存: {html_path}")

    # 控制台简要输出
    print()
    print("=" * 60)
    print(f"  尾盘选股完成 · 模式{mode_str} · {result.get('date','')}")
    print(f"  最终动作: {result.get('final_action','N/A')}")
    print(f"  耗时: {result.get('elapsed_sec',0)}s")
    top5 = result.get("final_top5")
    if top5 is not None and len(top5) > 0:
        print(f"  选出Top{len(top5)}只票：")
        for i, (_, row) in enumerate(top5.iterrows()):
            print(f"    {i+1}. {row.get('code','')} {row.get('name','')} "
                  f"买={row.get('close',0):.2f} "
                  f"策略{row.get('sell_strategy','')} "
                  f"目标={row.get('sell_target_price',0):.2f}")
    print(f"  报告: {', '.join(os.path.basename(p) for p in outputs)}")
    print("=" * 60)
    print(f"  ⚠️ {RISK_DISCLAIMER[:80]}...")
    print("=" * 60)


if __name__ == "__main__":
    main()
