"""wei_pan_core.orchestrator · 主流程编排（串联全部模块）。

对照 checklist M8.1 ~ M8.4。
流程：
  1) 外盘闸门 compute_weighted_gate → hold=空仓报告 / tighten/normal/loosen 继续
  2) 大盘状态 detect_regime → 获取 ml_weight / regime
  3) 全市场快照 fetch_all_stocks_daily_snapshot
  4) 候选池 build_candidate_pool（剔ST/停牌+新股标注）
  5) 硬门槛 run_hard_filter（10规则+K系数三类应用+cond9分时）
  6) 规则打分 run_rule_scoring（基础70+板块30）→ Top10
  7) ML重排 rerank_with_ml → Top5（弱市纯规则）
  8) 仓位分配（4:6）+ 次日三卖出策略 → 买卖价清单
  9) 报告输出（Markdown/HTML）

不直接写文件，返回结构化dict由formatter渲染。
"""

from __future__ import annotations

import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    TIME_MODE_K, AUTO_MODE_TIME, OUTPUT_TOP_N,
    SELL_STRATEGIES, PATHS, RISK_DISCLAIMER,
)
from .data_fetcher import fetch_all_stocks_daily_snapshot
from .stock_pool import build_candidate_pool
from .global_market import compute_weighted_gate, detect_regime
from .hard_filter import run_hard_filter
from .scorer import run_rule_scoring
from .ml_ranker import rerank_with_ml
from .sector_ranking import fetch_sector_rankings, fetch_stock_sector_map, compute_stock_contrib_rank


def _auto_mode(now: Optional[datetime] = None) -> str:
    """根据当前系统时间自动判定买入模式A/B/C。"""
    now = now or datetime.now()
    hhmm = now.hour * 100 + now.minute
    if hhmm < 1445:
        return "A"
    elif hhmm < 1452:
        return "B"
    else:
        return "C"


def _assign_positions(top5: pd.DataFrame, total_capital: float = 100000.0) -> pd.DataFrame:
    """4:6 仓位分配：前2票各20%（合计40%），后3票各12%（合计36%），留24%空仓。
    不足5票时按排名递减，剩余资金留空仓（保守）。
    """
    if len(top5) == 0:
        return top5
    df = top5.copy()
    n = len(df)
    # 4:6 = 前2票各20%(40%) + 后3票各12%(36%) + 24%空仓
    if n >= 5:
        weights = [0.20, 0.20, 0.12, 0.12, 0.12]
        weights = weights[:n]
    elif n == 4:
        weights = [0.22, 0.18, 0.12, 0.08]  # 合60%仓位，40%空仓
    elif n == 3:
        weights = [0.24, 0.16, 0.10]  # 合50%，50%空仓
    elif n == 2:
        weights = [0.24, 0.16]  # 合40%，60%空仓
    else:
        weights = [0.20]  # 合20%，80%空仓
    df["position_weight"] = weights
    df["position_capital"] = [w * total_capital for w in weights]
    df["position_shares"] = (df["position_capital"] / df["close"].astype(float)).astype(int)
    # 取整100股
    df["position_shares"] = (df["position_shares"] // 100) * 100
    df["position_capital"] = df["position_shares"] * df["close"].astype(float)
    return df


def _assign_sell_strategies(top5: pd.DataFrame) -> pd.DataFrame:
    """为每只票分配次日卖出策略。
    Top1 → 策略A（集合竞价卖出，最保守）
    Top2-3 → 策略B（9:30-9:45止盈止损窗口）
    Top4-5 → 策略C（10点前灵活止盈+追踪止盈）
    """
    if len(top5) == 0:
        return top5
    df = top5.copy()
    strategies = []
    sell_prices = []
    sell_times = []
    sell_details = []
    for i, row in df.iterrows():
        if i == 0:
            s = "A"
        elif i <= 2:
            s = "B"
        else:
            s = "C"
        cfg = SELL_STRATEGIES[s]
        close = float(row.get("close") or 0.0)
        # 计算卖出价区间
        if s == "A":
            # 集合竞价卖出：无法预知次日竞价价，用"竞价区间=今收±2%"标注
            sell_price_low = close * 0.98   # 最差竞价跌2%
            sell_price_high = close * 1.02  # 最好竞价涨2%
            sell_time = cfg.get("sell_time", "09:25")
            detail = f"策略A：09:25集合竞价直接卖出（竞价价≈{sell_price_low:.2f}~{sell_price_high:.2f}，今收{close:.2f}±2%）"
            sell_price = close  # 标记基准=今收，实际以次日09:25竞价为准
        elif s == "B":
            stop = close * (1 + cfg["stop_loss_pct"])
            tp = close * (1 + cfg["take_profit_pct"])
            sell_time = f"{cfg['window_start']}~{cfg['window_end']}"
            detail = f"策略B：止损{stop:.2f}(-1.5%) 止盈{tp:.2f}(+1.5%) 时间窗{sell_time}"
            sell_price = tp  # 目标卖出价
        else:  # C
            stop = close * (1 + cfg["hard_stop_loss_pct"])
            tp_fast = close * (1 + cfg["take_profit_fast_pct"])
            sell_time = cfg.get("final_close_time", "10:00")
            detail = f"策略C：硬止损{stop:.2f}(-1.8%) 快止盈{tp_fast:.2f}(+2%) 追踪触发+1.2%/回撤0.5% 10点前平仓"
            sell_price = tp_fast
        strategies.append(s)
        sell_prices.append(round(sell_price, 2))
        sell_times.append(sell_time)
        sell_details.append(detail)
    df["sell_strategy"] = strategies
    df["sell_target_price"] = sell_prices
    df["sell_time"] = sell_times
    df["sell_detail"] = sell_details
    return df


def run_wei_pan(
    *,
    mode: str = "auto",
    date: Optional[str] = None,
    total_capital: float = 100000.0,
    enable_ml: bool = False,
    code_mask: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """尾盘选股主流程入口。

    参数:
        mode:          "auto"(按时间自动A/B/C) / "A" / "B" / "C"
        date:          交易日(YYYY-MM-DD)，默认今天
        total_capital: 总资金(元)
        enable_ml:     是否启用ML重排（False=纯规则）
        code_mask:     限定股票代码列表（测试用；None=全市场）

    返回结构化dict给formatter。
    """
    start_ts = time.time()
    now = datetime.now(timezone(timedelta(hours=8)))
    if date is not None and date != now.strftime("%Y-%m-%d"):
        raise ValueError("run_wei_pan使用最新行情，只接受上海时区今天；不是历史回测入口")
    if mode == "auto":
        mode = _auto_mode(now)
    mode = mode.upper()
    K = TIME_MODE_K.get(mode, 1.0)
    date_str = date or now.strftime("%Y-%m-%d")

    result: Dict[str, Any] = {
        "mode": mode,
        "K": K,
        "date": date_str,
        "run_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "risk_disclaimer": RISK_DISCLAIMER,
        "steps": [],
        "errors": [],
    }

    # ========== Step 1: 外盘闸门 ==========
    try:
        gate = compute_weighted_gate(market_close_date=date_str)
        result["gate"] = gate
        result["steps"].append({
            "step": "1_gate", "ok": True,
            "action": gate["gate_action"],
            "weighted_pct": gate.get("weighted_pct"),
            "data_incomplete": gate["data_incomplete"],
        })
        if gate["gate_action"] == "hold":
            # 外盘暴跌 → 空仓
            result["final_top5"] = pd.DataFrame()
            result["final_action"] = "HOLD_EMPTY"
            result["elapsed_sec"] = round(time.time() - start_ts, 1)
            return result
    except Exception as e:
        result["errors"].append(f"gate: {e}")
        result["final_action"] = "ERROR_GATE"
        result["elapsed_sec"] = round(time.time() - start_ts, 1)
        return result

    # ========== Step 2: 大盘状态 ==========
    try:
        regime_info = detect_regime(regime_date=date_str)
        result["regime"] = regime_info
        result["steps"].append({
            "step": "2_regime", "ok": True,
            "regime": regime_info["regime"],
            "bull_hits": regime_info["bull_hits"],
            "weak_hits": regime_info["weak_hits"],
            "ml_weight": regime_info["ml_weight"],
        })
    except Exception as e:
        result["errors"].append(f"regime: {e}")
        regime_info = {"regime": "neutral", "ml_weight": 0.20, "bull_hits": 0, "weak_hits": 0}

    # ========== Step 3: 全市场快照 ==========
    try:
        snap = fetch_all_stocks_daily_snapshot(date=date_str, code_mask=code_mask)
        result["steps"].append({"step": "3_snapshot", "ok": snap["ok"], "rows": len(snap["data"])})
        if not snap["ok"] or len(snap["data"]) == 0:
            result["errors"].append("snapshot: 无行情数据")
            result["final_action"] = "ERROR_DATA"
            result["elapsed_sec"] = round(time.time() - start_ts, 1)
            return result
    except Exception as e:
        result["errors"].append(f"snapshot: {e}")
        result["final_action"] = "ERROR_DATA"
        result["elapsed_sec"] = round(time.time() - start_ts, 1)
        return result

    # ========== Step 4: 候选池 ==========
    try:
        pool = build_candidate_pool(snap["data"], regime_date=date_str)
        result["steps"].append({"step": "4_pool", "ok": pool["ok"], "stats": pool["stats"]})
    except Exception as e:
        result["errors"].append(f"pool: {e}")
        result["final_action"] = "ERROR_POOL"
        result["elapsed_sec"] = round(time.time() - start_ts, 1)
        return result

    # ========== Step 5: 硬门槛过滤（两阶段：先跳过cond9快速筛，再拉分时二次筛） ==========
    try:
        tighten = gate["global_tightness_factor"]
        loosen_days = gate.get("loosen_limitup_extra_days", 0)

        # 第一阶段：跳过cond9快速过滤（4000+票→几十~几百票）
        passed, filter_report = run_hard_filter(
            pool["data"], mode=mode,
            global_tightness_factor=tighten,
            loosen_limitup_extra_days=loosen_days,
            skip_cond9=True,  # 关键：首轮不查分时，先筛掉cond1-8+10
        )
        result["filter_report"] = {
            "total_input": filter_report.total_input,
            "passed": filter_report.passed,
            "dropped_by_cond": filter_report.dropped_by_cond,
            "thresholds": filter_report.extra,
            "phase1_note": "首轮跳过cond9(分时),待二次筛",
        }

        # 第二阶段：对通过首轮的票逐只拉分时数据，再跑cond9
        if filter_report.passed > 0:
            from .data_fetcher import fetch_intraday_minute
            passed_codes = passed["code"].astype(str).str.zfill(6).tolist()
            # 限制最多拉50只（防超时），按候选池顺序取前50
            fetch_codes = passed_codes[:50]
            intraday_map = {}
            for code in fetch_codes:
                try:
                    # 根据模式决定拉取截止时间
                    until_time = {"A": "14:30", "B": "14:45", "C": "14:52"}.get(mode, "14:45")
                    ir = fetch_intraday_minute(code, date=date_str, until_minute_str=until_time)
                    if ir.get("ok") and ir.get("data") is not None:
                        intraday_map[code] = ir["data"]
                except Exception:
                    pass  # 单票失败不阻塞，cond9兜底处理

            # 第二轮：有分时数据的票再跑一次cond9
            if filter_report.passed > 0:
                # Always complete phase two, even when every intraday request failed.
                passed, filter_report = run_hard_filter(
                    passed, mode=mode,
                    global_tightness_factor=tighten,
                    loosen_limitup_extra_days=loosen_days,
                    intraday_map=intraday_map,
                    skip_cond9=False,  # 二轮必须查cond9
                )
                result["filter_report"]["passed"] = filter_report.passed
                result["filter_report"]["dropped_by_cond"] = filter_report.dropped_by_cond
                result["filter_report"]["phase2_intraday_fetched"] = len(intraday_map)

        result["steps"].append({"step": "5_filter", "ok": True, "passed": filter_report.passed})
        if filter_report.passed == 0:
            result["final_top5"] = pd.DataFrame()
            result["final_action"] = "NO_PASS"
            result["elapsed_sec"] = round(time.time() - start_ts, 1)
            return result
    except Exception as e:
        result["errors"].append(f"filter: {e}")
        result["final_action"] = "ERROR_FILTER"
        result["elapsed_sec"] = round(time.time() - start_ts, 1)
        return result

    # ========== Step 6: 规则打分 ==========
    try:
        # 板块排名 & 贡献名次
        sector_df = fetch_sector_rankings(market_close_date=date_str)
        codes = passed["code"].astype(str).str.zfill(6).tolist()
        sector_map = fetch_stock_sector_map(codes[:30])  # 限制30票防超时
        contrib = compute_stock_contrib_rank(passed, sector_map)

        th = filter_report.extra
        # P0-3修正：scorer需要的是HS300"当日"涨幅（1日），不是5日累计
        # 从signals里取：5d_return换算近似日收益率 = (1+5d)^(1/5)-1
        hs300_5d = regime_info.get("signals", {}).get("hs300_5d_return")
        hs300_daily_ret = (hs300_5d / 5.0) if hs300_5d is not None else None  # 粗粒近似：5日均摊

        scored = run_rule_scoring(
            passed,
            pct_lo=th["pct_lo"], pct_hi=th["pct_hi"],
            vr_lo=th["vr_lo"], vr_hi=th["vr_hi"],
            tr_lo=th["tr_lo"], tr_hi=th["tr_hi"],
            limit_up_N=th["limit_up_N"],
            hs300_daily_ret=hs300_daily_ret,
            sector_rank_df=sector_df,
            stock_contrib_rank_pct=contrib,
            intraday_map=intraday_map if 'intraday_map' in dir() else None,
            intraday_mode=mode,
            top_n=10,
        )
        result["steps"].append({"step": "6_score", "ok": True, "n_top10": scored["summary_counts"]["n_top"]})
    except Exception as e:
        result["errors"].append(f"score: {e}")
        result["final_action"] = "ERROR_SCORE"
        result["elapsed_sec"] = round(time.time() - start_ts, 1)
        return result

    # ========== Step 7: ML重排 → Top5 ==========
    try:
        top5, ml_meta = rerank_with_ml(
            scored["top"],
            regime=regime_info["regime"],
            ml_weight=regime_info["ml_weight"],
            enable_ml=enable_ml,
        )
        result["ml_meta"] = ml_meta
        result["steps"].append({"step": "7_ml", "ok": True, "ml_used": ml_meta["ml_used"], "n_top5": len(top5)})
    except Exception as e:
        result["errors"].append(f"ml: {e}")
        # ML失败 → 直接用规则Top5
        top5 = scored["top"].head(OUTPUT_TOP_N).copy()
        result["ml_meta"] = {"regime": regime_info["regime"], "ml_used": False, "n_predicted": 0, "n_fallback": 0}
        result["steps"].append({"step": "7_ml", "ok": False, "error": str(e)})

    # ========== Step 8: 仓位 + 卖出策略 ==========
    try:
        top5 = _assign_positions(top5, total_capital=total_capital)
        top5 = _assign_sell_strategies(top5)
        result["final_top5"] = top5
        result["final_action"] = "SELECTED"
    except Exception as e:
        result["errors"].append(f"position: {e}")
        result["final_top5"] = top5
        result["final_action"] = "SELECTED_NO_POS"

    result["elapsed_sec"] = round(time.time() - start_ts, 1)
    return result


# =========================================================================
# 回测模式入口
# =========================================================================
def run_wei_pan_backtest(
    *,
    date: str,
    mode: str = "B",
    total_capital: float = 100000.0,
    enable_ml: bool = False,
    code_mask: Optional[List[str]] = None,
    snapshot_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """回测模式入口。与run_wei_pan的区别：
    - 接受外部预建快照（不调fetch_all_stocks_daily_snapshot，用重建的历史快照）
    - 使用reconstruct_snapshot + 历史intraday_minute
    - 附加P&L验证步骤
    """
    start_ts = time.time()
    now = datetime.now()
    mode = mode.upper() if mode != "auto" else _auto_mode(now)
    K = TIME_MODE_K.get(mode, 1.0)
    date_str = date

    result: Dict[str, Any] = {
        "mode": mode,
        "K": K,
        "date": date_str,
        "run_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "risk_disclaimer": RISK_DISCLAIMER,
        "is_backtest": True,
        "steps": [],
        "errors": [],
    }

    # ========== Step 1: 外盘闸门 ==========
    try:
        gate = compute_weighted_gate(market_close_date=date_str)
        result["gate"] = gate
        result["steps"].append({"step": "1_gate", "ok": True, "action": gate["gate_action"]})
        if gate["gate_action"] == "hold":
            result["final_top5"] = pd.DataFrame()
            result["final_action"] = "HOLD_EMPTY"
            result["elapsed_sec"] = round(time.time() - start_ts, 1)
            return result
    except Exception as e:
        result["errors"].append(f"gate: {e}")
        result["final_action"] = "ERROR_GATE"
        result["elapsed_sec"] = round(time.time() - start_ts, 1)
        return result

    # ========== Step 2: 大盘状态 ==========
    try:
        regime_info = detect_regime(regime_date=date_str)
        result["regime"] = regime_info
        result["steps"].append({"step": "2_regime", "ok": True, "regime": regime_info["regime"]})
    except Exception as e:
        result["errors"].append(f"regime: {e}")
        regime_info = {"regime": "neutral", "ml_weight": 0.20, "bull_hits": 0, "weak_hits": 0,
                        "signals": {}}

    # ========== Step 3: 快照（用重建器或外部传入） ==========
    try:
        if snapshot_df is not None:
            snap_data = snapshot_df
            snap_ok = True
        else:
            from .backtest_data_reconstructor import reconstruct_snapshot
            snap_res = reconstruct_snapshot(date=date_str, code_mask=code_mask)
            snap_ok = snap_res["ok"]
            snap_data = snap_res["data"]
            result["snapshot_stats"] = snap_res.get("stats", {})
        result["steps"].append({"step": "3_snapshot", "ok": snap_ok, "rows": len(snap_data)})
        if not snap_ok or len(snap_data) == 0:
            result["errors"].append("snapshot: 重建快照为空")
            result["final_action"] = "ERROR_DATA"
            result["elapsed_sec"] = round(time.time() - start_ts, 1)
            return result
    except Exception as e:
        result["errors"].append(f"snapshot: {e}")
        result["final_action"] = "ERROR_DATA"
        result["elapsed_sec"] = round(time.time() - start_ts, 1)
        return result

    # ========== Step 4: 候选池 ==========
    try:
        pool = build_candidate_pool(snap_data)
        result["steps"].append({"step": "4_pool", "ok": pool["ok"], "stats": pool["stats"]})
    except Exception as e:
        result["errors"].append(f"pool: {e}")
        result["final_action"] = "ERROR_POOL"
        result["elapsed_sec"] = round(time.time() - start_ts, 1)
        return result

    # ========== Step 5: 硬门槛（两阶段：先跳cond9，再拉分时二次筛） ==========
    try:
        tighten = gate["global_tightness_factor"]
        loosen_days = gate.get("loosen_limitup_extra_days", 0)

        passed, filter_report = run_hard_filter(
            pool["data"], mode=mode,
            global_tightness_factor=tighten,
            loosen_limitup_extra_days=loosen_days,
            skip_cond9=True,
        )
        result["filter_report"] = {
            "total_input": filter_report.total_input,
            "passed": filter_report.passed,
            "dropped_by_cond": filter_report.dropped_by_cond,
            "thresholds": filter_report.extra,
        }

        # 第二阶段：拉历史分时数据 + cond9二次筛
        if filter_report.passed > 0:
            from .data_fetcher import fetch_intraday_minute
            passed_codes = passed["code"].astype(str).str.zfill(6).tolist()
            fetch_codes = passed_codes[:50]
            intraday_map = {}
            for code in fetch_codes:
                try:
                    until_time = {"A": "14:30", "B": "14:45", "C": "14:52"}.get(mode, "14:45")
                    ir = fetch_intraday_minute(code, date=date_str, until_minute_str=until_time)
                    if ir.get("ok") and ir.get("data") is not None and len(ir["data"]) > 0:
                        intraday_map[code] = ir["data"]
                except Exception:
                    pass

            if filter_report.passed > 0:
                # Always complete phase two, even when every intraday request failed.
                passed, filter_report = run_hard_filter(
                    passed, mode=mode,
                    global_tightness_factor=tighten,
                    loosen_limitup_extra_days=loosen_days,
                    intraday_map=intraday_map,
                    skip_cond9=False,
                )
                result["filter_report"]["passed"] = filter_report.passed
                result["filter_report"]["dropped_by_cond"] = filter_report.dropped_by_cond
                result["filter_report"]["phase2_intraday_fetched"] = len(intraday_map)

        result["steps"].append({"step": "5_filter", "ok": True, "passed": filter_report.passed})
        if filter_report.passed == 0:
            result["final_top5"] = pd.DataFrame()
            result["final_action"] = "NO_PASS"
            result["elapsed_sec"] = round(time.time() - start_ts, 1)
            return result
    except Exception as e:
        result["errors"].append(f"filter: {e}")
        result["final_action"] = "ERROR_FILTER"
        result["elapsed_sec"] = round(time.time() - start_ts, 1)
        return result

    # ========== Step 6: 规则打分 ==========
    try:
        from .sector_ranking import fetch_sector_rankings, fetch_stock_sector_map, compute_stock_contrib_rank
        sector_df = fetch_sector_rankings(market_close_date=date_str)
        codes = passed["code"].astype(str).str.zfill(6).tolist()
        sector_map = fetch_stock_sector_map(codes[:30])
        contrib = compute_stock_contrib_rank(passed, sector_map)

        th = filter_report.extra
        hs300_5d = regime_info.get("signals", {}).get("hs300_5d_return")
        hs300_daily_ret = (hs300_5d / 5.0) if hs300_5d is not None else None

        scored = run_rule_scoring(
            passed,
            pct_lo=th["pct_lo"], pct_hi=th["pct_hi"],
            vr_lo=th["vr_lo"], vr_hi=th["vr_hi"],
            tr_lo=th["tr_lo"], tr_hi=th["tr_hi"],
            limit_up_N=th["limit_up_N"],
            hs300_daily_ret=hs300_daily_ret,
            sector_rank_df=sector_df,
            stock_contrib_rank_pct=contrib,
            intraday_map=intraday_map if 'intraday_map' in dir() else None,
            intraday_mode=mode,
            top_n=10,
        )
        result["steps"].append({"step": "6_score", "ok": True, "n_top10": scored["summary_counts"]["n_top"]})
    except Exception as e:
        result["errors"].append(f"score: {e}")
        result["final_action"] = "ERROR_SCORE"
        result["elapsed_sec"] = round(time.time() - start_ts, 1)
        return result

    # ========== Step 7: ML重排 → Top5 ==========
    try:
        top5, ml_meta = rerank_with_ml(
            scored["top"],
            regime=regime_info["regime"],
            ml_weight=regime_info["ml_weight"],
            enable_ml=enable_ml,
        )
        result["ml_meta"] = ml_meta
        result["steps"].append({"step": "7_ml", "ok": True, "ml_used": ml_meta["ml_used"], "n_top5": len(top5)})
    except Exception as e:
        result["errors"].append(f"ml: {e}")
        top5 = scored["top"].head(OUTPUT_TOP_N).copy()
        result["ml_meta"] = {"regime": regime_info["regime"], "ml_used": False, "n_predicted": 0, "n_fallback": 0}

    # ========== Step 8: 仓位 + 卖出策略 ==========
    try:
        top5 = _assign_positions(top5, total_capital=total_capital)
        top5 = _assign_sell_strategies(top5)
        result["final_top5"] = top5
        result["final_action"] = "SELECTED"
    except Exception as e:
        result["errors"].append(f"position: {e}")
        result["final_top5"] = top5
        result["final_action"] = "SELECTED_NO_POS"

    result["elapsed_sec"] = round(time.time() - start_ts, 1)
    return result
