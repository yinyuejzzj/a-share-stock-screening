"""wei_pan_core.stock_pool · 初始候选股票池（ST/停牌剔除 + 新股标注）。

验收对照 checklist M3.1 ~ M3.3。
接受来自 fetch_all_stocks_daily_snapshot() 的 DataFrame，做：
    1) 剔 ST / *ST / 停牌（cond10的基本筛——A/C额外更严/更宽条件由 hard_filter.py 按模式叠加）
    2) 新股上市 < new_stock_days 标为 new_stock=True（**不排除**，仅标注）
返回标准化 DataFrame，列校验函数 assert_columns 可被其他模块调用做早期防御。
不直接调 data_fetcher（解耦），主程序调 data_fetcher 后传入。
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, date as date_cls

from .config import HARD_RULES_MODE_B


REQUIRED_COLUMNS = [
    "code", "name", "close", "preclose", "pct_chg",
    "market_cap_yi", "turnover_rate", "vol_ratio",
    "MA5", "MA10", "MA20", "MA45",
    "has_limit_up_in_N_day",  # dict {N:int -> bool}
    "list_date", "is_st", "is_halt",
    "open", "high", "low", "volume", "amount_yi",
]


def assert_columns(df: pd.DataFrame, extra_ok: bool = True) -> None:
    """对输入DataFrame做早期列防御，缺列直接报清晰错误（防止上游接口变了静默产生错结果）。"""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[stock_pool.assert_columns] 缺少必需列: {missing}. "
            f"实际列: {list(df.columns)}"
        )


def _ipoday_to_trade_days_approx(list_date_str: str, today_date: date_cls) -> int:
    """上市日期字符串→距今天数（近似"交易日"用自然日×0.7换算，误差±3对<30天阈值标注足够安全）。"""
    if not list_date_str:
        return 9999  # 未知→当做老股，不标新股（保守，老股不影响淘汰；新股也只是标注不排除）
    s = str(list_date_str).replace("-", "").replace("/", "")[:8]
    try:
        ipo = datetime.strptime(s, "%Y%m%d").date()
    except Exception:
        return 9999
    natural = (today_date - ipo).days
    if natural <= 0:
        return 0
    # 自然日 × 0.7（交易日约占自然日的70%，365天≈250交易日≈0.684）
    return int(round(natural * 0.7))


def build_candidate_pool(
    df_snapshot: pd.DataFrame,
    regime_date: Optional[str] = None,
    extra_new_stock_days: Optional[int] = None,
) -> Dict[str, Any]:
    """从实时行情快照 → 初始候选池（ST停牌剔除 + 新股标注）。

    返回 {
        "ok": bool,
        "reason": str,
        "data": DataFrame（REQUIRED_COLUMNS + new_stock列）,
        "stats": {
            "total_input": int,
            "dropped_st": int,
            "dropped_halt": int,
            "dropped_invalid": int,
            "passed": int,
            "new_stock_flagged": int,
        }
    }
    """
    stats: Dict[str, int] = {
        "total_input": int(len(df_snapshot)),
        "dropped_st": 0,
        "dropped_halt": 0,
        "dropped_invalid": 0,
        "passed": 0,
        "new_stock_flagged": 0,
    }
    try:
        assert_columns(df_snapshot, extra_ok=True)
    except ValueError as ve:
        return {"ok": False, "reason": f"输入列不满足: {ve}", "data": df_snapshot, "stats": stats}

    today_date: date_cls = (
        datetime.strptime(regime_date, "%Y-%m-%d").date()
        if regime_date else datetime.now().date()
    )
    new_stock_threshold = int(extra_new_stock_days
                              if extra_new_stock_days is not None
                              else HARD_RULES_MODE_B["new_stock_days"])

    df = df_snapshot.copy()
    # 代码6位zfill标准化（腾讯返回如 "858" → "000858"）
    df["code"] = df["code"].astype(str).str.zfill(6)
    # 只保留6位纯数字代码（防御畸形行）
    valid_mask = df["code"].str.fullmatch(r"\d{6}", na=False)
    stats["dropped_invalid"] += int((~valid_mask).sum())
    df = df[valid_mask].copy()

    # M3.1 剔除 ST/*ST
    try:
        st_mask = df["is_st"].astype(bool)
    except Exception:
        # 若is_st列是str则用正则兜底
        st_mask = df["name"].astype(str).str.contains(r"ST|\*ST", case=False, na=False)
    stats["dropped_st"] += int(st_mask.sum())
    df = df[~st_mask].copy()

    # M3.1 剔除停牌（换手率/成交量为0近似）
    try:
        vol_num = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        halt_vol_mask = vol_num <= 0
    except Exception:
        halt_vol_mask = pd.Series(False, index=df.index)
    try:
        halt_bool_mask = df["is_halt"].astype(bool)
    except Exception:
        halt_bool_mask = pd.Series(False, index=df.index)
    halt_mask = halt_vol_mask | halt_bool_mask
    stats["dropped_halt"] += int(halt_mask.sum())
    df = df[~halt_mask].copy()

    # M3.2 新股标注（仅标注，不排除）
    trade_days_since_ipo = df["list_date"].apply(
        lambda s: _ipoday_to_trade_days_approx(s, today_date)
    )
    df["new_stock"] = (trade_days_since_ipo < new_stock_threshold).astype(bool)
    stats["new_stock_flagged"] += int(df["new_stock"].sum())

    stats["passed"] = int(len(df))
    df = df.reset_index(drop=True)
    return {"ok": True, "reason": "ok", "data": df, "stats": stats}
