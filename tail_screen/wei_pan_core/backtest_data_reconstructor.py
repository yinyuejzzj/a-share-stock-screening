"""wei_pan_core.backtest_data_reconstructor · 从历史日线K线重建指定日期的"全市场快照"。

核心解决：实时行情API（腾讯QT/Sina HQ）返回的是"今天"数据，
不能用于回测。本模块从历史日线K线（Sina scale=240 / akshare）重建快照。

已知限制（诚实标注）：
- 市值/换手率是近似值（历史日线不含市值，用总股本×收盘价估算）
- 全市场4000+票逐只拉取很慢（~30分钟），建议先用50-100只测试
- 新浪5分钟线datalen=500覆盖~2-3个交易日，仅能回测最近1-2天
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, date as date_cls, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .data_fetcher import _fetch_single_daily_k, _tencent_market_prefix, _new_session
from .config import PATHS

logger = logging.getLogger(__name__)


# 默认50只活跃股清单（覆盖各行业热门票，方便快速回测）
DEFAULT_BACKTEST_CODES = [
    # 新能源/锂电
    "300750", "002594", "002714", "002460", "300014",
    # 半导体/科技
    "688981", "603501", "300059", "002230", "300059",
    # 消费/医药
    "000858", "600519", "000568", "600276", "300015",
    # 金融
    "601318", "600036", "601166", "601398", "601628",
    # 周期/制造
    "601899", "600887", "601012", "000002", "600596",
    # 中小盘活跃
    "603662", "000895", "002714", "300751", "688561",
    # 军工
    "600760", "002179", "300699", "300223",
    # 地产
    "000002", "001914", "600048",
    # 其他
    "601615", "600585", "002415", "300124", "603160",
    "002475", "300458", "688111", "600436", "002555",
]


def _batch_get_market_caps(codes: List[str]) -> Dict[str, Dict[str, Any]]:
    """批量获取市值和流通股本（用腾讯实时行情，市值日变化极小可做回测近似）。

    返回 {code: {"market_cap_yi": float, "float_shares": float, "total_shares": float, "name": str}}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not codes:
        return out
    try:
        from .data_fetcher import _tencent_batch_quote, _tencent_market_prefix
        prefixed = [_tencent_market_prefix(c) for c in codes]
        rows, err = _tencent_batch_quote(prefixed)
        if err or not rows:
            return out
        for row in rows:
            code = str(row.get("code", "")).zfill(6)
            if code in codes or code in [c.zfill(6) for c in codes]:
                # 腾讯88字段中：field 44=总市值(亿), field 45=流通市值(亿), field 38=换手率%
                mc = float(row.get("market_cap_yi") or 0)
                fmc = float(row.get("float_market_cap_yi") or 0)
                tr = float(row.get("turnover_rate") or 0)
                name = str(row.get("name", code))
                # 逆推流通股本 = 流通市值(亿)×1e8 / 收盘价
                close = float(row.get("close", 0))
                float_shares = (fmc * 1e8 / close) if (close > 0 and fmc > 0) else 0
                total_shares = (mc * 1e8 / close) if (close > 0 and mc > 0) else 0
                out[code] = {
                    "market_cap_yi": mc,
                    "float_shares": float_shares,
                    "total_shares": total_shares,
                    "name": name,
                    "turnover_rate_from_qq": tr / 100.0 if tr > 1 else tr,  # QQ返回百分比
                }
    except Exception:
        pass
    return out


def reconstruct_snapshot(
    date: str,
    code_mask: Optional[List[str]] = None,
    max_stocks: int = 200,
) -> Dict[str, Any]:
    """从历史日线K线重建指定日期的"全市场快照"DataFrame。

    参数:
        date:       目标日期 "YYYY-MM-DD"
        code_mask:  限定代码列表（None=用默认50只活跃股清单）
        max_stocks: 最大拉取数（防超时）

    返回: {"ok":bool, "data":DataFrame, "reason":str, "stats":dict}
    DataFrame列与 fetch_all_stocks_daily_snapshot 对齐。
    """
    date_str = date
    target_date_prefix = date_str.replace("-", "")[:8]  # "20260903"
    codes = code_mask or DEFAULT_BACKTEST_CODES[:max_stocks]
    codes = [str(c).zfill(6) for c in codes[:max_stocks]]

    # 目标日期对象
    try:
        target_date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return {"ok": False, "data": pd.DataFrame(), "reason": f"日期格式错误: {date_str}", "stats": {}}

    rows: List[Dict[str, Any]] = []
    n_success = 0
    n_fail = 0
    n_not_found = 0

    # 批量获取市值/流通股本/名称（腾讯实时行情，1次HTTP全拿到）
    mkt_caps = _batch_get_market_caps(codes)

    for i, code in enumerate(codes):
        if i % 10 == 0:
            logger.info(f"reconstruct_snapshot: {i}/{len(codes)} codes processed")

        # 拉取60日历史日线（end_date控制akshare fallback的截止日期）
        try:
            hist = _fetch_single_daily_k(code, bars=60, end_date=target_date_obj)
            if hist is None or len(hist) < 5:
                n_fail += 1
                continue
        except Exception:
            n_fail += 1
            continue

        # 在历史K线中找目标日期的行
        hist["day_str"] = hist["day"].astype(str).str[:10].str.replace("-", "")
        # 找匹配目标日期的行（或最近的 <= 目标日期的行）
        match_rows = hist[hist["day_str"] == target_date_prefix]
        if len(match_rows) == 0:
            # 目标日期不是交易日（周末/节假日），取<=目标日期的最后一行
            hist["day_num"] = hist["day_str"].astype(int)
            before = hist[hist["day_num"] <= int(target_date_prefix)]
            if len(before) == 0:
                n_not_found += 1
                continue
            target_row = before.iloc[-1]
            actual_date = target_row["day_str"]
        else:
            target_row = match_rows.iloc[0]
            actual_date = target_date_prefix

        # 取目标日和前一天的数据
        closes = hist["close"].astype(float).values
        opens = hist["open"].astype(float).values
        highs = hist["high"].astype(float).values
        lows = hist["low"].astype(float).values
        volumes = hist["volume"].astype(float).values

        # 找目标行在hist中的索引
        target_idx = None
        for j in range(len(hist)):
            if hist.iloc[j]["day_str"] == actual_date:
                target_idx = j
                break
        if target_idx is None:
            n_not_found += 1
            continue

        close = closes[target_idx]
        preclose = closes[target_idx - 1] if target_idx > 0 else close
        pct_chg = (close - preclose) / preclose if preclose > 0 else 0.0
        open_p = opens[target_idx]
        high = highs[target_idx]
        low = lows[target_idx]
        volume = volumes[target_idx]
        amount_yi = (close * volume * 100) / 1e8  # 近似成交额（元/股×手×100股/手→元→亿）

        # MA计算（基于目标行及之前的N日收盘）
        ma5 = float(np.mean(closes[max(0, target_idx - 4):target_idx + 1])) if target_idx >= 4 else np.nan
        ma10 = float(np.mean(closes[max(0, target_idx - 9):target_idx + 1])) if target_idx >= 9 else np.nan
        ma20 = float(np.mean(closes[max(0, target_idx - 19):target_idx + 1])) if target_idx >= 19 else np.nan
        ma45 = float(np.mean(closes[max(0, target_idx - 44):target_idx + 1])) if target_idx >= 44 else np.nan

        # 量比 = 当日成交量 / 5日均量
        if target_idx >= 5:
            vol5_avg = float(np.mean(volumes[target_idx - 5:target_idx]))
            vol_ratio = volume / vol5_avg if vol5_avg > 0 else 1.0
        else:
            vol_ratio = 1.0

        # 涨停基因：N日内日涨幅≥9.8%
        dpc = np.diff(closes[:target_idx + 1]) / np.concatenate(([closes[0]], closes[:target_idx]))[:-1] if target_idx > 0 else np.array([])
        limit_mask = dpc >= 0.098
        last_i = len(limit_mask) - 1
        N_defaults = [20, 25, 30, 35, 36, 45]
        gene_dict = {}
        for N in N_defaults:
            start = max(0, last_i - N + 1)
            gene_dict[N] = bool(np.any(limit_mask[start:last_i + 1])) if last_i >= 0 else False

        # 从批量行情中获取名称/市值/流通股本
        mc_info = mkt_caps.get(code, {})
        name = mc_info.get("name", code)
        is_st = "ST" in name or "*ST" in name
        is_halt = volume <= 0

        # 市值：用总股本×回测日收盘价计算（而非今天的市值）
        total_shares = mc_info.get("total_shares", 0.0)
        if total_shares > 0:
            market_cap_yi = (total_shares * close) / 1e8  # 用回测日收盘价×总股本
        else:
            market_cap_yi = mc_info.get("market_cap_yi", 0.0)  # 退化用今天的市值

        float_shares = mc_info.get("float_shares", 0.0)

        # 换手率：优先用历史成交量/流通股本计算（正确的历史值）
        # 不用腾讯实时换手率（那是今天的，不是回测日的）
        if float_shares > 0:
            turnover_rate = (volume * 100) / float_shares  # volume是手, ×100=股
        else:
            turnover_rate = 0.08  # 默认8%（无法计算时给中值）

        list_date = ""  # 批量模式下不逐只拉上市日期（太慢）

        rows.append({
            "code": code,
            "name": name,
            "close": close,
            "preclose": preclose,
            "pct_chg": pct_chg,
            "market_cap_yi": market_cap_yi,
            "turnover_rate": turnover_rate,
            "vol_ratio": vol_ratio,
            "MA5": ma5,
            "MA10": ma10,
            "MA20": ma20,
            "MA45": ma45,
            "has_limit_up_in_N_day": gene_dict,
            "list_date": list_date,
            "is_st": is_st,
            "is_halt": is_halt,
            "open": open_p,
            "high": high,
            "low": low,
            "volume": volume,
            "amount_yi": amount_yi,
        })
        n_success += 1

    if not rows:
        return {"ok": False, "data": pd.DataFrame(),
                "reason": f"无票成功拉取 (fail={n_fail}, not_found={n_not_found})",
                "stats": {"total": len(codes), "success": 0, "fail": n_fail, "not_found": n_not_found}}

    df = pd.DataFrame(rows)
    df = df.reset_index(drop=True)
    return {
        "ok": True,
        "data": df,
        "reason": "ok",
        "stats": {"total": len(codes), "success": n_success, "fail": n_fail, "not_found": n_not_found},
    }


def fetch_next_day_kline(code6: str, after_date: str, bars: int = 5) -> Optional[pd.DataFrame]:
    """拉取指定日期之后的K线（用于验证盈亏）。

    返回目标日期之后第一个交易日的K线（含open/high/low/close）。
    """
    try:
        target_date_obj = datetime.strptime(after_date, "%Y-%m-%d").date()
        hist = _fetch_single_daily_k(code6, bars=bars, end_date=target_date_obj)
        if hist is None or len(hist) < 2:
            return None
        hist["day_str"] = hist["day"].astype(str).str[:10].str.replace("-", "")
        target_prefix = after_date.replace("-", "")[:8]
        # 找>目标日期的第一行
        hist["day_num"] = hist["day_str"].astype(int)
        after = hist[hist["day_num"] > int(target_prefix)]
        if len(after) == 0:
            return None
        return after.head(1)
    except Exception:
        return None
