#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EMA20 判断工具（单独程序）v1.0
按 EMA20 趋势与缩量回踩规则实现，输入股票代码即输出 Markdown 判断报告。

信号组合规则：
  ① 放量跌破清仓 → 双方并列，不出单一建议
  ② 建仓主信号 → 缩量回踩为主，趋势确认规则三条件为辅
  ③ 震荡市规则 → 采纳但仅作提示，不强制

依赖：akshare / pandas / numpy / requests（不依赖主项目其他模块）
用法：
  python ema20_judge.py 600596
  python ema20_judge.py 600596 --no-realtime
  python ema20_judge.py 600596 --out report.md
"""
import argparse
import re
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import requests
except ImportError:
    requests = None


# =========================================================================
# 风格参数集（长线短炒 / 1-3日超短线）
# =========================================================================
STYLE_PROFILES = {
    "long": {
        "bias_multiplier": 3.0,
        "bias_lower": 6.0,
        "bias_upper": 15.0,
        "clear_days": 3,
        "label": "长线短炒",
    },
    "short": {
        "bias_multiplier": 2.5,
        "bias_lower": 5.0,
        "bias_upper": 10.0,
        "clear_days": 2,
        "label": "1-3日超短线",
    },
}


# =========================================================================
# CONFIG 区（所有阈值集中于此，修改只改这里）
#   ⚠️ 标注"可配置"的参数为业界常用默认值，使用者应自行校准
# =========================================================================
CONFIG = {
    "EMA_PERIOD": 20,                      # EMA 周期（固定参考周期）
    "TOUCH_PCT": 0.02,                     # "碰到线"阈值 ±2%（⚠️可配置）
    "SHRINK_RATIO": 0.8,                   # "缩量" 当日量<5日均量×0.8（⚠️可配置）
    "EXPAND_RATIO": 1.5,                   # "放量" 当日量>5日均量×1.5（⚠️可配置）
    "BIAS_TOO_BIG": 8.0,                   # 乖离过大阈值 >8%（⚠️可配置）
    "OSC_CROSS_MIN": 4,                    # 震荡市穿越次数 ≥4（⚠️可配置）
    "OSC_AMP": 0.05,                       # 震荡市振幅 ±5%（⚠️可配置）
    "OSC_LOOKBACK": 20,                    # 震荡市回看天数
    "CLEAR_DAYS": 3,                       # 缩量回踩规则规则五：3日不站回清仓（固定参考周期）
    "HISTORY_DAYS": 60,                    # 历史数据天数（需算 EMA20+震荡检测）
    "MACD_FAST": 12,                       # MACD 快线（业界常用）
    "MACD_SLOW": 26,                       # MACD 慢线（业界常用）
    "MACD_SIGNAL": 9,                      # MACD 信号线（业界常用）
    "VOL_MA": 5,                           # 成交量均线周期
}

# 可配置参数清单（用于报告标注）
UNQUANTIFIED_PARAMS = {
    "TOUCH_PCT": "碰到线阈值 ±2%",
    "SHRINK_RATIO": "缩量倍数 0.8",
    "EXPAND_RATIO": "放量倍数 1.5",
    "BIAS_TOO_BIG": "乖离过大阈值 8%",
    "OSC_CROSS_MIN": "震荡市穿越次数 4 次",
    "OSC_AMP": "震荡市振幅 ±5%",
}

# 三档仅调节回踩距离和缩量要求。风格、趋势定义及风险规则保持独立。
SCREEN_PROFILES = {
    "strict": {"TOUCH_PCT": 0.01, "SHRINK_RATIO": 0.65},
    "standard": {"TOUCH_PCT": 0.02, "SHRINK_RATIO": 0.80},
    "relaxed": {"TOUCH_PCT": 0.03, "SHRINK_RATIO": 0.95},
}
ACTIVE_PROFILE = "standard"


def apply_profile(name: str) -> None:
    """选择工程起始参数；只收紧/放宽缩量回踩主信号的入围条件。"""
    global ACTIVE_PROFILE
    CONFIG.update(SCREEN_PROFILES[name])
    ACTIVE_PROFILE = name


RISK_DISCLAIMER = (
    "风险说明：本工具输出技术指标与规则匹配结果，不构成投资建议。"
    "可配置的参数采用默认值，使用者需自行校准。股市有风险，投资需谨慎。"
    "参数分档未经收益优化，不应据此推断胜率或收益。"
)


# =========================================================================
# 数据获取区
# =========================================================================
def fetch_data(code: str, days: int = 60, realtime: bool = True, as_of: str = None) -> pd.DataFrame:
    """获取 A 股日线数据（前复权），可选追加当日实时行情。
    取数失败抛 RuntimeError，绝不编造数据。
    """
    if ak is None:
        raise RuntimeError("未安装 akshare，无法获取历史数据。请 pip install akshare")

    if days < CONFIG["EMA_PERIOD"] + 5:
        raise ValueError("history-days 至少为 EMA_PERIOD + 5")
    cutoff = datetime.strptime(as_of, "%Y-%m-%d") if as_of else datetime.now()
    if cutoff.date() > datetime.now().date():
        raise ValueError("截止日不能晚于今天")
    if as_of:
        realtime = False  # 指定日期只使用截止日及以前的日线，不混入当前行情
    end_date = cutoff.strftime("%Y%m%d")
    start_date = (cutoff - timedelta(days=days * 2)).strftime("%Y%m%d")
    print(f"正在获取 {code} 历史数据（{start_date} ~ {end_date}）...")

    df = None
    last_err = None
    # 主源：东方财富（ak.stock_zh_a_hist）
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date, end_date=end_date, adjust="qfq",
            )
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2)

    # 备源：新浪（ak.stock_zh_a_daily），与 data_fetcher.py 一致
    if df is None or df.empty:
        print(f"  东方财富不可用({type(last_err).__name__})，尝试新浪数据源...")
        try:
            sina_code = f"sh{code}" if code.startswith(("6", "9")) else f"sz{code}"
            df = ak.stock_zh_a_daily(symbol=sina_code, adjust="qfq")
            if df is not None and not df.empty:
                # 新浪返回英文列名，统一映射
                df = df.rename(columns={
                    "date": "日期", "open": "开盘", "close": "收盘",
                    "high": "最高", "low": "最低", "volume": "成交量", "amount": "成交额",
                })
                last_err = None
        except Exception as e2:
            last_err = e2

    if df is None or df.empty:
        raise RuntimeError(f"akshare 取数失败(东方财富+新浪均失败): {last_err}. 请检查网络/股票代码。")

    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "涨跌幅": "pct_change",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] <= pd.Timestamp(cutoff.date())) &
            (df["date"] >= pd.Timestamp(start_date))]
    df = df.sort_values("date").tail(days).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("截止日及请求窗口内没有日线数据")
    if "pct_change" not in df.columns:
        df["pct_change"] = df["close"].pct_change() * 100
    if "amount" not in df.columns:
        df["amount"] = 0.0

    keep = ["date", "open", "close", "high", "low", "volume", "amount", "pct_change"]
    df = df[[c for c in keep if c in df.columns]]

    # 追加当日实时行情
    if realtime:
        today = datetime.now().date()
        last_date = df["date"].iloc[-1].date()
        if last_date < today:
            print(f"  历史数据截止 {last_date}，获取今日实时行情...")
            rt = get_realtime_quote(code)
            if rt:
                new_row = {
                    "date": pd.to_datetime(rt["date"]),
                    "open": rt["open"], "close": rt["price"],
                    "high": rt["high"], "low": rt["low"],
                    "volume": rt["volume"], "amount": rt["amount"],
                    "pct_change": rt["pct_change"],
                }
                # 避免重复
                if df["date"].iloc[-1] != new_row["date"]:
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    print(f"  已追加实时行情：{rt['name']} 价格 {rt['price']:.2f}")
                else:
                    # 同日，更新最新一根
                    df.loc[df.index[-1], "close"] = rt["price"]
                    df.loc[df.index[-1], "high"] = max(df.loc[df.index[-1], "high"], rt["high"])
                    df.loc[df.index[-1], "low"] = min(df.loc[df.index[-1], "low"], rt["low"])
                    df.loc[df.index[-1], "volume"] = rt["volume"]
                    print(f"  已更新今日实时：{rt['name']} 价格 {rt['price']:.2f}")
            else:
                print("  ⚠️ 实时行情获取失败，仅用历史数据判断")

    return df


def get_realtime_quote(code: str) -> dict:
    """新浪财经实时行情（复用 data_fetcher.py 已验证接口）。"""
    if requests is None:
        return None
    sina_code = f"sh{code}" if code.startswith(("6", "9")) else f"sz{code}"
    url = f"https://hq.sinajs.cn/list={sina_code}"
    headers = {"Referer": "https://finance.sina.com.cn"}
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.encoding = "gbk"
            match = re.search(r'"(.+)"', resp.text)
            if not match:
                continue
            fields = match.group(1).split(",")
            if len(fields) < 32:
                continue
            price = float(fields[3])
            pre_close = float(fields[2])
            return {
                "name": fields[0],
                "open": float(fields[1]),
                "pre_close": pre_close,
                "price": price,
                "high": float(fields[4]),
                "low": float(fields[5]),
                "volume": float(fields[8]),
                "amount": float(fields[9]),
                "date": fields[30],
                "time": fields[31],
                "pct_change": (price - pre_close) / pre_close * 100 if pre_close > 0 else 0,
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                print(f"  实时行情获取失败: {type(e).__name__}")
                return None


# =========================================================================
# 指标计算区
# =========================================================================
def compute_ema20(df: pd.DataFrame) -> pd.DataFrame:
    """EMA20 指数均线 + 乖离率。用 ewm(span=20)，非简单 MA。"""
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=CONFIG["EMA_PERIOD"], adjust=False).mean()
    df["bias"] = (df["close"] - df["ema20"]) / df["ema20"] * 100
    return df


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD: DIF = EMA12 - EMA26, DEA = EMA9(DIF), MACD柱 = (DIF-DEA)*2。"""
    df = df.copy()
    fast = df["close"].ewm(span=CONFIG["MACD_FAST"], adjust=False).mean()
    slow = df["close"].ewm(span=CONFIG["MACD_SLOW"], adjust=False).mean()
    df["dif"] = fast - slow
    df["dea"] = df["dif"].ewm(span=CONFIG["MACD_SIGNAL"], adjust=False).mean()
    df["macd"] = (df["dif"] - df["dea"]) * 2
    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """5 日均量 + 当日量比。"""
    df = df.copy()
    df["vol_ma5"] = df["volume"].rolling(window=CONFIG["VOL_MA"], min_periods=1).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma5"].replace(0, np.nan)
    return df


def compute_adaptive_bias_threshold(df: pd.DataFrame, style: str) -> float:
    """自适应乖离阈值：20日平均偏离 × 风格倍数，clip 到 [下限, 上限]。
    与 EMA20 同周期(20日)，逻辑自洽。
    """
    p = STYLE_PROFILES[style]
    dev_pct = (df["close"] - df["ema20"]).abs() / df["ema20"] * 100
    dev_ma20 = dev_pct.rolling(window=20, min_periods=10).mean()
    threshold = (dev_ma20 * p["bias_multiplier"]).clip(
        lower=p["bias_lower"], upper=p["bias_upper"])
    return float(threshold.iloc[-1])


# =========================================================================
# 信号检测区 —— 缩量回踩
#   每条信号标注来源规则编号，主信号与风险提示分别输出
# =========================================================================
def primary_signals(df: pd.DataFrame, style: str = "long") -> dict:
    """返回缩量回踩各条触发状态。"""
    if len(df) < CONFIG["EMA_PERIOD"] + 5:
        return {"error": "数据不足，无法计算缩量回踩规则信号"}

    bias_threshold = compute_adaptive_bias_threshold(df, style)

    last = df.iloc[-1]
    ema = last["ema20"]
    close = last["close"]
    low = last["low"]
    high = last["high"]
    vol = last["volume"]
    vol_ma5 = last["vol_ma5"]
    bias = last["bias"]

    touch_high = ema * (1 + CONFIG["TOUCH_PCT"])   # 线上方碰到线：low ≤ ema*1.02
    touch_low = ema * (1 - CONFIG["TOUCH_PCT"])    # 线下方碰到线：high ≥ ema*0.98
    is_shrink = (vol < vol_ma5 * CONFIG["SHRINK_RATIO"]) if vol_ma5 > 0 else False
    is_expand = (vol > vol_ma5 * CONFIG["EXPAND_RATIO"]) if vol_ma5 > 0 else False

    # 趋势判断：近 10 日 ema20 斜率向上 或 多数日在 ema20 上方
    recent = df.tail(10)
    slope_up = recent["ema20"].iloc[-1] > recent["ema20"].iloc[0]
    above_cnt = (recent["close"] > recent["ema20"]).sum()
    trend_up = slope_up or above_cnt >= 6

    # 规则一：缩量回踩定买点（三条件同时满足）
    r1_touch = low <= touch_high
    r1_not_break = close > ema
    r1_shrink = is_shrink
    r1 = (trend_up and r1_touch and r1_not_break and r1_shrink)

    # 规则二：触碰20日线未跌穿（确认条件）
    r2 = (low <= touch_high and close > ema)

    # 规则三：乖离过大先卖一半
    r3 = bias > bias_threshold

    # 规则四：破线先减半（14:50 检查，收盘<EMA20 先卖一半）
    r4_break = close < ema
    # 是否昨日曾破今日站回（接回信号）
    if len(df) >= 2:
        prev = df.iloc[-2]
        r4_reclaim = (prev["close"] < prev["ema20"]) and (close > ema)
    else:
        r4_reclaim = False
    r4 = r4_break  # 主信号：破线减半

    # 规则五：N日不站回全部清仓（N由风格决定）
    clear_days = STYLE_PROFILES[style]["clear_days"]
    consecutive_below = 0
    for i in range(len(df) - 1, -1, -1):
        if df.iloc[i]["close"] < df.iloc[i]["ema20"]:
            consecutive_below += 1
        else:
            break
    r5 = consecutive_below >= clear_days

    return {
        "规则一 缩量回踩定买点": {
            "triggered": bool(r1), "action": "建仓（买入）",
            "detail": f"趋势{'↑' if trend_up else '↓'}/碰到线{'是' if r1_touch else '否'}/未跌穿{'是' if r1_not_break else '否'}/缩量{'是' if r1_shrink else '否'}(量比{last['vol_ratio']:.2f})",
        },
        "规则二 触碰20日线未跌穿": {
            "triggered": bool(r2), "action": "确认买点（与规则一配套）",
            "detail": f"最低{low:.2f}≤{touch_high:.2f} 且 收盘{close:.2f}>EMA20 {ema:.2f}",
        },
        "规则三 乖离过大先卖一半": {
            "triggered": bool(r3), "action": "减仓一半",
            "detail": f"乖离率{bias:.2f}% {('>' if r3 else '<=')}{bias_threshold:.2f}%",
        },
        "规则四 破线先减半仓": {
            "triggered": bool(r4), "action": "先卖一半保护本金" + ("；昨日破今日站回→接回一半" if r4_reclaim else ""),
            "detail": f"收盘{close:.2f} {('<' if r4_break else '>=')} EMA20 {ema:.2f}",
        },
        f"规则五 {clear_days}日不站回全部清仓": {
            "triggered": bool(r5), "action": "全部清仓",
            "detail": f"连续收盘<EMA20 共{consecutive_below}日 {('≥' if r5 else '<')}{clear_days}日",
        },
        "_meta": {
            "ema": float(ema), "close": float(close), "low": float(low),
            "high": float(high), "vol": float(vol), "vol_ma5": float(vol_ma5),
            "bias": float(bias), "is_shrink": bool(is_shrink),
            "is_expand": bool(is_expand), "consecutive_below": int(consecutive_below),
            "trend_up": bool(trend_up), "bias_threshold": float(bias_threshold),
        },
    }


# =========================================================================
# 信号检测区 —— 趋势确认战法
#   EMA 与 MACD 辅助确认规则
# =========================================================================
def confirmation_signals(df: pd.DataFrame) -> dict:
    """返回趋势确认规则各信号触发状态。"""
    if len(df) < CONFIG["EMA_PERIOD"] + 5:
        return {"error": "数据不足，无法计算趋势确认规则信号"}

    last = df.iloc[-1]
    ema = last["ema20"]
    close = last["close"]
    low = last["low"]
    high = last["high"]
    vol = last["volume"]
    vol_ma5 = last["vol_ma5"]
    dea = last["dea"]
    dif = last["dif"]
    bias = last["bias"]

    touch_high = ema * (1 + CONFIG["TOUCH_PCT"])
    touch_low = ema * (1 - CONFIG["TOUCH_PCT"])
    is_shrink = (vol < vol_ma5 * CONFIG["SHRINK_RATIO"]) if vol_ma5 > 0 else False
    is_expand = (vol > vol_ma5 * CONFIG["EXPAND_RATIO"]) if vol_ma5 > 0 else False

    above = close > ema
    below = close < ema

    # 1. 线上持有
    s_hold = above
    # 2. 线上回踩是机会（进仓点）
    s_pullback_buy = above and (low <= touch_high) and (close > ema)
    # 3. 线下反抽是陷阱（出货点）
    s_rebound_trap = below and (high >= touch_low) and (close < ema)
    # 4. 三条件建仓：EMA20上方站稳 + MACD零轴上方 + 缩量
    s_three_cond = above and (dea > 0) and is_shrink
    # 5. 收盘不破EMA20拿着（持有规则，同 s_hold）
    # 6. 缩量假破位
    s_false_break = below and is_shrink
    # 7. 放量真破位（第二天必须走）
    s_real_break = below and is_expand

    return {
        "线上持有": {
            "triggered": bool(s_hold), "action": "持有" if s_hold else "不适用（线下）",
            "detail": f"收盘{close:.2f} {('>' if above else '<')} EMA20 {ema:.2f}",
        },
        "线上回踩机会(进仓点)": {
            "triggered": bool(s_pullback_buy), "action": "建仓（买入）",
            "detail": f"线上{'是' if above else '否'}/碰到线{'是' if low<=touch_high else '否'}/撑住{'是' if close>ema else '否'}",
        },
        "线下反抽陷阱(出货点)": {
            "triggered": bool(s_rebound_trap), "action": "卖出/不抄底",
            "detail": f"线下{'是' if below else '否'}/碰到线{'是' if high>=touch_low else '否'}/压住{'是' if close<ema else '否'}",
        },
        "三条件建仓(EMA20站稳+MACD零轴上方+缩量)": {
            "triggered": bool(s_three_cond), "action": "建仓（买入）",
            "detail": f"站稳{'是' if above else '否'}/MACD零轴上方{'是' if dea>0 else '否'}(DEA {dea:.3f})/缩量{'是' if is_shrink else '否'}",
        },
        "缩量假破位": {
            "triggered": bool(s_false_break), "action": "观察（可能假破位）",
            "detail": f"线下{'是' if below else '否'}/缩量{'是' if is_shrink else '否'}(量比{last['vol_ratio']:.2f})",
        },
        "放量真破位(第二天必须走)": {
            "triggered": bool(s_real_break), "action": "次日开盘清仓",
            "detail": f"线下{'是' if below else '否'}/放量{'是' if is_expand else '否'}(量比{last['vol_ratio']:.2f})",
        },
        "_meta": {
            "dea": float(dea), "dif": float(dif), "above": bool(above),
            "vol_ratio": float(last["vol_ratio"]),
        },
    }


def detect_oscillation(df: pd.DataFrame) -> dict:
    """震荡市检测（⚠️可配置，默认值）。仅作提示，不强制。
    近 OSC_LOOKBACK 日内 close 穿越 ema20 次数 ≥ OSC_CROSS_MIN 且振幅在 ±OSC_AMP 内。
    """
    look = df.tail(CONFIG["OSC_LOOKBACK"]).copy()
    if len(look) < CONFIG["OSC_LOOKBACK"]:
        return {"is_osc": False, "crosses": 0, "note": "数据不足"}
    crosses = 0
    signs = (look["close"] > look["ema20"]).astype(int).diff().abs()
    crosses = int(signs.sum())
    # 振幅：偏离 ema20 的最大百分比
    dev = (look["close"] - look["ema20"]) / look["ema20"] * 100
    max_dev = float(dev.abs().max())
    is_osc = (crosses >= CONFIG["OSC_CROSS_MIN"]) and (max_dev <= CONFIG["OSC_AMP"] * 100)
    return {
        "is_osc": bool(is_osc), "crosses": crosses, "max_dev_pct": max_dev,
        "note": f"近{CONFIG['OSC_LOOKBACK']}日穿越{crosses}次/最大偏离{max_dev:.2f}%",
    }


# =========================================================================
# 报告生成区
# =========================================================================
def build_report(code: str, df: pd.DataFrame, primary: dict, confirmation: dict, osc: dict, style: str = "long") -> str:
    """生成 Markdown 报告字符串。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last = df.iloc[-1]
    ema = last["ema20"]
    close = last["close"]
    bias = last["bias"]
    vol_ratio = last["vol_ratio"]
    dea = last["dea"]
    dif = last["dif"]
    macd_bar = last["macd"]

    cm = primary.get("_meta", {})
    bm = confirmation.get("_meta", {})

    lines = []
    lines.append(f"# {code} EMA20 判断报告\n")
    lines.append(f"> 生成时间：{now}  |  规则：EMA20 趋势与缩量回踩\n")
    p = STYLE_PROFILES[style]
    bias_threshold = primary.get("_meta", {}).get("bias_threshold", 0)
    lines.append(f"> 筛选档位：{ACTIVE_PROFILE} | 风格：{p['label']} | 自适应乖离阈值：{bias_threshold:.2f}%"
                 f"（{p['bias_multiplier']}倍20日平均偏离，上下限{p['bias_lower']:.0f}%/{p['bias_upper']:.0f}%）\n")
    lines.append("")

    # 一、当前价量快照
    lines.append("## 一、当前价量快照\n")
    lines.append(f"| 指标 | 数值 | 说明 |")
    lines.append(f"| --- | --- | --- |")
    lines.append(f"| 最新价 | {close:.2f} | 当日实时/最新收盘 |")
    lines.append(f"| EMA20 | {ema:.2f} | 指数均线(span={CONFIG['EMA_PERIOD']}) |")
    lines.append(f"| 乖离率 | {bias:.2f}% | (收盘-EMA20)/EMA20×100 |")
    lines.append(f"| 量比 | {vol_ratio:.2f} | 当日量/5日均量 |")
    lines.append(f"| MACD | DIF {dif:.3f} / DEA {dea:.3f} / 柱 {macd_bar:.3f} | DEA{'>' if dea>0 else '<'}0 即零轴{'上方' if dea>0 else '下方'} |")
    lines.append(f"| 日期 | {last['date'].strftime('%Y-%m-%d')} | 数据最新一日 |")
    lines.append("")
    lines.append("> 数据来源：akshare 日线（前复权）；未指定截止日且未禁用实时数据时，可能追加新浪行情\n")
    lines.append("")

    # 二、缩量回踩信号
    lines.append("## 二、缩量回踩信号\n")
    if "error" in primary:
        lines.append(f"⚠️ {primary['error']}\n")
    else:
        lines.append("| 规则 | 触发 | 依据数据 | 建议动作 |")
        lines.append("| --- | :---: | --- | --- |")
        for k, v in primary.items():
            if k.startswith("_"):
                continue
            trig = "✅ 是" if v["triggered"] else "— 否"
            lines.append(f"| {k} | {trig} | {v['detail']} | {v['action']} |")
        lines.append("")

    # 三、趋势确认信号
    lines.append("## 三、趋势确认信号\n")
    if "error" in confirmation:
        lines.append(f"⚠️ {confirmation['error']}\n")
    else:
        lines.append("| 信号 | 触发 | 依据数据 | 建议动作 |")
        lines.append("| --- | :---: | --- | --- |")
        for k, v in confirmation.items():
            if k.startswith("_"):
                continue
            trig = "✅ 是" if v["triggered"] else "— 否"
            lines.append(f"| {k} | {trig} | {v['detail']} | {v['action']} |")
        lines.append("")

    # 四、综合建议
    lines.append("## 四、综合建议\n")
    lines.append("（依用户已确认的冲突优先级）\n")

    # 建仓方向：缩量回踩规则为主，趋势确认规则为辅
    c1 = primary.get("规则一 缩量回踩定买点", {}).get("triggered", False)
    c2 = primary.get("规则二 触碰20日线未跌穿", {}).get("triggered", False)
    b3 = confirmation.get("三条件建仓(EMA20站稳+MACD零轴上方+缩量)", {}).get("triggered", False)
    bp = confirmation.get("线上回踩机会(进仓点)", {}).get("triggered", False)

    if c1 and c2:
        build_txt = "✅ 缩量回踩买点成立（主信号）"
        if b3:
            build_txt += " + 趋势确认规则三条件同步满足（辅助确认加强）→ 建仓信号较强"
        elif bp:
            build_txt += " + 趋势确认规则线上回踩机会（辅助确认）→ 可建仓"
        else:
            build_txt += "；趋势确认规则三条件未全部满足（辅助确认缺失），谨慎建仓"
    elif c1 or c2:
        build_txt = "△ 缩量回踩规则买点部分成立，待确认；趋势确认规则三条件：" + ("满足" if b3 else "未满足")
    else:
        build_txt = "— 缩量回踩买点未成立（主信号未触发）" + ("；趋势确认规则三条件满足但需缩量回踩规则主信号领" if b3 else "")

    lines.append(f"**建仓方向**（缩量回踩规则为主、趋势确认规则为辅）：{build_txt}\n")

    # 清仓方向：双方并列，不出单一建议
    clear_days = STYLE_PROFILES[style]["clear_days"]
    c5_key = f"规则五 {clear_days}日不站回全部清仓"
    c5 = primary.get(c5_key, {}).get("triggered", False)
    c4 = primary.get("规则四 破线先减半仓", {}).get("triggered", False)
    b_real = confirmation.get("放量真破位(第二天必须走)", {}).get("triggered", False)
    b_false = confirmation.get("缩量假破位", {}).get("triggered", False)

    lines.append(f"**清仓方向**（⚠️ 双方并列显示，不出单一综合建议，由你看数据自定）：\n")
    lines.append(f"- 趋势确认规则：{'✅ 放量跌破→第二天必须走' if b_real else '— 未触发放量跌破'}"
                 + (f'（缩量跌破→可能假破位，观察）' if b_false else ''))
    lines.append(f"- 缩量回踩规则：{'✅ 跌破' + str(clear_days) + '日不站回→全部清仓' if c5 else '— 未触发' + str(clear_days) + '日清仓'}"
                 + (f'（破线→先减半仓）' if c4 and not c5 else ''))
    lines.append("")

    # 震荡市提示
    lines.append(f"**震荡市提示**（趋势确认，仅作提示，不强制阻止）：")
    if osc.get("is_osc"):
        lines.append(f"⚠️ 当前判定为震荡市（{osc['note']}）。若近期已连续止损 2 次，趋势确认规则建议果断放弃该标的。"
                     "（注：工具无法获知你的实际止损记录，此处仅按规则提示）")
    else:
        lines.append(f"— 当前非震荡市（{osc['note']}），该规则不触发。")
    lines.append("")

    # 五、参数与免责
    lines.append("## 五、参数与免责\n")
    lines.append("**所用阈值清单**（可配置者用默认值，可在 CONFIG 区校准）：\n")
    lines.append("| 参数 | 当前值 | 配置说明 |")
    lines.append("| --- | --- | :---: |")
    lines.append(f"| EMA 周期 | {CONFIG['EMA_PERIOD']} | ✅ 固定参考周期 |")
    lines.append(f"| 碰到线阈值 | ±{CONFIG['TOUCH_PCT']*100:.0f}% | ⚠️ 可配置 |")
    lines.append(f"| 缩量倍数 | {CONFIG['SHRINK_RATIO']} | ⚠️ 可配置 |")
    lines.append(f"| 放量倍数 | {CONFIG['EXPAND_RATIO']} | ⚠️ 可配置 |")
    lines.append(f"| 乖离过大阈值 | 自适应 {bias_threshold:.2f}% | {p['bias_multiplier']}倍20日平均偏离，{p['bias_lower']:.0f}%-{p['bias_upper']:.0f}% |")
    lines.append(f"| 震荡市穿越次数 | {CONFIG['OSC_CROSS_MIN']} | ⚠️ 可配置 |")
    lines.append(f"| 震荡市振幅 | ±{CONFIG['OSC_AMP']*100:.0f}% | ⚠️ 可配置 |")
    lines.append(f"| 清仓倒计时 | {STYLE_PROFILES[style]['clear_days']}日 | {STYLE_PROFILES[style]['label']}风格 |")
    lines.append(f"| MACD 参数 | {CONFIG['MACD_FAST']},{CONFIG['MACD_SLOW']},{CONFIG['MACD_SIGNAL']} | ✅ 业界常用 |")
    lines.append("")
    lines.append(f"{RISK_DISCLAIMER}\n")

    return "\n".join(lines)


# =========================================================================
# 主入口
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="EMA20 判断工具")
    parser.add_argument("code", help="6 位股票代码，如 600596")
    parser.add_argument("--no-realtime", action="store_true", help="不取当日实时行情")
    parser.add_argument("--out", default="", help="输出 Markdown 文件路径（默认同目录自动命名）")
    parser.add_argument("--history-days", "--days", dest="days", type=int,
                        default=CONFIG["HISTORY_DAYS"], help="最多保留的历史日线行数，默认60")
    parser.add_argument("--as-of", default=None, help="日线截止日 YYYY-MM-DD；禁用当前实时行情")
    parser.add_argument("--profile", choices=list(SCREEN_PROFILES), default="standard",
                        help="缩量回踩筛选档位，与 --style 独立")
    parser.add_argument("--style", choices=["long", "short"], default="long",
                        help="交易风格：long=长线短炒 / short=1-3日超短线")
    args = parser.parse_args()

    code = args.code.strip()
    style = args.style
    apply_profile(args.profile)
    if args.days < CONFIG["EMA_PERIOD"] + 5:
        parser.error("--history-days 至少为 EMA_PERIOD + 5")
    if args.as_of:
        try:
            cutoff = datetime.strptime(args.as_of, "%Y-%m-%d")
            if cutoff.date() > datetime.now().date():
                parser.error("--as-of 不能晚于今天")
        except ValueError:
            parser.error("--as-of 格式应为 YYYY-MM-DD")
    if not re.match(r"^\d{6}$", code):
        print("错误：股票代码需为 6 位数字")
        sys.exit(1)

    # 1. 取数
    try:
        df = fetch_data(code, days=args.days, realtime=not args.no_realtime, as_of=args.as_of)
    except RuntimeError as e:
        print(f"❌ 取数失败：{e}")
        print("工具绝不编造数据，已终止。请检查网络/股票代码/akshare 安装。")
        sys.exit(2)

    if len(df) < CONFIG["EMA_PERIOD"] + 5:
        print(f"❌ 数据不足（仅 {len(df)} 行），需至少 {CONFIG['EMA_PERIOD']+5} 行")
        sys.exit(3)

    # 2. 计算指标
    df = compute_ema20(df)
    df = compute_macd(df)
    df = compute_volume_features(df)

    # 3. 信号检测
    bias_threshold = compute_adaptive_bias_threshold(df, style)
    print(f"风格：{STYLE_PROFILES[style]['label']} | 自适应乖离阈值：{bias_threshold:.2f}%")
    primary = primary_signals(df, style)
    confirmation = confirmation_signals(df)
    osc = detect_oscillation(df)

    # 4. 生成报告
    report = build_report(code, df, primary, confirmation, osc, style)

    report += f"\n> 请求截止日：{args.as_of or '当前'}；档位：{args.profile}。指定截止日只截断日线，不构成时点数据库或完整历史回测。\n"

    # 5. 写文件
    if args.out:
        out_path = args.out
    else:
        date_str = datetime.now().strftime("%Y%m%d")
        out_path = f"ema20判断报告_{code}_{date_str}_{style}.md"
    # 输出到脚本所在目录
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ 报告已生成：{out_path}\n")
    print("=" * 60)
    print(report[:2000] + ("\n...（完整内容见文件）" if len(report) > 2000 else ""))
    print("=" * 60)


if __name__ == "__main__":
    main()
