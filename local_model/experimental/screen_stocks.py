"""
选股筛选器 v2.0 - 基于9条标准筛选A股

数据源:
  - 股票列表: ak.stock_info_a_code_name()
  - 财务数据: ak.stock_yjbb_em() (季报+年报)
  - 历史K线: ak.stock_zh_a_daily() (新浪源)
  - 实时价格: 新浪 hq.sinajs.cn 接口

筛选标准:
  条件1: 所选季报营收同比>0 且 净利润同比>0（不直接观测订单）
  条件2: 股价 ≤ 10 元
  条件3: 季报+年报 营收同比>0 且 净利润同比>0
  条件4: 国企/央企/国资参股（行业代理标注）
  条件5: 流通市值 ≤ 500 亿（≤ 100 亿最佳）
  条件6: 窗口振幅、接近低点、均量比与近20日涨跌幅（阈值见配置）
  条件7+9: 分A池(PB≤2, PE≤20) / B池(PB≤3, PE≤30)
  条件8: 周线+月线阳线（放宽：周月线收盘上涨）
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import os
import sys
import requests
import re

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
MAX_PRICE = 10.0              # 条件2: 股价上限(元)
MAX_MARKET_CAP = 500          # 条件5: 流通市值上限(亿元)
BEST_MARKET_CAP = 100         # 条件5: 最佳流通市值(亿元)
CONSOLIDATION_YEARS = 5       # 条件6: 盘整年数(5年)
AMPLITUDE_LIMIT = 4.0         # 条件6: 振幅上限(最高/最低, 4倍以内)
VOLUME_RATIO_THRESHOLD = 1.0  # 条件6: 放量阈值(≥1.0即成交量不萎缩)
STAGNATION_MAX_CHANGE = 0.20  # 条件6: 滞涨最大涨幅(20%)
NEAR_LOW_FACTOR = 2.0         # 条件6: 股价接近5年低点(不超过低点的2.0倍)
A_PB_MAX = 2.0                # A池PB上限
A_PE_MAX = 20.0               # A池PE上限
B_PB_MAX = 3.0                # B池PB上限
B_PE_MAX = 30.0               # B池PE上限
WEEKLY_MONTHLY_STRICT = False  # 条件8: True=阳线, False=上涨(放宽)
OUTPUT_DIR = r"04_输出结果\01选股结果"

# 手工选择待尝试的报表期：(季报/报告期末, 年报期末)，按优先级从前到后。
# 期末日期不等于实际披露日期；不构成历史时点数据。
FINANCIAL_PERIODS = [
    ("20260630", "20251231"),
    ("20260331", "20251231"),
    ("20251231", "20241231"),
    ("20250930", "20241231"),
]

# 国资主导行业（作为国资代理标注）
STATE_DOMINANT_INDUSTRIES = [
    "银行", "保险", "石油", "煤炭", "电力", "电信", "铁路", "军工",
    "有色金属", "钢铁", "化工", "交通运输", "公用事业", "环保",
    "建筑施工", "建筑装饰", "电力设备", "国防军工",
]


# ==================== 工具函数 ====================
def retry_fetch(func, max_retries=3, delay=2, desc=""):
    """带重试的数据获取"""
    for attempt in range(max_retries):
        try:
            result = func()
            if result is not None and (not isinstance(result, pd.DataFrame) or not result.empty):
                return result
            if isinstance(result, pd.DataFrame) and result.empty:
                return result  # 空DataFrame也算有效结果
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print(f"    [{desc}] 获取失败: {type(e).__name__}")
                return None
    return None


def get_latest_financial_dates():
    """选择配置列表中首个有数据的报告期；不会自动发现最新报告期。"""
    for q_date, a_date in FINANCIAL_PERIODS:
        try:
            df_q = ak.stock_yjbb_em(date=q_date)
            if df_q is not None and not df_q.empty:
                print(f"  财务数据: 季报={q_date}({len(df_q)}条), 年报={a_date}")
                return q_date, a_date
        except Exception:
            continue
    return None, None


def fetch_realtime_prices(symbols, batch_size=80):
    """
    批量获取新浪实时行情
    返回 DataFrame: symbol, name, price, open, pre_close, high, low, volume, amount
    """
    results = []
    total_batches = (len(symbols) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(symbols))
        batch = symbols[start:end]

        # 构造新浪代码
        sina_codes = []
        for s in batch:
            if s.startswith("6") or s.startswith("9"):
                sina_codes.append(f"sh{s}")
            else:
                sina_codes.append(f"sz{s}")

        code_str = ",".join(sina_codes)
        url = f"https://hq.sinajs.cn/list={code_str}"

        try:
            resp = requests.get(url, headers={"Referer": "https://finance.sina.com.cn"}, timeout=15)
            resp.encoding = "gbk"

            # 解析每一行
            lines = resp.text.strip().split("\n")
            for line in lines:
                match = re.search(r'hq_str_(\w+)="(.+)"', line)
                if not match:
                    continue
                sina_code = match.group(1)
                fields = match.group(2).split(",")
                if len(fields) >= 32 and fields[0]:
                    results.append({
                        "symbol": sina_code[2:],  # 去掉 sh/sz 前缀
                        "name": fields[0],
                        "open": float(fields[1]) if fields[1] else 0,
                        "pre_close": float(fields[2]) if fields[2] else 0,
                        "price": float(fields[3]) if fields[3] else 0,
                        "high": float(fields[4]) if fields[4] else 0,
                        "low": float(fields[5]) if fields[5] else 0,
                        "volume": float(fields[8]) if fields[8] else 0,
                        "amount": float(fields[9]) if fields[9] else 0,
                        "date": fields[30],
                        "time": fields[31],
                    })
        except Exception as e:
            print(f"    批量获取失败(batch {batch_idx}): {type(e).__name__}")

        if (batch_idx + 1) % 5 == 0:
            print(f"    进度: {batch_idx+1}/{total_batches} 批")
            time.sleep(0.3)

    df = pd.DataFrame(results)
    if not df.empty:
        df["pct_change"] = (df["price"] - df["pre_close"]) / df["pre_close"] * 100
    return df


def get_outstanding_share(symbol):
    """获取单只股票的流通股数(从历史数据中提取)"""
    if symbol.startswith("6") or symbol.startswith("9"):
        sina_code = f"sh{symbol}"
    else:
        sina_code = f"sz{symbol}"

    try:
        df = ak.stock_zh_a_daily(symbol=sina_code, adjust="qfq")
        if df is not None and not df.empty and "outstanding_share" in df.columns:
            return df["outstanding_share"].iloc[-1]
    except Exception:
        pass
    return None


# ==================== 步骤1: 获取A股列表 ====================
def step1_get_stock_list():
    print("\n" + "=" * 60)
    print("📋 步骤1: 获取A股全列表")
    print("=" * 60)

    df = retry_fetch(lambda: ak.stock_info_a_code_name(), desc="股票列表")
    if df is None or df.empty:
        print("  ❌ 无法获取A股列表")
        return pd.DataFrame()

    n_total = len(df)

    # 过滤ST/*ST/退市
    df = df[~df["name"].str.contains("ST|退", na=False)].copy()
    # 过滤北交所(8/4/9开头)
    df = df[~df["code"].str.startswith(("8", "4", "9"))].copy()
    # 过滤B股(900开头)
    df = df[~df["code"].str.startswith("9")].copy()

    df = df.rename(columns={"code": "symbol", "name": "name"})
    df = df[["symbol", "name"]].reset_index(drop=True)

    print(f"  全A股: {n_total} 只")
    print(f"  过滤后(排除ST/退市/北交所): {len(df)} 只")
    return df


# ==================== 步骤2: 获取财务数据 ====================
def step2_get_financials(df):
    print("\n" + "=" * 60)
    print("📊 步骤2: 获取财务数据")
    print("=" * 60)

    q_date, a_date = get_latest_financial_dates()
    if q_date is None:
        print("  ❌ 无法获取财务数据")
        return pd.DataFrame()

    # 获取季报
    print(f"  获取季报({q_date})...")
    df_q = retry_fetch(lambda: ak.stock_yjbb_em(date=q_date), desc=f"季报{q_date}")
    if df_q is None or df_q.empty:
        print("  ❌ 季报数据获取失败")
        return pd.DataFrame()

    # 获取年报
    print(f"  获取年报({a_date})...")
    df_a = retry_fetch(lambda: ak.stock_yjbb_em(date=a_date), desc=f"年报{a_date}")

    # 标准化季报列
    q_col_map = {}
    for col in df_q.columns:
        if "股票代码" in col:
            q_col_map[col] = "symbol"
        elif "营业总收入-同比增长" in col:
            q_col_map[col] = "q_revenue_yoy"
        elif "净利润-同比增长" in col:
            q_col_map[col] = "q_profit_yoy"
        elif "每股净资产" in col:
            q_col_map[col] = "q_bps"
        elif "每股收益" in col:
            q_col_map[col] = "q_eps"
        elif "净资产收益率" in col:
            q_col_map[col] = "q_roe"
        elif "所处行业" in col:
            q_col_map[col] = "industry"
    df_q = df_q.rename(columns=q_col_map)

    # 标准化年报列
    if df_a is not None and not df_a.empty:
        a_col_map = {}
        for col in df_a.columns:
            if "股票代码" in col:
                a_col_map[col] = "symbol"
            elif "营业总收入-同比增长" in col:
                a_col_map[col] = "a_revenue_yoy"
            elif "净利润-同比增长" in col:
                a_col_map[col] = "a_profit_yoy"
        df_a = df_a.rename(columns=a_col_map)

    # 合并季报到主表
    needed_q = ["symbol"] + [c for c in ["q_revenue_yoy", "q_profit_yoy", "q_bps", "q_eps", "q_roe", "industry"] if c in df_q.columns]
    df = df.merge(df_q[needed_q], on="symbol", how="inner")
    print(f"  匹配季报: {len(df)} 只")

    # 合并年报
    if df_a is not None and not df_a.empty:
        needed_a = ["symbol"] + [c for c in ["a_revenue_yoy", "a_profit_yoy"] if c in df_a.columns]
        if len(needed_a) > 1:
            df = df.merge(df_a[needed_a], on="symbol", how="left")
            print(f"  匹配年报: {len(df)} 只")

    # 转换数值
    for col in ["q_revenue_yoy", "q_profit_yoy", "a_revenue_yoy", "a_profit_yoy", "q_bps", "q_eps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    #  akshare返回的增长率可能是百分数(如7.357157表示7.36%)，也可能是小数
    # 检查: 如果绝对值大多>1，说明是百分数形式
    for col in ["q_revenue_yoy", "q_profit_yoy"]:
        if col in df.columns:
            sample = df[col].dropna().head(100)
            if len(sample) > 0 and sample.abs().mean() > 1:
                # 已经是百分数形式(如7.36表示7.36%)，除以100转为小数
                df[col] = df[col] / 100

    for col in ["a_revenue_yoy", "a_profit_yoy"]:
        if col in df.columns:
            sample = df[col].dropna().head(100)
            if len(sample) > 0 and sample.abs().mean() > 1:
                df[col] = df[col] / 100

    print(f"  数据就绪: {len(df)} 只")
    return df


# ==================== 步骤3: 财务筛选(条件1+3) ====================
def step3_filter_financials(df):
    print("\n" + "=" * 60)
    print("📈 步骤3: 财务筛选(条件1+3)")
    print("=" * 60)

    n_before = len(df)

    # 条件1: 季报 营收同比>0 且 净利润同比>0
    if "q_revenue_yoy" in df.columns and "q_profit_yoy" in df.columns:
        df = df[(df["q_revenue_yoy"] > 0) & (df["q_profit_yoy"] > 0)].copy()
        print(f"  条件1(季报营收+净利双增>0): {len(df)} 只 (剔除 {n_before - len(df)})")

    n_before = len(df)

    # 条件3: 年报 营收同比>0 且 净利润同比>0 (缺失则保留)
    if "a_revenue_yoy" in df.columns and "a_profit_yoy" in df.columns:
        mask_pass = (df["a_revenue_yoy"] > 0) & (df["a_profit_yoy"] > 0)
        mask_missing = df["a_revenue_yoy"].isna()
        df = df[mask_pass | mask_missing].copy()
        n_pass = mask_pass.sum()
        n_missing = mask_missing.sum()
        print(f"  条件3(年报双增): {n_pass}只通过, {n_missing}只年报缺失(保留)")
        print(f"  小计: {len(df)} 只")

    return df


# ==================== 步骤4: 获取实时价格+估值筛选 ====================
def step4_filter_by_price_and_valuation(df):
    print("\n" + "=" * 60)
    print("💰 步骤4: 实时价格+估值筛选")
    print("=" * 60)

    symbols = df["symbol"].tolist()
    print(f"  批量获取 {len(symbols)} 只股票的实时行情...")

    # 批量获取实时行情
    df_rt = fetch_realtime_prices(symbols)
    if df_rt.empty:
        print("  ❌ 实时行情获取失败")
        return pd.DataFrame()

    print(f"  获取到 {len(df_rt)} 只股票的实时行情")

    # 合并到主表
    df = df.merge(df_rt[["symbol", "price", "pct_change", "high", "low", "open", "pre_close"]],
                  on="symbol", how="inner")
    print(f"  匹配行情: {len(df)} 只")

    # 条件2: 股价 ≤ 10
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"] > 0].copy()  # 排除停牌
    n_before = len(df)
    df = df[df["price"] <= MAX_PRICE].copy()
    print(f"  条件2(股价≤{MAX_PRICE}元): {len(df)} 只 (剔除 {n_before - len(df)})")

    # 计算PB和PE
    if "q_bps" in df.columns:
        df["pb"] = df["price"] / df["q_bps"]
    else:
        df["pb"] = np.nan

    if "q_eps" in df.columns:
        df["pe"] = df["price"] / df["q_eps"]
    else:
        df["pe"] = np.nan

    # 过滤有效PB/PE
    df_valid = df[df["pb"].notna() & (df["pb"] > 0) & df["pe"].notna() & (df["pe"] > 0) & (df["pe"] < 500)].copy()
    print(f"  有效PB/PE: {len(df_valid)} 只")

    # 条件7+9: 分A/B池
    pool_a = df_valid[(df_valid["pb"] <= A_PB_MAX) & (df_valid["pe"] <= A_PE_MAX)].copy()
    pool_b = df_valid[(df_valid["pb"] <= B_PB_MAX) & (df_valid["pe"] <= B_PE_MAX)].copy()

    print(f"  A池(PB≤{A_PB_MAX}, PE≤{A_PE_MAX}): {len(pool_a)} 只")
    print(f"  B池(PB≤{B_PB_MAX}, PE≤{B_PE_MAX}): {len(pool_b)} 只")

    # 合并(B池包含A池)
    pool_a["pool"] = "A"
    pool_b["pool"] = "B"
    df = pd.concat([pool_a, pool_b]).drop_duplicates(subset=["symbol"])

    return df


# ==================== 步骤5: 获取流通市值+过滤 ====================
def step5_filter_by_market_cap(df):
    print("\n" + "=" * 60)
    print("📊 步骤5: 流通市值筛选")
    print("=" * 60)

    print(f"  获取 {len(df)} 只股票的流通股数...")

    # 逐个获取流通股数
    outstanding_shares = {}
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        symbol = row["symbol"]
        share = get_outstanding_share(symbol)
        if share is not None and share > 0:
            outstanding_shares[symbol] = share

        if (idx + 1) % 20 == 0:
            print(f"    进度: {idx+1}/{total}")
            time.sleep(0.5)

    print(f"    获取到 {len(outstanding_shares)} 只股票的流通股数")

    # 计算流通市值
    df["outstanding_share"] = df["symbol"].map(outstanding_shares)
    df["outstanding_share"] = pd.to_numeric(df["outstanding_share"], errors="coerce")
    df["float_mv"] = df["price"] * df["outstanding_share"] / 1e8  # 转为亿元

    # 过滤
    df = df[df["float_mv"].notna() & (df["float_mv"] > 0)].copy()
    print(f"    有效市值: {len(df)} 只")

    n_before = len(df)
    df = df[df["float_mv"] <= MAX_MARKET_CAP].copy()
    print(f"  条件5(流通市值≤{MAX_MARKET_CAP}亿): {len(df)} 只 (剔除 {n_before - len(df)})")

    # 标注100亿最佳
    df["cap_level"] = df["float_mv"].apply(
        lambda x: "★100亿内" if x <= BEST_MARKET_CAP else f"{BEST_MARKET_CAP}-{MAX_MARKET_CAP}亿"
    )

    n_best = (df["float_mv"] <= BEST_MARKET_CAP).sum()
    print(f"    其中★100亿内: {n_best} 只")

    return df


# ==================== 步骤6: 技术形态筛选 ====================
def step6_filter_by_technicals(df):
    print("\n" + "=" * 60)
    print("📈 步骤6: 技术形态筛选(底部盘整+周月线齐涨)")
    print("=" * 60)
    print(f"  待筛选: {len(df)} 只")

    results = []
    partial_6_only = []
    partial_8_only = []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        symbol = row["symbol"]
        name = row["name"]
        progress = f"[{idx+1}/{total}]"

        # 获取历史K线
        if symbol.startswith("6") or symbol.startswith("9"):
            sina_code = f"sh{symbol}"
        else:
            sina_code = f"sz{symbol}"

        df_hist = retry_fetch(
            lambda sc=sina_code: ak.stock_zh_a_daily(symbol=sc, adjust="qfq"),
            max_retries=2, delay=1, desc=f"{symbol}历史K线"
        )

        if df_hist is None or df_hist.empty:
            print(f"    {progress} {name}({symbol}) ❌ 历史数据获取失败")
            continue

        # 标准化列名
        col_rename = {"date": "date", "open": "open", "high": "high", "low": "low",
                      "close": "close", "volume": "volume", "outstanding_share": "outstanding_share"}
        df_hist = df_hist.rename(columns=col_rename)

        # 条件6: 底部盘整+放量滞涨
        passed_6, reason_6 = check_consolidation(symbol, df_hist)

        # 条件8: 周月线齐涨
        result_8 = check_weekly_monthly(symbol, df_hist)
        passed_8 = not result_8.startswith("❌")

        if passed_6 and passed_8:
            entry = row.to_dict()
            entry["consolidation_info"] = reason_6
            entry["weekly_monthly_tag"] = result_8
            results.append(entry)
            print(f"    {progress} {name}({symbol}) ✅✅ | {reason_6} | {result_8}")
        elif passed_6 and not passed_8:
            partial_6_only.append((symbol, name, reason_6, result_8))
            print(f"    {progress} {name}({symbol}) ⚠️ 仅条件6通过 | {reason_6} | {result_8}")
        elif not passed_6 and passed_8:
            partial_8_only.append((symbol, name, reason_6, result_8))
            print(f"    {progress} {name}({symbol}) ⚠️ 仅条件8通过 | {reason_6} | {result_8}")
        else:
            print(f"    {progress} {name}({symbol}) ❌ {reason_6} | {result_8}")

        time.sleep(0.3)

    # 汇总
    if results:
        df_result = pd.DataFrame(results)
        print(f"\n  ✅ 全部通过: {len(df_result)} 只")
    else:
        df_result = pd.DataFrame()
        print(f"\n  ❌ 无股票全部通过")

    if partial_6_only:
        print(f"  ⚠️ 仅条件6通过: {len(partial_6_only)} 只")
        for s, n, r6, r8 in partial_6_only:
            print(f"     {n}({s}) | {r6} | {r8}")
    if partial_8_only:
        print(f"  ⚠️ 仅条件8通过: {len(partial_8_only)} 只")
        for s, n, r6, r8 in partial_8_only:
            print(f"     {n}({s}) | {r6} | {r8}")

    return df_result


def check_consolidation(symbol, df_hist):
    """检查底部盘整+放量滞涨"""
    if df_hist is None or len(df_hist) < 200:
        return False, "数据不足"

    df = df_hist.copy()
    if "date" not in df.columns:
        return False, "无日期列"

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 取最近CONSOLIDATION_YEARS年的数据
    cutoff_date = df["date"].max() - timedelta(days=CONSOLIDATION_YEARS * 365)
    df_5y = df[df["date"] >= cutoff_date].copy()

    if len(df_5y) < 300:
        return False, f"{CONSOLIDATION_YEARS}年数据不足({len(df_5y)}条)"

    # 计算振幅
    high_max = df_5y["high"].max()
    low_min = df_5y["low"].min()
    amplitude = high_max / low_min

    if amplitude > AMPLITUDE_LIMIT:
        return False, f"振幅{amplitude:.2f}>{AMPLITUDE_LIMIT}"

    # 股价是否接近5年低点(不超过5年低点的NEAR_LOW_FACTOR倍)
    latest_close = df["close"].iloc[-1]
    if latest_close > low_min * NEAR_LOW_FACTOR:
        return False, f"距5年低点较远({latest_close/low_min:.1f}x>{NEAR_LOW_FACTOR}x)"

    # 放量滞涨
    if "volume" not in df.columns:
        return False, "无成交量"

    recent_20 = df.tail(20)
    recent_60 = df.tail(60)

    if len(recent_20) < 20 or len(recent_60) < 60:
        return False, "近期数据不足"

    vol_ratio = recent_20["volume"].mean() / recent_60["volume"].mean()

    if vol_ratio < VOLUME_RATIO_THRESHOLD:
        return False, f"量比{vol_ratio:.2f}<{VOLUME_RATIO_THRESHOLD}"

    # 近20日涨幅
    price_start = recent_20["close"].iloc[0]
    price_end = recent_20["close"].iloc[-1]
    change = (price_end - price_start) / price_start

    if abs(change) > STAGNATION_MAX_CHANGE:
        return False, f"近20日涨幅{change:+.2%}>±{STAGNATION_MAX_CHANGE:.0%}"

    return True, f"振幅{amplitude:.2f},位置{latest_close/low_min:.1f}x低点,量比{vol_ratio:.2f},涨幅{change:+.2%}"


def check_weekly_monthly(symbol, df_hist):
    """检查周线+月线齐涨"""
    if df_hist is None or len(df_hist) < 100:
        return "❌数据不足"

    df = df_hist.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 重采样为周线
    try:
        df_weekly = df.set_index("date").resample("W").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

        df_monthly = df.set_index("date").resample("ME").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
    except Exception:
        return "❌重采样失败"

    if len(df_weekly) < 2 or len(df_monthly) < 2:
        return "❌周期数据不足"

    # 最近一根周K线
    latest_week = df_weekly.iloc[-1]
    weekly_yang = latest_week["close"] > latest_week["open"]
    weekly_up = latest_week["close"] > df_weekly.iloc[-2]["close"]

    # 最近一根月K线
    latest_month = df_monthly.iloc[-1]
    monthly_yang = latest_month["close"] > latest_month["open"]
    monthly_up = latest_month["close"] > df_monthly.iloc[-2]["close"]

    if WEEKLY_MONTHLY_STRICT:
        if weekly_yang and monthly_yang:
            return "✅周月阳线"
        elif weekly_up and monthly_up:
            return "⚠️放宽:周月上涨"
        else:
            return f"❌周({weekly_yang})月({monthly_yang})均非阳线"
    else:
        if weekly_up and monthly_up:
            return "✅周月上涨"
        else:
            return f"❌周({weekly_up})月({monthly_up})未齐涨"


# ==================== 步骤7: 国资标注 ====================
def step7_mark_state_ownership(df):
    print("\n" + "=" * 60)
    print("🏛️ 步骤7: 国资标注")
    print("=" * 60)

    if df.empty:
        return df

    # 方法1: 使用概念板块数据识别国资/央企
    state_stocks = set()
    try:
        print("  正在获取央企/国企改革概念股...")
        # 央企改革成分股
        try:
            df_gj = ak.stock_board_concept_cons_em(symbol="央企改革")
            if df_gj is not None and not df_gj.empty:
                state_stocks.update(df_gj["代码"].tolist())
                print(f"    央企改革: {len(df_gj)} 只")
        except Exception as e:
            print(f"    央企改革获取失败: {e}")

        # 国企改革成分股
        try:
            df_gq = ak.stock_board_concept_cons_em(symbol="国企改革")
            if df_gq is not None and not df_gq.empty:
                state_stocks.update(df_gq["代码"].tolist())
                print(f"    国企改革: {len(df_gq)} 只")
        except Exception as e:
            print(f"    国企改革获取失败: {e}")

        # 地方国资
        try:
            df_local = ak.stock_board_concept_cons_em(symbol="地方国资")
            if df_local is not None and not df_local.empty:
                state_stocks.update(df_local["代码"].tolist())
                print(f"    地方国资: {len(df_local)} 只")
        except Exception as e:
            pass

        print(f"    合并去重后国资股: {len(state_stocks)} 只")
    except Exception as e:
        print(f"  概念板块获取异常: {e}")

    # 方法2: 如果概念板块没有数据，回退到行业代理法
    if not state_stocks and "industry" in df.columns:
        print("  回退到行业代理法...")
        state_stocks = set(
            df[df["industry"].apply(
                lambda x: any(ind in str(x) for ind in STATE_DOMINANT_INDUSTRIES) if pd.notna(x) else False
            )]["symbol"].tolist()
        )

    # 标注
    df["is_state"] = df["symbol"].isin(state_stocks)
    n_state = df["is_state"].sum()
    print(f"  国资标注: {n_state} 只国资背景")

    if n_state == 0:
        print("  ⚠️ 未识别到国资股，可能数据源受限，建议人工复核")

    return df


# ==================== 主流程 ====================
def run_screening(test_symbol=None):
    start_time = time.time()

    print("=" * 60)
    print("🚀 A股选股筛选器 v2.0")
    print("=" * 60)
    print(f"  筛选标准: 9条 (含国资标注)")
    print(f"  股价上限: {MAX_PRICE}元 | 市值上限: {MAX_MARKET_CAP}亿")
    print(f"  估值分池: A(PB≤{A_PB_MAX},PE≤{A_PE_MAX}) B(PB≤{B_PB_MAX},PE≤{B_PE_MAX})")

    if test_symbol:
        print(f"\n🧪 测试模式: {test_symbol}")
        print("   (用于验证程序可用性)")

    # 步骤1: 获取A股列表
    df = step1_get_stock_list()
    if df.empty:
        return None

    if test_symbol:
        df = df[df["symbol"] == test_symbol].copy()
        if df.empty:
            print(f"  ❌ 未找到股票 {test_symbol}")
            return None

    # 步骤2: 获取财务数据
    df = step2_get_financials(df)
    if df.empty:
        return None

    # 步骤3: 财务筛选
    df = step3_filter_financials(df)
    if df.empty:
        return None

    # 步骤4: 实时价格+估值筛选
    df = step4_filter_by_price_and_valuation(df)
    if df.empty:
        return None

    # 步骤5: 流通市值筛选
    df = step5_filter_by_market_cap(df)
    if df.empty:
        return None

    # 保存候选列表(技术形态筛选前)
    candidates_before_tech = df.copy()

    # 步骤6: 技术形态筛选
    df = step6_filter_by_technicals(df)
    if df.empty:
        # 即使技术形态没通过，也保存候选列表
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        base_cols = ["symbol", "name", "price", "pb", "pe", "float_mv", "cap_level",
                     "q_revenue_yoy", "q_profit_yoy", "a_revenue_yoy", "a_profit_yoy",
                     "industry", "pool"]
        save_cols = [c for c in base_cols if c in candidates_before_tech.columns]
        candidates_before_tech[save_cols].to_csv(
            os.path.join(OUTPUT_DIR, f"候选_技术形态前_{timestamp}.csv"),
            index=False, encoding="utf-8-sig"
        )
        print(f"  📁 候选列表已保存(共{len(candidates_before_tech)}只)")
        return None

    # 步骤7: 国资标注
    df = step7_mark_state_ownership(df)

    # 分池输出
    print("\n" + "=" * 60)
    print("📦 最终股票池")
    print("=" * 60)

    pool_a = df[df["pool"] == "A"].copy()
    a1 = pool_a[pool_a["is_state"]].copy()
    a2 = pool_a[~pool_a["is_state"]].copy()

    pool_b = df[df["pool"] == "B"].copy()
    b1 = pool_b[pool_b["is_state"]].copy()
    b2 = pool_b[~pool_b["is_state"]].copy()

    print(f"\n  🅰️ A池 (PB≤{A_PB_MAX}, PE≤{A_PE_MAX}): {len(pool_a)} 只")
    print(f"     A1(国资): {len(a1)} 只  |  A2(非国资): {len(a2)} 只")
    print(f"\n  🅱️ B池 (PB≤{B_PB_MAX}, PE≤{B_PE_MAX}): {len(pool_b)} 只")
    print(f"     B1(国资): {len(b1)} 只  |  B2(非国资): {len(b2)} 只")

    # 条件4严格池 vs 宽松池
    strict_count = len(a1) + len(b1)
    relaxed_count = len(pool_a) + len(pool_b)
    print(f"\n  📊 严格9条(含国资): {strict_count} 只")
    print(f"  📊 宽松8条(不含国资): {relaxed_count} 只")

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # 保存候选列表(技术形态筛选前的全部)
    base_cols_all = ["symbol", "name", "price", "pb", "pe", "float_mv", "cap_level",
                    "q_revenue_yoy", "q_profit_yoy", "a_revenue_yoy", "a_profit_yoy",
                    "industry", "pool"]
    save_cols_all = [c for c in base_cols_all if c in candidates_before_tech.columns]
    candidates_before_tech[save_cols_all].to_csv(
        os.path.join(OUTPUT_DIR, f"候选_技术形态前_{timestamp}.csv"),
        index=False, encoding="utf-8-sig"
    )

    base_cols = ["symbol", "name", "price", "pb", "pe", "float_mv", "cap_level",
                 "q_revenue_yoy", "q_profit_yoy", "a_revenue_yoy", "a_profit_yoy",
                 "industry", "is_state", "weekly_monthly_tag", "consolidation_info"]

    for pool_name, pool_df in [("A1_国资", a1), ("A2_非国资", a2), ("B1_国资", b1), ("B2_非国资", b2)]:
        if not pool_df.empty:
            save_cols = [c for c in base_cols if c in pool_df.columns]
            pool_df[save_cols].to_csv(
                os.path.join(OUTPUT_DIR, f"{pool_name}_{timestamp}.csv"),
                index=False, encoding="utf-8-sig"
            )

    # 打印详细结果
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    for pool_name, pool_df in [("A1_国资", a1), ("A2_非国资", a2), ("B1_国资", b1), ("B2_非国资", b2)]:
        if not pool_df.empty:
            print(f"\n  📋 {pool_name} ({len(pool_df)} 只):")
            show_cols = ["symbol", "name", "price", "pb", "pe", "float_mv",
                         "q_revenue_yoy", "q_profit_yoy", "weekly_monthly_tag"]
            available = [c for c in show_cols if c in pool_df.columns]
            if available:
                print(pool_df[available].to_string(index=False))

    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed:.1f} 秒")
    print(f"📁 结果已保存至: {os.path.abspath(OUTPUT_DIR)}/")

    return df


# ==================== 入口 ====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A股选股筛选器 v2.0")
    parser.add_argument("--test", type=str, help="测试模式: 指定单只股票代码")
    parser.add_argument("--full", action="store_true", help="全量筛选")
    args = parser.parse_args()

    if args.test:
        print(f"🧪 测试模式: {args.test}")
        run_screening(test_symbol=args.test)
    elif args.full:
        run_screening()
    else:
        print("默认全量筛选...")
        run_screening()
