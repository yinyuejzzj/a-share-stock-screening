"""
技术指标计算模块 v2.0 - 计算技术指标作为机器学习特征
v2.0 优化：
  - 新增 OBV/WR/CCI/ATR/换手率等指标
  - 所有特征尽量归一化/标准化，消除价格量纲差异
  - 新增时间特征（星期几、月末等）
  - 新增市场指数特征（大盘涨跌影响）
  - 新增交叉特征（量价配合度）
"""
import pandas as pd
import numpy as np


def add_all_indicators(df: pd.DataFrame, market_df: pd.DataFrame = None, stock_code: str = None) -> pd.DataFrame:
    """
    为 DataFrame 添加全部技术指标特征

    参数:
        df:         个股行情数据，需包含 date, open, close, high, low, volume, amount, pct_change
        market_df:  大盘指数数据（可选），用于添加市场环境特征
        stock_code: 保留的兼容参数，公开版不加载缠论增强特征

    输出: 在原 DataFrame 基础上新增各技术指标列
    """
    df = df.copy()
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_kdj(df)
    df = add_bollinger(df)
    df = add_volume_features(df)
    df = add_price_features(df)
    # v2.0 新增指标
    df = add_obv(df)
    df = add_williams_r(df)
    df = add_cci(df)
    df = add_atr(df)
    df = add_turnover_features(df)
    df = add_cross_features(df)
    df = add_time_features(df)
    if market_df is not None:
        df = add_market_features(df, market_df)
        # v3.0: 市场环境标签（牛市/熊市/震荡市），失败不影响主流程
        try:
            df = add_regime_label(df, market_df)
        except Exception as e:
            print(f"[WARN] add_regime_label 失败: {e}")
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """计算移动平均线 MA5 / MA10 / MA20 / MA60，并归一化为偏离百分比"""
    for window in [5, 10, 20, 60]:
        df[f"ma{window}"] = df["close"].rolling(window=window).mean()
        # v2.0: 归一化为价格相对均线的偏离百分比（消除价格量纲）
        df[f"ma{window}_bias"] = (df["close"] - df[f"ma{window}"]) / df[f"ma{window}"] * 100

    # 均线多头排列信号：MA5 > MA10 > MA20
    df["ma_bullish"] = ((df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"])).astype(int)
    # 金叉信号：MA5 上穿 MA20
    df["ma_cross"] = ((df["ma5"] > df["ma20"]) & (df["ma5"].shift(1) <= df["ma20"].shift(1))).astype(int)
    # 均线斜率（趋势方向）
    df["ma5_slope"] = (df["ma5"] - df["ma5"].shift(3)) / df["ma5"].shift(3) * 100
    df["ma20_slope"] = (df["ma20"] - df["ma20"].shift(5)) / df["ma20"].shift(5) * 100
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 RSI 相对强弱指标"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    # RSI 超买超卖信号
    df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
    # v2.0: RSI 变化趋势
    df["rsi_change"] = df["rsi"].diff(3)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MACD 指标"""
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = 2 * (df["macd_dif"] - df["macd_dea"])
    df["macd_cross"] = ((df["macd_dif"] > df["macd_dea"]) & (df["macd_dif"].shift(1) <= df["macd_dea"].shift(1))).astype(int)
    # v2.0: MACD 归一化（相对价格的比例）
    df["macd_dif_norm"] = df["macd_dif"] / df["close"] * 100
    df["macd_hist_norm"] = df["macd_hist"] / df["close"] * 100
    return df


def add_kdj(df: pd.DataFrame) -> pd.DataFrame:
    """计算 KDJ 随机指标"""
    low_min = df["low"].rolling(window=9, min_periods=9).min()
    high_max = df["high"].rolling(window=9, min_periods=9).max()
    rsv = (df["close"] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50.0)

    k = pd.Series(np.zeros(len(df)), index=df.index)
    d = pd.Series(np.zeros(len(df)), index=df.index)
    k.iloc[0] = 50.0
    d.iloc[0] = 50.0
    for i in range(1, len(df)):
        k.iloc[i] = 2 / 3 * k.iloc[i - 1] + 1 / 3 * rsv.iloc[i]
        d.iloc[i] = 2 / 3 * d.iloc[i - 1] + 1 / 3 * k.iloc[i]

    df["kdj_k"] = k
    df["kdj_d"] = d
    df["kdj_j"] = 3 * k - 2 * d
    # v2.0: KDJ 金叉/死叉
    df["kdj_cross"] = ((df["kdj_k"] > df["kdj_d"]) & (df["kdj_k"].shift(1) <= df["kdj_d"].shift(1))).astype(int)
    return df


def add_bollinger(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """计算布林带"""
    df["boll_mid"] = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    df["boll_upper"] = df["boll_mid"] + 2 * std
    df["boll_lower"] = df["boll_mid"] - 2 * std
    df["boll_pct"] = (df["close"] - df["boll_lower"]) / (df["boll_upper"] - df["boll_lower"]) * 100
    # v2.0: 布林带宽度（波动收缩信号）
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"] * 100
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算成交量相关特征"""
    df["vol_ma5"] = df["volume"].rolling(window=5).mean()
    df["vol_ma10"] = df["volume"].rolling(window=10).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma5"]
    df["vol_change"] = df["volume"].pct_change()
    # v2.0: 量能扩张信号
    df["vol_surge"] = (df["vol_ratio"] > 2.0).astype(int)
    # v2.0: 成交额归一化
    df["amount_ma5"] = df["amount"].rolling(window=5).mean()
    df["amount_ratio"] = df["amount"] / df["amount_ma5"]
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算价格相关特征"""
    df["price_change_1d"] = df["close"].pct_change(1)
    df["price_change_5d"] = df["close"].pct_change(5)
    df["price_change_10d"] = df["close"].pct_change(10)
    df["high_low_ratio"] = (df["high"] - df["low"]) / df["close"] * 100
    df["volatility_5d"] = df["close"].pct_change().rolling(window=5).std()
    df["volatility_20d"] = df["close"].pct_change().rolling(window=20).std()
    # v2.0: 上影线/下影线长度（反映多空博弈）
    body = (df["close"] - df["open"]).abs()
    total_range = df["high"] - df["low"]
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / total_range.replace(0, np.nan) * 100
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / total_range.replace(0, np.nan) * 100
    df["body_ratio"] = body / total_range.replace(0, np.nan) * 100
    # v2.0: 连涨/连跌天数
    df["consecutive_up"] = _consecutive_days(df["close"].diff() > 0)
    df["consecutive_down"] = _consecutive_days(df["close"].diff() < 0)
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """v2.0: 计算 OBV（能量潮）指标"""
    obv = pd.Series(np.zeros(len(df)), index=df.index)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    df["obv"] = obv
    # OBV 归一化：相对5日均量
    df["obv_norm"] = obv / df["vol_ma5"] / 100
    # OBV 趋势
    df["obv_ma5"] = obv.rolling(window=5).mean()
    df["obv_trend"] = (obv > df["obv_ma5"]).astype(int)
    return df


def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """v2.0: 计算 Williams %R 指标"""
    high_max = df["high"].rolling(window=period).max()
    low_min = df["low"].rolling(window=period).min()
    df["wr"] = (high_max - df["close"]) / (high_max - low_min) * -100
    # 超买超卖
    df["wr_oversold"] = (df["wr"] < -80).astype(int)
    df["wr_overbought"] = (df["wr"] > -20).astype(int)
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """v2.0: 计算 CCI（顺势指标）"""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_dev = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (typical_price - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """v2.0: 计算 ATR（真实波幅）"""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=period).mean()
    # 归一化：ATR 相对价格比例
    df["atr_pct"] = df["atr"] / df["close"] * 100
    return df


def add_turnover_features(df: pd.DataFrame) -> pd.DataFrame:
    """v2.0: 计算换手率相关特征"""
    # 如果没有 turnover 数据，用 成交量/成交量均值 近似
    if "turnover_rate" not in df.columns:
        # 用量比作为换手率的代理
        df["turnover_proxy"] = df["volume"] / df["vol_ma5"]
    else:
        df["turnover_proxy"] = df["turnover_rate"]
    df["turnover_ma5"] = df["turnover_proxy"].rolling(window=5).mean()
    df["turnover_change"] = df["turnover_proxy"].pct_change()
    return df


def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """v2.0: 计算交叉特征（量价配合度等）"""
    # 量价齐升信号
    df["vol_price_up"] = ((df["price_change_1d"] > 0) & (df["vol_ratio"] > 1.2)).astype(int)
    # 量价背离信号（价涨量缩）
    df["vol_price_diverge"] = ((df["price_change_1d"] > 0.01) & (df["vol_ratio"] < 0.8)).astype(int)
    # RSI 与价格背离
    df["rsi_price_diverge"] = ((df["price_change_5d"] > 0.03) & (df["rsi"] < df["rsi"].shift(5))).astype(int)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """v2.0: 添加时间特征（星期效应、月末效应）"""
    dates = pd.to_datetime(df["date"])
    df["day_of_week"] = dates.dt.dayofweek  # 0=周一, 4=周五
    df["is_monday"] = (dates.dt.dayofweek == 0).astype(int)
    df["is_friday"] = (dates.dt.dayofweek == 4).astype(int)
    df["is_month_end"] = dates.dt.is_month_end.astype(int)
    df["is_month_start"] = dates.dt.is_month_start.astype(int)
    return df


def add_market_features(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """v3.0: 添加大盘指数特征（含5日相对强弱）"""
    market_df = market_df.copy()
    market_df["date"] = pd.to_datetime(market_df["date"])
    market_df = market_df[["date", "close", "pct_change"]].rename(columns={
        "close": "market_close",
        "pct_change": "market_pct_change",
    })
    market_df["market_change_5d"] = market_df["market_close"].pct_change(5)

    df = df.merge(market_df, on="date", how="left")
    # v3.0: 5日级别相对强弱 = 个股5日涨幅 - 大盘5日涨幅
    if "price_change_5d" not in df.columns:
        df["price_change_5d"] = df["close"].pct_change(5)
    df["relative_strength"] = df["price_change_5d"] - df["market_change_5d"]
    return df


def add_regime_label(df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    v3.0: 根据大盘（上证指数）或个股自身行为，为每日打上市场环境标签：
      - 'bull'     牛市：MA20 > MA60 且 20日涨幅 > -2%（均线多头且未急跌）
      - 'bear'     熊市：MA20 < MA60 且 20日涨幅 < -5%（均线空头且近20日跌幅>5%）
      - 'sideways' 震荡市：其他情况（含前60日MA未就绪时）

    参数:
        df:        个股行情数据，需含 date, close 列
        market_df: 大盘指数数据（可选），需含 date, close 列
                   - 提供且含 close → 用大盘计算 regime，regime_source='market_index'
                   - 否则回退用个股自身，regime_source='stock_fallback'

    输出: 在 df 基础上新增 'regime' 与 'regime_source' 两列，不修改入参
    """
    df = df.copy()

    def _classify(close_series: pd.Series) -> pd.Series:
        """根据 close 计算 MA20/MA60 与 20日收益，给出 regime 标签"""
        ma20 = close_series.rolling(window=20).mean()
        ma60 = close_series.rolling(window=60).mean()
        ret_20d = close_series.pct_change(20)
        # 默认 sideways，覆盖前60日 MA 未就绪的 NaN 情况
        regime = pd.Series("sideways", index=close_series.index)
        bull_mask = ((ma20 > ma60) & (ret_20d > -0.02)).fillna(False)
        bear_mask = ((ma20 < ma60) & (ret_20d < -0.05)).fillna(False)
        regime[bull_mask] = "bull"
        regime[bear_mask] = "bear"
        return regime

    df["date"] = pd.to_datetime(df["date"])

    if market_df is not None and "close" in market_df.columns:
        mkt = market_df.copy()
        mkt["date"] = pd.to_datetime(mkt["date"])
        mkt["regime"] = _classify(mkt["close"])
        regime_map = mkt[["date", "regime"]]
        df = df.merge(regime_map, on="date", how="left")
        # 大盘日期可能与个股不完全对齐，前向填充（限10日）后剩余 NaN 视为震荡
        df["regime"] = df["regime"].ffill(limit=10).fillna("sideways")
        df["regime_source"] = "market_index"
    else:
        df["regime"] = _classify(df["close"])
        df["regime_source"] = "stock_fallback"

    return df


def _consecutive_days(condition: pd.Series) -> pd.Series:
    """计算连续满足条件的天数"""
    result = pd.Series(np.zeros(len(condition)), index=condition.index)
    count = 0
    for i in range(len(condition)):
        if i == 0:
            count = 0
        elif condition.iloc[i]:
            count += 1
        else:
            count = 0
        result.iloc[i] = count
    return result


def add_label(df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.005) -> pd.DataFrame:
    """
    生成标签：未来 N 天涨跌幅
    v3.0: 长周期自动放大 threshold（参考 predict_longterm.py）
         threshold = max(0.005, forward_days * 0.001)，上限 0.15

    - 1 = 涨（涨幅 >= threshold）
    - 0 = 跌（涨幅 < threshold）

    参数:
        forward_days: 预测未来多少天的涨跌
        threshold:    涨跌分界线起点，默认 0.5%（长周期自动放大）
    """
    df = df.copy()
    effective_threshold = max(threshold, forward_days * 0.001)
    effective_threshold = min(effective_threshold, 0.15)
    df["future_return"] = df["close"].shift(-forward_days) / df["close"] - 1
    df["label"] = (df["future_return"] >= effective_threshold).astype(float)
    df.loc[df["future_return"].isna(), "label"] = np.nan
    return df


if __name__ == "__main__":
    from data_fetcher import get_stock_data

    df = get_stock_data("600519", days=200)
    df = add_all_indicators(df)
    df = add_label(df, forward_days=5)
    new_cols = [c for c in df.columns if c not in
                ["date", "open", "close", "high", "low", "volume", "amount", "pct_change",
                 "future_return", "label", "ma5", "ma10", "ma20", "ma60",
                 "boll_mid", "boll_upper", "boll_lower", "vol_ma5", "vol_ma10",
                 "amount_ma5", "obv", "obv_ma5", "atr", "turnover_proxy", "turnover_ma5"]]
    print(f"总特征数: {len(new_cols)}")
    print(f"新特征: {new_cols}")
