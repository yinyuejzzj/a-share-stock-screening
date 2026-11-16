"""Chronological random-forest research baseline; no broker integration."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .indicators import add_all_indicators, add_label

PROFILES = {"strict": 0.65, "standard": 0.60, "relaxed": 0.55}
FEATURES = [
    "ma5_bias", "ma10_bias", "ma20_bias", "ma60_bias", "ma_bullish",
    "ma5_slope", "ma20_slope", "rsi", "rsi_change", "macd_dif_norm",
    "macd_hist_norm", "kdj_k", "kdj_d", "kdj_j", "boll_pct",
    "vol_ratio", "price_change_1d", "price_change_5d", "price_change_10d",
    "high_low_ratio", "wr", "cci",
]


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def iso_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式须为 YYYY-MM-DD") from exc


def probability(value: str) -> float:
    number = float(value)
    if not 0.5 <= number <= 1:
        raise argparse.ArgumentTypeError("概率门槛须介于 0.5 和 1 之间")
    return number


def normalize_data(frame: pd.DataFrame, end_date: str, days: int,
                   expected_code: str | None = None) -> pd.DataFrame:
    frame = frame.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
        "涨跌幅": "pct_change"}).copy()
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"缺少列: {sorted(missing)}")
    if frame.columns.duplicated().any():
        raise ValueError("列名重复，请只保留一种中英文列名")
    for column in ("code", "symbol", "代码", "股票代码"):
        if column in frame:
            codes = frame[column].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
            if (codes.isna().any() or not codes.str.fullmatch(r"[0-9]{6}").all()
                    or codes.nunique() != 1 or (expected_code is not None and not codes.eq(expected_code).all())):
                raise ValueError(f"{column}列必须只含请求的单只股票代码，不能混合或错标")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.isna().any():
        raise ValueError("交易日日期不能缺失")
    if dates.dt.tz is not None:
        raise ValueError("date须为不带时区的交易日日期")
    frame["date"] = dates.dt.normalize()
    cutoff = pd.Timestamp(end_date)
    frame = frame[(frame["date"] <= cutoff) &
        (frame["date"] >= cutoff - pd.Timedelta(days=days))]
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("开高低收与成交量必须为有限数值，不接受缺失、NaN或无穷值")
    if frame.empty or (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("行情为空或包含非正价格")
    if (frame["volume"] < 0).any():
        raise ValueError("成交量不能为负")
    if ((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) |
            (frame["low"] > frame[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("OHLC不一致：最高价须覆盖开收低，最低价须不高于开收高")
    if "amount" not in frame:
        frame["amount"] = 0.0
    else:
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
    if "pct_change" not in frame:
        frame["pct_change"] = frame["close"].pct_change() * 100
    return frame.reset_index(drop=True)


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1)


def evaluate_and_predict(frame: pd.DataFrame, forward_days: int = 1,
                         return_threshold: float = 0.005,
                         min_probability: float = 0.60) -> dict:
    enriched = add_all_indicators(frame)
    enriched = add_label(enriched, forward_days=forward_days, threshold=return_threshold)
    features = [name for name in FEATURES if name in enriched]
    enriched[features] = enriched[features].replace([np.inf, -np.inf], np.nan)
    labeled = enriched.dropna(subset=features + ["label"])
    if len(labeled) < 200:
        raise ValueError("指标与标签清理后至少需要 200 条日线，建议提供 1500 个自然日行情")
    split = int(len(labeled) * 0.8)
    # The last training label must end before validation starts.
    train_end = split - forward_days
    if train_end < 100:
        raise ValueError("预测步长过长，隔离标签后训练样本不足100条")
    training = labeled.iloc[:train_end]
    validation = labeled.iloc[split:]
    if len(training) < 100 or len(validation) < 20 or training["label"].nunique() < 2:
        raise ValueError("训练样本/验证样本不足或训练标签只有一个类别")
    model = build_model()
    model.fit(training[features], training["label"].astype(int))
    predicted = model.predict(validation[features])
    valid_accuracy = float(accuracy_score(validation["label"], predicted))
    balanced = float(balanced_accuracy_score(validation["label"], predicted))
    majority_baseline = float(validation["label"].value_counts(normalize=True).max())
    # After reporting a held-out score, refit on all labeled observations.
    model.fit(labeled[features], labeled["label"].astype(int))
    latest = enriched.iloc[[-1]]
    if latest[features].isna().any().any():
        raise ValueError("最新一根日线的指标缺失，不能替换成更早日期冒充最新预测")
    up_index = list(model.classes_).index(1)
    score = float(model.predict_proba(latest[features])[0, up_index])
    return {
        "as_of": str(latest["date"].iloc[0].date()),
        "forward_trading_days": forward_days,
        "label_return_threshold": min(max(return_threshold, forward_days * 0.001), 0.15),
        "model_probability": score, "min_probability": min_probability,
        "selected": score >= min_probability,
        "validation_accuracy": valid_accuracy,
        "validation_balanced_accuracy": balanced,
        "validation_majority_baseline": majority_baseline,
        "training_rows": len(training), "validation_rows": len(validation),
        "refit_labeled_rows": len(labeled), "feature_count": len(features),
        "note": "概率未经独立校准；标签0包括未达到涨幅门槛的上涨，并不等于下跌。",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="本地技术指标预测实验：固定随机森林与时间顺序留出验证")
    result.add_argument("--code", required=True, help="6位股票代码，仅作为股票标识")
    result.add_argument("--csv", type=Path, help="本地日线CSV；省略则调用AkShare")
    result.add_argument("--end-date", type=iso_date, default=date.today().isoformat(),
        help="数据截止日 YYYY-MM-DD，默认今天；休市日使用之前最后一根K线")
    result.add_argument("--days", type=positive_int, default=1500, help="回看自然日数，默认1500")
    result.add_argument("--forward-days", type=positive_int, default=1, help="预测交易日步长，常用1/3/5")
    result.add_argument("--return-threshold", type=float, default=0.005, help="标签涨幅门槛小数，默认0.005")
    result.add_argument("--profile", choices=PROFILES, default="standard")
    result.add_argument("--min-probability", type=probability, help="覆盖三档的概率门槛")
    result.add_argument("--output", type=Path, help="可选JSON输出路径")
    return result


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    arguments = parser().parse_args(argv)
    if re.fullmatch(r"[0-9]{6}", arguments.code) is None:
        parser().error("--code 必须是6位数字")
    if not np.isfinite(arguments.return_threshold) or not 0 <= arguments.return_threshold <= 0.15:
        parser().error("--return-threshold 必须介于0和0.15之间")
    if date.fromisoformat(arguments.end_date) > date.today():
        parser().error("--end-date 不能在未来")
    try:
        if arguments.csv:
            raw = pd.read_csv(arguments.csv, dtype={column: str for column in ("code", "symbol", "代码", "股票代码")})
        else:
            import akshare as ak
            end = date.fromisoformat(arguments.end_date)
            raw = ak.stock_zh_a_hist(symbol=arguments.code, period="daily",
                start_date=(end - timedelta(days=arguments.days)).strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"), adjust="qfq")
        frame = normalize_data(raw, arguments.end_date, arguments.days, expected_code=arguments.code)
        result = evaluate_and_predict(frame, arguments.forward_days,
            arguments.return_threshold, arguments.min_probability or PROFILES[arguments.profile])
        result.update(code=arguments.code, profile=arguments.profile)
        content = json.dumps(result, ensure_ascii=False, indent=2)
        if arguments.output:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(content + "\n", encoding="utf-8")
        print(content)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f"运行失败: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
