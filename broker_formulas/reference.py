"""Offline reference calculations for the three daily technical formula subsets.

This calculates documented conditions; it is not a TDX formula interpreter.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROFILES = {
    "strict": {"touch": .01, "shrink": .65, "gap_pct": .5,
               "ret_lo": .025, "ret_hi": .06, "dev_limit": .03, "amp_min": .03, "gene_days": 20},
    "standard": {"touch": .02, "shrink": .80, "gap_pct": 0.,
                 "ret_lo": .02, "ret_hi": .07, "dev_limit": .05, "amp_min": .03, "gene_days": 30},
    "relaxed": {"touch": .03, "shrink": .95, "gap_pct": 0.,
                "ret_lo": .01, "ret_hi": .085, "dev_limit": .08, "amp_min": .02, "gene_days": 45},
}
ALIASES = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
           "收盘": "close", "成交量": "volume", "代码": "code", "股票代码": "code", "symbol": "code"}
MIN_HISTORY = {"ema": 25, "gap": 2, "tail": 60}


def iso_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式应为 YYYY-MM-DD") from exc
    if parsed > date.today():
        raise argparse.ArgumentTypeError("截止日不能晚于今天")
    return parsed.isoformat()


def normalize(frame: pd.DataFrame, as_of: str | None = None) -> pd.DataFrame:
    """Validate a single security's daily OHLCV, cutting future rows first."""
    df = frame.rename(columns=ALIASES).copy()
    if df.columns.duplicated().any():
        raise ValueError("列名归一化后重复")
    required = {"date", "open", "high", "low", "close", "volume"}
    if required - set(df):
        raise ValueError(f"缺少日线列: {sorted(required - set(df))}")
    try:
        dates = pd.to_datetime(df["date"], errors="raise")
        if dates.dt.tz is not None:
            raise ValueError("请提供不带时区的本地交易日期")
        df["date"] = dates.dt.normalize()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("日线日期无效或含时区；请提供 YYYY-MM-DD 本地交易日期") from exc
    if df["date"].isna().any():
        raise ValueError("日期不能缺失")
    if as_of is not None:
        df = df[df["date"] <= pd.Timestamp(as_of)]
    if df.empty:
        raise ValueError("截止日及之前没有数据")
    if "code" in df and (df["code"].isna().any() or df["code"].astype(str).nunique() != 1):
        raise ValueError("CSV 必须只包含一只股票；不能混合多只股票计算均线")
    if df["date"].duplicated().any():
        raise ValueError("日线日期重复")
    numeric = ["open", "high", "low", "close", "volume"]
    for key in numeric:
        df[key] = pd.to_numeric(df[key], errors="coerce")
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise ValueError("OHLCV 含非数字、缺失或非有限数值")
    if (df[numeric[:-1]] <= 0).any().any() or (df["volume"] < 0).any():
        raise ValueError("价格必须为正，成交量不能为负")
    if ((df["high"] < df[["open", "low", "close"]].max(axis=1)) |
            (df["low"] > df[["open", "high", "close"]].min(axis=1))).any():
        raise ValueError("OHLC 高低价关系不成立")
    return df.sort_values("date").reset_index(drop=True)


def evaluate(frame: pd.DataFrame, kind: str, profile: str = "standard",
             as_of: str | None = None) -> pd.DataFrame:
    """Return a selected boolean for every input bar after normalization.

    EMA is seeded with the first available close, matching pandas adjust=False.
    Comparison with clients requires the same history start and price adjustment.
    """
    df = normalize(frame, as_of)
    p = PROFILES[profile]
    close, volume = df["close"], df["volume"]
    count = pd.Series(np.arange(1, len(df) + 1), index=df.index)
    if kind == "ema":
        df["ema20"] = close.ewm(span=20, adjust=False).mean()
        df["volume_ma5"] = volume.rolling(5).mean()
        trend = (df["ema20"] > df["ema20"].shift(9)) | (
            (close > df["ema20"]).rolling(10).sum() >= 6)
        selected = ((count >= 25) & (volume > 0) & (df["volume_ma5"] > 0) & trend &
                    (df["low"] <= df["ema20"] * (1 + p["touch"])) &
                    (close > df["ema20"]) & (volume < df["volume_ma5"] * p["shrink"]))
    elif kind == "gap":
        df["previous_high"] = df["high"].shift(1)
        df["gap_above_high_pct"] = (df["open"] / df["previous_high"] - 1) * 100
        selected = ((count >= 2) & (volume > 0) & (df["previous_high"] > 0) &
                    (df["open"] > df["previous_high"] * (1 + p["gap_pct"] / 100)))
    elif kind == "tail":
        df["return_1d"] = (close - close.shift(1)) / close.shift(1)
        ma5, ma10, ma20 = (close.rolling(n).mean() for n in (5, 10, 20))
        df["ma20_deviation"] = (close - ma20) / ma20
        df["amplitude"] = (df["high"] - df["low"]) / close.shift(1)
        df["strong_return_count"] = (df["return_1d"] >= .098).rolling(p["gene_days"]).sum()
        selected = ((count >= 60) & (volume > 0) & (close.shift(1) > 0) &
                    df["return_1d"].between(p["ret_lo"], p["ret_hi"]) &
                    df["ma20_deviation"].between(-p["dev_limit"], p["dev_limit"]) &
                    (df["amplitude"] >= p["amp_min"]) & (df["strong_return_count"] > 0) &
                    (ma5 > ma10) & (ma10 > ma20))
    else:
        raise ValueError("kind 必须为 ema/gap/tail")
    df["history_ready"] = count >= MIN_HISTORY[kind]
    df["selected"] = selected.fillna(False).astype(bool)
    return df


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["ema", "gap", "tail"], required=True)
    parser.add_argument("--csv", type=Path, required=True, help="单只股票日线 OHLCV，不联网")
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--as-of", type=iso_date, default=date.today().isoformat(), help="含该日的截止日")
    parser.add_argument("--out", type=Path, help="可选：保存每根日线的参考计算结果CSV")
    args = parser.parse_args(argv)
    try:
        frame = pd.read_csv(args.csv, dtype={key: str for key in ("code", "代码", "股票代码", "symbol")})
        result = evaluate(frame, args.kind, args.profile, args.as_of)
        latest = result.iloc[-1]
        summary = {"kind": args.kind, "profile": args.profile, "requested_as_of": args.as_of,
                   "actual_last_date": latest["date"].strftime("%Y-%m-%d"),
                   "rows": len(result), "selected": bool(latest["selected"]),
                   "minimum_history": MIN_HISTORY[args.kind],
                   "status": "evaluated" if latest["history_ready"] else "insufficient_history",
                   "scope": "daily technical subset; not broker client validation or complete Python strategy"}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.out, index=False, encoding="utf-8-sig")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"输入或计算失败: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
