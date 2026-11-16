"""跳空高开筛选：当前快照或带日期的离线快照 + 未复权日线。

从原先固定日期脚本整理，缺口定义始终为目标日开盘 > 上一交易日最高。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

PROFILES = {
    "strict": {"min_market_cap_yi": 30, "max_market_cap_yi": 80, "max_pb": 5, "min_gap_pct": 0.5},
    "standard": {"min_market_cap_yi": 20, "max_market_cap_yi": 100, "max_pb": 10, "min_gap_pct": 0},
    "relaxed": {"min_market_cap_yi": 10, "max_market_cap_yi": 200, "max_pb": 15, "min_gap_pct": 0},
}
ALIASES = {"代码": "code", "名称": "name", "日期": "date", "总市值": "market_cap",
           "市净率": "pb", "成交量": "volume", "开盘": "open", "最高": "high",
           "最低": "low", "收盘": "close", "快照日期": "snapshot_date"}


def iso_date(value):
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式必须为 YYYY-MM-DD") from exc
    if parsed > date.today():
        raise argparse.ArgumentTypeError("日期不能晚于今天")
    return parsed.isoformat()


def normalize(frame):
    out = frame.rename(columns=ALIASES).copy()
    if "code" in out:
        out["code"] = out["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return out


def filter_snapshot(snapshot, target_date, params):
    """估值/市值必须来自同日快照，不能以最新估值冒充历史筛选。"""
    df = normalize(snapshot)
    required = {"code", "name", "market_cap", "pb", "volume", "snapshot_date"}
    if required - set(df):
        raise ValueError(f"快照缺列: {sorted(required - set(df))}")
    days = pd.to_datetime(df["snapshot_date"], errors="raise").dt.strftime("%Y-%m-%d")
    if not days.eq(target_date).all():
        raise ValueError("快照日期必须全部等于 --date；拒绝混用最新估值与历史行情")
    if df["code"].duplicated().any():
        raise ValueError("快照每个股票代码只能有一行")
    for key in ("market_cap", "pb", "volume"):
        df[key] = pd.to_numeric(df[key], errors="coerce")
    cap = df["market_cap"] / 1e8
    valid = (~df["name"].astype(str).str.contains("ST|退", case=False, na=False)
             & cap.ge(params["min_market_cap_yi"]) & cap.lt(params["max_market_cap_yi"])
             & df["pb"].gt(0) & df["pb"].le(params["max_pb"]) & df["volume"].gt(0))
    return df.loc[valid].copy()


def gap_values(bars, target_date, min_gap_pct=0):
    """使用上一条实际交易日K线，非简单日历日期减一。缺失目标日不回退。"""
    df = normalize(bars)
    required = {"date", "open", "high", "close"}
    if required - set(df):
        raise ValueError(f"日线缺列: {sorted(required - set(df))}")
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if df["date"].duplicated().any():
        raise ValueError("同一股票日线存在重复日期")
    df = df[df["date"] <= target_date].sort_values("date").reset_index(drop=True)
    if len(df) < 2 or df.iloc[-1]["date"] != target_date:
        return None
    previous, current = df.iloc[-2], df.iloc[-1]
    opening, previous_high, previous_close = (float(current["open"]), float(previous["high"]), float(previous["close"]))
    if not all(math.isfinite(v) and v > 0 for v in (opening, previous_high, previous_close)):
        return None
    if opening <= previous_high * (1 + min_gap_pct / 100):
        return None
    return {"date": target_date, "previous_date": previous["date"], "open": opening,
            "previous_high": previous_high, "previous_close": previous_close,
            "gap_above_high_pct": (opening / previous_high - 1) * 100,
            "open_vs_previous_close_pct": (opening / previous_close - 1) * 100}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--date", type=iso_date, default=date.today().isoformat(), help="目标交易日 YYYY-MM-DD")
    parser.add_argument("--snapshot", type=Path, help="带 snapshot_date 列的同日股票快照CSV")
    parser.add_argument("--bars", type=Path, help="含 code,date,open,high,close 的未复权日线CSV")
    parser.add_argument("--live", action="store_true", help="读取最新快照，仅允许今天；需要联网")
    parser.add_argument("--codes", help="可选：逗号分隔股票代码")
    parser.add_argument("--lookback-days", type=int, default=40, help="日线回看自然日数，默认40")
    parser.add_argument("--out", type=Path, default=Path("outputs/gap_results.csv"))
    parser.add_argument("--show-config", action="store_true", help="显示参数后退出，不联网")
    for key in PROFILES["standard"]:
        parser.add_argument("--" + key.replace("_", "-"), type=float)
    return parser


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    params = {key: getattr(args, key) if getattr(args, key) is not None else value
              for key, value in PROFILES[args.profile].items()}
    if not all(math.isfinite(value) for value in params.values()):
        parser.error("参数必须为有限数字")
    if not (0 <= params["min_market_cap_yi"] < params["max_market_cap_yi"] and params["max_pb"] > 0
            and params["min_gap_pct"] >= 0 and args.lookback_days >= 2):
        parser.error("市值区间应递增，PB>0，缺口>=0，回看天数>=2")
    metadata = {"profile": args.profile, "target_date": args.date, "parameters": params,
                "market_cap_unit": "yuan", "gap_unit": "percentage points", "mode": "live" if args.live else "csv"}
    if args.show_config:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    if args.live:
        if args.date != date.today().isoformat() or args.snapshot or args.bars:
            parser.error("--live 只读取今天最新快照，不能搭配历史日期或CSV；历史筛选请提供同日快照和日线")
        import akshare as ak
        snapshot = normalize(ak.stock_zh_a_spot_em())
        snapshot["snapshot_date"] = args.date
        metadata["snapshot_date_basis"] = "retrieval_date_unverified_provider_session"
        metadata["limitation"] = "快照交易日期未经接口验证；休市/盘中可能返回旧值或未完成K线，不作为回测证据"
        all_bars = None
    else:
        if not args.snapshot or not args.bars:
            parser.error("请提供 --snapshot 和 --bars，或使用 --live")
        snapshot = pd.read_csv(args.snapshot, dtype=str)
        all_bars = normalize(pd.read_csv(args.bars, dtype=str))
        if "code" not in all_bars:
            parser.error("日线CSV缺少code列")
    candidates = filter_snapshot(snapshot, args.date, params)
    if args.codes:
        codes = {code.strip().zfill(6) for code in args.codes.split(",")}
        candidates = candidates[candidates["code"].isin(codes)]
    results, errors = [], []
    for _, row in candidates.iterrows():
        try:
            if args.live:
                start = (date.fromisoformat(args.date) - timedelta(days=args.lookback_days)).strftime("%Y%m%d")
                bars = ak.stock_zh_a_hist(symbol=row["code"], period="daily", start_date=start,
                                        end_date=args.date.replace("-", ""), adjust="")
            else:
                bars = all_bars[all_bars["code"].eq(row["code"])].copy()
                if "date" not in bars:
                    raise ValueError("日线CSV缺少date列")
                bar_dates = pd.to_datetime(bars["date"], errors="raise")
                lower = pd.Timestamp(args.date) - pd.Timedelta(days=args.lookback_days)
                bars = bars[bar_dates.ge(lower)]
            match = gap_values(bars, args.date, params["min_gap_pct"])
            if match:
                results.append({"code": row["code"], "name": row["name"], **match,
                                "market_cap_yi": row["market_cap"] / 1e8, "pb": row["pb"]})
            normalized = normalize(bars)
            if "date" in normalized and not pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d").eq(args.date).any():
                errors.append({"code": row["code"], "reason": "目标交易日不存在，未回退到其它日期"})
        except Exception as exc:
            errors.append({"code": row["code"], "reason": str(exc)})
    columns = ["code", "name", "date", "previous_date", "open", "previous_high", "previous_close",
               "gap_above_high_pct", "open_vs_previous_close_pct", "market_cap_yi", "pb"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(results, columns=columns).sort_values("gap_above_high_pct", ascending=False)
    result.to_csv(args.out, index=False, encoding="utf-8-sig")
    metadata.update({"candidates": len(candidates), "matches": len(result), "errors": errors,
                     "generated_at": datetime.now().isoformat(timespec="seconds")})
    args.out.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"候选{len(candidates)}只，匹配{len(result)}只，数据异常{len(errors)}只；已输出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
