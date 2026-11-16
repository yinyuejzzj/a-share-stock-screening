"""wei_pan_core.sector_ranking · 板块排名 + 个股板块内贡献名次。

对照 checklist M6.3 板块热度30分的数据依赖。
能力：
  1) fetch_sector_rankings() → DataFrame[sector_name, ret_3d_pct, ret_5d_pct, count_stocks]
     优先 akshare stock_board_industry_*；失败时返回None（scorer有50%折中兜底）
  2) fetch_stock_sector_map(codes) → dict {code: sector_name}
     用 akshare stock_individual_info_em 或新浪/腾讯个股所属板块
  3) compute_stock_contrib_rank(df, sector_map) → dict {code: percentile 0~1}
     个股当日涨幅在所属板块所有票里的百分位（0最差/1最好）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def fetch_sector_rankings(market_close_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """获取行业板块3日/5日涨幅排名。

    返回 DataFrame[sector_name, ret_3d_pct, ret_5d_pct, count_stocks] 或 None。
    数据不可用时返回None（scorer.py有50%折中兜底），绝不伪造数据。
    """
    # 优先 akshare
    try:
        import akshare as ak
        # 行业板块实时行情（含当日涨跌幅）
        brd = ak.stock_board_industry_name_em()
        if brd is not None and len(brd) > 0:
            df = brd[["板块名称", "涨跌幅"]].copy()
            df.columns = ["sector_name", "ret_1d_pct"]
            df["sector_name"] = df["sector_name"].astype(str).str.strip()

            # 尝试获取真实的3日/5日历史涨幅
            ret_3d = {}
            ret_5d = {}
            try:
                # 逐板块拉取历史日线，计算3日/5日累计收益
                for sn in df["sector_name"].tolist()[:30]:  # 限制30个板块防超时
                    try:
                        hist = ak.stock_board_industry_hist_em(
                            symbol=sn, period="日",
                            start_date=(market_close_date or "20260101").replace("-","")[:8],
                            end_date=(market_close_date or "20261231").replace("-","")[:8],
                            adjust=""
                        )
                        if hist is not None and len(hist) >= 5:
                            closes = pd.to_numeric(hist["收盘"], errors="coerce").dropna().values
                            if len(closes) >= 5:
                                ret_3d[sn] = float((closes[-1] - closes[-4]) / closes[-4])
                                ret_5d[sn] = float((closes[-1] - closes[-6]) / closes[-6])
                            elif len(closes) >= 3:
                                ret_3d[sn] = float((closes[-1] - closes[-3]) / closes[-3])
                    except Exception:
                        continue
            except Exception:
                pass

            if ret_3d:
                df["ret_3d_pct"] = df["sector_name"].map(ret_3d)
                df["ret_5d_pct"] = df["sector_name"].map(ret_5d)
            else:
                # 真实历史数据获取失败 → 返回None，让scorer走50%折中兜底
                # 绝不用1日×3/×5伪造（已修正AI幻觉）
                return None

            df["count_stocks"] = 0
            df = df[["sector_name", "ret_3d_pct", "ret_5d_pct", "count_stocks"]]
            df = df.dropna(subset=["ret_3d_pct"])
            return df
    except Exception:
        pass
    # 兜底：返回None（scorer.py 有50%折中机制），绝不伪造数据
    return None


def fetch_stock_sector_map(codes: List[str]) -> Dict[str, str]:
    """获取每只股票所属的行业板块名。

    返回 {code: sector_name}，拿不到的code不在dict中。
    """
    out: Dict[str, str] = {}
    if not codes:
        return out
    try:
        import akshare as ak
        # 尝试批量获取个股所属板块
        for code in codes:
            code = str(code).zfill(6)
            try:
                info = ak.stock_individual_info_em(symbol=code)
                if info is not None and len(info) > 0:
                    # info是DataFrame，行=指标名, 值
                    # 找行业行
                    for _, row in info.iterrows():
                        label = str(row.iloc[0]) if len(row) > 0 else ""
                        val = str(row.iloc[1]) if len(row) > 1 else ""
                        if "行业" in label and val and val != "nan":
                            out[code] = val.strip()
                            break
            except Exception:
                continue
    except Exception:
        pass
    return out


def compute_stock_contrib_rank(
    snapshot_df: pd.DataFrame,
    sector_map: Dict[str, str],
) -> Dict[str, float]:
    """计算每只股票在所属板块内的当日涨幅百分位（0最差/1最好）。

    依赖 snapshot_df 的 code + pct_chg 列。
    板块内只有1只票时给0.5（无参照）。
    """
    out: Dict[str, float] = {}
    if snapshot_df is None or len(snapshot_df) == 0 or not sector_map:
        return out

    df = snapshot_df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["sector"] = df["code"].map(sector_map)

    for sector_name, grp in df.groupby("sector"):
        if sector_name is None or pd.isna(sector_name) or str(sector_name).strip() == "":
            continue
        n = len(grp)
        if n == 1:
            code = str(grp.iloc[0]["code"])
            out[code] = 0.5
            continue
        # 按 pct_chg 排名（升序→第0名最差→第n-1名最好）
        sorted_grp = grp.sort_values(by="pct_chg", ascending=True)
        ranks = {str(row["code"]): i / (n - 1) for i, (_, row) in enumerate(sorted_grp.iterrows())}
        out.update(ranks)
    return out
