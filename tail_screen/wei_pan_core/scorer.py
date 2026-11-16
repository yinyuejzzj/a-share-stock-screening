"""wei_pan_core.scorer · 规则打分模块（基础分70 + 板块热度30 = 满分100）。

对照 checklist M6.1 ~ M6.4：
- 基础分项70分（SCORE_BASE_WEIGHTS）：涨停基因(15) / 涨幅位置(10) / 量比位置(8) / 换手健康(7) / 均线强度(10) / 量价力度(8) / 跑赢HS300(7) / 分时质量(5)
- 板块热度30分（SCORE_SECTOR_WEIGHTS）：所属行业3日涨幅排名(15) / 5日排名(10) / 个股当日在板块内贡献排名(5)
- 板块映射 & 排名数据：data_fetcher.fetch_sector_rankings() → 返回 DataFrame[板块, 3d_ret, 5d_ret, stock_ret...]
- 输出Top10供下一层ML重排（规则×0.8 + ML×0.2 → 合并final_score_100制；弱市时只有规则）。

审计修正 §3.5 最终分数归一（统一为百分制后再加权）：
    rule_norm = rule_total / 70 * 100
    final    = rule_norm (弱市) or rule_norm*0.8 + ml_proba*100*0.2 (非弱市)
本模块只算规则分（70/100制输出两列rule_total_70 和 rule_norm_100，方便后续流水线）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SCORE_BASE_WEIGHTS, SCORE_SECTOR_WEIGHTS


# =========================================================================
# 小工具：线性位置评分（x∈[lo,hi]区间里的位置越靠越 → 得满分；位置越低→得0分）
# =========================================================================
def _lin_position(x: float, lo: float, hi: float, invert: bool = False) -> float:
    """x在区间[lo, hi]中的相对位置 [0, 1]。超出边界会 clip 到 0/1。"""
    if hi <= lo:
        return 1.0  # 退化区间：满分（不扣分）
    t = (x - lo) / (hi - lo)
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) if invert else t


# =========================================================================
# 基础70分计算（逐行打分，无外部依赖，可离线mock，便于单元测试）
# =========================================================================
def _score_one_base(
    row: pd.Series,
    *,
    pct_lo: float, pct_hi: float,
    vr_lo: float, vr_hi: float,
    tr_lo: float, tr_hi: float,
    limit_up_N: int,
    hs300_daily_ret: Optional[float],
    intraday_ok: bool,
    intraday_climb: float,  # 14:00到买入窗累计爬升率(正为优)，缺则用0.5默认中分
    filter_th: Dict[str, Any],
) -> Tuple[float, Dict[str, float], List[str]]:
    """单票基础分项70分 → (total_70, partials_分, reasons_for_display)。"""
    W = SCORE_BASE_WEIGHTS
    pct = float(row.get("pct_chg") or 0.0)
    vr = float(row.get("vol_ratio") or 0.0)
    tr = float(row.get("turnover_rate") or 0.0)
    close = float(row.get("close") or 0.0)
    ma5 = float(row.get("MA5") or np.nan)
    ma10 = float(row.get("MA10") or np.nan)
    ma20 = float(row.get("MA20") or np.nan)
    preclose = float(row.get("preclose") or 0.0)
    high = float(row.get("high") or np.nan)
    low = float(row.get("low") or np.nan)
    opens = float(row.get("open") or np.nan)

    partials: Dict[str, float] = {}
    reasons: List[str] = []

    # ① 涨停基因15分（按涨停距离分档：近20日=满15分；近30日=12分；近45日=8分；更远=5分）
    from .config import LIMITUP_GENE_SCORE_TABLE
    raw_gene = row.get("has_limit_up_in_N_day") or {}
    # JSON object keys become strings after a cache round-trip. Normalize once
    # so identical observations retain identical scores on subsequent runs.
    gene = {int(k): bool(v) for k, v in raw_gene.items()
            if str(k).isdigit()} if isinstance(raw_gene, dict) else {}
    gene_n_day_bool = bool(gene.get(limit_up_N, False))
    if gene_n_day_bool:
        # 按最近涨停窗口分档打分（越近涨停=含金量越高）
        if isinstance(gene, dict) and gene:
            true_windows = sorted([k for k, v in gene.items() if isinstance(k, int) and bool(v)])
            if true_windows:
                nearest = true_windows[0]  # 最近的涨停日窗口
                # 查表分档
                gene_score = 5  # 默认：超过45日也有一丝基因
                for window, score in sorted(LIMITUP_GENE_SCORE_TABLE.items()):
                    if nearest <= window:
                        gene_score = score
                        break
                partials["limit_up_gene"] = float(gene_score)
                reasons.append(f"gene{gene_score}分(最近涨停{nearest}日)")
            else:
                partials["limit_up_gene"] = float(W["limit_up_gene"])
                reasons.append(f"gene15分(满)")
        else:
            partials["limit_up_gene"] = float(W["limit_up_gene"])
            reasons.append(f"gene15分(满)")
    else:
        partials["limit_up_gene"] = 3.0
        reasons.append("gene告警3分(不该出现)")

    # ② 涨幅位置10分：离区间上沿越近（越强势）分越高
    pos = _lin_position(pct, pct_lo, pct_hi, invert=False)  # 越接近上沿→1.0
    partials["pct_chg_position"] = round(pos * W["pct_chg_position"], 3)
    reasons.append(f"涨幅位{pos:.0%}→{partials['pct_chg_position']:.1f}分")

    # ③ 量比位置8分：量比越接近区间中值分最高（过低不活跃/过高可能出货→均扣分）——用"倒三角"评分：距离中值越近→分越高
    vr_mid = (vr_lo + vr_hi) / 2.0
    vr_dist = abs(vr - vr_mid) / max(1e-6, (vr_hi - vr_lo) / 2.0)  # 0=完美, 1=已触边界
    vr_score_t = float(np.clip(1.0 - vr_dist, 0.0, 1.0))
    partials["vol_ratio_strength"] = round(vr_score_t * W["vol_ratio_strength"], 3)
    reasons.append(f"量比{vr:.2f}→{partials['vol_ratio_strength']:.1f}分")

    # ④ 换手健康度7分：和量比同理，健康区间中间最稳，两端扣
    tr_mid = (tr_lo + tr_hi) / 2.0
    tr_dist = abs(tr - tr_mid) / max(1e-6, (tr_hi - tr_lo) / 2.0)
    tr_t = float(np.clip(1.0 - tr_dist, 0.0, 1.0))
    partials["turnover_health"] = round(tr_t * W["turnover_health"], 3)
    reasons.append(f"换手{tr:.2%}→{partials['turnover_health']:.1f}分")

    # ⑤ 均线强度10分：按MA5>MA10>MA20>MA45完整度累加
    ma_pairs = [(ma5, ma10), (ma10, ma20), (ma20, float(row.get("MA45") or np.nan))]
    ok_cnt = 0
    total_actual = 0
    for a, b in ma_pairs:
        if np.isnan(a) or np.isnan(b):
            continue
        total_actual += 1
        if a > b:
            ok_cnt += 1
        elif a == b:
            ok_cnt += 0.5
    if total_actual == 0:
        ma_t = 0.0
    else:
        ma_t = ok_cnt / total_actual
    partials["ma_strength"] = round(ma_t * W["ma_strength"], 3)
    # 距离MA20偏离越小越贴近趋势则给点加成（±5%区间给满，越界扣）
    if not np.isnan(ma20) and close > 0 and ma20 > 0:
        dev = abs((close - ma20) / ma20)
        dev_t = float(np.clip(1.0 - dev / 0.10, 0.0, 1.0))  # 10%偏差→0分
        # 把偏差纳入ma_strength比例的一部分（简单：ma_t取 0.7×order + 0.3×dev）
        blend = 0.7 * ma_t + 0.3 * dev_t
        partials["ma_strength"] = round(blend * W["ma_strength"], 3)
    reasons.append(f"均线{ok_cnt:.0f}/{total_actual}→{partials['ma_strength']:.1f}分")

    # ⑥ 量价力度8分：阳线+收上影短（close/high越接近1越好）+量>MA5volume简单化用 vol_ratio >1.1
    #   简化打分：
    #     · candle up 且 close≥open → 得基础0.45；否则0.15
    #     · close/high 越接近1越好：(close/high-0.96)/0.04 clip 0/1 → 0.35权重
    #     · 振幅=|high-low|/preclose 大于阈值：0.20权重
    candle_up = close > opens if not np.isnan(opens) else close > preclose
    cu_t = 0.45 if candle_up else 0.15
    ch_t = float(np.clip(((close / high) - 0.96) / 0.04, 0.0, 1.0)) if (high > 0 and not np.isnan(high)) else 0.5
    amp = (high - low) / preclose if (preclose > 0 and not np.isnan(high) and not np.isnan(low)) else 0.0
    amp_t = float(np.clip(amp / 0.06, 0.0, 1.0))  # 6%振幅满
    vp_t = cu_t * 0.45 + ch_t * 0.35 + amp_t * 0.20
    partials["vol_price_sync"] = round(float(np.clip(vp_t, 0.0, 1.0)) * W["vol_price_sync"], 3)
    reasons.append(f"量价{partials['vol_price_sync']:.1f}分(阳={candle_up})")

    # ⑦ 跑赢HS300当日 7分：若信号缺失则给4.2折中分（60%）
    if hs300_daily_ret is None:
        partials["outperform_hs300"] = round(W["outperform_hs300"] * 0.6, 3)
        reasons.append("跑赢300=4.2分(信号缺60%折中)")
    else:
        diff = pct - float(hs300_daily_ret)
        # diff>0记正（跑赢），需要≥+1%满；-1%及以下0；中间线性
        out_t = _lin_position(diff, -0.01, 0.01, invert=False)
        partials["outperform_hs300"] = round(out_t * W["outperform_hs300"], 3)
        reasons.append(f"300超{pct - float(hs300_daily_ret):+.2%}→{partials['outperform_hs300']:.1f}分")

    # ⑧ 分时质量5分：cond9过了且爬升越多越高
    climb_t = 0.5  # 未知折中 50%
    if not np.isnan(intraday_climb):
        # climb∈[-0.012, 0.015] 做线性映射 0→1；< -0.012 →0；>0.015→1
        climb_t = _lin_position(float(intraday_climb), -0.012, 0.015)
    # cond9没通过（intraday_ok=False）→ 分时扣分最多50%
    if not intraday_ok:
        climb_t *= 0.5
    partials["minute_shape"] = round(climb_t * W["minute_shape"], 3)
    reasons.append(f"分时={partials['minute_shape']:.1f}分(爬升{intraday_climb if not np.isnan(intraday_climb) else 'N/A'})")

    # 汇总70
    total = round(sum(partials.values()), 3)
    return total, partials, reasons


# =========================================================================
# 板块30分（需要外部拉板块排名；若缺数据则所有票平均分15/30 = 50%，不影响排序的相对公平性）
# =========================================================================
def _score_one_sector(
    row: pd.Series,
    *,
    sector_rank_df: Optional[pd.DataFrame],  # [sector_name, sector_code, ret_3d_pct, ret_5d_pct, count_stocks]
    stock_contrib_rank_pct: Optional[Dict[str, float]],  # code -> percentile(0最差/1最好)
    missing_fallback_pct: float = 0.5,
) -> Tuple[float, Dict[str, float], List[str]]:
    W = SCORE_SECTOR_WEIGHTS
    partials: Dict[str, float] = {}
    reasons: List[str] = []
    code = str(row.get("code") or "").zfill(6)
    sector = row.get("sector")
    if sector_rank_df is None or sector is None or pd.isna(sector) or str(sector).strip() == "":
        # 缺失 → 每类按 missing_fallback_pct (默认50%)得折中位置分
        partials["sector_rank_3d"] = round(W["sector_rank_3d"] * missing_fallback_pct, 3)
        partials["sector_rank_5d"] = round(W["sector_rank_5d"] * missing_fallback_pct, 3)
        partials["stock_contrib"] = round(W["stock_contrib"] * missing_fallback_pct, 3)
        return round(sum(partials.values()), 3), partials, ["板块30=折中50%"]

    # 找所属板块行（支持sector字段做df行匹配，sector列名'板块'或'sector_name'或'sector'）
    sr = sector_rank_df
    hit = None
    for col in ("sector_name", "板块", "sector", "name"):
        if col in sr.columns:
            m = sr[sr[col].astype(str) == str(sector)]
            if len(m) > 0:
                hit = m.iloc[0]
                break
    if hit is None:
        partials["sector_rank_3d"] = round(W["sector_rank_3d"] * missing_fallback_pct, 3)
        partials["sector_rank_5d"] = round(W["sector_rank_5d"] * missing_fallback_pct, 3)
        partials["stock_contrib"] = round(W["stock_contrib"] * missing_fallback_pct, 3)
        return round(sum(partials.values()), 3), partials, ["板块30=匹配失败折中50%"]

    # 板块rank分（ret越高→分越高，用ret值线性clip到[-5%,+10%]区间作为"全市场板块排名"近似）
    r3 = float(pd.to_numeric(hit.get("ret_3d_pct") if "ret_3d_pct" in hit.index else hit.get("3日涨跌幅") if "3日涨跌幅" in hit.index else hit.get("ret_3d") if "ret_3d" in hit.index else np.nan, errors="coerce"))
    r5 = float(pd.to_numeric(hit.get("ret_5d_pct") if "ret_5d_pct" in hit.index else hit.get("5日涨跌幅") if "5日涨跌幅" in hit.index else hit.get("ret_5d") if "ret_5d" in hit.index else np.nan, errors="coerce"))
    t3 = _lin_position(r3, -0.05, 0.10) if not np.isnan(r3) else missing_fallback_pct
    t5 = _lin_position(r5, -0.08, 0.18) if not np.isnan(r5) else missing_fallback_pct
    partials["sector_rank_3d"] = round(t3 * W["sector_rank_3d"], 3)
    partials["sector_rank_5d"] = round(t5 * W["sector_rank_5d"], 3)

    # 个股当日在板块内涨幅贡献排名（0~1）
    contrib = float(stock_contrib_rank_pct.get(code)) if stock_contrib_rank_pct and (code in stock_contrib_rank_pct) else None
    if contrib is None:
        # 缺失折中，用pct相对其他股票在候选池里的位置估算（用涨幅百分比再clip）
        contrib = missing_fallback_pct
    contrib = float(np.clip(contrib, 0.0, 1.0))
    partials["stock_contrib"] = round(contrib * W["stock_contrib"], 3)

    reasons = [
        f"板块3D涨{r3 if not np.isnan(r3) else 'N/A'}→{partials['sector_rank_3d']:.1f}分",
        f"板块5D涨{r5 if not np.isnan(r5) else 'N/A'}→{partials['sector_rank_5d']:.1f}分",
        f"个股贡献名次{contrib:.0%}→{partials['stock_contrib']:.1f}分",
    ]
    return round(sum(partials.values()), 3), partials, reasons


# =========================================================================
# 对外：run_rule_scoring(filtered_df) → scored_df（加新列） + Top10_df
# =========================================================================
def run_rule_scoring(
    filtered_df: pd.DataFrame,
    *,
    # —— 打分所需阈值（通常来自 hard_filter.run_hard_filter 返回的 report.extra，保持一致）
    pct_lo: float,
    pct_hi: float,
    vr_lo: float,
    vr_hi: float,
    tr_lo: float,
    tr_hi: float,
    limit_up_N: int,
    # —— 大盘/板块信息
    hs300_daily_ret: Optional[float] = None,
    sector_rank_df: Optional[pd.DataFrame] = None,
    stock_contrib_rank_pct: Optional[Dict[str, float]] = None,
    # —— 逐票intraday形态（用于分时评分项）
    intraday_map: Optional[Dict[str, pd.DataFrame]] = None,
    intraday_mode: str = "B",
    # —— TopN
    top_n: int = 10,
) -> Dict[str, Any]:
    """对硬门槛通过后的候选池逐票打分 → Top10（规则分）。

    返回: {
        "scored": DataFrame（全部已打分）,
        "top":    DataFrame（top N）,
        "summary_counts": {"n_input": int, "n_top": int},
        "score_columns": [新列名列表],
    }
    """
    intraday_map = intraday_map or {}
    if len(filtered_df) == 0:
        return {
            "scored": filtered_df.copy(),
            "top": filtered_df.copy(),
            "summary_counts": {"n_input": 0, "n_top": 0},
            "score_columns": [],
        }

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 遍历打分
        base_scores: List[float] = []
        sector_scores: List[float] = []
        total_scores_70_30: List[float] = []
        rule_norm_100: List[float] = []
        base_reasons: List[List[str]] = []
        sector_reasons_list: List[List[str]] = []
        base_partials_list: List[Dict[str, float]] = []
        sector_partials_list: List[Dict[str, float]] = []

        for _, r in filtered_df.iterrows():
            code = str(r.get("code") or "").zfill(6)
            idf = intraday_map.get(code)
            # 估算intraday爬升（如果有分时就用窗口内爬升；否则用NaN折中）
            climb: float = np.nan
            shape_ok = True
            if idf is not None and len(idf) >= 5:
                try:
                    # 买入窗口前的尾盘爬升：统一用14:00-14:45间的累计涨幅（简化跨mode）
                    tmp = idf.copy()
                    tmp["t"] = pd.to_datetime(tmp["time"], errors="coerce", format="mixed").dt.strftime("%H:%M")
                    sub = tmp[(tmp["t"] >= "14:00") & (tmp["t"] <= "14:45")]
                    if len(sub) >= 2:
                        s = float(sub["close"].iloc[0]); e = float(sub["close"].iloc[-1])
                        if s > 0:
                            climb = (e - s) / s
                except Exception:
                    pass
                # 形态ok与否用"14:00到买入窗的尾端是否上升"简单估计（硬门槛已过cond9，但评分看幅度）
                if not np.isnan(climb) and climb < -0.005:
                    shape_ok = False
            # 基础70
            btotal, bpartials, breaons = _score_one_base(
                r, pct_lo=pct_lo, pct_hi=pct_hi,
                vr_lo=vr_lo, vr_hi=vr_hi,
                tr_lo=tr_lo, tr_hi=tr_hi,
                limit_up_N=limit_up_N,
                hs300_daily_ret=hs300_daily_ret,
                intraday_ok=shape_ok, intraday_climb=climb,
                filter_th={},
            )
            # 板块30
            stotal, spartials, sreasons = _score_one_sector(
                r, sector_rank_df=sector_rank_df,
                stock_contrib_rank_pct=stock_contrib_rank_pct,
            )
            base_scores.append(round(btotal, 3))
            sector_scores.append(round(stotal, 3))
            total_70_30 = round(btotal + stotal, 3)
            total_scores_70_30.append(total_70_30)
            # 审计修正：规则归一到100制（给后续ML重排用）
            rule_norm_100.append(round(total_70_30 / 100 * 100, 3))  # 70+30=100，所以直接=total
            base_reasons.append(breaons)
            sector_reasons_list.append(sreasons)
            base_partials_list.append(bpartials)
            sector_partials_list.append(spartials)

        scored = filtered_df.copy()
        scored["rule_base_70"] = base_scores
        scored["rule_sector_30"] = sector_scores
        scored["rule_total_100"] = total_scores_70_30          # = base + sector
        scored["rule_norm_100"] = rule_norm_100                # 与上相等（70+30=100；为明确语义保持两列）
        scored["rule_bp_dict"] = base_partials_list
        scored["rule_sp_dict"] = sector_partials_list
        scored["rule_why"] = [b + s for b, s in zip(base_reasons, sector_reasons_list)]

        cols_new = ["rule_base_70", "rule_sector_30", "rule_total_100", "rule_norm_100",
                    "rule_bp_dict", "rule_sp_dict", "rule_why"]
        scored = scored.sort_values(by="rule_total_100", ascending=False, kind="stable").reset_index(drop=True)
        top_df = scored.head(top_n).copy()
        return {
            "scored": scored,
            "top": top_df,
            "summary_counts": {"n_input": int(len(filtered_df)), "n_top": int(len(top_df))},
            "score_columns": cols_new,
        }
