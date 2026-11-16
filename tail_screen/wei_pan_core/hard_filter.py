"""wei_pan_core.hard_filter · 10条硬门槛 + 三类K系数应用。

验收清单对照 M5.1 ~ M5.5。

三类K系数应用规则（审计修正 K-application-mapping §3.6）：
  ① 区间型 (pct/market_cap/vol/turnover/ma_window)：
        lo' = lo_B × K；hi' = hi_B ÷ K （收紧时区间变窄）
  ② 天数窗口型 (涨停基因N天)：
        N = ceil(N_B / K) （K>1 变短更严；K<1 变长更松）
  ③ 布尔叠加型 (cond7~cond9 均线/分时/模式A叠加)：
        按模式写死分支：A=叠加全部；B=标准；C=放宽为"或"条件 / 关闭新股排除

本模块与外盘/闸门解耦，主程序先把「闸门动作」翻译成 tighten_factor + 窗口扩展 传入本模块。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import HARD_RULES_MODE_B, TIME_MODE_K as MODE_K


# =========================================================================
# K 系数三类应用
# =========================================================================
def _apply_k_interval(lo_b: float, hi_b: float, K: float, extra_tighten: float = 1.0):
    """区间型K应用：lo'=lo×K×T, hi'=hi÷(K×T)。 T=global_tightness_factor（默认1.0=无附加影响）。"""
    T = float(extra_tighten or 1.0)
    KT = max(0.2, K * T)  # 避免KT过近0导致上界溢出
    lo = lo_b * KT
    hi = hi_b / KT
    return lo, hi


def _apply_k_window_ceil(n_b: int, K: float, extra_add_days: int = 0) -> int:
    """天数窗口型K：N = ceil(N_B / K) + extra_add_days。loosen时+5日叠加由主程序传extra。"""
    K_pos = max(0.2, float(K))
    base = int(math.ceil(int(n_b) / K_pos))
    return max(1, base + int(extra_add_days or 0))


# =========================================================================
# 模式专用 "布尔叠加" 分支（审计修正 cond9_time_window_by_mode）
# =========================================================================
def _cond7_ma_checks(row: pd.Series, K: float) -> Tuple[bool, List[str]]:
    """cond7 多线形态：A=严格多头；B=MA5>MA10>MA20；C=MA5>MA20 或 MA5>MA10（放宽"或"）。"""
    ma5 = float(row.get("MA5") or np.nan)
    ma10 = float(row.get("MA10") or np.nan)
    ma20 = float(row.get("MA20") or np.nan)
    ma45 = float(row.get("MA45") or np.nan)
    notes: List[str] = []
    if any(np.isnan(x) for x in (ma5, ma10, ma20)):
        return False, ["cond7_MA缺失"]
    if K >= 1.19:  # A=1.20 最严 → 严格多头排列 + MA45抬升（≥前日MA45近似：这里用"MA20>MA45"代替）
        ok = (ma5 > ma10 > ma20) and (ma20 > ma45)
        notes.append("cond7_A严格多头(MA5>MA10>MA20>MA45)")
    elif K <= 0.86:  # C=0.85 最松 → MA5>MA20 "或" MA5>MA10（任一即可）
        ok = (ma5 > ma20) or (ma5 > ma10)
        notes.append("cond7_C放宽(MA5>MA20 或 MA5>MA10)")
    else:  # B=1.00 标准
        ok = (ma5 > ma10 > ma20)
        notes.append("cond7_B(MA5>MA10>MA20)")
    return ok, notes


def _cond10_extra_strictness(
    new_stock: bool,
    is_kcb_cyb: bool,
    K: float,
    *,
    close: float,
    ma45: float,
    pct_chg: float,
) -> Tuple[bool, List[str]]:
    """cond10 扩展：新股/科创板/形态叠加（A追加；B默认；C放松排除）。"""
    notes: List[str] = []
    passed = True
    if K >= 1.19:  # 严格：新股排除 + 科创/创业排除（保守规避±20%涨停波动）+ 收盘需站MA45上方（长线支撑）
        if new_stock:
            passed = False; notes.append("cond10_A排除新股")
        if is_kcb_cyb:
            passed = False; notes.append("cond10_A排除科创板/创业板(±20%)")
        if not np.isnan(ma45) and close < ma45:
            passed = False; notes.append("cond10_A收盘价需在MA45上方")
        return passed, notes or ["cond10_A通过"]
    if K <= 0.86:  # 宽松：C 不主动排除新股；不排除科创；不追MA45
        notes.append("cond10_C宽松(新股/科创可过)")
        return True, notes
    # B：中性——新股pct不能过强（防止连板尾端接棒），但不直接排除（"或"分支）
    if new_stock and pct_chg > 0.095:
        passed = False; notes.append("cond10_B尾端强势连板新股排除(pct>9.5%)")
    return passed, notes or ["cond10_B中性通过"]


# =========================================================================
# cond9 分时形态（审计修正：cond9_time_window_by_mode）
# =========================================================================
def _cond9_intraday_shape(
    intraday_df: Optional[pd.DataFrame],
    K: float,
    *,
    mode: str,        # "A" | "B" | "C"
    window_start: str = "14:00",
    close_price: float,
) -> Tuple[bool, Optional[str]]:
    """cond9：14:00~买入窗分时走势判断。
    A窗口14:30 → 看14:00-14:29（窗口前30min），需稳定爬升≥0.5%，且不出现-0.3%回落；
    B窗口14:45 → 看14:00-14:44（窗口前45min），需稳≥0.4%，回落容差-0.5%；
    C窗口14:52 → 看14:30-14:51（窗口前22min），不暴跌≥-1.2%即可（稳健兜底）。
    若数据缺失、格式不完整或样本不足：所有模式返回False（不过）。
    """
    if intraday_df is None or len(intraday_df) < 5:
        # 无分时数据
        return False, "cond9_分时数据缺失→不通过"
    try:
        df = intraday_df.copy()
        t_raw = pd.to_datetime(df["time"], errors="coerce", format="mixed")
        df["t"] = t_raw.dt.strftime("%H:%M")
    except Exception:
        return False, "cond9_分时解析异常→不通过"
    # 模式化窗口截取（cond9_time_window_by_mode 规定）
    if mode == "A":
        beg, end = "14:00", "14:29"
        need_up = 0.005
        max_drop = 0.003
    elif mode == "B":
        beg, end = "14:00", "14:44"
        need_up = 0.004
        max_drop = 0.005
    else:  # C
        beg, end = "14:30", "14:51"
        need_up = 0.0
        max_drop = 0.012  # 0.012 ≈ -1.2% 容差

    sub = df[(df["t"] >= beg) & (df["t"] <= end)]
    if len(sub) < 3:
        return False, f"cond9_[{beg}-{end}]样本数<3→不通过"
    try:
        price_start = float(sub["close"].iloc[0])
        price_end = float(sub["close"].iloc[-1])
        high_in = float(sub["high"].max())
        low_in = float(sub["low"].min())
    except Exception:
        return False, "cond9_字段缺失→不通过"

    if not all(np.isfinite(v) and v > 0 for v in (price_start, price_end, high_in, low_in)):
        return False, "cond9_无有效分时价格→不通过"
    total_up = (price_end - price_start) / price_start
    # 回落：最高到收盘的最大回撤，近似用(high_in - price_end) / high_in
    pullback = (high_in - price_end) / high_in if high_in > 0 else 0.0
    # C模式：只要求不暴跌（total_up ≥ -max_drop 即价格端没掉 -1.2%）
    if mode == "C":
        ok = (total_up >= -max_drop)
        msg = f"cond9_C[{beg}-{end}] 涨幅{total_up:+.2%} ≥ -1.2%: {'PASS' if ok else 'FAIL'}"
        return ok, msg
    # A/B：需累计爬升达标 + 回落容差不超
    ok_up = total_up >= need_up
    ok_pullback = pullback <= max_drop
    ok = bool(ok_up and ok_pullback)
    msg = (
        f"cond9_{mode}[{beg}-{end}]爬升{total_up:+.2%}/需≥{need_up:+.0%}✓{ok_up} "
        f"回撤{pullback:.2%}/≤{max_drop:.0%}✓{ok_pullback} → {'PASS' if ok else 'FAIL'}"
    )
    return ok, msg


# =========================================================================
# 综合过滤
# =========================================================================
@dataclass
class FilterReport:
    total_input: int = 0
    passed: int = 0
    dropped_by_cond: Dict[str, int] = None  # cond1..10 各踢掉多少
    extra: Dict[str, Any] = None            # K/窗口等参数（用于记录）

    def __post_init__(self):
        self.dropped_by_cond = self.dropped_by_cond or {}
        self.extra = self.extra or {}


def _apply_thresholds_for_mode(mode: str, global_tighten: float, loosen_extra_days: int) -> Dict[str, Any]:
    """生成"模式×闸门"后的最终阈值表（从config中按别名取B模式基准值，缺失用合理默认）。"""
    K = float(MODE_K.get(mode, 1.0))
    cfg = HARD_RULES_MODE_B

    # 按config真实键名取基础区间值
    pct_lo_b, pct_hi_b = cfg.get("pct_chg", (0.02, 0.07))
    cap_lo_b, cap_hi_b = cfg.get("market_cap", (30, 200))
    vr_lo_b, vr_hi_b = cfg.get("vol_ratio", (1.1, 5.0))
    tr_lo_b, tr_hi_b = cfg.get("turnover_rate", (0.05, 0.15))
    limit_up_N_day_b = int(cfg.get("limit_up_days", 30))

    # config中未显式定义ma20偏差与振幅系数的话，用设计默认值
    ma20_dev_band = cfg.get("ma20_deviation_band", (-0.05, 0.05))  # 价格在MA20±5%内

    # ①区间型K（global_tighten=1.1再乘一层）
    pct_lo, pct_hi = _apply_k_interval(pct_lo_b, pct_hi_b, K, global_tighten)
    cap_lo, cap_hi = _apply_k_interval(cap_lo_b, cap_hi_b, K, global_tighten)
    vr_lo, vr_hi = _apply_k_interval(vr_lo_b, vr_hi_b, K, global_tighten)
    tr_lo, tr_hi = _apply_k_interval(tr_lo_b, tr_hi_b, K, global_tighten)
    # A signed band around zero tightens by dividing both ends, not multiplying its negative side.
    band_factor = max(.2, K * float(global_tighten))
    dev_lo, dev_hi = (float(v) / band_factor for v in ma20_dev_band)

    # ②天数窗口型（注意loosen_extra_days只对涨停窗口N_day生效）
    limit_up_N = _apply_k_window_ceil(limit_up_N_day_b, K, loosen_extra_days)
    if not 1 <= limit_up_N <= 60:
        raise ValueError("有效涨停窗口需在1至60个交易日内，请调整limit_up_days或模式")

    # A/C模式下科创板/创业板前缀列表（A直接禁止，C允许）
    kcb_cyb_prefixes = ("688", "300")

    return dict(
        K=K,
        mode=mode,
        global_tighten=global_tighten,
        loosen_extra_days=loosen_extra_days,
        pct_lo=pct_lo, pct_hi=pct_hi,                 # cond1
        cap_lo=cap_lo, cap_hi=cap_hi,                 # cond2
        vr_lo=vr_lo, vr_hi=vr_hi,                     # cond3
        tr_lo=tr_lo, tr_hi=tr_hi,                     # cond4
        limit_up_N=limit_up_N,                        # cond5 天数
        dev_lo=dev_lo, dev_hi=dev_hi,                 # cond6
        kcb_cyb_prefixes=kcb_cyb_prefixes,
        amplitude_min=float(cfg.get("amplitude_min", .03)),
    )


def run_hard_filter(
    candidate_pool_df: pd.DataFrame,
    *,
    mode: str = "B",
    global_tightness_factor: float = 1.0,
    loosen_limitup_extra_days: int = 0,
    intraday_map: Optional[Dict[str, pd.DataFrame]] = None,
    close_price_map: Optional[Dict[str, float]] = None,
    skip_cond9: bool = False,
) -> Tuple[pd.DataFrame, FilterReport]:
    """对候选池执行10条硬门槛 + K系数三类应用 + cond9分时。

    参数:
        skip_cond9: True=跳过cond9（第一轮筛cond1-8+10，后续再补cond9二次筛）
                    用于orchestrator两阶段过滤（先无分时快速筛，再对剩余票拉分时）
    返回 (passed_df, report)
    passed_df 新增列：passed_reason（命中规则简述，用于后面top5 formatter展示）
    """
    # 防御性输入
    if candidate_pool_df is None or len(candidate_pool_df) == 0:
        return candidate_pool_df, FilterReport(total_input=0, passed=0)

    mode = mode.upper() if isinstance(mode, str) else mode
    if mode not in ("A", "B", "C"):
        raise ValueError(f"[hard_filter] mode必须A/B/C，收到: {mode!r}")

    th = _apply_thresholds_for_mode(mode, float(global_tightness_factor or 1.0),
                                    int(loosen_limitup_extra_days or 0))
    intraday_map = intraday_map or {}
    close_map = dict(close_price_map or {})

    K = th["K"]
    N = th["limit_up_N"]

    report = FilterReport(total_input=int(len(candidate_pool_df)), extra=dict(th))
    dropped = report.dropped_by_cond

    passed_rows: List[pd.Series] = []
    passed_reasons: List[str] = []

    for _, r in candidate_pool_df.iterrows():
        reasons: List[str] = []
        ok_all = True

        # === cond1 涨跌幅区间 ===
        pct = float(r.get("pct_chg") or 0.0)
        if not (th["pct_lo"] <= pct <= th["pct_hi"]):
            dropped["cond1_pct"] = dropped.get("cond1_pct", 0) + 1
            ok_all = False
        else:
            reasons.append(f"涨跌幅{pct:+.2%}∈[{th['pct_lo']:+.2%},{th['pct_hi']:+.2%}]")

        # === cond2 市值区间 ===
        cap = float(r.get("market_cap_yi") or np.nan)
        if np.isnan(cap) or not (th["cap_lo"] <= cap <= th["cap_hi"]):
            dropped["cond2_cap"] = dropped.get("cond2_cap", 0) + 1
            ok_all = False
        else:
            reasons.append(f"市值{cap:.0f}亿∈[{th['cap_lo']:.0f},{th['cap_hi']:.0f}]")

        # === cond3 量比区间 ===
        vr = float(r.get("vol_ratio") or np.nan)
        if np.isnan(vr) or not (th["vr_lo"] <= vr <= th["vr_hi"]):
            dropped["cond3_volratio"] = dropped.get("cond3_volratio", 0) + 1
            ok_all = False
        else:
            reasons.append(f"量比{vr:.2f}∈[{th['vr_lo']:.2f},{th['vr_hi']:.2f}]")

        # === cond4 换手率区间 ===
        tr = float(r.get("turnover_rate") or np.nan)
        if np.isnan(tr) or not (th["tr_lo"] <= tr <= th["tr_hi"]):
            dropped["cond4_turnover"] = dropped.get("cond4_turnover", 0) + 1
            ok_all = False
        else:
            reasons.append(f"换手率{tr:.2%}∈[{th['tr_lo']:.2%},{th['tr_hi']:.2%}]")

        # === cond5 近N日有涨停（涨停基因）（天数K型） ===
        gene_dict = r.get("has_limit_up_in_N_day") or {}
        has_gene = False
        if isinstance(gene_dict, dict):
            # JSON cache restores dictionary keys as strings. Never borrow a longer-window hit.
            has_gene = bool(gene_dict.get(N, gene_dict.get(str(N), False)))
        if not has_gene:
            dropped["cond5_gene"] = dropped.get("cond5_gene", 0) + 1
            ok_all = False
        else:
            reasons.append(f"近{N}日有涨停(gene)")

        # === cond6 价格距MA20偏差区间 ===
        close = float(r.get("close") or 0.0)
        ma20 = float(r.get("MA20") or np.nan)
        if not ok_all:
            pass
        if close <= 0 or np.isnan(ma20):
            dropped["cond6_ma20"] = dropped.get("cond6_ma20", 0) + 1
            ok_all = False
        else:
            dev = (close - ma20) / ma20
            if not (th["dev_lo"] <= dev <= th["dev_hi"]):
                dropped["cond6_ma20"] = dropped.get("cond6_ma20", 0) + 1
                ok_all = False
            else:
                reasons.append(f"MA20偏差{dev:+.2%}∈[{th['dev_lo']:+.2%},{th['dev_hi']:+.2%}]")

        # === cond7 均线形态（布尔叠加型K：A/B/C 不同分支） ===
        ma_ok, ma_note = _cond7_ma_checks(r, K)
        if not ma_ok:
            dropped["cond7_ma_shape"] = dropped.get("cond7_ma_shape", 0) + 1
            ok_all = False
        else:
            reasons.extend(ma_note)

        # === cond8 日内振幅：(high - low) / preclose ≥ 配置下限 ===
        preclose = float(r.get("preclose") or np.nan)
        high = float(r.get("high") or np.nan)
        low = float(r.get("low") or np.nan)
        if preclose <= 0 or np.isnan(high) or np.isnan(low):
            dropped["cond8_amp"] = dropped.get("cond8_amp", 0) + 1
            ok_all = False
        else:
            amp = (high - low) / preclose
            amp_min = th["amplitude_min"]
            if amp < amp_min:
                dropped["cond8_amp"] = dropped.get("cond8_amp", 0) + 1
                ok_all = False
            else:
                reasons.append(f"振幅{amp:+.2%}≥{amp_min:+.2%}")

        # === cond9 分时形态（按mode截取不同时间窗；需逐票单独拉1min分时） ===
        code = str(r.get("code") or "").zfill(6)
        cprice = close_map.get(code, close)
        if skip_cond9 and intraday_map.get(code) is None:
            # 第一轮：无分时数据时跳过cond9（不踢人，后续二次筛）
            shape_ok = True
            shape_note = "cond9_待二次筛(首轮跳过)"
        else:
            shape_ok, shape_note = _cond9_intraday_shape(
                intraday_map.get(code),
                K,
                mode=mode,
                close_price=cprice,
            )
        if not shape_ok:
            dropped["cond9_intraday"] = dropped.get("cond9_intraday", 0) + 1
            ok_all = False
        else:
            if shape_note:
                reasons.append(shape_note)
            else:
                reasons.append(f"cond9_{mode}分时形态OK")

        # === cond10 新股/科创等布尔叠加型 ===
        code_prefix = code[:3]
        is_kcb_cyb = code_prefix in th["kcb_cyb_prefixes"]
        new_stock = bool(r.get("new_stock"))
        strict_ok, strict_note = _cond10_extra_strictness(
            new_stock=new_stock, is_kcb_cyb=is_kcb_cyb, K=K,
            close=close, ma45=float(r.get("MA45") or np.nan),
            pct_chg=pct,
        )
        if not strict_ok:
            dropped["cond10_strict"] = dropped.get("cond10_strict", 0) + 1
            ok_all = False
        else:
            reasons.extend(strict_note)

        # 汇总
        if ok_all:
            row_out = r.copy()
            row_out["filter_reasons"] = "；".join(reasons)
            passed_rows.append(row_out)

    if not passed_rows:
        out_df = candidate_pool_df.iloc[0:0].copy()
    else:
        out_df = pd.DataFrame(passed_rows).reset_index(drop=True)
    report.passed = len(out_df)
    return out_df, report
