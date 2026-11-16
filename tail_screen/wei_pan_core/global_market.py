"""wei_pan_core.global_market · 外盘情绪总闸 + 大盘状态ML权重判定。

验收对照 checklist M4.1 ~ M4.5。
两项能力：
  1) compute_weighted_gate(market_close_date) → 计算外盘加权涨跌幅 + 闸门动作hold/tighten/normal/loosen
     · 单指数缺失时按剩余权重重新归一（设计§3.2规则1）
     · 美股整组缺失时：标记data_incomplete + 走tighten（不直接hold空仓，设计§3.2规则2）
     · 返回结构化 dict：gate_action / global_tightness_factor / weighted_pct / group_rows / data_incomplete
  2) detect_regime(regime_date) → bull/neutral/weak（审计修正：命中≥2 且 > 对侧命中才算，平手→neutral）
     · 返回 regime + bull_hits / weak_hits 每条件命中明细（用于formatter报告段展示）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .config import GLOBAL_GATE, REGIME_RULES


# =========================================================================
# 🛡️ 对外M4.1~M4.4 外盘情绪总闸（加权+缺失归一）
# =========================================================================
def compute_weighted_gate(market_close_date: Optional[str] = None) -> Dict[str, Any]:
    """计算外盘情绪闸门 → {gate_action, global_tightness_factor, weighted_pct, groups, data_incomplete, reason}

    gate_action:
        "hold"   = 外盘加权 ≤ -2% （今日空仓，不选股）
        "tighten"= 外盘加权在 -2% ~ -1%，或者美股整组缺失走数据异常"保守收紧"
        "normal" = 正常区间 -1% ~ +1%
        "loosen" = ≥ +1%，条件略松
    """
    # 延迟import，防止模块在非网络环境import失败（测试mock数据时不需要fetcher）
    from .data_fetcher import fetch_global_indices

    gate_cfg = GLOBAL_GATE
    thresholds = gate_cfg["thresholds"]
    date = market_close_date or datetime.now().strftime("%Y-%m-%d")

    fetch_res = fetch_global_indices(market_close_date=date)
    groups = list(fetch_res.get("data", []) or [])

    # 按组拿 group_pct / total_weight，处理整组缺失的收紧分支
    available_weights: List[float] = []    # 实际拿到数据的组的原始总权重
    available_pcts: List[float] = []       # 对应组涨跌幅
    data_incomplete = False
    tighten_forced_by_us_missing = False

    for g in groups:
        w = float(g.get("total_weight", 0.0))
        gp = g.get("group_pct")
        success = bool(g.get("success"))
        if success and gp is not None:
            available_weights.append(w)
            available_pcts.append(float(gp))
        else:
            # 组失败
            whole_policy = g.get("whole_group_missing_policy")
            if whole_policy == "data_incomplete_abnormal_tighten":
                # 美股整组失败：标记异常 + 保守强制收紧（闸门最终最小=tighten）
                data_incomplete = True
                tighten_forced_by_us_missing = True
                # 注意：该组权重不加入分母（不归一，因为不知道是涨是跌，保守按缺失处理）
            # 其他组（JP/KR/EU的单个或整组缺失）→直接从剩余组归一（分母÷sum(available_weights)）
            data_incomplete = True  # 只要有任何组拿不到就标不完整（在报告显示）

    # 归一 + 加权
    if not available_weights:
        # 一组都没拿到 → 保守直接tighten（没法判断外盘）
        weighted_pct = None
    else:
        wsum = float(sum(available_weights))
        if wsum <= 0:
            weighted_pct = None
        else:
            wp = 0.0
            for w, p in zip(available_weights, available_pcts):
                # P2-10修正：极端值校验（单指数日涨幅>15%或<-15%几乎必为数据错误）
                if abs(p) > 0.15:
                    data_incomplete = True
                    # 该组数据异常→标记不完整但不参与加权（保守跳过）
                    continue
                wp += (w / wsum) * p
            weighted_pct = wp if wp != 0.0 else None

    # 闸门动作判定（设计§3.2 阈值4档）
    crash = float(thresholds["crash_hold"])        # -0.02
    t_lo, t_hi = thresholds["tighten_band"]         # (-0.02, -0.01)
    n_lo, n_hi = thresholds["normal_band"]          # (-0.01, 0.01)
    loosen = float(thresholds["loosen_above"])      # 0.01

    gate_action = "normal"
    global_tightness_factor = float(gate_cfg["normal_factor"])  # 1.0
    loosen_limitup_extra_days = 0

    if tighten_forced_by_us_missing:
        # 美股缺失：数据异常 → 即使加权算出来>0也强制收紧（最保守）
        gate_action = "tighten"
        global_tightness_factor = float(gate_cfg["tighten_factor"])  # 1.1
    elif weighted_pct is None:
        # 全部没拿到 → tighten（等价于：宁可不买也别踩雷）
        gate_action = "tighten"
        global_tightness_factor = float(gate_cfg["tighten_factor"])
    else:
        if weighted_pct <= crash:
            gate_action = "hold"
            global_tightness_factor = 0.0  # hold=不选股，因子为0形式化
        elif t_lo < weighted_pct <= t_hi:
            gate_action = "tighten"
            global_tightness_factor = float(gate_cfg["tighten_factor"])  # 1.10
        elif n_lo < weighted_pct <= n_hi:
            gate_action = "normal"
            global_tightness_factor = float(gate_cfg["normal_factor"])  # 1.00
        else:  # ≥ loosen(+1%)
            gate_action = "loosen"
            global_tightness_factor = float(gate_cfg["loosen_factor"])  # 0.95
            loosen_limitup_extra_days = int(gate_cfg.get("loosen_limitup_window_add_days", 0))  # +5日

    # 每只指数归一前后权重明细（方便formatter报告显示，调试友好）
    detail_rows = []
    if available_weights:
        wsum = sum(available_weights)
    else:
        wsum = 0.0
    for g in groups:
        orig_w = float(g.get("total_weight", 0.0))
        gp = g.get("group_pct")
        success = bool(g.get("success"))
        renorm_w = (orig_w / wsum) if (success and wsum > 0 and gp is not None) else None
        detail_rows.append({
            "group_id": g.get("id"),
            "members": list(g.get("members", [])),
            "orig_w": orig_w,
            "orig_weight": orig_w,
            "renorm_w": renorm_w,
            "renorm_weight": renorm_w,
            "group_pct": gp,
            "success": success,
            "constituents": list(g.get("constituents", [])),
        })

    return {
        "gate_action": gate_action,                          # "hold" / "tighten" / "normal" / "loosen"
        "global_tightness_factor": global_tightness_factor,  # 0.0 / 1.1 / 1.0 / 0.95
        "loosen_limitup_extra_days": loosen_limitup_extra_days,  # loosen时涨停窗口+5日
        "weighted_pct": weighted_pct,                        # 实际加权涨跌幅（None=全部缺失）
        "data_incomplete": data_incomplete,                  # 报告顶部⚠️
        "groups_detail": detail_rows,                        # 逐组+逐指数明细
        "reason": "ok",
        "fallback_used": list(fetch_res.get("fallback_used", []) or []),
        "market_close_date": date,
    }


# =========================================================================
# 🛡️ 对外 M4.5 大盘状态判定（审计修正：bull/weak冲突相对命中数）
# =========================================================================
def _evaluate_conditions(cond_list: List[Tuple[str, Any]], signals: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
    """对bull或weak的4条条件，逐条与signals比对 → 命中数 + 命中明细。"""
    hits = 0
    detail: List[Dict[str, Any]] = []
    for op_name, threshold in cond_list:
        ok = False
        actual = signals.get(op_name.split("_gt")[0].split("_lt")[0].split("_streak")[0] + ("_streak_down" if "_streak_down" in op_name else ""))
        # 兼容命名差异：统一从signals中按"操作关键词"取
        try:
            if op_name == "hs300_5d_gain_gt":
                v = signals.get("hs300_5d_return")
                ok = (v is not None) and float(v) > float(threshold)
                actual = v
            elif op_name == "hs300_5d_gain_lt":
                v = signals.get("hs300_5d_return")
                ok = (v is not None) and float(v) < float(threshold)
                actual = v
            elif op_name == "hs300_3d_streak_down":
                v = signals.get("hs300_3d_streak_down")
                ok = (v is True) and bool(threshold) is True
                actual = v
            elif op_name == "limit_up_count_gt":
                v = signals.get("limit_up_count")
                ok = (v is not None) and int(v) > int(threshold)
                actual = v
            elif op_name == "limit_down_count_lt":
                v = signals.get("limit_down_count")
                ok = (v is not None) and int(v) < int(threshold)
                actual = v
            elif op_name == "limit_down_count_gt":
                v = signals.get("limit_down_count")
                ok = (v is not None) and int(v) > int(threshold)
                actual = v
            elif op_name == "northbound_flow_gt":
                v = signals.get("northbound_flow_yuan")
                ok = (v is not None) and float(v) > float(threshold)
                actual = v
            elif op_name == "northbound_flow_lt":
                v = signals.get("northbound_flow_yuan")
                ok = (v is not None) and float(v) < float(threshold)
                actual = v
            else:
                ok = False
        except Exception:
            ok = False
        if ok:
            hits += 1
        detail.append({"cond": op_name, "threshold": threshold, "actual": actual, "hit": ok})
    return hits, detail


def detect_regime(regime_date: Optional[str] = None) -> Dict[str, Any]:
    """返回 {
        "regime": "bull" / "neutral" / "weak",
        "bull_hits": int, "weak_hits": int,
        "bull_detail": [...], "weak_detail": [...],  # （给formatter逐条件展示）
        "ml_weight": float (0.0 or 0.2),
        "hit_threshold": int (2),
        "signals": dict,  # 原始信号（调试）
        "fallback_used": list,
    }

    算法（审计修正 §3.3 relative_hit_count）：
        bull≥2 AND bull>weak → bull；
        weak≥2 AND weak>bull → weak；
        其余一切 → neutral（含两边平手=2的场景，保守判中性）。
    """
    from .data_fetcher import fetch_hs300_regime_signals

    date = regime_date or datetime.now().strftime("%Y-%m-%d")
    r = fetch_hs300_regime_signals(regime_date=date)
    signals_raw: Dict[str, Any] = dict(r.get("data", {}) or {})
    fallback_used = list(r.get("fallback_used", []) or [])

    thresh = int(REGIME_RULES.get("hit_threshold", 2))
    ml_cfg = REGIME_RULES.get("ml_weight", {"bull": 0.20, "neutral": 0.20, "weak": 0.00})

    bull_hits, bull_detail = _evaluate_conditions(list(REGIME_RULES["bull"]), signals_raw)
    weak_hits, weak_detail = _evaluate_conditions(list(REGIME_RULES["weak"]), signals_raw)

    if bull_hits >= thresh and bull_hits > weak_hits:
        regime = "bull"
    elif weak_hits >= thresh and weak_hits > bull_hits:
        regime = "weak"
    else:
        regime = "neutral"  # 包括平手（2:2）或两边都没到阈值

    ml_weight = float(ml_cfg.get(regime, 0.20))
    return {
        "regime": regime,
        "bull_hits": bull_hits,
        "weak_hits": weak_hits,
        "hit_threshold": thresh,
        "bull_detail": bull_detail,
        "weak_detail": weak_detail,
        "ml_weight": ml_weight,
        "signals": signals_raw,
        "fallback_used": fallback_used,
        "date": date,
    }
