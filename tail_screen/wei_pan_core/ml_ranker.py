"""wei_pan_core.ml_ranker · ML辅助重排模块（审计修正公式）。

对照 checklist M7.1 ~ M7.3。
接受 scorer.py 的 Top10（rule_total_100已归一），逐票调 predictor 获取上涨概率，
按审计修正公式做最终重排：
    rule_norm = rule_total_100  （scorer已输出70+30=100制，直接用）
    final = rule_norm                        (regime == "weak")  → 弱市纯规则
    final = rule_norm*0.8 + ml_proba*100*0.2 (非弱市)            → 正常/牛市blend

predictor 不可用/超时时，ml_proba=0.5（中性），final=rule_norm*0.8+10（不改变相对排序）。
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ML_RERANK_CONFIG, PATHS


def _try_predict_single(code: str, forward_days: int = 1, timeout_sec: int = 20) -> Optional[float]:
    """调 predict_batch.predict_single 获取单票上涨概率。

    返回 ml_proba ∈ [0, 1]，失败/超时返回 None（由调用方兜底0.5）。
    注意：predict_batch.predict_single 会从零训练模型（耗时30-60s/票），
    实盘时应优先加载预训练模型（joblib缓存），否则10票=10分钟不可用。
    """
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"predict_single timeout after {timeout_sec}s")

    try:
        root = PATHS["root"]
        if root not in sys.path:
            sys.path.insert(0, root)
        from predict_batch import predict_single

        # 设置超时（仅Unix可用signal；Windows用threading兜底）
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_sec)
        except (AttributeError, ValueError):
            pass  # Windows无SIGALRM，跳过（靠predictor内部超时或用户耐心）

        result = predict_single(symbol=str(code).zfill(6), forward_days=forward_days)

        try:
            signal.alarm(0)  # 取消闹钟
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (AttributeError, ValueError):
            pass

        if result and isinstance(result, dict):
            pred = result.get("prediction") or {}
            up = pred.get("up_prob") or pred.get("up_probability")
            if up is not None:
                return float(np.clip(up, 0.0, 1.0))
            cp = pred.get("calibrated_prob")
            if cp is not None:
                return float(np.clip(cp, 0.0, 1.0))
            label = pred.get("label") or pred.get("prediction")
            if label is not None:
                return 0.6 if str(label).lower() in ("up", "涨", "1", "true") else 0.4
        return None
    except TimeoutError:
        return None
    except Exception:
        return None


def rerank_with_ml(
    top10_df: pd.DataFrame,
    *,
    regime: str = "neutral",
    ml_weight: float = 0.20,
    forward_days: int = 1,
    timeout_per_stock: int = 30,
    enable_ml: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """对规则Top10做ML重排 → Top5（带final_score）。

    参数:
        top10_df:     scorer 输出的 Top10，必须含 rule_total_100 列
        regime:       detect_regime 返回的 regime ("bull"/"neutral"/"weak")
        ml_weight:    ML权重（0.20默认；弱市自动设0）
        forward_days: predictor预测天数
        timeout_per_stock: 单票预测超时（秒），超时用0.5兜底
        enable_ml:    外部开关（False=完全禁用ML，纯规则排）

    返回:
        (top5_df, meta)
        top5_df 新增列: ml_proba, final_score, ml_source, rank_reason
        meta: {regime, ml_used, n_predicted, n_fallback, rule_weight, ml_weight}
    """
    if len(top10_df) == 0:
        return top10_df.copy(), {"regime": regime, "ml_used": False, "n_predicted": 0, "n_fallback": 0}

    df = top10_df.copy().reset_index(drop=True)

    # 弱市 → 禁用ML（审计修正 R3红线）
    disable_regimes = set(ML_RERANK_CONFIG.get("disable_when_regime", ["weak"]))
    ml_active = enable_ml and regime not in disable_regimes

    rule_w = float(ML_RERANK_CONFIG.get("rule_weight", 0.80))
    ml_w = float(ML_RERANK_CONFIG.get("ml_weight", 0.20)) if ml_active else 0.0

    ml_probas: List[Optional[float]] = []
    ml_sources: List[str] = []
    n_predicted = 0
    n_fallback = 0

    if ml_active:
        for _, row in df.iterrows():
            code = str(row.get("code") or "").zfill(6)
            proba = _try_predict_single(code, forward_days=forward_days, timeout_sec=timeout_per_stock)
            if proba is not None:
                ml_probas.append(proba)
                ml_sources.append("predictor")
                n_predicted += 1
            else:
                ml_probas.append(0.5)  # 中性兜底
                ml_sources.append("fallback_0.5")
                n_fallback += 1
    else:
        # 弱市或外部禁用 → 全部走纯规则
        ml_probas = [0.5] * len(df)
        ml_sources = ["disabled_" + regime] * len(df)

    df["ml_proba"] = ml_probas
    df["ml_source"] = ml_sources

    # 审计修正公式：
    #   rule_norm = rule_total_100（已经是100制）
    #   final = rule_norm*0.8 + ml_proba*100*0.2  (非弱市)
    #   final = rule_norm                          (弱市)
    rule_norm = df["rule_total_100"].astype(float).values
    if ml_active:
        final = rule_norm * rule_w + np.array(ml_probas, dtype=float) * 100.0 * ml_w
    else:
        final = rule_norm.copy()

    df["final_score"] = np.round(final, 3)
    df["rank_reason"] = df.apply(
        lambda r: f"规则{r['rule_total_100']:.1f}×{rule_w:.0%}"
                   + (f"+ML{r['ml_proba']:.0%}×{ml_w:.0%}={r['final_score']:.1f}" if ml_active else f"=纯规则{r['final_score']:.1f}"),
        axis=1,
    )

    # 按final_score降序取Top5
    df = df.sort_values(by="final_score", ascending=False, kind="stable").reset_index(drop=True)
    top5 = df.head(int(ML_RERANK_CONFIG.get("output_top_n", 5))).copy()

    meta = {
        "regime": regime,
        "ml_used": ml_active,
        "n_predicted": n_predicted,
        "n_fallback": n_fallback,
        "rule_weight": rule_w,
        "ml_weight": ml_w,
    }
    return top5, meta
