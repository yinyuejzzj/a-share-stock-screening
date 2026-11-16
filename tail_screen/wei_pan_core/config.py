"""wei_pan_core.config · 配置中心（唯一阈值入口 v1.1，审计6项修正版）

验收对照 checklist M1.1 ~ M1.8。
修改任何阈值 / 权重 / 列表 **只改这个文件**，不要去各模块实现里硬编码数字。
"""

from __future__ import annotations

import os
import math
from typing import Dict, Tuple, Any


# =========================================================================
# 🏷️ 版本信息（全局报告 / 日志必带，用于核对设计文档版本）
# =========================================================================
CONFIG_VERSION = "1.1.0"
DESIGN_DOC_VERSION = "v1.1（审计6项修正通过）"
# 审计强制风险声明，formatter.py 三格式报告都必须带这段话（R2 红线）
RISK_DISCLAIMER = (
    "本工具输出技术筛选清单，不构成投资建议或收益承诺。"
    "参数档位仅供工程试验，尚无经独立验证的收益或胜率结论。"
    "实际结果受数据质量、交易成本、停牌和涨跌停限制等影响。"
)

ILLEGAL_WORDS = ["保证获利", "必赚", "稳赢", "稳赚", "无风险", "保本"]  # R1 检查关键词


# =========================================================================
# ⏰ M1.1 三买入时点严格度系数 + Auto模式时间判定表
# =========================================================================
TIME_MODE_K: Dict[str, float] = {
    "A": 1.20,   # < 14:45，最严格
    "B": 1.00,   # 14:45 ~ 14:52，默认
    "C": 0.85,   # ≥ 14:52，最宽松
}

# Auto模式：按当前系统时间映射买入模式（14:45分界 / 14:52分界）
AUTO_MODE_TIME: Dict[str, str] = {
    "before_1445": "A",   # 系统时间 hhmm < 1445
    "1445_1452":  "B",    # 1445 ≤ hhmm < 1452
    "after_1452": "C",    # hhmm ≥ 1452
}
# 默认输出Top数量（主程序 --top 参数覆盖）
OUTPUT_TOP_N = 5


# =========================================================================
# 🧱 M1.2 第1层 · 10条硬门槛（模式B基准值）
#     三类参数应用规则：
#       区间型 (lo, hi) → 应用K时：lo*K / hi÷K
#       天数型  int N  → ceil(N_B / K)
#       布尔叠加型  dict → hard_filter按time_mode写死分支（不用K乘除）
# =========================================================================
# Only keys consumed by the published filter are listed here.
# CLI profile selection updates this dictionary in place.
HARD_RULES_MODE_B: Dict[str, Any] = {
    "pct_chg": (.02, .07),        # decimal return, .02 means 2%
    "market_cap": (30, 200),      # CNY 100 million
    "vol_ratio": (1.1, 5.),
    "turnover_rate": (.05, .15),  # decimal, .05 means 5%
    "limit_up_days": 30,          # trading-session window
    "ma20_deviation_band": (-.05, .05),
    "amplitude_min": .03,
    "new_stock_days": 30,         # approximate trading-session age
}


# =========================================================================
# 📊 M1.3 第2层 · 打分权重（基础分70 + 板块热度30 = 100）
# =========================================================================
SCORE_BASE_WEIGHTS: Dict[str, int] = {
    "limit_up_gene":       15,   # 涨停基因（最高权重，7项基础分）
    "pct_chg_position":    10,   # 当日涨幅在允许区间内位置
    "vol_ratio_strength":   8,   # 量比位置
    "turnover_health":      7,   # 换手健康度
    "ma_strength":         10,   # 均线强度
    "vol_price_sync":       8,   # 量价力度
    "outperform_hs300":     7,   # 跑赢沪深300
    "minute_shape":         5,   # 分时形态质量
}
# 合计校验：70
_SUM_BASE = sum(SCORE_BASE_WEIGHTS.values())
assert _SUM_BASE == 70, f"SCORE_BASE_WEIGHTS 合计应为70，当前={_SUM_BASE}，请检查config.py"

SCORE_SECTOR_WEIGHTS: Dict[str, int] = {
    "sector_rank_3d":  15,   # 所属行业板块 3 日涨幅排名
    "sector_rank_5d":  10,   # 所属行业板块 5 日涨幅排名
    "stock_contrib":    5,   # 个股当日在所属板块内的涨幅贡献排名
}
_SUM_SECTOR = sum(SCORE_SECTOR_WEIGHTS.values())
assert _SUM_SECTOR == 30, f"SCORE_SECTOR_WEIGHTS 合计应为30，当前={_SUM_SECTOR}，请检查config.py"

# 涨停基因距离型分档（对应设计§4.1）
LIMITUP_GENE_SCORE_TABLE: Dict[int, int] = {
    20: 15,   # 近20日涨停 → 满
    30: 12,   # 近30日 → 12
    45:  8,   # 近45日 → 8
    # 超过45 → 0（由scorer.py兜底）
}


# =========================================================================
# 🤖 M1.4 第3层 · ML辅助重排配置（审计修正公式，R3 红线）
# =========================================================================
ML_RERANK_CONFIG: Dict[str, Any] = {
    "only_on_top_n":      10,     # 只跑规则前10
    "rule_weight":        0.80,   # 规则权重
    "ml_weight":          0.20,   # ML权重
    "disable_when_regime": ["weak"],  # 弱市时 ML=0
    # ✅ 审计修正：先规则归一到100制再加权（旧公式 rule/0.8 绝对禁止）
    "rule_max_score":     70,     # 规则分满分基准，归一公式：rule_norm = rule / 70 * 100
    "output_top_n":       OUTPUT_TOP_N,
}


# =========================================================================
# 🌍 M1.5 外盘情绪总闸（审计修正：整组缺失差异化降级）
# =========================================================================
GLOBAL_GATE: Dict[str, Any] = {
    "groups": [
        {
            "id": "US",
            "members": ["^DJI", "^IXIC", "^GSPC"],  # 道琼斯 / 纳斯达克 / 标普500
            "intra_method": "avg",                 # 组内三者简单平均
            "total_weight": 0.50,
            # 审计强制：美股整组取不到时，标记"数据异常+收紧"，不直接判空仓
            "whole_group_missing_policy": "data_incomplete_abnormal_tighten",
        },
        {"id": "HK", "members": ["HSI"],        "intra_method": "first", "total_weight": 0.20},
        {"id": "JP", "members": ["N225"],       "intra_method": "first", "total_weight": 0.10},
        {"id": "KR", "members": ["KS11"],       "intra_method": "first", "total_weight": 0.10},
        {
            "id": "EU",
            "members": ["FTSE", "DAX"],  # 富时100 + 德国DAX
            "intra_method": "avg",
            "total_weight": 0.10,
            "whole_group_missing_policy": "renormalize_remaining",  # 同单个缺失
        },
    ],
    "thresholds": {
        "crash_hold":   -0.02,          # ≤-2% → 空仓 hold
        "tighten_band": (-0.02, -0.01), # 之间 → tighten 下限×1.1
        "normal_band":  (-0.01, 0.01),  # 之间 → normal 系数=1.0
        "loosen_above":  0.01,          # ≥+1% → loosen 下限×0.95+涨停窗口+5
    },
    # 系数对应
    "tighten_factor": 1.10,
    "normal_factor":  1.00,
    "loosen_factor":  0.95,
    "loosen_limitup_window_add_days": 5,
}


# =========================================================================
# 📈 M1.6 大盘状态判定（审计修正：bull/weak冲突用命中相对优势解）
# =========================================================================
REGIME_RULES: Dict[str, Any] = {
    "bull": [  # 命中≥2 且 bull>weak 才算强市
        ("hs300_5d_gain_gt",           0.02),      # ① HS300 近 5 日 > +2%
        ("limit_up_count_gt",            60),      # ② 涨停家数 > 60
        ("limit_down_count_lt",           5),      # ③ 跌停家数 < 5
        ("northbound_flow_gt",   5_0000_0000),    # ④ 北向净流入 > 50 亿（元）
    ],
    "weak": [  # 命中≥2 且 weak>bull 才算弱市
        ("hs300_5d_gain_lt",          -0.02),      # ① HS300 近 5 日 < -2%
        ("hs300_3d_streak_down",       True),      # ② 近 3 日连跌
        ("limit_down_count_gt",          20),      # ③ 跌停家数 > 20
        ("northbound_flow_lt",  -3_0000_0000),    # ④ 北向净流出 > 30 亿
    ],
    # 审计修正：冲突判定规则
    #   命中≥2 且严格 > 对侧命中 → 对应regime；否则→neutral
    "resolve": "relative_hit_count",
    "hit_threshold": 2,
    "ml_weight": {"bull": 0.20, "neutral": 0.20, "weak": 0.00},
}


# =========================================================================
# 💸 M1.7 三卖出策略止损/止盈参数
# =========================================================================
SELL_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "A": {
        "hard_stop_loss_pct": -0.02,    # -2% 硬止损（竞价低开超此值也必须卖）
        "recommend_sell_multiplier": 1.00,  # 其他情况以竞价价直接卖（正常）
        "sell_time": "09:25",
    },
    "B": {
        "stop_loss_pct":    -0.015,     # -1.5%
        "take_profit_pct":   0.015,     # +1.5%
        "window_start":     "09:30",
        "window_end":       "09:45",    # 9:45时间止损
    },
    "C": {
        "hard_stop_loss_pct":        -0.018,   # -1.8% 硬止损
        "take_profit_fast_pct":       0.020,   # +2% 快速止盈
        "take_profit_trailing_on":    True,    # +1.2%+高点回撤0.5%止盈
        "trailing_trigger_pct":       0.012,
        "trailing_drawdown_pct":      0.005,
        "vol_break_stop_loss":        True,    # 亏损中放量破-1%也止损
        "final_close_time":           "10:00", # 10点强制平仓
    },
}


# =========================================================================
# 📂 M1.8 输出 / 缓存 / 日志 路径
# =========================================================================
def _root() -> str:
    """Return the tail_screen module directory."""
    # config.py 路径 = gupiao/wei_pan_core/config.py
    core_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(core_dir)  # gupiao/


ROOT_DIR = _root()

PATHS: Dict[str, str] = {
    "root":            ROOT_DIR,
    "result_base":     os.path.join(ROOT_DIR, "04_输出结果", "04尾盘选股结果"),
    "market_snap_dir": os.path.join(ROOT_DIR, "04_输出结果", "04尾盘选股结果", "_market_snapshots"),
    "data_cache":      os.path.join(ROOT_DIR, "缓存", "weipan_data_cache"),
    "model_cache":     os.path.join(ROOT_DIR, "04_输出结果", "03模型缓存", "wei_pan_ml_cache"),
    "log_dir":         os.path.join(ROOT_DIR, "logs", "weipan"),
    # 与外部predictor的边界协议缓存（放ml_cache下）
    "ml_predict_cache": os.path.join(ROOT_DIR, "04_输出结果", "03模型缓存", "wei_pan_ml_cache", "prediction_cache"),
}

# 自动创建目录（防止首次运行报不存在）
for _p in ("result_base", "data_cache", "model_cache", "log_dir", "market_snap_dir", "ml_predict_cache"):
    try:
        os.makedirs(PATHS[_p], exist_ok=True)
    except Exception:  # pragma: no cover - 创建失败交给上层日志提示，不阻止config加载
        pass


# =========================================================================
# 🎛️ 数据源优先级 + 失败重试（data_fetcher.py 读取）
# =========================================================================
FETCHER_CONFIG: Dict[str, Any] = {
    "source_priority":   ["akshare", "sina", "tencent"],
    "request_timeout":   12,       # 秒
    "max_retries":       2,        # 每次重试间隔2s/4s（退避）
    "cache_use_for_day": True,     # 同一交易日重复运行，优先读本地缓存（不打接口）
    "user_agent":        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
}


# =========================================================================
# 🧮 公共辅助：K系数应用工具（hard_filter.py 直接调）
#   对应设计§2.1的三类参数K应用规则，全局唯一实现，避免各模块重复写
# =========================================================================
def apply_k_to_interval(interval: Tuple[float, float], K: float,
                        global_tightness: float = 1.0) -> Tuple[float, float]:
    """类型A·区间型K应用：lo = K * interval[0]；hi = interval[1] / K。

    global_tightness 外盘总闸再叠加：lo × global_tightness（≥1更严，<1更松）
    hi 除以 global_tightness（≥1时hi更低更严，<1时hi更高更松）。
    """
    lo, hi = interval
    lo_new = lo * K * global_tightness
    hi_new = hi / K / global_tightness if global_tightness > 0 else hi
    return (round(lo_new, 6), round(hi_new, 6))


def apply_k_to_window_days(N_B: int, K: float, extra_days_from_loosen: int = 0) -> int:
    """类型B·窗口天数型K应用：N = ceil(N_B / K) + loosen放宽。

    extra_days_from_loosen：外盘宽松时GLOBAL_GATE["loosen_limitup_window_add_days"]加5日。
    """
    if K <= 0:
        return int(N_B) + extra_days_from_loosen  # 防御
    return math.ceil(N_B / K) + extra_days_from_loosen


# Three baseline sets; mode/time and market-gate multipliers are applied afterwards.
SCREEN_PROFILES = {
    "strict": {
        "pct_chg": (.025, .06), "market_cap": (40, 150),
        "vol_ratio": (1.3, 4.), "turnover_rate": (.06, .12),
        "limit_up_days": 20, "ma20_deviation_band": (-.03, .03),
        "amplitude_min": .03,
    },
    "standard": {
        "pct_chg": (.02, .07), "market_cap": (30, 200),
        "vol_ratio": (1.1, 5.), "turnover_rate": (.05, .15),
        "limit_up_days": 30, "ma20_deviation_band": (-.05, .05),
        "amplitude_min": .03,
    },
    "relaxed": {
        "pct_chg": (.01, .085), "market_cap": (20, 300),
        "vol_ratio": (1., 6.), "turnover_rate": (.03, .20),
        "limit_up_days": 45, "ma20_deviation_band": (-.08, .08),
        "amplitude_min": .02,
    },
}


def select_profile(name: str) -> Dict[str, Any]:
    """Set the active baseline in place, preserving imported references to the dict."""
    if name not in SCREEN_PROFILES:
        raise ValueError(f"未知参数档位: {name}")
    HARD_RULES_MODE_B.update(SCREEN_PROFILES[name])
    return dict(SCREEN_PROFILES[name])


select_profile("standard")
