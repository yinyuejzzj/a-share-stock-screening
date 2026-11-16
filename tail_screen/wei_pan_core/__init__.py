"""wei_pan_core - 尾盘选股程序核心子包（与项目其余模块解耦，除ML重排外不依赖外部）。

模块清单：
    config       所有阈值/权重/路径常量（唯一入口，改阈值只改这里）
    data_fetcher 行情数据获取（akshare优先，新浪HTTP + 腾讯qt 双层fallback）
    stock_pool   候选股票池（ST/停牌剔除，新股标注）
    global_market 外盘情绪总闸 + 大盘状态（ML权重）判定
    hard_filter  10条硬门槛过滤 + 三买入时点K系数应用
    scorer       规则打分（基础70 + 板块热度30）→ Top10
    rerank_ml    ML辅助重排（仅Top10，弱市ML=0，审计修正归一公式）
    strategy_A/B/C  三卖出策略次日评估
    formatter    三格式输出（Markdown / Excel / JSON快照）
"""

__all__ = [
    "config",
    "data_fetcher",
    "stock_pool",
    "global_market",
    "sector_heat",
    "hard_filter",
    "scorer",
    "rerank_ml",
    "strategy_A",
    "strategy_B",
    "strategy_C",
    "formatter",
]

# 版本号与设计文档对齐（任何时候程序输出都要打这个版本号，便于核对审计修正）
__version__ = "1.1.0"
__design_doc_version__ = "v1.1（已通过审计6项修正）"
