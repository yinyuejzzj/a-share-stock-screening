# A股筛选方法与交易策略整理：Python实现与本地使用

将几类 A 股筛选思路整理成可修改参数、在本地运行的 Python 工具。重点记录策略条件、代码实现和使用方法，另附部分技术条件的通达信选股公式，便于在券商软件中建立初筛清单。

## 策略介绍

| 方法 | 筛选思路 | Python 实现 | 使用说明 |
|---|---|---|---|
| **尾盘趋势与活跃度筛选** | 涨幅、市值、量比、换手、涨幅日历史、均线与分时条件过滤，再按规则和板块信息排序 | [核心模块](tail_screen/wei_pan_core) / [命令行入口](tail_screen/wei_pan_cli.py) | [尾盘使用说明](tail_screen/README.md) |
| **本地技术指标模型** | 固定随机森林按时间顺序验证，再用概率门槛筛选；原九条件筛选及自动模型源码另存实验目录 | [模型入口](local_model/cli.py) / [原实验代码](local_model/experimental) | [模型与原九条件筛选说明](local_model/README.md) |
| **EMA 趋势中的缩量回踩** | 判断向上趋势中价格是否接近 EMA、成交量是否收缩，同时提示破位、乖离和震荡状态 | [EMA 判断代码](ema/ema20_judge.py) | [EMA 使用说明](ema/README.md) |
| **跳空高开筛选** | 开盘严格突破上一交易日最高价，再叠加市值、PB、非 ST/退市及正成交量条件 | [跳空筛选代码](gap_screen/screen_gapup.py) | [跳空使用说明](gap_screen/README.md) |

[策略与交易规则详解](docs/strategies.md)介绍每类方法的信号、参数取舍、结果阅读方式和适用边界。筛选信号不等于成交指令；程序不自动交易，参数版本也未经收益优化。

## Python 实现与本地使用

建议 Python 3.10+。下载仓库并进入根目录，在 Windows PowerShell 运行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

以下命令直接调用虚拟环境 Python，无需先激活环境。在线取数依赖行情服务，当天日线可能尚未收盘；历史研究应手动指定已收盘日。

### 尾盘：先看参数，再用少量代码检查取数

```powershell
.venv\Scripts\python.exe tail_screen/wei_pan_cli.py --profile standard --mode B --show-config
.venv\Scripts\python.exe tail_screen/wei_pan_cli.py --profile strict --mode B --codes 000001,600000 --format both
```

输出本地 Markdown/HTML 候选报告，附生效阈值及规则说明。`--mode A/B/C` 控制时间窗口和附加形态，`--profile` 控制参数档位，两者分开选择。此入口使用最新快照，不能把 `--date` 改成过去日期来回测。

### 本地模型：指定股票、历史窗口和预测步长

```powershell
.venv\Scripts\python.exe -m local_model.cli --code 000001 --days 1500 --forward-days 3 --profile standard --output outputs/model_result.json
```

输出实际行情截止日、时间顺序留出验证指标、模型概率及是否达到门槛。也可用 `--csv` 提供单股票日线、`--end-date` 指定截止日；CSV 格式见 [模块说明](local_model/README.md)。默认入口是简化研究基线；原九条件规则与自动模型保留在 `experimental/`，使用方式和限制单独说明。

### EMA：查看缩量回踩及辅助提示

```powershell
.venv\Scripts\python.exe ema/ema20_judge.py 000001 --profile standard --history-days 120
.venv\Scripts\python.exe ema/ema20_judge.py 000001 --profile strict --as-of 2026-09-28 --history-days 120 --style short
```

默认只使用日线，报告保存在 `ema/`；`--out` 可改路径。`--as-of` 指定日线截止日，`--style` 控制乖离和破线提示，与三档筛选参数独立。

### 跳空：先用离线样例，再选择最新行情

```powershell
.venv\Scripts\python.exe gap_screen/screen_gapup.py --date 2026-09-28 --snapshot examples/gap_snapshot.csv --bars examples/gap_bars.csv --profile relaxed --out outputs/gap_demo.csv
.venv\Scripts\python.exe gap_screen/screen_gapup.py --live --profile standard --codes 000001,600000
```

第一条使用合成数据，输出 CSV 与参数 JSON，适合检查安装。真实历史筛选必须提供同日估值快照和未复权日线；最新市值/PB不能冒充历史值。示例股票代码仅用于演示命令。

## 参数修改与推荐版本

各模块都提供 `strict`（强限制）、`standard`（标准）、`relaxed`（弱限制）。它们是方便比较的参数起点，不是统一的风险等级；收紧条件也不保证收益更高。

| 模块 | 三档主要改变什么 | 自定义方式 |
|---|---|---|
| 尾盘 | 涨幅、市值、量比、换手、涨幅日窗口、MA20 偏差、振幅门槛 | `config.py` 中 `SCREEN_PROFILES`，先用 `--show-config` 查看 |
| 本地模型 | 模型概率门槛 0.65 / 0.60 / 0.55 | `--min-probability`；原九条件筛选的常量另见模块说明 |
| EMA | 回踩容差 1% / 2% / 3%，缩量上限 0.65 / 0.80 / 0.95 | `SCREEN_PROFILES`；周期等参数在 `CONFIG` |
| 跳空 | 市值区间、PB 上限、开盘超过昨高的幅度 | `--min-market-cap-yi`、`--max-market-cap-yi`、`--max-pb`、`--min-gap-pct` |

[参数与日期指南](docs/parameters.md)列出字段名、单位、具体参考值和历史日期的支持范围。

## 券商软件中的条件选股公式

[通达信公式与对应 Python 计算](broker_formulas/README.md)提供 EMA 缩量回踩、跳空突破、尾盘日线技术初筛三类可复制公式，以及 CSV 离线计算入口。

这部分用于快速初筛。尾盘分时和外盘判断、完整评分、财务条件、机器学习概率仍以原 Python 模块为准；公式与完整策略的差异逐项列在说明中。不同券商版本的数据、函数和 EMA 初始化口径可能不同，当前未在具体券商客户端完成兼容性验证。

## 检查记录与使用边界

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m unittest local_model.test_offline -v
```

[验证记录](docs/validation.md)与[系统检查记录](docs/systematic-review.md)说明已检查的行为和已修复的问题。源码测试、合成数据测试及公式结构检查不能证明策略收益，也不能替代真实客户端测试。

仓库不包含个人资料、原始截图、行情/模型缓存或交易账户信息。运行输出和日志保存在本地。真实研究还需考虑数据修订、复权、历史股票池、交易费用、滑点及不可成交情形。
