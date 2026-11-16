# A 股选股与信号研究工具

四组 Python 工具：尾盘条件筛选、本地预测模型、EMA 均线判断、跳空高开筛选。用于研究不同规则如何改变候选清单，提供可修改参数和强限制、标准、弱限制三个参考版本。

这些档位是便于比较的参数起点，没有经过独立样本验证为最优。强限制意味着部分条件更收紧，不意味着收益更高或风险更低。程序不自动交易。

## 包含内容

| 模块 | 入口 | 功能与范围 |
|---|---|---|
| [尾盘筛选](tail_screen/README.md) | `tail_screen/wei_pan_cli.py` | 市值、涨幅、量比、换手、涨停窗口、均线和分时条件，规则评分输出候选清单 |
| [本地模型](local_model/README.md) | 详见模块说明 | 指标计算、时序划分训练和实验模型源码 |
| [EMA 判断](ema/README.md) | `ema/ema20_judge.py` | 均线附近缩量回踩、量能及震荡提示 |
| [跳空高开](gap_screen/README.md) | `gap_screen/screen_gapup.py` | 开盘突破上一交易日最高价，叠加市值与 PB 过滤 |

## 安装与运行

建议 Python 3.10+，在仓库根目录建立虚拟环境：

```bash
python -m venv .venv
# Windows PowerShell
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

先查看配置，再运行筛选：

```bash
python tail_screen/wei_pan_cli.py --profile strict --show-config
python gap_screen/screen_gapup.py --profile standard --show-config
python ema/ema20_judge.py --help
```

依赖范围是安装起点，不是所有组合都经验证；网络接口与字段可能变化。可先运行离线示例检查环境：

```bash
python gap_screen/screen_gapup.py --date 2026-09-28 --snapshot examples/gap_snapshot.csv --bars examples/gap_bars.csv --profile relaxed --out outputs/gap_demo.csv
python -m unittest discover -s tests -v
python -m unittest local_model.test_offline -v
```

`examples/` 中的样例为合成数据，不能当作历史选股结果。

## 哪些条件和日期可以修改

各模块说明均给出：真实生效的参数名、单位、默认值、三档建议、命令行覆盖方法和日期限制。汇总见 [参数与日期指南](docs/parameters.md)。

- **筛选档位**：`strict`（强限制）、`standard`（标准）、`relaxed`（弱限制）。不同模块的档位针对不同信号，不能跨模块当作统一风险等级。
- **日期**：日期参数不会凭空提供历史估值、历史股票池或历史分钟行情。最新行情筛选与按历史截止日计算分别处理，具体支持范围以模块说明为准。
- **输出**：运行结果、行情缓存、模型文件和日志在本地生成，不提交到仓库。

## 验证范围与局限

本仓库不宣称具备实盘验证过的胜率、收益率或无前视偏差的完整回测。离线测试用于检查参数切换、日期边界和核心规则，不验证投资效果，也不保证第三方行情接口一直可用。

若要比较档位效果，应固定股票池、样本区间和数据时点，按时间切分训练/验证/测试，计入手续费、滑点、涨跌停及停牌等不可成交情形；调参只使用训练/验证区间，保留独立测试区间。最新名单与最新估值不能直接代表历史时点。

公开版本从本地研究代码整理，保留必要的模块逻辑，并在模块说明中列出公开入口与原实验代码的区别。仓库不包含 Rust 缠论引擎、加密货币交易工程、个人资料、原始截图或个人记忆文件。
