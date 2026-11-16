# 跳空高开筛选

[策略定义与数据要求](../docs/strategies.md#4-跳空高开) · [跳空突破条件选股公式与Python参考计算](../broker_formulas/README.md)

券商公式只判断技术缺口和正成交量；完整市值、PB、名称状态过滤仍使用本模块，或在客户端另加相同口径的条件。

从固定日期筛选脚本整理，保留“市值 + PB + 非 ST/退市 + 开盘突破上一交易日最高价”的核心逻辑。排除成交量不大于零的数据。新增日期与参数入口，避免修改日期后仍拿最新估值当作历史估值。

## 推荐参数版本

以下为工程上的筛选起点，未用独立样本验证最优性；严格不代表更高收益。

| 参数 | 强限制 strict | 标准 standard（默认） | 弱限制 relaxed | 命令行覆盖 |
|---|---:|---:|---:|---|
| 最低总市值（亿元，含） | 30 | 20 | 10 | `--min-market-cap-yi` |
| 最高总市值（亿元，不含） | 80 | 100 | 200 | `--max-market-cap-yi` |
| PB 上限（PB 必须 > 0） | 5 | 10 | 15 | `--max-pb` |
| 开盘超过昨高的最低幅度（%） | 0.5 | 0 | 0 | `--min-gap-pct` |

这里 0.5 表示 0.5%，不是 50%。开盘必须严格高于“上一交易日最高价 × (1 + 幅度/100)”，等于阈值不入选；所有版本保留严格缺口定义。PB 与市值只表示筛选偏好，不能单独判断股票便宜或风险低。

`--lookback-days` 默认 40 个自然日，可从 20–90 调整以覆盖长假或停牌后的上一交易日。程序不会在目标交易日没有 K 线时自动改成其它日期。

## 使用

在仓库根目录运行：

```bash
python gap_screen/screen_gapup.py --profile strict --show-config
python gap_screen/screen_gapup.py --live --profile standard --codes 000001,600000
python gap_screen/screen_gapup.py --date 2026-09-28 --snapshot examples/gap_snapshot.csv --bars examples/gap_bars.csv --profile relaxed --out outputs/gap_demo.csv
python gap_screen/screen_gapup.py --date 2026-09-28 --snapshot examples/gap_snapshot.csv --bars examples/gap_bars.csv --max-pb 6 --max-market-cap-yi 120
```

示例 CSV 是合成数据，用于演示接口，不是真实选股记录。

### 历史筛选

`--date YYYY-MM-DD` 指定目标交易日。必须同时提供该日股票快照和未复权历史日线：

- 快照列：`code,name,market_cap,pb,volume,snapshot_date`。总市值单位为元；成交量只用于判断 > 0。所有快照日期必须等于目标日，每代码一行。
- 日线列：`code,date,open,high,close`，每个代码至少包括目标日和上一实际交易日；不能把不同时点的复权价格混在一起。
- 不得将最新市值/PB填成历史日期。程序能检查日期列是否一致，不能确认用户提供的数据确实来自当时。
- 历史股票池需包括当时的可交易股票；仅用当前上市名单会产生幸存者偏差。

### 最新行情

`--live` 使用 AkShare 最新快照和当日日线，只允许当前日期。快照接口没有经过本程序验证的交易日期字段，因此获取日期不等于确认过的行情交易日。休市、盘中未完成日线、接口延迟均可能造成不可用或不一致；程序不将这种结果当作历史回测。日线没有目标日的股票会跳过并记入 JSON。

输出 CSV 与同名 JSON；JSON 记录日期、实际参数、入围数和数据异常。网络异常可能导致部分股票缺失，空结果不证明全市场没有信号。

列名归一化后不能重复；快照代码必须为6位数字，名称不能缺失，成交量必须为有限非负数。日线价格需为有限正数，且最高价不能低于开盘/收盘，提供最低价时也会检查低价关系。不满足这些条件属于数据错误，不会当作正常的“无缺口”结果；逐股日线错误会进入 JSON 的 `errors`。
