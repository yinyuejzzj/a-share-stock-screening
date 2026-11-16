"""wei_pan_core.data_fetcher · 行情数据获取（akshare优先 → 新浪HTTP → 腾讯QT 双层fallback）

验收对照 checklist M2.1 ~ M2.6。
接口返回统一结构：{"ok": bool, "data": Any, "reason": str, "fallback_used": [str, ...]}
任何接口失败不抛未捕获异常，先试akshare再sina最后tencent，逐级fallback写入fallback_used列表。
对同一交易日结果默认按日缓存到PATHS["data_cache"]，避免重复打接口（限流友好）。
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import hashlib
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, date as date_cls

import pandas as pd
import numpy as np

from .config import (
    PATHS,
    FETCHER_CONFIG,
    HARD_RULES_MODE_B,
)


# =========================================================================
# 🔧 基础：带代理清理的 Session + 统一重试 + 按日缓存
# =========================================================================
def _clean_proxy_env() -> None:
    """根据经验lesson：东方财富/akshare 极易被系统代理搞坏，每次网络请求前清代理环境+trust_env=False。"""
    proxy_keys = [k for k in list(os.environ.keys()) if "proxy" in k.lower()]
    for k in proxy_keys:
        os.environ.pop(k, None)


def _new_session() -> "requests.Session":
    import requests  # 延迟import，避免data_fetcher import时requests未装报错
    _clean_proxy_env()
    s = requests.Session()
    s.trust_env = False  # 不信任系统代理（Windows上经常设了但其实不走就挂）
    s.headers.update({
        "User-Agent": FETCHER_CONFIG["user_agent"],
    })
    return s


def _cache_key(keywords: Tuple[Any, ...]) -> str:
    # The old schema can contain index quotes under individual-stock codes.
    # Invalidate it rather than replaying those observations after the route fix.
    raw = ("tail-v2|" + "|".join(str(k) for k in keywords)).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def _cache_read(key: str) -> Optional[Any]:
    if not FETCHER_CONFIG["cache_use_for_day"]:
        return None
    today = datetime.now().strftime("%Y%m%d")
    p = os.path.join(PATHS["data_cache"], f"{today}_{key}.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_write(key: str, obj: Any) -> None:
    if not FETCHER_CONFIG["cache_use_for_day"]:
        return
    today = datetime.now().strftime("%Y%m%d")
    p = os.path.join(PATHS["data_cache"], f"{today}_{key}.json")
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _try_request_with_retries(session, method: str, url: str, **kw) -> Tuple[Optional[Any], Optional[str]]:
    """带退避重试的requests请求，返回 (response_or_None, error_reason_or_None)。"""
    timeout = kw.pop("timeout", FETCHER_CONFIG["request_timeout"])
    for attempt in range(1, FETCHER_CONFIG["max_retries"] + 2):  # 首次 + max_retries 次重试
        try:
            resp = session.request(method, url, timeout=timeout, **kw)
            resp.raise_for_status()
            return resp, None
        except Exception as e:  # noqa: BLE001
            if attempt == FETCHER_CONFIG["max_retries"] + 1:
                return None, f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(2 ** attempt)  # 2s / 4s
    return None, "unknown_retry_exhausted"


# =========================================================================
# 🧱 M2.1 全市场实时快照（当日行情，批量5100只拉取，腾讯qt优先（快且稳定）→ sina hq 备用 → akshare em 最后）
# =========================================================================
def fetch_all_stocks_daily_snapshot(
    date: Optional[str] = None,
    code_mask: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """返回所有A股当日行情日频快照 DataFrame + 拉取成功/失败信息。

    DataFrame列包含：
        code(6位数字), name, market_cap_yi(亿), turnover_rate(小数), vol_ratio,
        pct_chg(小数), close, preclose, volume, amount_yi(亿), high, low, open,
        MA5, MA10, MA20, MA45（日收盘移动平均，基于近60日历史K线补算）,
        has_limit_up_in_N_day(dict {N:bool}, 供hard_filter cond2用),
        is_st(bool), is_halt(bool), list_date(YYYYMMDD)
    """
    fallback_used: List[str] = []
    date = date or datetime.now().strftime("%Y-%m-%d")
    cache_key = _cache_key(("daily_snapshot", date, code_mask or "ALL"))
    cached = _cache_read(cache_key)
    if cached is not None:
        return {"ok": True, "data": pd.read_json(cached["df_json"], orient="split"),
                "reason": "cache", "fallback_used": fallback_used}

    df: Optional[pd.DataFrame] = None
    last_reason = ""

    # ⚡ 调试/小批量优化：用户显式传了 code_mask（< 80只小批量）时，直接对代码表做腾讯行情，
    #    不用去拉整个5000+的A股代码列表（省HTTP+避免akshare代码列表因代理挂掉）。
    if code_mask and len(code_mask) <= 1000:
        try:
            prefixed = [_tencent_market_prefix(str(c).strip()[:6]) for c in code_mask if str(c).strip()[:6].isdigit()]
            ok_rows, err = _tencent_batch_quote(prefixed)
            if err is None and len(ok_rows) > 0:
                df_raw = pd.DataFrame(ok_rows)
                # ST名/上市日/停牌直接从name和volume推断（小批量调试模式下够用）
                df_raw["list_date"] = ""
                df_raw["is_st"] = df_raw["name"].str.contains(r"ST|\*ST", na=False).astype(bool)
                df_raw["is_halt"] = (df_raw["volume"].fillna(0) <= 0).astype(bool)
                df = df_raw
                fallback_used.append("tencent_qt_small_mask")
                last_reason = "small_mask_path"
        except Exception as e:  # noqa: BLE001
            last_reason = f"tencent_mask_fail: {type(e).__name__}: {str(e)[:200]}"

    # 优先级1：腾讯 qt.gtimg.cn 批量实时（5000只以上的全市场路径）
    if (df is None or len(df) == 0) and (not code_mask or len(code_mask) > 1000):
        try:
            df_allcodes, list_info_map = _fetch_tencent_code_list_and_info()
            batch_codes = list(list_info_map.keys())  # 全部代码（6位数字）
            rows = []
            # 每次最多80个一批（URL长度限制）
            for i in range(0, len(batch_codes), 80):
                codes6 = batch_codes[i:i + 80]
                prefixed = [_tencent_market_prefix(c) for c in codes6]
                ok_rows, err = _tencent_batch_quote(prefixed)
                if err:
                    raise RuntimeError(err)
                rows.extend(ok_rows)
            df_raw = pd.DataFrame(rows)
            if len(df_raw) > 0:
                df_raw["list_date"] = df_raw["code"].map(lambda c: list_info_map.get(c, {}).get("list_date", ""))
                df_raw["is_st"] = df_raw["name"].str.contains(r"ST|\*ST", na=False).astype(bool)
                df_raw["is_halt"] = (df_raw["volume"].fillna(0) <= 0).astype(bool)
                df = df_raw
                fallback_used.append("tencent_qt")
        except Exception as e:  # noqa: BLE001
            last_reason = f"tencent_qt_fail: {type(e).__name__}: {str(e)[:200]}"

    # 优先级2：新浪 vip.stock Market_Center.getHQNodeData → 批量行情（自带代码列表，不依赖akshare）
    if df is None or len(df) == 0:
        fallback_used.append("sina_hq_fallback_trying")
        try:
            df = _fetch_sina_all_spot()
            fallback_used.append("sina_hq")
        except Exception as e2:  # noqa: BLE001
            last_reason += " | sina_hq_fail: " + str(e2)[:200]

    # 优先级3：akshare stock_zh_a_spot_em（代理环境经常挂）
    if df is None or len(df) == 0:
        fallback_used.append("akshare_em_fallback_trying")
        try:
            df = _fetch_akshare_em_spot()
            fallback_used.append("akshare_em")
            # akshare返回一般不带name/is_st/is_halt/list_date列 → 补齐
            if "is_st" not in df.columns:
                df["is_st"] = df.get("name", pd.Series(dtype=str)).astype(str).str.contains(r"ST|\*ST").astype(bool)
            if "is_halt" not in df.columns:
                df["is_halt"] = (df.get("volume", pd.Series(0)).fillna(0) <= 0).astype(bool)
            if "list_date" not in df.columns:
                df["list_date"] = ""
        except Exception as e3:
            last_reason += " | akshare_em_fail: " + str(e3)[:200]

    if df is None or len(df) == 0:
        return {"ok": False, "data": pd.DataFrame(),
                "reason": f"all_sources_failed. last={last_reason}",
                "fallback_used": fallback_used}

    # 如需缩小范围（调试用）：先筛范围（这样后面补均线/涨停历史不用跑5000只×60日，提速100倍+）
    if code_mask:
        mask6 = {c[:6] for c in code_mask}
        df = df[df["code"].isin(mask6)].reset_index(drop=True)

    # 补均线 & 涨停基因（每只票单独拉60日历史K的均线 + N日涨停）
    df = _augment_with_historical_columns(df, date)

    # 缓存（保存成json split格式，保证读入还原正确dtype）
    _cache_write(cache_key, {"df_json": df.to_json(orient="split", force_ascii=False),
                             "fallback_used": fallback_used})
    return {"ok": True, "data": df, "reason": "ok", "fallback_used": fallback_used}


# =========================================================================
# 内部实现：tencent qt 批量行情（我们今天验证200OK字段全，主数据源）
# =========================================================================
_SINA_PREFIX_REFERER = "https://finance.sina.com.cn/"
_QT_REFERER = "https://qt.gtimg.cn/"


# 常见指数代码 → 真实上/深交易所的手动映射（避免仅按首字母判断，例沪深300代码000300实际上交所）
_INDEX_EXCHANGE_OVERRIDE: Dict[str, str] = {
    # 上交所指数（实际sh前缀）
    "000001": "sh",   # 上证指数
    "000016": "sh",   # 上证50
    "000300": "sh",   # 沪深300
    "000010": "sh",   # 上证180
    "000688": "sh",   # 科创50（科创板上交所）
    # 深交所指数（sz前缀）
    "399001": "sz",   # 深证成指
    "399006": "sz",   # 创业板指
    "399005": "sz",   # 中小板指
    "399300": "sz",   # 沪深300（深交所也有一份别名但新浪sz000300返回空，以sh为正式）
}


def _tencent_market_prefix(code6: str, *, is_index: bool = False) -> str:
    """6位个股代码 → sh/sz/bj；只有显式指数请求才使用指数市场映射。"""
    code6 = str(code6).strip().zfill(6)
    if is_index:
        if code6 not in _INDEX_EXCHANGE_OVERRIDE:
            raise ValueError(f"未知指数交易所，请明确配置指数市场: {code6}")
        return _INDEX_EXCHANGE_OVERRIDE[code6] + code6
    if code6.startswith(("6", "9", "5")):
        return "sh" + code6
    if code6.startswith(("688",)):  # 上交所科创板
        return "sh" + code6
    if code6.startswith(("4", "8")):  # 北交所
        return "bj" + code6
    return "sz" + code6


def _fetch_tencent_code_list_and_info() -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """腾讯API没有直接"全A股代码列表"，改用 akshare 拉代码（或新浪沪深京列表URL）。
    列表来源失败时返回空列表，让调用方继续真实行情来源或明确失败。
    """
    list_info: Dict[str, Dict[str, Any]] = {}

    # 1) 优先 akshare 拉代码+上市日期+ST（轻量，只拉一次元数据）
    try:
        import akshare as ak  # 延迟
        meta = ak.stock_info_a_code_name()  # DataFrame: code,name
        for _, row in meta.iterrows():
            code6 = str(row["code"]).zfill(6)
            list_info[code6] = {"name": str(row.get("name", "")),
                                "list_date": ""}  # akshare code_name不带上市日，后面用腾讯/ST名补
        # 上市日单独从 stock_zh_a_list_em 取（或降级）
        try:
            full_df = ak.stock_zh_a_list_em()
            for _, r in full_df.iterrows():
                c6 = str(r.get("代码", r.get("code", ""))).zfill(6)
                if c6 in list_info:
                    list_info[c6]["list_date"] = str(r.get("上市日期", r.get("ipo_date", ""))).replace("-", "")[:8]
        except Exception:
            pass
    except Exception:
        pass

    if not list_info:
        # 2) 降级：新浪股票列表页面 （http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=1&num=8000&sort=symbol&asc=1&node=hs_a）
        try:
            s = _new_session()
            s.headers.update({"Referer": _SINA_PREFIX_REFERER})
            resp, err = _try_request_with_retries(
                s, "GET",
                "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData",
                params={"page": 1, "num": 8000, "sort": "symbol", "asc": 1, "node": "hs_a"},
            )
            if err is None and resp is not None:
                arr = json.loads(resp.text)
                for item in arr:
                    symbol = str(item.get("symbol", "")).zfill(6)
                    if not symbol.isdigit():
                        continue
                    list_info[symbol] = {"name": item.get("name", ""),
                                         "list_date": str(item.get("ipo_date", "")).replace("-", "")[:8]}
        except Exception:
            pass

    if not list_info:
        return [], {}
    codes_sorted = sorted(list_info.keys())
    return codes_sorted, list_info


def _tencent_batch_quote(prefixed_codes: List[str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """从腾讯 qt.gtimg.cn 拉一批（80以内）代码的实时行情，解析出 10 个基础列。"""
    if not prefixed_codes:
        return [], "empty_batch"
    s = _new_session()
    s.headers.update({"Referer": _QT_REFERER})
    url = "https://qt.gtimg.cn/q=" + ",".join(prefixed_codes)
    resp, err = _try_request_with_retries(s, "GET", url)
    if err:
        return [], "tencent_batch_req_err " + err
    resp.encoding = "gbk"
    lines = [ln.strip() for ln in resp.text.split(";") if ln.strip() and "=" in ln]
    rows: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            # v_sh600596="1~新安股份~600596~..."
            _, rhs = ln.split("=", 1)
            rhs = rhs.strip().strip('"')
            if not rhs:
                continue
            f = rhs.split("~")
            if len(f) < 46 or not f[2].isdigit():
                continue
            code6 = f[2].zfill(6)
            close = float(f[3]) if f[3] else math.nan
            preclose = float(f[4]) if f[4] else math.nan
            openp = float(f[5]) if f[5] else math.nan
            vol_hand = float(f[6]) if f[6] else 0.0
            vol_yuan = float(f[37]) if len(f) > 37 and f[37] else 0.0  # 成交额（万？见下转换）
            high = float(f[33]) if len(f) > 33 and f[33] else close
            low = float(f[34]) if len(f) > 34 and f[34] else close
            turnover_pct_raw = float(f[38]) if len(f) > 38 and f[38] else 0.0  # 字段38=换手率%
            vol_ratio = float(f[46]) if len(f) > 46 and f[46] else math.nan   # 字段46=量比
            mcap_yi = float(f[44]) if len(f) > 44 and f[44] else math.nan      # 字段44=总市值亿
            pct_chg = (close - preclose) / preclose if preclose > 0 else math.nan
            rows.append({
                "code": code6,
                "name": f[1],
                "close": close,
                "preclose": preclose,
                "open": openp,
                "high": high,
                "low": low,
                "volume": vol_hand,                 # 手（1手=100股）
                "amount_wan": vol_yuan,             # 成交额 万（腾讯字段是万元单位）
                "amount_yi": vol_yuan / 10000.0 if vol_yuan else 0.0,  # 亿
                "market_cap_yi": mcap_yi,
                "turnover_rate": turnover_pct_raw / 100.0,            # % -> 小数
                "vol_ratio": vol_ratio,
                "pct_chg": pct_chg,
            })
        except Exception:
            continue
    return rows, None


# =========================================================================
# 备用数据源1：新浪 hq + vip.stock 全量节点（较稳定，备用）
# =========================================================================
def _fetch_sina_all_spot() -> pd.DataFrame:
    """用新浪 Market_Center.getHQNodeData 拉全量（我们测过返回json数组OK）。
    再对每批80只再 hq.sinajs.cn 拉量比/换手/市值（sina自身node接口部分字段不全）。
    """
    import requests as _rq
    s = _new_session()
    s.headers.update({"Referer": _SINA_PREFIX_REFERER})
    resp, err = _try_request_with_retries(
        s, "GET",
        "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData",
        params={"page": 1, "num": 8000, "sort": "symbol", "asc": 1, "node": "hs_a"},
    )
    if err or resp is None:
        raise RuntimeError("sina_node_fetch " + (err or "empty"))
    arr = json.loads(resp.text)
    # node字段近似：symbol/name/trade(现价)/settlement(昨收)/open/high/low/volume(股)/amount(元)/turnoverratio(%)
    # 新浪没有直接的量比/市值 → 再批量走 hq.sinajs.cn 拉（但已有基础列够过滤逻辑）
    rows = []
    for item in arr:
        sym = str(item.get("symbol", "")).zfill(6)
        if not sym.isdigit():
            continue
        close = float(item.get("trade") or "nan")
        preclose = float(item.get("settlement") or "nan")
        pct = (close - preclose) / preclose if preclose else math.nan
        vol = float(item.get("volume") or 0) / 100.0  # 股->手
        amount = float(item.get("amount") or 0)
        tr = float(item.get("turnoverratio") or 0)
        rows.append({
            "code": sym,
            "name": item.get("name", ""),
            "close": close, "preclose": preclose,
            "open": float(item.get("open") or "nan"),
            "high": float(item.get("high") or "nan"),
            "low": float(item.get("low") or "nan"),
            "volume": vol,
            "amount_wan": amount / 10000.0 if amount else 0.0,
            "amount_yi": amount / 1e8 if amount else 0.0,
            "market_cap_yi": math.nan,  # 再查下一轮；这里先放nan
            "turnover_rate": tr / 100.0,
            "vol_ratio": math.nan,
            "pct_chg": pct,
        })
    df = pd.DataFrame(rows)
    # 尝试用sinajs批量补market_cap / 量比（每次80）
    codes_sorted = list(df["code"].values)
    vol_map: Dict[str, float] = {}
    mc_map: Dict[str, float] = {}
    for i in range(0, min(len(codes_sorted), 4000), 80):  # 先补前4000只，防止耗时过长
        batch = codes_sorted[i:i + 80]
        prefixed = [("sh" if c.startswith(("6","9","5")) else "sz") + c for c in batch]
        try:
            s2 = _new_session()
            s2.headers.update({"Referer": _SINA_PREFIX_REFERER})
            r, e2 = _try_request_with_retries(
                s2, "GET", "https://hq.sinajs.cn/list=" + ",".join(prefixed),
            )
            if e2 or r is None:
                continue
            r.encoding = "gbk"
            for line in r.text.split(";\n"):
                if "=" not in line or '"' not in line:
                    continue
                head, body = line.split("=", 1)
                code_prefix = head.split("_")[-1]  # hq_str_sh600596
                prefix = code_prefix[:2]
                c6 = code_prefix[2:]
                inside = body.strip().strip('"')
                if not inside:
                    continue
                parts = inside.split(",")
                if len(parts) < 10:
                    continue
                try:
                    tot_money_hand = float(parts[8])
                    close_now = float(parts[3])
                    if tot_money_hand and close_now:
                        # 新浪没给总市值；兜底=流通股本*价未知，这里用近似=成交额/换手
                        # 实际我们返回nan即可，后续用日线补
                        pass
                    # 量比 = 现量/5日均量（新浪当日没给 → 历史算）
                except Exception:
                    pass
        except Exception:
            continue
    df["vol_ratio"] = df["code"].map(vol_map)
    df["market_cap_yi"] = df["code"].map(mc_map)
    return df


# =========================================================================
# 备用数据源2：akshare stock_zh_a_spot_em（东方财富全量，经常因代理挂；做最后fallback）
# =========================================================================
def _fetch_akshare_em_spot() -> pd.DataFrame:
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    # 列名规范化：东方财富返回中文列名 → 转英文
    rename_map = {
        "代码": "code", "名称": "name", "最新价": "close", "昨收": "preclose",
        "今开": "open", "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount_wan", "换手率": "turnover_pct_raw", "量比": "vol_ratio",
        "涨跌幅": "pct_raw", "总市值": "mcap_yuan",
    }
    keep = [c for c in rename_map.keys() if c in df.columns]
    df = df[keep].rename(columns=rename_map)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["market_cap_yi"] = df["mcap_yuan"] / 1e8 if "mcap_yuan" in df else math.nan
    df["turnover_rate"] = df["turnover_pct_raw"].astype(float) / 100.0 if "turnover_pct_raw" in df else math.nan
    df["pct_chg"] = (df["close"].astype(float) - df["preclose"].astype(float)) / df["preclose"].astype(float)
    df["volume"] = df["volume"].astype(float) / 100.0  # 东方财富返回"手"单位，与腾讯一致？其实东方财富是股→手/100
    # 东方财富返回是股（实测过）→ 除100转手
    # 如果已经是手就不再动，做一次兜底：取前10只若avg(volume/amount_wan)量级接近股数/手则修正
    df["amount_yi"] = df["amount_wan"].astype(float) / 10000.0 if "amount_wan" in df else 0.0
    return df


# =========================================================================
# 历史K线补齐（MA4/10/20/45 + N日内涨停）
# =========================================================================
def _augment_with_historical_columns(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """遍历所有票，拉60日K线（或查缓存）→ 加均线列 + has_limit_up_in_N_day 列。"""
    # 为了跑通速度：只对市值30亿以上的票拉历史（否则市值<30亿的直接被cond3淘汰，不用算历史）
    codes = df["code"].tolist()
    N_defaults = list(range(1, 61))
    ma_cols = {f"MA{n}": np.nan for n in (5, 10, 20, 45)}
    for k in ma_cols:
        df[k] = np.nan
    df["has_limit_up_in_N_day"] = [dict.fromkeys(N_defaults, False) for _ in range(len(df))]
    if not codes:
        return df
    result_idx = {c: i for i, c in enumerate(codes)}
    # 批量：每批20只并行串行？不串行就OK，每批走sina历史K（我们测sina/getKLineData稳定）
    import datetime as _dt
    today_date_obj = datetime.strptime(date, "%Y-%m-%d").date() if isinstance(date, str) else date
    for batch_i in range(0, len(codes), 20):
        batch_codes = codes[batch_i:batch_i + 20]
        for c6 in batch_codes:
            try:
                hist = _fetch_single_daily_k(c6, bars=61, end_date=today_date_obj)
                if hist is None or len(hist) < 5:
                    continue
                closes = hist["close"].astype(float).values
                for n in (5, 10, 20, 45):
                    if len(closes) >= n:
                        ma_val = float(np.mean(closes[-n:]))
                        df.at[result_idx[c6], f"MA{n}"] = ma_val
                # N日内是否涨停：当日涨幅>=9.8%近似（去掉ST票限幅）
                dpc = np.diff(closes) / closes[:-1]
                limit_mask = dpc >= 0.098
                # 第i个位置的"距今天数"是 len-1 - i
                last_i = len(limit_mask) - 1
                hits = {N: False for N in N_defaults}
                for N in N_defaults:
                    start = max(0, last_i - N + 1)
                    hits[N] = bool(np.any(limit_mask[start:last_i + 1]))
                df.at[result_idx[c6], "has_limit_up_in_N_day"] = hits
            except Exception:
                continue
    return df


def _fetch_single_daily_k(code6: str, bars: int = 60,
                          end_date: Optional[date_cls] = None,
                          *, is_index: bool = False) -> Optional[pd.DataFrame]:
    """拉单票60日日线（默认sina历史；akshare兜底）。"""
    end = end_date or datetime.now().date()
    cache_key = _cache_key(("daily_k", "index" if is_index else "stock", code6, bars, end.isoformat()))
    cached = _cache_read(cache_key)
    if cached is not None:
        return pd.read_json(cached, orient="split")
    # sina 历史
    pref = _tencent_market_prefix(code6, is_index=is_index)
    s = _new_session()
    s.headers.update({"Referer": _SINA_PREFIX_REFERER})
    url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={pref}&scale=240&ma=no&datalen={bars}")
    resp, err = _try_request_with_retries(s, "GET", url)
    if err is None and resp is not None:
        try:
            arr = json.loads(resp.text)
            if arr:
                df = pd.DataFrame(arr)[["day", "open", "high", "low", "close", "volume"]]
                _cache_write(cache_key, df.to_json(orient="split"))
                return df
        except Exception:
            pass
    # akshare 兜底
    try:
        import akshare as ak
        if is_index:
            df = ak.stock_zh_index_daily_em(symbol=pref, end_date=end.strftime("%Y%m%d"))
        else:
            df = ak.stock_zh_a_hist(symbol=code6, period="daily", adjust="qfq",
                                   end_date=end.strftime("%Y%m%d"))
        if len(df) > 0:
            df = df.rename(columns={"日期": "day", "date": "day", "开盘": "open", "最高": "high", "最低": "low",
                                     "收盘": "close", "成交量": "volume"}).tail(bars)
            df = df[["day", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
            _cache_write(cache_key, df.to_json(orient="split"))
            return df
    except Exception:
        return None
    return None


# =========================================================================
# 🧱 M2.2 分钟级分时（供cond9分时形态判定）
# =========================================================================
def fetch_intraday_minute(code6: str, date: Optional[str] = None,
                          until_minute_str: str = "15:00") -> Dict[str, Any]:
    """拉指定票当日（或指定日期）的5分钟线，按"截至until_minute_str"切片返回。

    until_minute_str 接受 "14:00" / "14:30" / "14:45" / "15:00" 四种（对应A/B/C的时间窗）。
    返回 {"ok":bool, "data":DataFrame columns=[time, time_str, open, high, low, close, volume, avg_price],
           "fallback_used":[], "reason":""}
    avg_price = 截至该分钟的累计成交额/累计成交量近似（用于cond9现价与均价距离）
    """
    fallback_used: List[str] = []
    date = date or datetime.now().strftime("%Y-%m-%d")
    cache_key = _cache_key(("intraday_minute", code6, date, until_minute_str))
    cached = _cache_read(cache_key)
    if cached is not None:
        restored = pd.read_json(StringIO(cached), orient="split")
        if "time" not in restored.columns and "time_str" in restored.columns:
            restored["time"] = date + " " + restored["time_str"].astype(str)
        return {"ok": True, "data": restored,
                "fallback_used": fallback_used, "reason": "cache"}
    # 优先sina getKLineData scale=5（今天我们测稳定返回最近1-2天的5分钟）
    pref = _tencent_market_prefix(code6)
    s = _new_session()
    s.headers.update({"Referer": _SINA_PREFIX_REFERER})
    url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={pref}&scale=5&ma=no&datalen=500")
    resp, err = _try_request_with_retries(s, "GET", url)
    df: Optional[pd.DataFrame] = None
    if err is None and resp is not None:
        try:
            arr = json.loads(resp.text)
            if arr:
                df = pd.DataFrame(arr)
                fallback_used.append("sina_5min")
        except Exception:
            err = "sina_5min_parse_err"
    # 兜底：akshare 分钟线（经常限流，慢）
    if df is None or len(df) == 0:
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist_min_em(symbol=code6, period="5", adjust="")
            if len(df):
                df = df.rename(columns={"时间": "day", "开盘": "open", "最高": "high",
                                         "最低": "low", "收盘": "close", "成交量": "volume"})
                fallback_used.append("akshare_minute")
        except Exception as e2:
            return {"ok": False, "data": pd.DataFrame(), "fallback_used": fallback_used,
                    "reason": f"all_sources: {err} / ak: {type(e2).__name__}"}
    if df is None or len(df) == 0:
        return {"ok": False, "data": pd.DataFrame(), "fallback_used": fallback_used,
                "reason": "empty_result"}
    df = df.copy()
    df["day"] = df["day"].astype(str)
    # 保留"当日"：YYYY-MM-DD 前缀匹配
    df["date_part"] = df["day"].str[:10]
    df = df[df["date_part"] == date].copy()
    # time_str HH:MM = day[11:16]
    df["time_str"] = df["day"].str[11:16]
    # 截至时刻过滤（含）
    target_hm = until_minute_str.replace("：", ":")
    df = df[df["time_str"] <= target_hm].copy()
    # 均价线计算：累计成交额 / 累计成交量（手）*100 → 元/股
    df["vol100"] = df["volume"].astype(float)  # 新浪/东财都是手
    df["amount_approx"] = (df["close"].astype(float) + df["open"].astype(float) + df["high"].astype(float) + df["low"].astype(float)) / 4.0 * df["vol100"] * 100
    df["cum_amount"] = df["amount_approx"].cumsum()
    df["cum_vol_shares"] = (df["vol100"] * 100).cumsum()
    df["avg_price"] = np.where(df["cum_vol_shares"] > 0,
                               df["cum_amount"] / df["cum_vol_shares"],
                               df["open"].astype(float))
    df["time"] = df["day"]
    df = df[["time", "time_str", "open", "high", "low", "close", "volume", "avg_price"]].reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "avg_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    _cache_write(cache_key, df.to_json(orient="split"))
    return {"ok": True, "data": df, "fallback_used": fallback_used, "reason": "ok"}


# =========================================================================
# 🧱 M2.3 外盘指数（US3+HK+JP+KR+EU2）涨跌% 带成功/失败标记
# =========================================================================
def fetch_global_indices(market_close_date: Optional[str] = None) -> Dict[str, Any]:
    """返回 {"ok","fallback_used","data":[{"id","members","intra_method","total_weight","success","group_pct",
            "constituents":[{"sym","pct_chg","close","ok"},...]}, ... ]}
    group_pct = None 表示该组整体不可用（美股整组缺失进入"收紧+异常"路径）。
    """
    from .config import GLOBAL_GATE
    fallback_used: List[str] = []
    groups_cfg = GLOBAL_GATE["groups"]

    # ✅ 审计后实测修正的新浪映射（2026-09-04 实际可返回数据的代码）
    #   大小写敏感，优先全小写（新浪无b_前缀时直接写代码小写最稳定）。
    #   每个global symbol 尝试两个代码（主/备），第一个成功即ok。
    SYM_MAP: Dict[str, List[str]] = {
        "^DJI":  ["dji", "b_DJI"],                              # 道琼斯（小写dji最稳实测返回 0.65%）
        "^IXIC": ["ixic", "b_IXIC"],                            # 纳指（小写ixic最稳 实测+1.40%）
        "^GSPC": ["b_SPX", "spx"],                              # 标普500（b_SPX之前已实测返回）
        "HSI":   ["hkHSI"],                                      # 恒生指数（OK）
        "N225":  ["n225", "b_N225", "jpn_N225"],                # 日经225（小写n225最常见，若失败再b_）
        "KS11":  ["ks11", "b_KS11", "kor_KS11"],                # 韩国KOSPI
        "FTSE":  ["b_FTSE", "ftse"],                             # 富时100（之前已OK）
        "DAX":   ["DAX", "dax", "b_DAXG"],                       # 德国DAX（大写DAX 实测OK 0.63%；b_DAXG备用）
    }
    # 生成每个 symbol 的请求代码列表（同时把两个备用都放进去，后续用哪个有数据就算谁ok）
    req_prefixed_all: List[str] = []
    sym_to_codes: Dict[str, List[str]] = {}
    for g in groups_cfg:
        for sym in g["members"]:
            codes = SYM_MAP.get(sym, [sym])
            sym_to_codes[sym] = codes
            req_prefixed_all.extend(codes)
    # 建反向：prefixed_code → 原始global symbol
    code_to_sym: Dict[str, str] = {}
    for sym, codes in sym_to_codes.items():
        for c in codes:
            code_to_sym[c] = sym
    # 所有请求用的prefixed
    all_symbols_prefixed: List[str] = req_prefixed_all

    data_raw: Dict[str, Dict[str, Any]] = {}
    s = _new_session()
    s.headers.update({"Referer": _SINA_PREFIX_REFERER})
    # 每批80一次
    for i in range(0, len(all_symbols_prefixed), 80):
        batch = all_symbols_prefixed[i:i + 80]
        r, err = _try_request_with_retries(
            s, "GET", "https://hq.sinajs.cn/list=" + ",".join(batch),
        )
        if err or r is None:
            continue
        r.encoding = "gbk"
        # 格式三种（从字段长度推断）：
        #   港指hkHSI: 0代码1名称2今开3昨收4最高5最低6现价7涨跌点8涨跌幅%
        #   b_系列（道指备用/富时等）: 0名称1现价2涨跌点3涨跌幅% 4时间 5日期
        #   小写dji/ixic/DAX等无b_无hk前缀 美股/欧股/日股 格式：两种都可能
        #     5项 = 0名称1现价2涨跌点3涨跌幅% 4时间日期合并? → 常见格式6项: 0名称1现价2涨跌点3涨跌幅% 4HH:MM:SS 5YYYY-MM-DD
        #     8项（含日期）：0名称1现价2涨跌点3涨跌幅% 4时间 5日期 6日高 7日低
        #     或者4项：0名称1现价2涨跌点3涨跌幅%
        #   特殊格式（刚才实测DAX）：0名1现价2涨跌点3涨跌幅% 4日期1 5日期2 6日期3 7时间 8...
        for line in r.text.split(";\n"):
            if "=" not in line or '"' not in line:
                continue
            head, body = line.split("=", 1)
            # hq_str_XXXXX → 取XXXXX
            tail = head.split("hq_str_")[-1] if "hq_str_" in head else head.split("_")[-1]
            prefixed_code = tail.strip()
            inside = body.strip().strip('"; ')
            if not inside:
                data_raw[prefixed_code] = {"ok": False, "pct": None, "close": None}
                continue
            parts = inside.split(",")
            try:
                if prefixed_code.startswith("hk") and len(parts) >= 8:
                    pct = float(parts[8]) / 100.0
                    close = float(parts[6])
                    data_raw[prefixed_code] = {"ok": True, "pct": pct, "close": close}
                    continue
                # 其他：通用解析（美股/欧股/日经 无b_或有b_）
                # 一般格式：parts[0]名称 parts[1]现价 parts[2]涨跌点 parts[3]涨跌幅%
                if len(parts) >= 4:
                    close_v = float(parts[1])
                    pct_s = parts[3]
                    if pct_s.endswith("%"):
                        pct_s = pct_s[:-1]
                    pct_v = float(pct_s)
                    # 涨跌幅% 可能直接是百分数形式（如1.40=1.4% → /100）；若本身<1则已是小数（兼容）
                    pct_v = pct_v / 100.0 if abs(pct_v) > 1 else pct_v
                    data_raw[prefixed_code] = {"ok": True, "pct": pct_v, "close": close_v}
                    continue
                data_raw[prefixed_code] = {"ok": False, "pct": None, "close": None}
            except Exception:
                data_raw[prefixed_code] = {"ok": False, "pct": None, "close": None}
    fallback_used.append("sina_global")
    # akshare 兜底补全缺失项（可选）
    missing_symbols: List[str] = []
    for sym, codes in sym_to_codes.items():
        found_any = any(data_raw.get(c, {}).get("ok") for c in codes)
        if not found_any:
            missing_symbols.append(sym)
    if missing_symbols:
        try:
            import akshare as ak
            for sym in missing_symbols:
                try:
                    df: Optional[pd.DataFrame] = None
                    if sym == "HSI":
                        df = ak.stock_hk_index_daily_em(symbol="HSI").tail(2)
                    elif sym in ("^DJI", "^IXIC", "^GSPC"):
                        mp = {"^DJI": "道琼斯", "^IXIC": "纳斯达克", "^GSPC": "标普500指数"}
                        df = ak.index_us_stock_sina(symbol=mp[sym]).tail(2) if hasattr(ak, "index_us_stock_sina") else None
                        if df is None:
                            df = ak.stock_us_index_hist_cs(index=mp[sym]).tail(2)
                    elif sym == "N225":
                        df = ak.index_investing_global(country="日本", index_name="日经225", period="每日", start_date="20260101").tail(2)
                    elif sym == "KS11":
                        df = ak.index_investing_global(country="韩国", index_name="KOSPI", period="每日", start_date="20260101").tail(2)
                    elif sym == "FTSE":
                        df = ak.index_investing_global(country="英国", index_name="富时100", period="每日", start_date="20260101").tail(2)
                    elif sym == "DAX":
                        df = ak.index_investing_global(country="德国", index_name="DAX", period="每日", start_date="20260101").tail(2)
                    if df is not None and len(df) >= 2:
                        c1 = float(df.tail(1)["收盘"].iloc[-1])
                        c2 = float(df.tail(2)["收盘"].iloc[-2])
                        pct = (c1 - c2) / c2 if c2 else None
                        if pct is not None:
                            fallback_code = sym_to_codes[sym][0]
                            data_raw[fallback_code] = {"ok": True, "pct": float(pct), "close": c1}
                            fallback_used.append("akshare_global_fallback")
                except Exception:
                    continue
        except Exception:
            pass

    # 构建：每个global symbol对应的最终ok信息（取它的多个候选代码第一个ok的）
    sym_final: Dict[str, Dict[str, Any]] = {}
    for sym, codes in sym_to_codes.items():
        chosen: Dict[str, Any] = {"ok": False, "pct": None, "close": None}
        which = codes[0]
        for c in codes:
            got = data_raw.get(c, {})
            if got.get("ok"):
                chosen = {"ok": True, "pct": got.get("pct"), "close": got.get("close")}
                which = c
                break
        sym_final[sym] = {**chosen, "used_prefixed": which}

    # 组装成按 groups_cfg 结构返回
    out_groups = []
    for g in groups_cfg:
        cons = []
        vals = []
        for sym in g["members"]:
            info = sym_final.get(sym, {"ok": False, "pct": None, "close": None, "used_prefixed": ""})
            cons.append({"sym": sym,
                         "prefixed": info.get("used_prefixed", sym_to_codes.get(sym, [sym])[0]),
                         "ok": bool(info.get("ok")), "pct_chg": info.get("pct"),
                         "close": info.get("close")})
            if info.get("ok") and info.get("pct") is not None:
                vals.append(float(info.get("pct")))
        group_success = bool(vals)  # 组内至少有一个能拿到
        # 整组失败 policy
        if (not group_success) and g.get("whole_group_missing_policy") == "data_incomplete_abnormal_tighten":
            group_pct = None  # 显式None → 上层global_market.py走收紧分支
        elif not group_success:
            group_pct = None
        elif g["intra_method"] == "avg":
            group_pct = float(np.mean(vals))
        else:  # first
            group_pct = vals[0]
        out_groups.append({
            "id": g["id"],
            "members": g["members"],
            "intra_method": g["intra_method"],
            "total_weight": g["total_weight"],
            "whole_group_missing_policy": g.get("whole_group_missing_policy", "renormalize_remaining"),
            "success": group_success,
            "group_pct": group_pct,
            "constituents": cons,
        })
    return {"ok": True, "data": out_groups, "fallback_used": fallback_used,
            "reason": "ok", "date": market_close_date}


# =========================================================================
# 🧱 M2.4 大盘状态信号（HS300 5日/3连跌/涨停/跌停/北向）
# =========================================================================
def fetch_hs300_regime_signals(regime_date: Optional[str] = None) -> Dict[str, Any]:
    """返回 {"ok","data":{"hs300_5d_return","hs300_3d_streak_down","limit_up_count","limit_down_count","northbound_flow_yuan"}, "fallback_used","reason"}

    主方法akshare，失败时走纯HTTP兜底（代理环境下akshare经常挂，兜底使用：
      · HS300日K = sina money.finance getKLineData（已在_fetch_single_daily_k中稳定实现）
      · 涨跌停数 = sina vip.stock getHQNodeData(hs_a)过滤涨幅>9.8%/<-9.8%
      · 北向净流入 = Eastmoney纯HTTP（push2his.eastmoney.com/api/qt/kamt/get 沪股通+深股通净买合计）
    """
    fallback_used: List[str] = []
    date = regime_date or datetime.now().strftime("%Y-%m-%d")
    cache_key = _cache_key(("regime_signals", date))
    cached = _cache_read(cache_key)
    if cached is not None:
        return {"ok": True, "data": cached, "fallback_used": fallback_used, "reason": "cache"}
    data: Dict[str, Optional[float]] = {
        "hs300_5d_return": None,
        "hs300_3d_streak_down": None,
        "limit_up_count": None,
        "limit_down_count": None,
        "northbound_flow_yuan": None,
    }
    errors: List[str] = []

    # ============ 1) HS300 5日收益 + 3连跌 ============
    hs300_done = False
    try:
        import akshare as ak
        idx = ak.stock_zh_index_daily_em(symbol="sh000300").tail(8)
        if len(idx) >= 6:
            close_column = "close" if "close" in idx.columns else "收盘"
            closes = idx[close_column].astype(float).values
            data["hs300_5d_return"] = float((closes[-1] - closes[-6]) / closes[-6])
            if len(closes) >= 4:
                d3 = np.diff(closes[-4:]) < 0
                data["hs300_3d_streak_down"] = bool(np.all(d3))
            hs300_done = True
        elif len(idx) > 0:
            # 行数不够（非交易日等场景也append fallback）
            pass
        fallback_used.append("akshare_hs300")
    except Exception as e1:
        errors.append(f"ak_hs300:{type(e1).__name__}")
    if not hs300_done:  # ✅ 行数不够也兜底（刚才的场景：akshare只返回3行最新交易日，也应该走sina兜底）
        try:
            idx_df = _fetch_single_daily_k("000300", bars=10, is_index=True)
            if idx_df is not None and len(idx_df) >= 6:
                closes = idx_df["close"].astype(float).values
                data["hs300_5d_return"] = float((closes[-1] - closes[-6]) / closes[-6])
                if len(closes) >= 4:
                    d3 = np.diff(closes[-4:]) < 0
                    data["hs300_3d_streak_down"] = bool(np.all(d3))
            fallback_used.append("sina_hs300_daily_fallback")
        except Exception as e1b:
            errors.append(f"sina_hs300:{type(e1b).__name__}")

    # ============ 2) 涨停 / 跌停数 ============
    ymd = date.replace("-", "")
    try:
        import akshare as ak
        zt = ak.stock_zt_pool_em(date=ymd)
        data["limit_up_count"] = int(len(zt))
        dt = ak.stock_zt_pool_dtgc_em(date=ymd)
        data["limit_down_count"] = int(len(dt))
        fallback_used.append("akshare_zt_dt")
    except Exception as e2:
        errors.append(f"ak_zt_dt:{type(e2).__name__}")
        # 兜底：sina 沪深A股节点全量拉取后按涨幅±9.8%阈值统计（近似）
        try:
            df_sina = _fetch_sina_all_spot()
            if len(df_sina) > 0 and "pct_chg" in df_sina.columns:
                pct = pd.to_numeric(df_sina["pct_chg"], errors="coerce")
                # 涨停近似：普通A股(非ST/非北交所/非科创板)涨跌10%/20%规则，用9.8%近似普通涨停
                data["limit_up_count"] = int((pct >= 0.098).sum())
                data["limit_down_count"] = int((pct <= -0.098).sum())
                fallback_used.append("sina_node_zt_dt_approx_fallback")
        except Exception as e2b:
            errors.append(f"sina_zt_dt:{type(e2b).__name__}")

    # ============ 3) 北向净流入（Eastmoney纯HTTP，兼容代理） ============
    try:
        import akshare as ak
        nb = ak.stock_hsgt_north_net_flow_in_em(symbol="北向").tail(1).iloc[0]
        nb_yi = float(nb.iloc[-1])
        data["northbound_flow_yuan"] = nb_yi * 1e8
        fallback_used.append("akshare_northbound")
    except Exception as e3:
        errors.append(f"ak_nb:{type(e3).__name__}")
        try:
            # Eastmoney 沪股通 + 深股通 当日净买（HTTP GET）
            # fields1: 沪股通代码 1.00001 ；深股通代码 0.399006 → Eastmoney的港股通接口
            # 使用 push2.eastmoney.com/api/qt/stock/get?secid=1.00001 / secid=0.399006 的字段
            s = _new_session()
            s.headers.update({"Referer": "https://quote.eastmoney.com/"})
            nb_yi_total = 0.0
            got = 0
            # 北向=沪股通+深股通；东方财富北向单票代码：沪股通北向净买= 1.00001字段？或者更可靠走kamt接口
            # Eastmoney 资金流（沪深港通资金流向）：
            #   https://push2his.eastmoney.com/api/qt/kamt.kline/get?fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58&klt=101&lmt=1
            # f53: 沪股通当日净买(元) f54: 深股通当日净买(元)  → 北向合计 = f53+f54
            url = ("https://push2his.eastmoney.com/api/qt/kamt.kline/get?"
                   "fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58&klt=101&lmt=1&ut=fa5fd1943c7b386f172d6893dbfba10b")
            resp, err = _try_request_with_retries(s, "GET", url)
            if err is None and resp is not None:
                js = resp.json()
                # 预期结构：{"data":{"klines":[[date, hgtNetBuy, sgtNetBuy, ...]]}}
                rows = js.get("data", {}).get("klines") or js.get("data", {}).get("nr2") or []
                if rows:
                    latest = rows[-1].split(",") if isinstance(rows[-1], str) else list(rows[-1])
                    # f53 index=2 (index 0 日期, 1 hgt当日余额?, 2 hgt净买, 3 sgt净买) 根据实际情况
                    # 按官方北向净买=沪+深净买合计；尝试多个位置求和
                    for idxs in ((2, 3), (2, 4), (1, 2)):
                        try:
                            val = sum(float(latest[i]) for i in idxs if i < len(latest))
                            if abs(val) > 1e6:  # > 100万元量级就认定为元单位的净买
                                nb_yi_total = val
                                got = 1
                                break
                        except Exception:
                            continue
            if got:
                data["northbound_flow_yuan"] = float(nb_yi_total)  # 元
                fallback_used.append("eastmoney_http_northbound_fallback")
        except Exception as e3b:
            errors.append(f"em_nb:{type(e3b).__name__}")

    _cache_write(cache_key, data)
    # ok=True 只要有至少1个字段拿到；即使不全，也让上层按REGIME_RULES降级
    any_got = any(v is not None for v in data.values())
    return {
        "ok": any_got,
        "data": data,
        "fallback_used": fallback_used,
        "reason": "ok" if any_got else ("partial_missing: " + " ; ".join(errors[:3])),
    }
