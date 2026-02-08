#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权大单分析器 V5.5（最终整合版 + CSV 时间对齐）
- 全局多腿聚合（不限 size）
- 大单导出 CSV（含中文 condition 描述）
- ✅ CSV 中的 `time` 字段与 JSON 完全一致（如 "10:15:22"）
- ✅ Excel 打开不自动转换时间（通过前缀 ' 强制文本）
- Massive 原始数据缓存复用
- Finnhub 期权链缓存支持
"""

import os
import json
import csv
import time
import requests
from datetime import datetime, date, timedelta, time as dtime
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from typing import List, Dict, Any
import holidays
# ======================================================================================
# 🔑 API 密钥（从环境变量读取）
# ======================================================================================

MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

if not MASSIVE_API_KEY:
    raise RuntimeError("❌ MASSIVE_API_KEY 未设置！")

# 初始化 Massive 客户端（必须）
from massive import RESTClient
massive_client = RESTClient(MASSIVE_API_KEY)

# Finnhub 客户端（可选）
finnhub_client = None
if FINNHUB_API_KEY:
    import finnhub
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
else:
    print("⚠️ FINNHUB_API_KEY 未设置，将跳过期权链缓存构建（需已有缓存）")

# ======================================================================================
# ⚙️ 全局配置
# ======================================================================================

UNDERLYINGS = ["NVDA", "SPY", "MU","AVGO","GOOG"]

MIN_DELTA_OI = 300
MIN_VOLUME = 500
MIN_NOMINAL_FOR_BIG = 1_000_000

RTH_START = dtime(9, 30)
RTH_END = dtime(16, 0)
VALID_EXCHANGES = {300, 301, 302, 303, 304, 307, 308, 309, 312, 313, 315, 316, 318, 319, 320, 322, 323, 325}

EXCHANGE_NAME_MAP = {
    300: "NYSE American", 301: "BOX", 302: "CBOE", 303: "MIAX Emerald",
    304: "EDGX", 307: "GEMX", 308: "ISE", 309: "MRX", 312: "MIAX",
    313: "NYSE Arca", 315: "MIAX Pearl", 316: "Nasdaq NOM", 318: "MIAX Sapphire",
    319: "Nasdaq BX", 320: "MEMX", 322: "C2", 323: "PHLX", 325: "BZX",
}

CACHE_DIR = "finnhub_cache"
RAW_CACHE_DIR = "massive_raw_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RAW_CACHE_DIR, exist_ok=True)

# ======================================================================================
# 🇨🇳 条件码中文映射（用于 CSV 导出）
# ======================================================================================

CONDITION_DESC_ZH_MAP = {
    "Canceled": "已取消",
    "Late and Out Of Sequence": "延迟且乱序",
    "Last and Canceled": "最后一笔且已取消",
    "Late": "延迟成交",
    "Opening Trade and Canceled": "开盘成交且已取消",
    "Opening Trade, Late, and Out Of Sequence": "开盘成交、延迟且乱序",
    "Only Trade and Canceled": "唯一成交且已取消",
    "Opening Trade and Late": "开盘成交且延迟",
    "Automatic Execution": "自动执行",
    "Reopening Trade": "重新开盘成交",
    "Intermarket Sweep Order (ISO)": "跨市场扫单（ISO）",
    "Single Leg Auction Non ISO": "单腿非ISO拍卖",
    "Single Leg Auction ISO": "单腿ISO拍卖",
    "Single Leg Cross Non ISO": "单腿非ISO交叉交易",
    "Single Leg Cross ISO": "单腿ISO交叉交易",
    "Single Leg Floor Trade": "单腿场内交易",
    "Multi Leg auto-electronic trade": "多腿自动电子交易",
    "Multi Leg Auction": "多腿拍卖",
    "Multi Leg Cross": "多腿交叉交易",
    "Multi Leg floor trade": "多腿场内交易",
    "Multi Leg vs Single Leg (auto)": "多腿对单腿（自动）",
    "Stock-Options Auction": "股票-期权拍卖",
    "Multi Leg Auction vs Single Leg": "多腿拍卖对单腿",
    "Multi Leg Floor vs Single Leg": "多腿场内对单腿",
    "Stock-Options auto-electronic": "股票-期权自动电子交易",
    "Stock-Options Cross": "股票-期权交叉交易",
    "Stock-Options floor trade": "股票-期权场内交易",
    "Stock-Options vs Single Leg (auto)": "股票-期权对单腿（自动）",
    "Stock-Options Auction vs Single Leg": "股票-期权拍卖对单腿",
    "Stock-Options Floor vs Single Leg": "股票-期权场内对单腿",
    "Proprietary Multi Leg Floor Trade (≥3 legs)": "自营多腿场内交易（≥3腿）",
    "Multilateral Compression Trade": "多边压缩交易",
    "Extended Hours Trade": "盘后交易"
}

# ======================================================================================
# 📅 智能交易日判定
# ======================================================================================

def get_last_trading_day() -> str:
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)
    today_et = now_et.date()

    # 美国股市节假日
    us_holidays = holidays.US(state='NY', years=range(2020, 2030))

    # 如果今天是交易日且已收盘（>16:00），则今天可算
    if today_et.weekday() < 5 and today_et not in us_holidays:
        if now_et.time() > datetime.strptime("16:00", "%H:%M").time():
            return today_et.strftime("%Y-%m-%d")

    # 否则从昨天开始找
    for i in range(1, 8):
        d = today_et - timedelta(days=i)
        if d.weekday() < 5 and d not in us_holidays:
            return d.strftime("%Y-%m-%d")
    
    raise RuntimeError("找不到最近交易日")

TRADE_DATE = get_last_trading_day()
print(f"📅 自动选择交易日: {TRADE_DATE}")

# ======================================================================================
# 🧩 工具函数
# ======================================================================================

def clean_symbol(symbol: str) -> str:
    return symbol.replace(".", "").replace("-", "")

def finnhub_to_massive_ticker(symbol: str, expiry: str, call_put: str, strike: float) -> str:
    clean_sym = clean_symbol(symbol)
    exp_clean = expiry.replace("-", "")[2:]  # 2025-12-19 → 251219
    cp = "C" if call_put.lower() == "call" else "P"
    strike_int = int(round(strike * 1000))
    strike_str = f"{strike_int:08d}"
    return f"O:{clean_sym}{exp_clean}{cp}{strike_str}"

def parse_option_ticker(ticker: str):
    if not ticker.startswith("O:"):
        raise ValueError(f"无效期权 ticker: {ticker}")
    s = ticker[2:]
    if len(s) < 15:
        raise ValueError(f"ticker 长度不足: {ticker}")
    strike_part = s[-8:]
    cp_char = s[-9]
    date_part = s[-15:-9]
    symbol_part = s[:-15]
    try:
        year = int(date_part[:2])
        month = int(date_part[2:4])
        day = int(date_part[4:6])
        full_year = 2000 + year
        expiration = f"{full_year:04d}-{month:02d}-{day:02d}"
        call_put = "Call" if cp_char.upper() == "C" else "Put"
        strike = int(strike_part) / 1000.0
        return {"symbol": symbol_part, "expiration": expiration, "call_put": call_put, "strike": strike}
    except Exception as e:
        raise ValueError(f"解析 ticker 失败 {ticker}: {e}")

# ======================================================================================
# 📋 OPRA Condition Rules
# ======================================================================================

CONDITION_RULES = {
    201: (False, False, False, "Canceled"),
    202: (True,  False, False, "Late and Out Of Sequence"),
    203: (False, False, False, "Last and Canceled"),
    204: (True,  False, False, "Late"),
    205: (False, False, False, "Opening Trade and Canceled"),
    206: (True,  False, False, "Opening Trade, Late, and Out Of Sequence"),
    207: (False, False, False, "Only Trade and Canceled"),
    208: (True,  True,  False, "Opening Trade and Late"),
    209: (True,  True,  False, "Automatic Execution"),
    210: (True,  True,  False, "Reopening Trade"),
    219: (True,  True,  False, "Intermarket Sweep Order (ISO)"),
    227: (True,  True,  False, "Single Leg Auction Non ISO"),
    228: (True,  True,  False, "Single Leg Auction ISO"),
    229: (True,  True,  False, "Single Leg Cross Non ISO"),
    230: (True,  True,  False, "Single Leg Cross ISO"),
    231: (True,  True,  False, "Single Leg Floor Trade"),
    232: (True,  True,  True,  "Multi Leg auto-electronic trade"),
    233: (True,  True,  True,  "Multi Leg Auction"),
    234: (True,  True,  True,  "Multi Leg Cross"),
    235: (True,  True,  True,  "Multi Leg floor trade"),
    236: (True,  True,  True,  "Multi Leg vs Single Leg (auto)"),
    237: (True,  True,  True,  "Stock-Options Auction"),
    238: (True,  True,  True,  "Multi Leg Auction vs Single Leg"),
    239: (True,  True,  True,  "Multi Leg Floor vs Single Leg"),
    240: (True,  True,  True,  "Stock-Options auto-electronic"),
    241: (True,  True,  True,  "Stock-Options Cross"),
    242: (True,  True,  True,  "Stock-Options floor trade"),  # TLFT（已忽略）
    243: (True,  True,  True,  "Stock-Options vs Single Leg (auto)"),
    244: (True,  True,  True,  "Stock-Options Auction vs Single Leg"),
    245: (True,  True,  True,  "Stock-Options Floor vs Single Leg"),
    246: (True,  False, True,  "Proprietary Multi Leg Floor Trade (≥3 legs)"),
    247: (True,  False, True,  "Multilateral Compression Trade"),
    248: (True,  False, False, "Extended Hours Trade"),
}

def analyze_trade_conditions(conditions):
    if not conditions:
        return True, True, False, "No condition codes"
    is_valid = ohlc_eligible = is_combo = False
    descs = []
    for cond in conditions:
        rule = CONDITION_RULES.get(cond)
        if rule is None:
            continue
        valid, ohlc, combo, desc = rule
        if valid:
            is_valid = True
        if ohlc:
            ohlc_eligible = True
        if combo:
            is_combo = True
        descs.append(f"{cond}:{desc}")
    unique_descs = list(dict.fromkeys(descs))
    return is_valid, ohlc_eligible, is_combo, "; ".join(unique_descs)

# ======================================================================================
# 🧮 OHLC 聚合
# ======================================================================================

def aggregate_ohlc_from_trades(trades):
    from collections import defaultdict
    bars = defaultdict(list)
    for t in trades:
        ts_sec = t.sip_timestamp / 1e9
        dt = datetime.fromtimestamp(ts_sec)
        minute_key = dt.replace(second=0, microsecond=0)
        bars[minute_key].append(t)
    ohlc_list = []
    for minute in sorted(bars):
        trade_batch = bars[minute]
        prices = [t.price for t in trade_batch]
        sizes = [t.size for t in trade_batch]
        ohlc_list.append({
            "datetime": minute.isoformat(),
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(sizes),
            "vwap": round(sum(p * s for p, s in zip(prices, sizes)) / sum(sizes), 4)
        })
    return ohlc_list

# ======================================================================================
# 📦 Finnhub 缓存构建（仅期权链）
# ======================================================================================

def build_and_save_cache_for_symbol(symbol: str, trade_date: str):
    print(f"📥 [{symbol}] 开始构建 {trade_date} 期权链缓存...")
    if not FINNHUB_API_KEY:
        print(f"⚠️ [{symbol}] 无 Finnhub Key，跳过缓存构建")
        return

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/stock/option-chain",
            params={"symbol": symbol.upper(), "token": FINNHUB_API_KEY},
            timeout=20
        )
        if resp.status_code != 200:
            print(f"❌ HTTP {resp.status_code}: {resp.text}")
            return

        data = resp.json()
        all_expiries = data.get("data", [])
        print(f"📊 {symbol} 共返回 {len(all_expiries)} 个到期日")

        cache = {}
        total_contracts = 0

        for expiry_block in all_expiries:
            if not isinstance(expiry_block, dict):
                continue

            expiry_date = expiry_block.get("expirationDate")
            if not expiry_date:
                continue

            options = expiry_block.get("options", {})
            calls = options.get("CALL", [])
            puts = options.get("PUT", [])

            for opt in calls:
                oi = opt.get("openInterest", 0)
                vol = opt.get("volume", 0)
                strike = opt.get("strike", 0)
                if oi > 0 or vol > 0:
                    ticker = finnhub_to_massive_ticker(symbol, expiry_date, "CALL", strike)
                    cache[ticker] = {
                        "open_interest": int(oi),
                        "volume": int(vol),
                        "strike": float(strike),
                        "call_put": "call",
                        "expiry": expiry_date
                    }
                    total_contracts += 1

            for opt in puts:
                oi = opt.get("openInterest", 0)
                vol = opt.get("volume", 0)
                strike = opt.get("strike", 0)
                if oi > 0 or vol > 0:
                    ticker = finnhub_to_massive_ticker(symbol, expiry_date, "PUT", strike)
                    cache[ticker] = {
                        "open_interest": int(oi),
                        "volume": int(vol),
                        "strike": float(strike),
                        "call_put": "put",
                        "expiry": expiry_date
                    }
                    total_contracts += 1

    except Exception as e:
        print(f"⚠️ [{symbol}] 缓存构建失败: {e}")
        traceback.print_exc()
        return

    sym_dir = os.path.join(CACHE_DIR, symbol.upper())
    os.makedirs(sym_dir, exist_ok=True)
    path = os.path.join(sym_dir, f"{symbol}_{trade_date}.json")
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"✅ [{symbol}] 期权缓存已保存 ({total_contracts} 合约)")

# ======================================================================================
# 🚀 Massive 大单分析（带缓存复用）
# ======================================================================================

def sanitize_filename(name: str) -> str:
    return name.replace(":", "_").replace("/", "_").replace("\\", "_").replace("*", "_")

def load_or_fetch_massive_trades(ticker: str, trade_date: str, underlying_symbol: str):
    """加载缓存或从 Massive 拉取并缓存"""
    safe_ticker = sanitize_filename(ticker)
    cache_path = os.path.join(RAW_CACHE_DIR, underlying_symbol.upper(), f"{safe_ticker}_{trade_date}.json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
                return cached_data
        except Exception as e:
            print(f"⚠️ 缓存损坏，重新拉取 {ticker}: {e}")

    # 否则拉取
    print(f"📡 拉取新数据: {ticker}")
    et = pytz.timezone("US/Eastern")
    target_date = datetime.fromisoformat(trade_date).date()
    start_et = et.localize(datetime.combine(target_date, dtime.min))
    end_et = start_et + timedelta(days=1)
    start_utc_iso = start_et.astimezone(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc_iso = end_et.astimezone(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_valid_trades = []
    ticker_info = parse_option_ticker(ticker)

    try:
        for raw_trade in massive_client.list_trades(
            ticker=ticker,
            timestamp_gte=start_utc_iso,
            timestamp_lt=end_utc_iso,
            sort="timestamp",
            order="asc",
            limit=50000
        ):
            conditions = getattr(raw_trade, 'conditions', []) or []
            is_valid, ohlc_eligible, is_combo, cond_desc = analyze_trade_conditions(conditions)

            ts_sec = raw_trade.sip_timestamp / 1e9
            dt_utc = datetime.fromtimestamp(ts_sec, tz=pytz.UTC)
            dt_et = dt_utc.astimezone(et)

            if not (RTH_START <= dt_et.time() <= RTH_END and dt_et.weekday() < 5):
                continue
            if raw_trade.exchange not in VALID_EXCHANGES:
                continue
            if not raw_trade.price or not raw_trade.size or raw_trade.price <= 0 or raw_trade.size <= 0:
                continue

            nominal = raw_trade.size * raw_trade.price * 100

            if is_valid:
                all_valid_trades.append({
                    "ticker": ticker,
                    "time": dt_et.strftime("%H:%M:%S"),  # ← JSON 中就是这个字符串
                    "size": raw_trade.size,
                    "price": raw_trade.price,
                    "nominal": round(nominal, 2),
                    "exchange": EXCHANGE_NAME_MAP.get(raw_trade.exchange, f"Unknown({raw_trade.exchange})"),
                    "is_combo_leg": is_combo,
                    "condition_codes": conditions,
                    "condition_desc": cond_desc,
                    **ticker_info
                })

    except Exception as e:
        raise RuntimeError(f"Massive 获取成交失败: {e}")

    # 保存缓存
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(all_valid_trades, f, indent=2)

    return all_valid_trades

def process_single_ticker_massive(ticker: str, trade_date: str, min_nominal: float, underlying_symbol: str,
                                  cache_today: Dict[str, Any], cache_yesterday: Dict[str, Any]):
    big_trades = []
    clean_trades_for_ohlc = []

    trades_data = load_or_fetch_massive_trades(ticker, trade_date, underlying_symbol)
    ticker_info = parse_option_ticker(ticker)

    et = pytz.timezone("US/Eastern")
    target_date = datetime.fromisoformat(trade_date).date()

    # 获取 OI 数据（Finnhub 的 openInterest 是 T-1 收盘数据）
    oi_today = cache_today.get(ticker, {}).get("open_interest", 0)
    oi_yesterday = cache_yesterday.get(ticker, {}).get("open_interest", 0)
    delta_oi = oi_today - oi_yesterday

    for record in trades_data:
        dt_et = et.localize(datetime.strptime(f"{trade_date} {record['time']}", "%Y-%m-%d %H:%M:%S"))
        ts_utc = dt_et.astimezone(pytz.UTC).timestamp()

        class FakeTrade:
            def __init__(self, price, size, sip_ts):
                self.price = price
                self.size = size
                self.sip_timestamp = int(sip_ts * 1e9)

        fake_raw = FakeTrade(record["price"], record["size"], ts_utc)

        if record["nominal"] >= min_nominal:
            big_record = {
                "ticker": ticker,
                "time": record["time"],  # ← 保持原样！
                "size": record["size"],
                "price": record["price"],
                "nominal": record["nominal"],
                "exchange": record["exchange"],
                "is_combo_leg": record["is_combo_leg"],
                "condition_codes": record["condition_codes"],
                "condition_desc": record["condition_desc"],
                "oi_today": oi_today,
                "oi_yesterday": oi_yesterday,
                "delta_oi": delta_oi,
                **ticker_info
            }

            if record["is_combo_leg"] and 242 not in record["condition_codes"]:
                big_record["combo_note"] = "Candidate for post-aggregation multi-leg grouping"

            big_trades.append(big_record)

        # OHLC eligible?
        if any(c in [209, 210, 219, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 243, 244, 245]
               for c in record["condition_codes"]):
            clean_trades_for_ohlc.append(fake_raw)

    ohlc = aggregate_ohlc_from_trades(clean_trades_for_ohlc)
    return big_trades, ohlc


def process_symbol(symbol: str, trade_date: str):
    print(f"\n{'='*60}")
    print(f"🚀 开始处理标的: {symbol}")
    print(f"{'='*60}")

    output_dir = symbol.upper()
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{symbol.lower()}_{trade_date}"

    cache_yesterday = load_cache(symbol, (datetime.fromisoformat(trade_date) - timedelta(days=1)).strftime("%Y-%m-%d"))
    cache_today = load_cache(symbol, trade_date)

    if not cache_today:
        print(f"⚠️ [{symbol}] 今日缓存缺失！请先运行缓存构建")
        return

    candidate_tickers = set()
    for ticker in cache_today:
        oi_today = cache_today[ticker].get("open_interest", 0)
        vol_today = cache_today[ticker].get("volume", 0)
        oi_yest = cache_yesterday.get(ticker, {}).get("open_interest", 0)
        delta_oi = oi_today - oi_yest
        if delta_oi >= MIN_DELTA_OI or vol_today >= MIN_VOLUME:
            candidate_tickers.add(ticker)

    print(f"🎯 [{symbol}] 候选池: {len(candidate_tickers)} 个合约")

    all_big = []
    all_ohlc = []
    failed = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                process_single_ticker_massive,
                t,
                trade_date,
                MIN_NOMINAL_FOR_BIG,
                symbol,
                cache_today,
                cache_yesterday
            ): t
            for t in candidate_tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                big, ohlc = future.result()
                all_big.extend(big)
                all_ohlc.extend(ohlc)
            except Exception as e:
                failed.append(ticker)
                print(f"❌ [{symbol}] {ticker} 失败: {e}")

    # 排序
    all_big.sort(key=lambda x: x["nominal"], reverse=True)
    all_ohlc.sort(key=lambda x: x["datetime"])

    # 多腿策略聚合
    print(f"🔍 [{symbol}] 开始多腿策略聚合（不限 size）...")
    strategies = group_multi_leg_strategies(all_big)

    # === 保存大单 JSON ===
    big_path = os.path.join(output_dir, f"{prefix}_big_trades.json")
    with open(big_path, "w") as f:
        json.dump(all_big, f, indent=2)
    print(f"✅ [{symbol}] 大单已保存 ({len(all_big)} 笔)")

    # === 新增：导出大单 CSV（含中文条件描述 + OI 变化）===
    if all_big:
        csv_path = os.path.join(output_dir, f"{prefix}_big_trades.csv")
        fieldnames = [
            "ticker", "symbol", "expiration", "call_put", "strike",
            "time", "size", "price", "nominal", "exchange",
            "is_combo_leg", "condition_codes", "condition_desc_zh",
            "oi_today", "oi_yesterday", "delta_oi"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in all_big:
                row = {k: record.get(k, "") for k in fieldnames}
                row["condition_desc_zh"] = translate_condition_to_zh(record.get("condition_desc", ""))
                row["condition_codes"] = ",".join(map(str, record.get("condition_codes", [])))
                # ✅ 强制 time 为文本（Excel 兼容）
                row["time"] = "'" + record["time"]
                writer.writerow(row)
        print(f"✅ [{symbol}] 大单 CSV 已导出（含中文条件 + OI 变化）: {csv_path}")

    # === 保存策略汇总 ===
    if strategies:
        strategy_path = os.path.join(output_dir, f"{prefix}_multi_leg_strategies.json")
        with open(strategy_path, "w") as f:
            json.dump(strategies, f, indent=2)
        print(f"✅ [{symbol}] 多腿策略已保存 ({len(strategies)} 个策略)")

    # === 保存 OHLC ===
    if all_ohlc:
        ohlc_json = os.path.join(output_dir, f"{prefix}_ohlc.json")
        ohlc_csv = os.path.join(output_dir, f"{prefix}_ohlc.csv")
        with open(ohlc_json, "w") as f:
            json.dump(all_ohlc, f, indent=2)
        with open(ohlc_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["datetime", "open", "high", "low", "close", "volume", "vwap"])
            writer.writeheader()
            writer.writerows(all_ohlc)
        print(f"✅ [{symbol}] OHLC 已保存 ({len(all_ohlc)} 根K线)")

    if failed:
        fail_path = os.path.join(output_dir, f"{prefix}_failed.json")
        with open(fail_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"❗ [{symbol}] {len(failed)} 个合约失败")

def get_cache_path(symbol: str, trade_date: str) -> str:
    sym_dir = os.path.join(CACHE_DIR, symbol.upper())
    return os.path.join(sym_dir, f"{symbol}_{trade_date}.json")

def load_cache(symbol: str, trade_date: str) -> dict:
    path = get_cache_path(symbol, trade_date)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ [{symbol}] 缓存加载失败: {e}")
    return {}

# ======================================================================================
# 🔗 全局多腿策略聚合（核心新增）
# ======================================================================================

def group_multi_leg_strategies(big_trades: List[Dict]) -> List[Dict]:
    """
    聚合多腿策略：基于 (symbol, expiration, time, exchange)
    不限制 size，支持任意比例
    """
    from collections import defaultdict

    combo_candidates = [
        t for t in big_trades
        if any(232 <= c <= 246 for c in t.get("condition_codes", []))
    ]

    groups = defaultdict(list)
    for t in combo_candidates:
        key = (
            t["symbol"],
            t["expiration"],
            t["time"],
            t["exchange"]
        )
        groups[key].append(t)

    strategies = []
    for key, legs in groups.items():
        if len(legs) < 2:
            continue

        for leg in legs:
            leg["is_combo_resolved"] = True
            leg["combo_group_key"] = str(key)

        call_count = sum(1 for l in legs if l["call_put"] == "Call")
        put_count = sum(1 for l in legs if l["call_put"] == "Put")
        strikes = sorted(set(l["strike"] for l in legs))

        if call_count > 0 and put_count > 0:
            if len(strikes) == 2 and abs(strikes[0] - strikes[1]) < 1:
                strategy_type = "Straddle"
            else:
                strategy_type = "Strangle / Risk Reversal"
        elif call_count >= 2:
            if len(strikes) == 2:
                strategy_type = "Vertical Call Spread"
            elif len(strikes) == 3:
                strategy_type = "Call Butterfly"
            else:
                strategy_type = f"Call Multi-leg ({len(legs)} legs)"
        elif put_count >= 2:
            if len(strikes) == 2:
                strategy_type = "Vertical Put Spread"
            elif len(strikes) == 3:
                strategy_type = "Put Butterfly"
            else:
                strategy_type = f"Put Multi-leg ({len(legs)} legs)"
        else:
            strategy_type = f"Unknown Multi-leg ({len(legs)} legs)"

        strategy = {
            "strategy_type": strategy_type,
            "group_key": str(key),
            "leg_count": len(legs),
            "total_nominal": sum(l["nominal"] for l in legs),
            "common_time": legs[0]["time"],
            "common_exchange": legs[0]["exchange"],
            "legs": legs
        }
        strategies.append(strategy)

    return strategies

# ======================================================================================
# 🇨🇳 翻译 condition_desc 为中文
# ======================================================================================

def translate_condition_to_zh(condition_desc: str) -> str:
    """将英文 condition_desc 转为中文"""
    if not condition_desc:
        return ""
    parts = []
    for item in condition_desc.split("; "):
        if ":" in item:
            code, desc_en = item.split(":", 1)
            desc_zh = CONDITION_DESC_ZH_MAP.get(desc_en.strip(), desc_en)
            parts.append(f"{code}:{desc_zh}")
        else:
            parts.append(item)
    return "; ".join(parts)

# ======================================================================================
# 🚀 主处理流程
# ======================================================================================

def process_symbol(symbol: str, trade_date: str):
    print(f"\n{'='*60}")
    print(f"🚀 开始处理标的: {symbol}")
    print(f"{'='*60}")

    output_dir = symbol.upper()
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{symbol.lower()}_{trade_date}"

    cache_yesterday = load_cache(symbol, (datetime.fromisoformat(trade_date) - timedelta(days=1)).strftime("%Y-%m-%d"))
    cache_today = load_cache(symbol, trade_date)

    if not cache_today:
        print(f"⚠️ [{symbol}] 今日缓存缺失！请先运行缓存构建")
        return

    candidate_tickers = set()
    for ticker in cache_today:
        oi_today = cache_today[ticker].get("open_interest", 0)
        vol_today = cache_today[ticker].get("volume", 0)
        oi_yest = cache_yesterday.get(ticker, {}).get("open_interest", 0)
        delta_oi = oi_today - oi_yest
        if delta_oi >= MIN_DELTA_OI or vol_today >= MIN_VOLUME:
            candidate_tickers.add(ticker)

    print(f"🎯 [{symbol}] 候选池: {len(candidate_tickers)} 个合约")

    all_big = []
    all_ohlc = []
    failed = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                process_single_ticker_massive,
                t,
                trade_date,
                MIN_NOMINAL_FOR_BIG,
                symbol,
                cache_today,
                cache_yesterday
            ): t
            for t in candidate_tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                big, ohlc = future.result()
                all_big.extend(big)
                all_ohlc.extend(ohlc)
            except Exception as e:
                failed.append(ticker)
                print(f"❌ [{symbol}] {ticker} 失败: {e}")

    # 排序
    all_big.sort(key=lambda x: x["nominal"], reverse=True)
    all_ohlc.sort(key=lambda x: x["datetime"])

    # 多腿策略聚合
    print(f"🔍 [{symbol}] 开始多腿策略聚合（不限 size）...")
    strategies = group_multi_leg_strategies(all_big)

    # === 保存大单 JSON ===
    big_path = os.path.join(output_dir, f"{prefix}_big_trades.json")
    with open(big_path, "w") as f:
        json.dump(all_big, f, indent=2)
    print(f"✅ [{symbol}] 大单已保存 ({len(all_big)} 笔)")

    # === 新增：导出大单 CSV（含中文条件描述 + OI 变化）===
    if all_big:
        csv_path = os.path.join(output_dir, f"{prefix}_big_trades.csv")
        fieldnames = [
            "ticker", "symbol", "expiration", "call_put", "strike",
            "time", "size", "price", "nominal", "exchange",
            "is_combo_leg", "condition_codes", "condition_desc_zh",
            "oi_today", "oi_yesterday", "delta_oi"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in all_big:
                row = {k: record.get(k, "") for k in fieldnames}
                row["condition_desc_zh"] = translate_condition_to_zh(record.get("condition_desc", ""))
                row["condition_codes"] = ",".join(map(str, record.get("condition_codes", [])))
                # ✅ 强制 time 为文本（Excel 兼容）
                row["time"] = "'" + record["time"]
                writer.writerow(row)
        print(f"✅ [{symbol}] 大单 CSV 已导出（含中文条件 + OI 变化）: {csv_path}")

    # === 保存策略汇总 ===
    if strategies:
        strategy_path = os.path.join(output_dir, f"{prefix}_multi_leg_strategies.json")
        with open(strategy_path, "w") as f:
            json.dump(strategies, f, indent=2)
        print(f"✅ [{symbol}] 多腿策略已保存 ({len(strategies)} 个策略)")

    # === 保存 OHLC ===
    if all_ohlc:
        ohlc_json = os.path.join(output_dir, f"{prefix}_ohlc.json")
        ohlc_csv = os.path.join(output_dir, f"{prefix}_ohlc.csv")
        with open(ohlc_json, "w") as f:
            json.dump(all_ohlc, f, indent=2)
        with open(ohlc_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["datetime", "open", "high", "low", "close", "volume", "vwap"])
            writer.writeheader()
            writer.writerows(all_ohlc)
        print(f"✅ [{symbol}] OHLC 已保存 ({len(all_ohlc)} 根K线)")

    if failed:
        fail_path = os.path.join(output_dir, f"{prefix}_failed.json")
        with open(fail_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"❗ [{symbol}] {len(failed)} 个合约失败")

# ======================================================================================
# 🎮 主控逻辑
# ======================================================================================

def main():
    print("=" * 60)
    print("🚀 期权大单分析系统（V5.5：CSV 时间与 JSON 严格对齐）")
    print(f"📅 分析日期: {TRADE_DATE}")
    print("💡 CSV 中 time 字段 = JSON 中 time 字符串，Excel 显示正确")
    print("=" * 60)

    for symbol in UNDERLYINGS:
        cache_path = get_cache_path(symbol, TRADE_DATE)
        cache_exists = os.path.exists(cache_path) and os.path.getsize(cache_path) > 0

        if cache_exists:
            print(f"✅ [{symbol}] Finnhub 缓存已存在")
        else:
            if FINNHUB_API_KEY:
                print(f"📥 [{symbol}] 构建 Finnhub 期权链缓存...")
                build_and_save_cache_for_symbol(symbol, TRADE_DATE)
            else:
                print(f"❌ [{symbol}] 无缓存且无 Finnhub Key，跳过")
                continue

        try:
            process_symbol(symbol, TRADE_DATE)
        except Exception as e:
            print(f"💥 [{symbol}] 分析失败: {e}")
            traceback.print_exc()

    print("\n🎉 全部完成！")

if __name__ == "__main__":
    main()