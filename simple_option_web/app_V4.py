# ====================== 完整可运行 app.py ======================
import os
import sys
import json
import traceback
import pandas as pd
import requests
from datetime import datetime, date, timedelta, timezone
import numpy as np
from flask import Flask, request, render_template

# ---------- 基础配置 ----------
os.environ["FLASK_ENV"] = "development"
API_KEY = os.environ.get("FINNHUB_API_KEY")
if not API_KEY:
    print("❌ FATAL: 请设置环境变量 FINNHUB_API_KEY", file=sys.stderr)
    sys.exit(1)

HIGH_GAMMA_TICKERS = {"NVDA", "SPY", "MU", "AVGO", "GOOG","SNDK","ORCL","GLD"}
app = Flask(__name__)
RAW_DIR = "raw_data"
REPORT_DIR = "reports"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

beijing_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))

# ---------- 工具函数 ----------
def get_week_range(d: date):
    monday = d - timedelta(days=d.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday

def classify_expiration(exp_str: str) -> str:
    try:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
    except:
        return "无效"
    today = beijing_time.date()
    this_mon, this_sun = get_week_range(today)
    if this_mon <= exp_date <= this_sun:
        return "本周"
    next_mon = this_sun + timedelta(days=1)
    next_sun = next_mon + timedelta(days=6)
    if next_mon <= exp_date <= next_sun:
        return "下周"
    if exp_date.year == today.year and exp_date.month == today.month:
        return "本月"
    return "远期"

def get_stock_price(symbol: str):
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol.upper(), "token": API_KEY},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            price = data.get("c") or data.get("l") or data.get("pc")
            return float(price) if price else None
    except Exception as e:
        print(f"   ⚠️ 获取股价失败: {e}")
    return None

# ---------- IV / Gamma墙 / MaxPain ----------
def iv_surface(df: pd.DataFrame, spot: float) -> float:
    atm_mask = df["moneyness"] == "ATM 区域"
    if atm_mask.sum() < 3:
        atm_mask = df["moneyness"].isin(["ATM 区域", "Call ITM / Put OTM",
                                         "Call OTM / Put ITM"])
    df_atm = df[atm_mask].copy()
    if df_atm.empty:
        return np.nan

    def _dte(exp):
        try:
            return (datetime.strptime(exp, "%Y-%m-%d").date() -
                    beijing_time.date()).days
        except Exception:
            return 7
    df_atm["dte"] = df_atm["expiration"].apply(_dte).clip(lower=1)

    df_atm["sum_premium"] = (df_atm["call_premium"] + df_atm["put_premium"]).replace(0, np.nan)
    avg_premium = df_atm["sum_premium"].mean()
    avg_dte   = df_atm["dte"].mean()
    T = avg_dte / 365.25
    iv = avg_premium / (0.4 * np.sqrt(T) * spot) if spot and T else np.nan
    return max(0.05, min(0.9, iv))


def gamma_wall(df: pd.DataFrame, spot: float):
    df["call_gamma"] = df["call_OI"] * df["call_premium"]
    df["put_gamma"]  = df["put_OI"]  * df["put_premium"]
    call_wall = df.loc[df["call_gamma"].idxmax(), "strike"] if df["call_gamma"].max() > 0 else None
    put_wall  = df.loc[df["put_gamma"].idxmax(), "strike"]  if df["put_gamma"].max() > 0  else None
    return dict(call_wall=call_wall, put_wall=put_wall,
                call_gamma=df["call_gamma"].max(),
                put_gamma =df["put_gamma"].max())


def max_pain(df: pd.DataFrame):
    """
    先按 strike 聚合 OI，再算经典 Max-Pain
    """
    # 1. 每个 strike 的合计 OI
    oi_sum = df.groupby("strike")[["call_OI", "put_OI"]].sum()
    strikes = oi_sum.index.values
    oi_call = oi_sum["call_OI"]
    oi_put  = oi_sum["put_OI"]

    # 2. 计算每个 strike 的总 pain
    pain = []
    for k in strikes:
        loss = ((strikes[strikes > k] - k) * oi_call[strikes > k]).sum() + \
               ((k - strikes[strikes < k]) * oi_put[strikes < k]).sum()
        pain.append(loss)

    # 3. 返回 pain 最小的价格
    idx = np.argmin(pain)
    return float(strikes[idx]), pain[idx]


def oi_weighted_strike(df: pd.DataFrame):
    total_oi = (df["call_OI"] + df["put_OI"]).sum()
    if total_oi == 0:
        return np.nan
    return ((df["strike"] * (df["call_OI"] + df["put_OI"])).sum() / total_oi)


# ---------- 追加写入波动区间 ----------
# ---------- 仅用于「本周到期」的波动区间计算 ----------
# ---------- 仅用于「本周到期」的全量优化计算 ----------
def append_vol_range_to_existing_report(today_str: str, symbols: list, df_full: pd.DataFrame):
    summary_path = os.path.join(REPORT_DIR, f"📊 主力信号_{today_str}.xlsx")
    if not os.path.exists(summary_path):
        return

    this_mon, this_sun = get_week_range(beijing_time.date())
    weekly_mask = df_full["expiration"].apply(
        lambda e: this_mon <= datetime.strptime(e, "%Y-%m-%d").date() <= this_sun
    )
    df_week = df_full[weekly_mask].copy()
    if df_week.empty:
        return

    results = []
    for sym in symbols:
        df = df_week[df_week["symbol"] == sym].copy()
        spot = get_stock_price(sym) or df["strike"].mean()

        # 1. 用 raw gamma 算墙
        df["call_gamma_dollar"] = df["call_gamma"] * 100.0 * spot * 0.01
        df["put_gamma_dollar"]  = df["put_gamma"]  * 100.0 * spot * 0.01

        call_gd = df.groupby("strike")["call_gamma_dollar"].sum()
        put_gd  = df.groupby("strike")["put_gamma_dollar"].sum()
        call_wall = call_gd.idxmax() if not call_gd.empty else None
        put_wall  = put_gd.idxmax()  if not put_gd.empty else None

        print(df_full.head(100))
        # ===== 调试：看同一 strike 到底混了几期 =====
        debug_df = df[df["strike"].isin([174.0, 175.0])][["expiration", "strike", "call_iv"]]
        print(f"\n>>> {sym}  strike=174/175  原始多期混合：")
        print(debug_df.sort_values(["strike", "expiration"]))
        print(f"该 strike 平均 IV={df[df['strike']==174.0]['call_iv'].mean():.2f}%\n")
        # ===== 调试：打印 ATM 区间原始数据 =====
        atm_df = df[(df["strike"].between(spot * 0.95, spot * 1.05))]
        print(f"\n>>> {sym} 现货={spot}  ATM 区间 0.95-1.05:")
        print(atm_df[["strike", "call_iv", "call_gamma"]].drop_duplicates().sort_values("strike"))
        print(f"ATM 档数={len(atm_df)}, 平均 IV={atm_df['call_iv'].mean():.2f}%\n")
        # ===== 结束调试 =====
        
        # 2. 用 raw IV 算 ATM 波动率 & 区间
        atm_mask = (df["strike"].between(spot * 0.8, spot * 1.2))
        iv_raw  = df[atm_mask]["call_iv"].mean()
        iv_raw  = min(iv_raw, 200.0)   # 超过 200 % 一律当 200 %
        iv_avg  = iv_raw / 100.0
        if pd.isna(iv_avg) or iv_avg <= 0:
            iv_avg = 0.25        # 保底
        T_day  = 1 / 365.25
        T_week = 7 / 365.25
        std_day  = iv_avg * np.sqrt(T_day)  * spot
        std_week = iv_avg * np.sqrt(T_week) * spot

        # 3. Max-Pain & OI 加权（只取合理价内/价外）
        reasonable = df[df["strike"].between(spot * 0.5, spot * 1.5)]
        mp, _ = max_pain(reasonable)
        oiw   = oi_weighted_strike(reasonable)

        results.append({
            "symbol": sym,
            "spot": round(spot, 2),
            "iv_pct": round(iv_avg * 100, 1),
            "1d_low":  round(spot - std_day, 2),
            "1d_high": round(spot + std_day, 2),
            "1w_low":  round(spot - std_week, 2),
            "1w_high": round(spot + std_week, 2),
            "call_wall": call_wall,
            "put_wall": put_wall,
            "max_pain": round(mp, 2),
            "oi_weighted": round(oiw, 2),
            "weekly_call_key": call_wall,
            "weekly_put_key": put_wall
        })

    if not results:
        return

    df_vol = pd.DataFrame(results)
    with pd.ExcelWriter(summary_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df_vol.to_excel(writer, sheet_name="📈 波动区间", index=False)
    print(f"   ✅ 已追加【本周-raw-gamma-IV】波动区间到 {os.path.basename(summary_path)}")
def calculate_net_bias_index(call_net, put_net):
    total_abs = abs(call_net) + abs(put_net)
    if total_abs == 0:
        return 0.0
    net_bias = (call_net - put_net) / total_abs
    return max(-1.0, min(1.0, round(net_bias, 2)))

def fetch_and_save_raw_chain(symbol, today_str):
    raw_path_json = os.path.join(RAW_DIR, symbol, f"{today_str}.json")
    
    if os.path.exists(raw_path_json):
        print(f"📂 {symbol} 今日原始数据已存在，跳过 API 请求")
        try:
            with open(raw_path_json, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception as e:
            print(f"   ⚠️ 读取 JSON 失败: {e}，将重新请求")
            raw_data = None
    else:
        print(f"📡 正在请求 {symbol} 原始期权链...")
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/stock/option-chain",
                params={"symbol": symbol.upper(), "token": API_KEY},
                timeout=20
            )
            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}"
            raw_data = resp.json()
            os.makedirs(os.path.dirname(raw_path_json), exist_ok=True)
            with open(raw_path_json, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)
            print(f"   💾 已保存原始 JSON: {raw_path_json}")
        except Exception as e:
            print(f"   ❌ 请求失败: {e}")
            traceback.print_exc()
            return None, str(e)

    all_opts = []
    for exp in raw_data.get("data", []):
        exp_date = exp.get("expirationDate")
        if not exp_date:
            continue

        opts = exp.get("options", {})
        call_list = opts.get("CALL", [])
        put_list = opts.get("PUT", [])

        for c in call_list:
            premium = c.get("lastPrice")
            bid = c.get("bid", 0)
            ask = c.get("ask", 0)
            vol = c.get("volume", 0)

            if premium is not None and premium > 0:
                if premium >= ask and ask > 0:
                    aggressor_vol = vol
                elif premium <= bid and bid > 0:
                    aggressor_vol = -vol
                else:
                    aggressor_vol = 0
            else:
                if bid > 0 and ask > 0:
                    premium = (bid + ask) / 2
                else:
                    premium = 0.0
                aggressor_vol = 0

            try:
                c.update({
                    "type": "CALL",
                    "expiration": exp_date,
                    "premium": float(premium),
                    "aggressor_vol": int(aggressor_vol),
                    "gamma": float(c.get("gamma", 0)),          # ← 新增
                    "impliedVolatility": float(c.get("impliedVolatility", 0))  # ← 可选
                })
                all_opts.append(c)
            except (ValueError, TypeError) as e:
                print(f"   ⚠️ 跳过无效 CALL 合约: {e}, data={c}")

        for p in put_list:
            premium = p.get("lastPrice")
            bid = p.get("bid", 0)
            ask = p.get("ask", 0)
            vol = p.get("volume", 0)

            if premium is not None and premium > 0:
                if premium >= ask and ask > 0:
                    aggressor_vol = vol
                elif premium <= bid and bid > 0:
                    aggressor_vol = -vol
                else:
                    aggressor_vol = 0
            else:
                if bid > 0 and ask > 0:
                    premium = (bid + ask) / 2
                else:
                    premium = 0.0
                aggressor_vol = 0

            try:
                p.update({
                    "type": "PUT",
                    "expiration": exp_date,
                    "premium": float(premium),
                    "aggressor_vol": int(aggressor_vol),
                    "gamma": float(c.get("gamma", 0)),          # ← 新增
                    "impliedVolatility": float(c.get("impliedVolatility", 0)) 
                })
                all_opts.append(p)
            except (ValueError, TypeError) as e:
                print(f"   ⚠️ 跳过无效 PUT 合约: {e}, data={p}")

    print(f"   📦 总合约数量: {len(all_opts)}")
    if all_opts:
        print(f"   📦 示例合约 keys: {list(all_opts[0].keys())}")

    if not all_opts:
        print(f"   ⚠️ {symbol} 无有效期权合约数据")
        return pd.DataFrame(), None

    df_raw = pd.DataFrame(all_opts)
    print(f"   📊 df_raw columns: {list(df_raw.columns)}")
    print(f"   📊 df_raw shape: {df_raw.shape}")

    df_raw["aggressor_vol"] = pd.to_numeric(df_raw["aggressor_vol"], errors="coerce").fillna(0)
    for col in ["premium", "volume", "openInterest"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0)

    parquet_path = os.path.join(RAW_DIR, symbol, f"{today_str}.parquet")
    df_raw.to_parquet(parquet_path, index=False)
    return df_raw, None

def load_previous_oi(symbol, today_str):
    from datetime import datetime
    import os

    try:
        today = datetime.strptime(today_str, "%Y-%m-%d").date()
    except ValueError:
        return None

    symbol_dir = os.path.join(RAW_DIR, symbol)
    if not os.path.exists(symbol_dir):
        return None

    date_files = []
    for f in os.listdir(symbol_dir):
        if f.endswith(".json"):
            date_part = f.replace(".json", "")
            if date_part == today_str:
                continue
            try:
                file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
                if file_date < today:
                    date_files.append((file_date, date_part))
            except ValueError:
                continue

    if not date_files:
        return None

    latest_date, latest_date_str = max(date_files, key=lambda x: x[0])
    parquet_path = os.path.join(symbol_dir, f"{latest_date_str}.parquet")

    df_hist = None
    need_rebuild = False

    if os.path.exists(parquet_path):
        try:
            df_hist = pd.read_parquet(parquet_path)
            if "type" in df_hist.columns:
                pass
            else:
                cols_needed = ["expiration", "strike", "call_OI", "put_OI"]
                if all(col in df_hist.columns for col in cols_needed):
                    print(f"   ✅ 直接使用宽格式历史数据 ({latest_date_str})")
                    return df_hist[cols_needed].copy()
                else:
                    need_rebuild = True
        except Exception as e:
            print(f"   ⚠️ 读取历史 parquet 失败: {e}")
            need_rebuild = True
    else:
        need_rebuild = True

    if need_rebuild:
        print(f"   🛠️ 正在重建 {symbol} 的 {latest_date_str} 数据...")
        df_raw, err = fetch_and_save_raw_chain(symbol, latest_date_str)
        if df_raw is None or df_raw.empty or "type" not in df_raw.columns:
            print(f"   ❌ 重建失败或缺少 'type' 列")
            return None
        df_hist = df_raw

    required_cols = {"expiration", "strike", "openInterest", "type"}
    if not required_cols.issubset(df_hist.columns):
        print(f"   ❌ 历史数据缺少必要列: {required_cols - set(df_hist.columns)}")
        return None

    try:
        df_wide = df_hist.pivot_table(
            index=["expiration", "strike"],
            columns="type",
            values="openInterest",
            aggfunc="first"
        ).fillna(0)

        if "CALL" not in df_wide.columns:
            df_wide["CALL"] = 0
        if "PUT" not in df_wide.columns:
            df_wide["PUT"] = 0

        df_wide = df_wide.reset_index()
        df_wide.rename(columns={"CALL": "call_OI", "PUT": "put_OI"}, inplace=True)

        result = df_wide[["expiration", "strike", "call_OI", "put_OI"]].copy()
        print(f"   ✅ 成功生成宽格式历史 OI（日期: {latest_date_str}），shape: {result.shape}")
        return result

    except Exception as e:
        print(f"   ❌ Pivot 失败: {e}")
        traceback.print_exc()
        return None

# ============ 新增：Gamma-based 支撑/阻力标记函数 ============
def mark_levels_and_signals(group, oi_thresh, call_money_thresh, put_money_thresh, is_weekly=False):
    """
    is_weekly: 是否为本周到期分组，决定是否用 Gamma Proxy 计算支撑/阻力
    """
    if is_weekly:
        group['call_gamma_proxy'] = group['call_OI'] * group['call_premium']
        group['put_gamma_proxy'] = group['put_OI'] * group['put_premium']

        valid_call = group[group['call_gamma_proxy'] > 0]
        valid_put = group[group['put_gamma_proxy'] > 0]

        call_peak_strike = None
        put_peak_strike = None

        if not valid_call.empty:
            call_peak_strike = valid_call.loc[valid_call['call_gamma_proxy'].idxmax(), 'strike']
        if not valid_put.empty:
            put_peak_strike = valid_put.loc[valid_put['put_gamma_proxy'].idxmax(), 'strike']
    else:
        call_peak_strike = group.loc[group['call_OI'].idxmax(), 'strike'] if not group.empty and group['call_OI'].sum() > 0 else None
        put_peak_strike = group.loc[group['put_OI'].idxmax(), 'strike'] if not group.empty and group['put_OI'].sum() > 0 else None

    group['阻力位'] = group['strike'].apply(lambda x: '🔴' if x == call_peak_strike else '')
    group['支撑位'] = group['strike'].apply(lambda x: '🟢' if x == put_peak_strike else '')

    # === 信号生成逻辑（不变）===
    call_signal_list = []
    put_signal_list = []

    for _, row in group.iterrows():
        call_sigs = []
        put_sigs = []

        if pd.notna(row['Δcall_OI']) and abs(row['Δcall_OI']) >= oi_thresh and row['call_volume'] > 0:
            direction = '🟢Call增' if row['Δcall_OI'] > 0 else '🔴Call减'
            call_sigs.append((abs(row['Δcall_OI']), f"{direction} ({int(row['Δcall_OI']):+})"))

        if row['call_estimated_flow'] >= call_money_thresh:
            val_k = row['call_estimated_flow'] / 1000
            call_sigs.append((row['call_estimated_flow'], f"🟢Call新资 (+${val_k:.0f}K)"))
        elif row['call_estimated_flow'] <= -call_money_thresh:
            val_k = row['call_estimated_flow'] / 1000
            call_sigs.append((abs(row['call_estimated_flow']), f"🔴Call撤资 (${val_k:.0f}K)"))

        if pd.notna(row['Δput_OI']) and abs(row['Δput_OI']) >= oi_thresh and row['put_volume'] > 0:
            direction = '🟢Put增' if row['Δput_OI'] > 0 else '🔴Put减'
            put_sigs.append((abs(row['Δput_OI']), f"{direction} ({int(row['Δput_OI']):+})"))

        if row['put_estimated_flow'] >= put_money_thresh:
            val_k = row['put_estimated_flow'] / 1000
            put_sigs.append((row['put_estimated_flow'], f"🟢Put新资 (+${val_k:.0f}K)"))
        elif row['put_estimated_flow'] <= -put_money_thresh:
            val_k = row['put_estimated_flow'] / 1000
            put_sigs.append((abs(row['put_estimated_flow']), f"🔴Put撤资 (${val_k:.0f}K)"))

        call_sigs.sort(key=lambda x: x[0], reverse=True)
        put_sigs.sort(key=lambda x: x[0], reverse=True)

        call_text = "\n".join([sig[1] for sig in call_sigs]) if call_sigs else ""
        put_text = "\n".join([sig[1] for sig in put_sigs]) if put_sigs else ""

        call_signal_list.append(call_text)
        put_signal_list.append(put_text)

    group['Call 主力信号'] = call_signal_list
    group['Put 主力信号'] = put_signal_list
    return group

# ============ 路由 ============
@app.route('/', methods=['GET', 'POST'])
def index():
    print("\n🌐 收到请求:", request.method)
    if request.method == 'POST':
        symbols_input = request.form.get("symbols", "").strip()
        oi_threshold_input = request.form.get("oi_threshold", "1000").strip()
        call_to_input = request.form.get("call_turnover_threshold", "500000").strip()
        put_to_input = request.form.get("put_turnover_threshold", "500000").strip()

        try:
            oi_threshold = int(oi_threshold_input)
        except ValueError:
            oi_threshold = 1000

        try:
            call_money_threshold = max(0, int(call_to_input))
        except ValueError:
            call_money_threshold = 500_000

        try:
            put_money_threshold = max(0, int(put_to_input))
        except ValueError:
            put_money_threshold = 500_000

        if not symbols_input:
            return render_template(
                "index.html",
                error="请输入股票代码",
                oi_threshold=oi_threshold,
                call_turnover_threshold=call_money_threshold,
                put_turnover_threshold=put_money_threshold
            )

        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        beijing_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
        today_str = beijing_time.strftime("%Y-%m-%d")
        print(f"📅 日期: {today_str}, 股票: {symbols}")
        print(f"   📌 阈值: OI={oi_threshold}, Call 净资=${call_money_threshold:,}, Put 净资=${put_money_threshold:,}")

        analysis_dfs = []
        for sym in symbols:
            df_raw, err = fetch_and_save_raw_chain(sym, today_str)
            if df_raw is None:
                print(f"⚠️ {sym} 跳过: {err}")
                continue

            required_cols = {"expiration", "strike", "type"}
            if not required_cols.issubset(df_raw.columns):
                print(f"⚠️ {sym} 数据缺少必要列，跳过。现有列: {list(df_raw.columns)}")
                continue

            if df_raw.empty:
                print(f"⚠️ {sym} 无有效期权数据，跳过")
                continue

            print(f"   📊 {sym} 原始数据形状: {df_raw.shape}")

            df_pivot = df_raw.pivot_table(
                index=["expiration", "strike"],
                columns="type",
                values=["openInterest", "volume", "premium", "aggressor_vol", "gamma", "impliedVolatility"],
                aggfunc="first"          # gamma 用 first/mean 都行
            ).fillna(0)
            print(">>> pivot 后实际列名：", df_pivot.columns.tolist())
            df_pivot.columns = [
                'call_OI', 'put_OI',
                'call_volume', 'put_volume',
                'call_premium', 'put_premium',
                'call_aggressor_vol', 'put_aggressor_vol',
                'call_gamma', 'put_gamma',
                'call_iv', 'put_iv'
            ]
            df_today = df_pivot.reset_index()
            df_today.rename(columns={
                'openInterest_CALL': 'call_OI',
                'openInterest_PUT': 'put_OI',
                'volume_CALL': 'call_volume',
                'volume_PUT': 'put_volume',
                'premium_CALL': 'call_premium',
                'premium_PUT': 'put_premium',
                'aggressor_vol_CALL': 'call_aggressor_vol',
                'aggressor_vol_PUT': 'put_aggressor_vol'
            }, inplace=True)

            for col in ['call_OI', 'put_OI', 'call_volume', 'put_volume', 
                        'call_premium', 'put_premium', 'call_aggressor_vol', 'put_aggressor_vol']:
                if col not in df_today.columns:
                    df_today[col] = 0

            df_today["symbol"] = sym
            df_today["call_trading_value"] = df_today["call_volume"] * df_today["call_premium"] * 100
            df_today["put_trading_value"] = df_today["put_volume"] * df_today["put_premium"] * 100
            df_today["call_aggressor_flow"] = df_today["call_aggressor_vol"] * df_today["call_premium"] * 100
            df_today["put_aggressor_flow"] = df_today["put_aggressor_vol"] * df_today["put_premium"] * 100

            df_yest = load_previous_oi(sym, today_str)
            if df_yest is not None and not df_yest.empty:
                df_yest = df_yest.rename(columns={"call_OI": "call_OI_yest", "put_OI": "put_OI_yest"})
                df_today = pd.merge(
                    df_today,
                    df_yest[["expiration", "strike", "call_OI_yest", "put_OI_yest"]],
                    on=["expiration", "strike"],
                    how="left"
                )
            else:
                print(f"   ⚠️ {sym} 历史 OI 数据不可用，ΔOI 将基于 0 计算")

            for col in ["call_OI_yest", "put_OI_yest"]:
                if col not in df_today.columns:
                    df_today[col] = 0.0
                else:
                    df_today[col] = df_today[col].fillna(0.0)

            df_today["Δcall_OI"] = df_today["call_OI"] - df_today["call_OI_yest"]
            df_today["Δput_OI"] = df_today["put_OI"] - df_today["put_OI_yest"]
            df_today["call_estimated_flow"] = df_today["Δcall_OI"] * df_today["call_premium"] * 100
            df_today["put_estimated_flow"] = df_today["Δput_OI"] * df_today["put_premium"] * 100

            spot_price = get_stock_price(sym)
            df_today["spot_price"] = spot_price

            def classify_strike_moneyness(strike, spot):
                if spot is None:
                    return "Unknown"
                lower_atm = spot * 0.98
                upper_atm = spot * 1.02
                if strike < lower_atm:
                    return "Call ITM / Put OTM"
                elif strike > upper_atm:
                    return "Call OTM / Put ITM"
                else:
                    return "ATM 区域"

            df_today["moneyness"] = df_today["strike"].apply(
                lambda s: classify_strike_moneyness(s, spot_price)
            )

            today = beijing_time.date()
            this_mon, this_sun = get_week_range(today)

            def is_expiration_this_week(exp_str):
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    return this_mon <= exp_date <= this_sun
                except:
                    return False

            df_today["is_weekly"] = df_today["expiration"].apply(is_expiration_this_week)

            df_today = df_today.groupby(['symbol', 'expiration']).apply(
                lambda g: mark_levels_and_signals(
                    g,
                    oi_threshold,
                    call_money_threshold,
                    put_money_threshold,
                    is_weekly=g["is_weekly"].iloc[0]
                )
            ).reset_index(drop=True)

            analysis_dfs.append(df_today)

        if not analysis_dfs:
            return render_template(
                "index.html",
                error="无有效数据",
                oi_threshold=oi_threshold,
                call_turnover_threshold=call_money_threshold,
                put_turnover_threshold=put_money_threshold
            )

        df_full = pd.concat(analysis_dfs, ignore_index=True)
        df_full["到期分组"] = df_full["expiration"].apply(classify_expiration)

        summary_path = os.path.join(REPORT_DIR, f"📊 主力信号_{today_str}.xlsx")
        with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
            workbook = writer.book

            fmt_green = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
            fmt_red = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
            fmt_wrap = workbook.add_format({'text_wrap': True, 'valign': 'top'})

            group_order = {"本周": 0, "下周": 1, "本月": 2, "远期": 3}

            for sym in df_full["symbol"].unique():
                df_sym = df_full[df_full["symbol"] == sym].copy()
                if df_sym.empty:
                    continue

                df_sym["__sort"] = df_sym["到期分组"].map(lambda g: group_order.get(g, 99))
                df_out = df_sym.sort_values(["__sort", "expiration", "strike"]).reset_index(drop=True)
                df_out.drop(columns=["__sort"], inplace=True)

                # ====== 新表格布局：Call 左，Put 右 ======
                common_cols = ["到期分组", "expiration", "strike", "moneyness"]
                call_cols = [
                    "call_OI", "Δcall_OI", "call_volume", "call_premium",
                    "call_trading_value", "call_estimated_flow", "阻力位"
                ]
                put_cols = [
                    "put_OI", "Δput_OI", "put_volume", "put_premium",
                    "put_trading_value", "put_estimated_flow", "支撑位"
                ]

                new_col_order = common_cols + call_cols + put_cols
                for col in new_col_order:
                    if col not in df_out.columns:
                        df_out[col] = ""

                df_display = df_out[new_col_order].copy()

                header_df = pd.DataFrame([df_display.columns], columns=df_display.columns)
                final_df = pd.concat([header_df, df_display], ignore_index=True)

                sheet_name = sym[:31]
                final_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                ws = writer.sheets[sheet_name]

                # 合并顶部标题
                call_end_letter = chr(ord('A') + len(common_cols) + len(call_cols) - 1)
                put_start_letter = chr(ord('A') + len(common_cols) + len(call_cols))
                put_end_letter = chr(ord('A') + len(common_cols) + len(call_cols) + len(put_cols) - 1)

                title_fmt = workbook.add_format({
                    'bold': True, 'align': 'center', 'valign': 'vcenter',
                    'font_size': 12, 'bg_color': '#E0E0E0'
                })

                ws.merge_range(f"A1:{call_end_letter}1", "Call Open Trend", title_fmt)
                ws.merge_range(f"{put_start_letter}1:{put_end_letter}1", "Put Open Trend", title_fmt)

                # 条件格式和列宽
                col_list = list(final_df.columns)
                delta_call_idx = col_list.index("Δcall_OI")
                call_flow_idx = col_list.index("call_estimated_flow")
                delta_put_idx = col_list.index("Δput_OI")
                put_flow_idx = col_list.index("put_estimated_flow")

                data_start_row = 2
                last_row = len(final_df) + 1

                for idx, cond_col in [(delta_call_idx, '>'), (delta_call_idx, '<'),
                                      (call_flow_idx, '>'), (call_flow_idx, '<'),
                                      (delta_put_idx, '>'), (delta_put_idx, '<'),
                                      (put_flow_idx, '>'), (put_flow_idx, '<')]:
                    criteria = cond_col
                    fmt = fmt_green if criteria == '>' else fmt_red
                    ws.conditional_format(data_start_row, idx, last_row, idx,
                                          {'type': 'cell', 'criteria': criteria, 'value': 0, 'format': fmt})

                for i, col in enumerate(col_list):
                    if col in ["阻力位", "支撑位"]:
                        ws.set_column(i, i, 8)
                    elif "premium" in col:
                        ws.set_column(i, i, 10)
                    elif "estimated_flow" in col or "trading_value" in col:
                        ws.set_column(i, i, 14)
                    elif "volume" in col or "OI" in col:
                        ws.set_column(i, i, 12)
                    else:
                        width = max(12, len(str(col)) + 2)
                        if col == "moneyness":
                            width = 18
                        ws.set_column(i, i, width)

            # === 主力信号汇总 sheet：拆分为独立 Call / Put 表格，按 |ΔOI| 降序 ===
            signal_df = df_full[
                (df_full["Call 主力信号"] != "") | (df_full["Put 主力信号"] != "")
            ].copy()

            if not signal_df.empty:
                for sym in signal_df["symbol"].unique():
                    sym_signal = signal_df[signal_df["symbol"] == sym].copy()
                    sheet_name = f"🔥 主力信号 - {sym}"[:31]
                    ws = workbook.add_worksheet(sheet_name)

                    # ====== 左侧：Call 信号（按 Δcall_OI 实际值降序）======
                    call_mask = sym_signal["Call 主力信号"] != ""
                    call_rows = sym_signal[call_mask].copy()
                    if not call_rows.empty:
                        # 按 Δcall_OI 实际值从大到小排序：大幅增仓在前，大幅减仓在后
                        call_rows = call_rows.sort_values(
                            by="Δcall_OI",
                            ascending=False,
                            na_position='last'
                        )
                        call_cols = [
                            "到期分组", "expiration", "strike", "moneyness",
                            "call_OI", "Δcall_OI", "call_volume", "call_premium",
                            "call_trading_value", "call_estimated_flow"
                        ]
                        call_data = call_rows[call_cols]

                        # 写入标题
                        ws.merge_range(0, 0, 0, len(call_cols) - 1, "🟢 Call 主力信号", title_fmt)
                        for j, col in enumerate(call_cols):
                            ws.write(1, j, col)

                        # 写入数据
                        for i, (_, row) in enumerate(call_data.iterrows()):
                            for j, val in enumerate(row):
                                ws.write(i + 2, j, val)

                        # 列宽
                        for j, col in enumerate(call_cols):
                            width = 16
                            if "trading_value" in col or "estimated_flow" in col:
                                width = 18
                            elif col == "moneyness":
                                width = 14
                            ws.set_column(j, j, width)

                        left_end_col = len(call_cols) - 1
                    else:
                        left_end_col = -1

                    # ====== 右侧：Put 信号（按 Δput_OI 实际值降序）======
                    put_mask = sym_signal["Put 主力信号"] != ""
                    put_rows = sym_signal[put_mask].copy()
                    if not put_rows.empty:
                        # 按 Δput_OI 实际值从大到小排序：Put 大幅增仓（看跌情绪强）在前
                        put_rows = put_rows.sort_values(
                            by="Δput_OI",
                            ascending=False,
                            na_position='last'
                        )
                        put_cols = [
                            "到期分组", "expiration", "strike", "moneyness",
                            "put_OI", "Δput_OI", "put_volume", "put_premium",
                            "put_trading_value", "put_estimated_flow"
                        ]
                        put_data = put_rows[put_cols]

                        start_col = left_end_col + 2  # 中间空一列
                        ws.merge_range(0, start_col, 0, start_col + len(put_cols) - 1, "🔴 Put 主力信号", title_fmt)
                        for j, col in enumerate(put_cols):
                            ws.write(1, start_col + j, col)

                        for i, (_, row) in enumerate(put_data.iterrows()):
                            for j, val in enumerate(row):
                                ws.write(i + 2, start_col + j, val)

                        for j, col in enumerate(put_cols):
                            width = 16
                            if "trading_value" in col or "estimated_flow" in col:
                                width = 18
                            elif col == "moneyness":
                                width = 14
                            ws.set_column(start_col + j, start_col + j, width)

            # === 汇总统计 ===
            for sym in df_full["symbol"].unique():
                df_sym = df_full[df_full["symbol"] == sym]
                summary_by_group = df_sym.groupby('到期分组').agg(
                    symbol=('symbol', 'first'),
                    call_estimated_flow_sum=('call_estimated_flow', 'sum'),
                    put_estimated_flow_sum=('put_estimated_flow', 'sum'),
                    call_trading_value=('call_trading_value', 'sum'),
                    put_trading_value=('put_trading_value', 'sum'),
                    call_oi_change_sum=('Δcall_OI', 'sum'),
                    put_oi_change_sum=('Δput_OI', 'sum'),
                    call_vol_sum=('call_volume', 'sum'),
                    put_vol_sum=('put_volume', 'sum')
                ).reset_index()

                summary_by_group['__sort'] = summary_by_group['到期分组'].map(lambda g: group_order.get(g, 99))
                summary_by_group = summary_by_group.sort_values('__sort').drop(columns=['__sort']).reset_index(drop=True)

                summary_by_group['净估算资金流 (C-P)'] = (
                    summary_by_group['call_estimated_flow_sum'] - 
                    summary_by_group['put_estimated_flow_sum']
                )
                summary_by_group['P/C 成交额比'] = (
                    summary_by_group['put_trading_value'] / 
                    summary_by_group['call_trading_value'].replace(0, np.nan)
                ).round(2)

                def fmt_millions(x):
                    if pd.isna(x) or x == 0:
                        return "$0"
                    if abs(x) >= 1e9:
                        return f"${x/1e9:.1f}B"
                    elif abs(x) >= 1e6:
                        return f"${x/1e6:.1f}M"
                    elif abs(x) >= 1e3:
                        return f"${x/1e3:.0f}K"
                    else:
                        return f"${x:.0f}"

                summary_by_group['Call 净资金'] = summary_by_group['call_estimated_flow_sum'].apply(fmt_millions)
                summary_by_group['Put 净资金'] = summary_by_group['put_estimated_flow_sum'].apply(fmt_millions)
                summary_by_group['净资金流 (C-P)'] = summary_by_group['净估算资金流 (C-P)'].apply(fmt_millions)
                summary_by_group['净倾向指数'] = summary_by_group.apply(
                    lambda r: calculate_net_bias_index(r['call_estimated_flow_sum'], r['put_estimated_flow_sum']), axis=1
                )

                summary_out = summary_by_group[[
                    '到期分组',
                    'Call 净资金',
                    'Put 净资金',
                    '净资金流 (C-P)',
                    'P/C 成交额比',
                    '净倾向指数'
                ]]

                sheet_name = f"📊 {sym} 汇总"
                if len(sheet_name) > 31:
                    sheet_name = f"📊 {sym[:25]}汇总"
                summary_out.to_excel(writer, sheet_name=sheet_name, index=False)
                ws_summary = writer.sheets[sheet_name]
                for i, col in enumerate(summary_out.columns):
                    width = max(14, len(str(col)) + 2)
                    ws_summary.set_column(i, i, width)
                net_bias_col = summary_out.columns.get_loc('净倾向指数')
                ws_summary.conditional_format(1, net_bias_col, len(summary_out), net_bias_col, {
                    'type': '3_color_scale',
                    'min_type': 'num', 'min_value': -1,
                    'mid_type': 'num', 'mid_value': 0,
                    'max_type': 'num', 'max_value': 1,
                    'min_color': "#FF4500",
                    'mid_color': "#FFFF00",
                    'max_color': "#008000"
                })
                # ============ 新增：为每个股票单独保存“主力信号”页 ============
        if not signal_df.empty:
            for sym in signal_df["symbol"].unique():
                sym_signal = signal_df[signal_df["symbol"] == sym].copy()

                # 准备 Call 信号数据
                call_mask = sym_signal["Call 主力信号"] != ""
                call_rows = sym_signal[call_mask].copy()
                call_data = None
                if not call_rows.empty:
                    call_rows = call_rows.sort_values(by="Δcall_OI", ascending=False, na_position='last')
                    call_cols = [
                        "到期分组", "expiration", "strike", "moneyness",
                        "call_OI", "Δcall_OI", "call_volume", "call_premium",
                        "call_trading_value", "call_estimated_flow", "Call 主力信号"
                    ]
                    call_data = call_rows[call_cols]

                # 准备 Put 信号数据
                put_mask = sym_signal["Put 主力信号"] != ""
                put_rows = sym_signal[put_mask].copy()
                put_data = None
                if not put_rows.empty:
                    put_rows = put_rows.sort_values(by="Δput_OI", ascending=False, na_position='last')
                    put_cols = [
                        "到期分组", "expiration", "strike", "moneyness",
                        "put_OI", "Δput_OI", "put_volume", "put_premium",
                        "put_trading_value", "put_estimated_flow", "Put 主力信号"
                    ]
                    put_data = put_rows[put_cols]

                # 如果都没有信号，跳过
                if call_data is None and put_data is None:
                    continue

                # 创建单独的 Excel 文件
                symbol_dir = os.path.join(REPORT_DIR, sym)
                os.makedirs(symbol_dir, exist_ok=True)
                signal_file = os.path.join(symbol_dir, f"{sym}_{today_str}_主力信号.xlsx")

                with pd.ExcelWriter(signal_file, engine='xlsxwriter') as sig_writer:
                    workbook = sig_writer.book
                    title_fmt = workbook.add_format({
                        'bold': True, 'align': 'center', 'valign': 'vcenter',
                        'font_size': 12, 'bg_color': '#E0E0E0'
                    })
                    fmt_wrap = workbook.add_format({'text_wrap': True, 'valign': 'top'})

                    # 写入 Call 表
                    if call_data is not None:
                        sheet_name = "🟢 Call 主力信号"
                        call_data.to_excel(sig_writer, sheet_name=sheet_name, index=False)
                        ws = sig_writer.sheets[sheet_name]
                        ws.set_row(0, 20, title_fmt)  # 标题行格式
                        for i, col in enumerate(call_data.columns):
                            width = 16
                            if "trading_value" in col or "estimated_flow" in col:
                                width = 18
                            elif col == "moneyness":
                                width = 14
                            elif "信号" in col:
                                width = 25
                            ws.set_column(i, i, width, fmt_wrap)

                    # 写入 Put 表
                    if put_data is not None:
                        sheet_name = "🔴 Put 主力信号"
                        put_data.to_excel(sig_writer, sheet_name=sheet_name, index=False)
                        ws = sig_writer.sheets[sheet_name]
                        ws.set_row(0, 20, title_fmt)
                        for i, col in enumerate(put_data.columns):
                            width = 16
                            if "trading_value" in col or "estimated_flow" in col:
                                width = 18
                            elif col == "moneyness":
                                width = 14
                            elif "信号" in col:
                                width = 25
                            ws.set_column(i, i, width, fmt_wrap)

                print(f"   💾 已保存主力信号到: {signal_file}") 
        append_vol_range_to_existing_report(today_str, symbols, df_full)
        return render_template(
            "index.html",
            success=f"✅ 成功！报告: {os.path.basename(summary_path)}",
            oi_threshold=oi_threshold,
            call_turnover_threshold=call_money_threshold,
            put_turnover_threshold=put_money_threshold
        )

        
        
        
        
    return render_template(
        "index.html",
        oi_threshold=1000,
        call_turnover_threshold=500000,
        put_turnover_threshold=500000
    )


if __name__ == '__main__':
    print("🚀 启动服务，请访问 http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
    
    # ====================== 追加写入原有 Excel ======================
