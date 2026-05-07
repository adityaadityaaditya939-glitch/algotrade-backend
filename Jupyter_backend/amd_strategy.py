"""
AMD Strategy — Accumulation, Manipulation, Distribution
Smart Money Concept (SMC)

Phase 1 — ACCUMULATION  : Detect sideways range on 30min chart
           Range = price consolidating in tight band (ATR-based, low volatility)

Phase 2 — MANIPULATION  : Detect liquidity sweep on 15min chart
           Price breaks ABOVE range high → stop hunt on longs → SHORT setup
           Price breaks BELOW range low  → stop hunt on shorts → LONG setup
           Must be a FAKE breakout (price closes back inside range ≤3 candles)

Phase 3 — DISTRIBUTION  : Enter on 15min after sweep confirmation
           RR = 1:2
           Stop = just beyond the sweep wick
           Target = opposite side of range × RR
"""

import pandas as pd
import numpy as np
import httpx, asyncio
from datetime import datetime, timedelta

BINANCE_REST = "https://api.binance.com"

# ─── Binance historical fetcher ───────────────────────────────────────────────
async def fetch_historical(symbol="ETHUSDT", interval="30m", days=180):
    """Fetch up to 3 years of OHLCV from Binance REST (no API key needed)."""
    end_ms   = int(datetime.utcnow().timestamp() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_data = []

    print(f"[AMD] Fetching {days}d of {symbol} {interval} from Binance…")
    async with httpx.AsyncClient(timeout=20) as client:
        cur = start_ms
        while cur < end_ms:
            try:
                r = await client.get(f"{BINANCE_REST}/api/v3/klines", params={
                    "symbol": symbol, "interval": interval,
                    "startTime": cur, "endTime": end_ms, "limit": 1000,
                })
                r.raise_for_status()
                data = r.json()
                if not data: break
                all_data.extend(data)
                cur = int(data[-1][6]) + 1
                if len(data) < 1000: break
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"[AMD] Fetch error: {e}"); break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qv","trades","tb","tq","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df = df[~df.index.duplicated(keep="first")]
    print(f"[AMD] {len(df)} candles: {df.index[0]} → {df.index[-1]}")
    return df[["open","high","low","close","volume"]]


# ─── Phase 1: Accumulation range detection (30min) ───────────────────────────
def detect_accumulation_ranges(df, lookback=20, max_range_pct=3.0, min_range_pct=0.5, atr_period=14):
    """
    Identify sideways consolidation ranges.
    max_range_pct / min_range_pct are in percent of price.
    """
    df = df.copy()
    hl  = df["high"] - df["low"]
    hcp = (df["high"] - df["close"].shift()).abs()
    lcp = (df["low"]  - df["close"].shift()).abs()
    df["atr"] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).ewm(span=atr_period, adjust=False).mean()

    df["range_high"]  = np.nan
    df["range_low"]   = np.nan
    df["range_valid"] = False
    df["range_pct"]   = np.nan

    for i in range(lookback, len(df)):
        w      = df.iloc[i - lookback : i + 1]
        w_high = w["high"].max()
        w_low  = w["low"].min()
        price  = df["close"].iloc[i]
        rng_h  = w_high - w_low
        rng_pct = rng_h / price * 100

        # ATR contraction check — must be below recent average
        atr_now  = df["atr"].iloc[i]
        atr_mean = df["atr"].iloc[max(0,i-lookback):i].mean()
        atr_ok   = (atr_now < atr_mean * 0.85) if atr_mean > 0 else False

        if min_range_pct <= rng_pct <= max_range_pct and atr_ok:
            df.at[df.index[i], "range_high"]  = round(w_high, 4)
            df.at[df.index[i], "range_low"]   = round(w_low,  4)
            df.at[df.index[i], "range_valid"] = True
            df.at[df.index[i], "range_pct"]   = round(rng_pct, 3)

    return df


# ─── Phase 2: Liquidity sweep detection (15min) ───────────────────────────────
def detect_liquidity_sweeps(df_15m, range_high, range_low, sweep_buffer=0.001, max_sweep_candles=3, rr_ratio=2.0):
    """
    Detect fake breakouts (liquidity sweeps) above/below the accumulation range.
    sweep_buffer: how far beyond the range counts as a sweep (0.1% default)
    """
    df = df_15m.copy()
    df["sweep_type"]   = None
    df["sweep_extreme"] = np.nan
    df["entry_price"]  = np.nan
    df["stop_loss"]    = np.nan
    df["take_profit"]  = np.nan
    df["sweep_candle"] = False

    high_thresh = range_high * (1 + sweep_buffer)
    low_thresh  = range_low  * (1 - sweep_buffer)
    rng_height  = range_high - range_low

    i = 1
    while i < len(df) - max_sweep_candles:
        candle = df.iloc[i]

        # ── BEARISH SWEEP: spike above range high, close back inside ─────────
        if candle["high"] > high_thresh:
            for j in range(i, min(i + max_sweep_candles + 1, len(df))):
                fut = df.iloc[j]
                if fut["close"] < range_high:
                    sweep_wick = round(candle["high"], 4)
                    entry      = round(fut["close"], 4)
                    sl         = round(sweep_wick * 1.0015, 4)   # 0.15% above wick
                    risk       = sl - entry
                    tp         = round(entry - risk * rr_ratio, 4)  # 1:rr_ratio RR

                    df.at[df.index[j], "sweep_type"]    = "bearish"
                    df.at[df.index[j], "sweep_extreme"]  = sweep_wick
                    df.at[df.index[j], "entry_price"]   = entry
                    df.at[df.index[j], "stop_loss"]     = sl
                    df.at[df.index[j], "take_profit"]   = tp
                    df.at[df.index[j], "sweep_candle"]  = True
                    i = j + 1
                    break
            else:
                i += 1

        # ── BULLISH SWEEP: spike below range low, close back inside ──────────
        elif candle["low"] < low_thresh:
            for j in range(i, min(i + max_sweep_candles + 1, len(df))):
                fut = df.iloc[j]
                if fut["close"] > range_low:
                    sweep_wick = round(candle["low"], 4)
                    entry      = round(fut["close"], 4)
                    sl         = round(sweep_wick * 0.9985, 4)   # 0.15% below wick
                    risk       = entry - sl
                    tp         = round(entry + risk * rr_ratio, 4)  # 1:rr_ratio RR

                    df.at[df.index[j], "sweep_type"]    = "bullish"
                    df.at[df.index[j], "sweep_extreme"]  = sweep_wick
                    df.at[df.index[j], "entry_price"]   = entry
                    df.at[df.index[j], "stop_loss"]     = sl
                    df.at[df.index[j], "take_profit"]   = tp
                    df.at[df.index[j], "sweep_candle"]  = True
                    i = j + 1
                    break
            else:
                i += 1
        else:
            i += 1

    return df


# ─── Phase 3: Distribution backtest engine ────────────────────────────────────
def run_amd_backtest(df_signals, initial_capital=10_000.0, risk_pct=0.02, commission=0.001):
    capital = initial_capital
    trades  = []
    equity  = []
    in_trade = False
    entry = sl = tp = size = 0
    side = entry_time = None

    rows = df_signals.reset_index() if "open_time" not in df_signals.columns else df_signals

    for _, row in rows.iterrows():
        ts_ms = int(row["open_time"].timestamp() * 1000)
        price = float(row["close"])

        if in_trade:
            sl_hit = (side=="SHORT" and price>=sl) or (side=="LONG" and price<=sl)
            tp_hit = (side=="SHORT" and price<=tp) or (side=="LONG" and price>=tp)

            if sl_hit or tp_hit:
                reason  = "TAKE PROFIT" if tp_hit else "STOP LOSS"
                exit_p  = tp if tp_hit else sl
                pnl_raw = (entry-exit_p)*size if side=="SHORT" else (exit_p-entry)*size
                net_pnl = round(pnl_raw - exit_p*size*commission, 2)
                capital = round(capital + net_pnl, 2)
                trades.append({
                    "num":        len(trades)+1,
                    "side":       side,
                    "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"),
                    "exit_time":  row["open_time"].strftime("%Y-%m-%d %H:%M"),
                    "entry":      round(entry,2),
                    "exit":       round(exit_p,2),
                    "sl":         round(sl,2),
                    "tp":         round(tp,2),
                    "pnl":        net_pnl,
                    "reason":     reason,
                    "win":        net_pnl > 0,
                    "duration_h": round((row["open_time"]-entry_time).total_seconds()/3600,1),
                    "strategy":   "AMD",
                })
                in_trade = False

        equity.append({"x": ts_ms, "y": round(capital,2)})

        if not in_trade and bool(row.get("sweep_candle", False)):
            ep = row["entry_price"]; sp = row["stop_loss"]; tpp = row["take_profit"]
            if any(pd.isna(x) for x in [ep, sp, tpp]): continue
            ep, sp, tpp = float(ep), float(sp), float(tpp)
            sl_dist = abs(ep - sp)
            if sl_dist <= 0: continue
            sz = min(capital * risk_pct / sl_dist, capital * 0.95 / ep)
            if sz <= 0: continue
            cost = ep * sz * commission
            if cost > capital: continue

            in_trade   = True
            side       = "SHORT" if row["sweep_type"]=="bearish" else "LONG"
            entry      = ep; sl = sp; tp = tpp
            size       = round(sz, 6)
            entry_time = row["open_time"]
            capital    = round(capital - cost, 2)

    return trades, equity, round(capital, 2)


# ─── Stats ────────────────────────────────────────────────────────────────────
def calc_amd_stats(trades, equity_curve, start_capital=10_000.0):
    if not trades:
        return {k:0 for k in ["total_trades","win_rate","total_pnl","total_return",
                               "sharpe","max_drawdown","profit_factor","avg_win",
                               "avg_loss","expectancy","take_profits","stop_losses","final_capital"]}
    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p>0]
    losses = [p for p in pnls if p<0]
    final  = equity_curve[-1]["y"] if equity_curve else start_capital
    eq     = pd.Series([e["y"] for e in equity_curve])
    rets   = eq.pct_change().dropna()
    sharpe = round(rets.mean()/rets.std()*(17520**0.5),2) if rets.std()>0 else 0.0
    peak   = eq.cummax()
    mdd    = round(((eq-peak)/peak*100).min(),2)
    gw     = sum(wins); gl = abs(sum(losses))
    return {
        "total_trades":  len(trades),
        "win_rate":      round(len(wins)/len(pnls)*100,1),
        "total_pnl":     round(sum(pnls),2),
        "total_return":  round((final-start_capital)/start_capital*100,2),
        "sharpe":        sharpe,
        "max_drawdown":  mdd,
        "profit_factor": round(gw/gl,2) if gl>0 else 0.0,
        "avg_win":       round(sum(wins)/len(wins),2)   if wins   else 0.0,
        "avg_loss":      round(sum(losses)/len(losses),2) if losses else 0.0,
        "expectancy":    round(sum(pnls)/len(pnls),2),
        "take_profits":  sum(1 for t in trades if "TAKE" in t["reason"]),
        "stop_losses":   sum(1 for t in trades if "STOP" in t["reason"]),
        "final_capital": round(final,2),
    }


# ─── Live signal detector ─────────────────────────────────────────────────────
async def detect_live_amd_signal(symbol="ETHUSDT", range_lookback=20,
                                  max_range_pct=3.0, sweep_buffer=0.001, rr_ratio=2.0):
    """Real-time AMD signal check. Returns current phase + signal if any."""
    try:
        df_30m = await fetch_historical(symbol, "30m", days=14)
        df_15m = await fetch_historical(symbol, "15m", days=7)
        if df_30m.empty or df_15m.empty:
            return {"signal":"NONE","reason":"No data"}

        df_r = detect_accumulation_ranges(df_30m, lookback=range_lookback,
                                           max_range_pct=max_range_pct)
        valid = df_r[df_r["range_valid"]==True]
        if valid.empty:
            return {"signal":"NONE","reason":"No accumulation range detected"}

        latest     = valid.iloc[-1]
        rh         = float(latest["range_high"])
        rl         = float(latest["range_low"])
        range_ts   = latest.name
        cur_price  = float(df_15m["close"].iloc[-1])

        df_15m_w   = df_15m[df_15m.index >= range_ts].copy()
        if len(df_15m_w) < 3:
            return {"signal":"WATCHING","phase":"Accumulation","range_high":round(rh,2),
                    "range_low":round(rl,2),"current_price":round(cur_price,2),
                    "description":"Range forming. Watching for manipulation sweep."}

        swept = detect_liquidity_sweeps(df_15m_w, rh, rl, sweep_buffer, rr_ratio=rr_ratio)
        sigs  = swept[swept["sweep_candle"]==True]

        if sigs.empty:
            return {
                "signal":"WATCHING","alert":False,"symbol":symbol,
                "phase":"Accumulation",
                "range_high":round(rh,2),"range_low":round(rl,2),
                "range_pct":round(float(latest["range_pct"]),3),
                "current_price":round(cur_price,2),
                "range_formed":str(range_ts),
                "description":"Range identified. Watching for liquidity sweep.",
                "timestamp":datetime.utcnow().isoformat(),
            }

        sig  = sigs.iloc[-1]
        side = "SHORT" if sig["sweep_type"]=="bearish" else "LONG"
        return {
            "signal":        side,
            "alert":         True,
            "alert_type":    "AMD_LIQUIDITY_SWEEP",
            "symbol":        symbol,
            "phase":         "Distribution",
            "sweep_type":    sig["sweep_type"],
            "range_high":    round(rh,2),
            "range_low":     round(rl,2),
            "entry":         round(float(sig["entry_price"]),2),
            "stop_loss":     round(float(sig["stop_loss"]),2),
            "take_profit":   round(float(sig["take_profit"]),2),
            "current_price": round(cur_price,2),
            "rr_ratio":      f"1:{rr_ratio}",
            "timeframe":     "15m",
            "strength":      "STRONG",
            "description":   f"Liquidity sweep of range {'high' if sig['sweep_type']=='bearish' else 'low'}. Enter {side} on distribution.",
            "timestamp":     datetime.utcnow().isoformat(),
        }
    except Exception as e:
        import traceback
        return {"signal":"ERROR","error":str(e),"trace":traceback.format_exc()}


# ─── Full backtest runner (called by backend API) ─────────────────────────────
async def run_full_amd_backtest(symbol="ETHUSDT", days=180, range_lookback=20,
                                 max_range_pct=3.0, min_range_pct=0.5,
                                 sweep_buffer=0.001, initial_capital=10_000.0,
                                 risk_per_trade=0.02, rr_ratio=2.0):
    try:
        print(f"[AMD] Full backtest: {symbol} {days}d")
        df_30m = await fetch_historical(symbol, "30m", days=days)
        df_15m = await fetch_historical(symbol, "15m", days=days)
        if df_30m.empty or df_15m.empty:
            return {"error": "Failed to fetch Binance data"}

        # Detect ranges
        df_r      = detect_accumulation_ranges(df_30m, lookback=range_lookback,
                                                max_range_pct=max_range_pct,
                                                min_range_pct=min_range_pct)
        valid_r   = df_r[df_r["range_valid"]==True]

        # Deduplicate ranges
        range_events, prev_h, prev_l = [], None, None
        for ts, row in valid_r.iterrows():
            rh, rl = float(row["range_high"]), float(row["range_low"])
            if rh!=prev_h or rl!=prev_l:
                range_events.append({"ts":ts,"high":rh,"low":rl})
                prev_h, prev_l = rh, rl

        # Collect all sweep signals on 15min
        sig_cols = ["sweep_type","sweep_extreme","entry_price","stop_loss","take_profit","sweep_candle"]
        all_sigs = df_15m.copy()
        for c in sig_cols:
            all_sigs[c] = None if c in ["sweep_type"] else (False if c=="sweep_candle" else np.nan)

        range_lines = []
        for evt in range_events:
            end_ts = evt["ts"] + pd.Timedelta(days=5)
            w15 = df_15m[(df_15m.index>=evt["ts"]) & (df_15m.index<=end_ts)].copy()
            if len(w15) < 5: continue
            swept = detect_liquidity_sweeps(w15, evt["high"], evt["low"], sweep_buffer, rr_ratio=rr_ratio)
            for sig_ts, sig_row in swept[swept["sweep_candle"]==True].iterrows():
                if sig_ts in all_sigs.index:
                    for c in sig_cols:
                        all_sigs.at[sig_ts, c] = sig_row[c]
            range_lines.append({"ts":int(evt["ts"].timestamp()*1000),
                                 "high":round(evt["high"],2),"low":round(evt["low"],2)})

        # Backtest
        all_sigs_reset = all_sigs.reset_index()
        trades, equity_curve, final = run_amd_backtest(all_sigs_reset, initial_capital, risk_per_trade)
        stats = calc_amd_stats(trades, equity_curve, initial_capital)

        # Chart data (last 300 15min candles)
        chart_data = []
        for ts, row in df_15m.tail(300).iterrows():
            t = int(ts.timestamp()*1000)
            chart_data.append({"x":t,"y":[round(row["open"],2),round(row["high"],2),
                                            round(row["low"],2), round(row["close"],2)]})

        sampled_eq = equity_curve[::4] if len(equity_curve)>4 else equity_curve
        if equity_curve:
            if not sampled_eq or sampled_eq[0]!=equity_curve[0]: sampled_eq=[equity_curve[0]]+sampled_eq
            if sampled_eq[-1]!=equity_curve[-1]: sampled_eq=sampled_eq+[equity_curve[-1]]

        print(f"[AMD] Done: {len(trades)} trades, {stats['total_return']}% return")
        return {
            "strategy":     "AMD",
            "symbol":       symbol,
            "days":         days,
            "total_ranges": len(range_events),
            "stats":        stats,
            "trades":       trades,
            "equity_curve": sampled_eq,
            "chart_data":   chart_data,
            "range_lines":  range_lines[-20:],
            "params":       {"range_lookback":range_lookback,"max_range_pct":max_range_pct,
                             "sweep_buffer":sweep_buffer,"risk_per_trade":risk_per_trade,
                             "rr_ratio":f"1:{rr_ratio}","entry_tf":"15m","range_tf":"30m"},
        }
    except Exception as e:
        import traceback
        return {"error":str(e),"trace":traceback.format_exc()}