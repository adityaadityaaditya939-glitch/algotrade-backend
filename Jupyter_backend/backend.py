"""
AlgoTrade Pro — Backend
Exact replication of notebook V2 backtest: +5%, 6 trades, 50% WR
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import glob, os, asyncio

app = FastAPI(title="AlgoTrade Pro API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Auth integration ──────────────────────────────────────────────────────────
try:
    from auth import router as auth_router, orders_router, init_db
    app.include_router(auth_router)
    app.include_router(orders_router)
    @app.on_event("startup")
    def startup_db():
        try:   init_db()
        except Exception as e: print(f"DB not available: {e}")
except ImportError as e:
    print(f"Running without auth: {e}")

# ── Realtime integration ──────────────────────────────────────────────────────
try:
    from realtime import (
        start_realtime_tasks, ws_manager,
        live_prices, candle_cache, get_live_chart_data,
        fetch_candles, add_indicators as rt_add_indicators
    )
    REALTIME_ENABLED = True

    @app.on_event("startup")
    async def startup_realtime():
        await start_realtime_tasks()
        print("[Realtime] Started ✅")

    # ── WebSocket endpoint — frontend connects here for live updates ──────────
    @app.websocket("/ws/live")
    async def websocket_live(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                # Keep connection alive, receive any client messages
                data = await websocket.receive_text()
                # Client can send {"subscribe": "ETHUSDT"} to change pair
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception as e:
            ws_manager.disconnect(websocket)

    # ── REST: live candle chart data for a symbol ──────────────────────────────
    @app.get("/api/live/candles")
    async def live_candles(symbol: str = "BTCUSDT", candles: int = 150):
        return get_live_chart_data(symbol, candles)

    # ── REST: current live prices for all pairs ────────────────────────────────
    @app.get("/api/live/prices")
    async def live_prices_endpoint():
        return {
            "prices":  live_prices.get_all(),
            "signals": {s: candle_cache.get_signal(s) for s in ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT"]},
            "ts":      int(__import__("time").time() * 1000),
        }

    # ── REST: signal for a specific symbol ────────────────────────────────────
    @app.get("/api/live/signal")
    async def live_signal(symbol: str = "BTCUSDT"):
        signal = candle_cache.get_signal(symbol)
        price  = live_prices.get(symbol)
        return {"symbol": symbol, "signal": signal, "price": price}

except ImportError as e:
    REALTIME_ENABLED = False
    print(f"[Realtime] Not available: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS  (exact copy from notebook add_indicators())
# ═══════════════════════════════════════════════════════════════════════════════
def add_indicators(df):
    df = df.copy()

    # EMAs
    df['ema_50']  = df['close'].ewm(span=50,  adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # EMA spread  ← notebook uses ema_spread_abs (%) with min_ema_spread=0.03 (%)
    df['ema_spread']     = (df['ema_50'] - df['ema_200']) / df['ema_200'] * 100
    df['ema_spread_abs'] = df['ema_spread'].abs()

    # RSI 14
    delta = df['close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    rs    = (gain.ewm(span=14, adjust=False).mean() /
             loss.ewm(span=14, adjust=False).mean().replace(0, np.nan))
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR 14
    hl  = df['high'] - df['low']
    hcp = (df['high'] - df['close'].shift()).abs()
    lcp = (df['low']  - df['close'].shift()).abs()
    df['atr'] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).ewm(span=14, adjust=False).mean()

    # MACD
    e12 = df['close'].ewm(span=12, adjust=False).mean()
    e26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']      = e12 - e26
    df['macd_sig']  = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']

    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20 * 100
    df['bb_pos']   = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20']

    # Momentum
    df['ret_1h']  = df['close'].pct_change(1)  * 100
    df['ret_4h']  = df['close'].pct_change(4)  * 100
    df['ret_24h'] = df['close'].pct_change(24) * 100

    # ADX 14  (exact notebook formula)
    plus_dm  = df['high'].diff()
    minus_dm = df['low'].diff(-1).abs()
    plus_dm  = plus_dm.where((plus_dm > minus_dm)  & (plus_dm  > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr       = pd.concat([df['high'] - df['low'],
                           (df['high'] - df['close'].shift()).abs(),
                           (df['low']  - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr14    = tr.ewm(span=14, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['adx']      = dx.ewm(span=14, adjust=False).mean()
    df['plus_di']  = plus_di
    df['minus_di'] = minus_di

    # Crossover signals  (exact notebook)
    df['cross_up']   = (df['ema_50'] > df['ema_200']) & (df['ema_50'].shift() <= df['ema_200'].shift())
    df['cross_down'] = (df['ema_50'] < df['ema_200']) & (df['ema_50'].shift() >= df['ema_200'].shift())

    return df.dropna()

# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST  (exact copy of notebook run_backtest() — V2 params)
# ═══════════════════════════════════════════════════════════════════════════════
def run_backtest(df,
                 initial_capital = 10_000.0,
                 risk_per_trade  = 0.02,
                 atr_sl_mult     = 3.5,
                 commission      = 0.001,
                 min_ema_spread  = 0.03,   # % — notebook default
                 rsi_min         = 50.0,
                 rr_ratio        = 2.0,
                 adx_min         = 25.0,
                 trail_trigger   = 0.66,
                 signal_filter   = None,
                 label           = 'V2'):

    capital     = initial_capital
    position    = 0.0
    entry_price = 0.0
    entry_time  = None
    stop_loss   = 0.0
    take_profit = 0.0
    direction   = None
    sl_moved_be = False
    trades      = []
    equity_curve= []
    in_trade    = False

    for ts, row in df.iterrows():
        price = float(row['close'])

        # Mark-to-market equity  (matches notebook exactly)
        if direction == 'long':
            equity = capital + position * price
        elif direction == 'short':
            equity = capital + position * (entry_price - price)
        else:
            equity = capital
        equity_curve.append({'ts': int(ts.timestamp() * 1000), 'equity': round(equity, 2)})

        # ── Trail → breakeven ──
        if in_trade and not sl_moved_be:
            if direction == 'short':
                tp_dist       = entry_price - take_profit
                trigger_price = entry_price - tp_dist * trail_trigger
                if price <= trigger_price:
                    stop_loss   = entry_price
                    sl_moved_be = True
            elif direction == 'long':
                tp_dist       = take_profit - entry_price
                trigger_price = entry_price + tp_dist * trail_trigger
                if price >= trigger_price:
                    stop_loss   = entry_price
                    sl_moved_be = True

        # ── SL / TP check ──
        if in_trade:
            sl_hit = tp_hit = False
            if direction == 'long':
                sl_hit = price <= stop_loss
                tp_hit = price >= take_profit
            elif direction == 'short':
                sl_hit = price >= stop_loss
                tp_hit = price <= take_profit

            if sl_hit or tp_hit:
                reason = 'TAKE PROFIT' if tp_hit else ('BREAKEVEN' if sl_moved_be else 'STOP LOSS')
                if direction == 'long':
                    pnl = (price - entry_price) * position
                else:
                    pnl = (entry_price - price) * position
                comm    = price * position * commission
                capital = equity - comm
                trades.append({
                    'num':        len(trades) + 1,
                    'side':       direction,
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'exit_time':  ts.strftime('%Y-%m-%d %H:%M'),
                    'entry':      round(entry_price, 2),
                    'exit':       round(price, 2),
                    'sl':         round(stop_loss, 2),
                    'tp':         round(take_profit, 2),
                    'pnl':        round(pnl - comm, 2),
                    'reason':     reason,
                    'win':        (pnl - comm) > 0,
                    'duration_h': round((ts - entry_time).total_seconds() / 3600, 1),
                })
                position = 0.0; in_trade = False; direction = None; sl_moved_be = False

        # ── Golden Cross → LONG ──
        if bool(row['cross_up']):
            if in_trade and direction == 'short':   # close short first
                pnl = (entry_price - price) * position
                comm = price * position * commission
                capital = equity - comm
                trades.append({
                    'num': len(trades)+1, 'side': 'short',
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'exit_time':  ts.strftime('%Y-%m-%d %H:%M'),
                    'entry': round(entry_price,2), 'exit': round(price,2),
                    'sl': round(stop_loss,2), 'tp': round(take_profit,2),
                    'pnl': round(pnl-comm,2), 'reason': 'SIGNAL', 'win': (pnl-comm)>0,
                    'duration_h': round((ts-entry_time).total_seconds()/3600,1),
                })
                position=0.0; in_trade=False; direction=None; sl_moved_be=False

            spread_ok = float(row['ema_spread_abs']) >= min_ema_spread
            rsi_ok    = float(row['rsi'])            >= rsi_min
            adx_ok    = float(row['adx'])            >= adx_min
            ml_ok     = True if signal_filter is None else bool(signal_filter.get(ts, False))

            if spread_ok and rsi_ok and adx_ok and ml_ok and not in_trade:
                atr         = float(row['atr'])
                sl_dist     = atr * atr_sl_mult
                stop_loss   = price - sl_dist
                take_profit = price + sl_dist * rr_ratio
                risk_amt    = capital * risk_per_trade
                position    = min(risk_amt / sl_dist, (capital * 0.95) / price)
                cost        = position * price * (1 + commission)
                if cost <= capital and position > 0:
                    capital -= cost
                    entry_price = price; entry_time = ts
                    direction = 'long'; in_trade = True; sl_moved_be = False

        # ── Death Cross → SHORT ──
        elif bool(row['cross_down']):
            if in_trade and direction == 'long':   # close long first
                pnl = (price - entry_price) * position
                comm = price * position * commission
                capital = equity - comm
                trades.append({
                    'num': len(trades)+1, 'side': 'long',
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'exit_time':  ts.strftime('%Y-%m-%d %H:%M'),
                    'entry': round(entry_price,2), 'exit': round(price,2),
                    'sl': round(stop_loss,2), 'tp': round(take_profit,2),
                    'pnl': round(pnl-comm,2), 'reason': 'SIGNAL', 'win': (pnl-comm)>0,
                    'duration_h': round((ts-entry_time).total_seconds()/3600,1),
                })
                position=0.0; in_trade=False; direction=None; sl_moved_be=False

            spread_ok = float(row['ema_spread_abs']) >= min_ema_spread
            adx_ok    = float(row['adx'])            >= adx_min

            if spread_ok and adx_ok and not in_trade:
                atr         = float(row['atr'])
                sl_dist     = atr * atr_sl_mult
                stop_loss   = price + sl_dist
                take_profit = price - sl_dist * rr_ratio
                risk_amt    = capital * risk_per_trade
                position    = min(risk_amt / sl_dist, (capital * 0.95) / price)
                cost        = position * price * commission
                if capital > cost and position > 0:
                    capital -= cost
                    entry_price = price; entry_time = ts
                    direction = 'short'; in_trade = True; sl_moved_be = False

    # ── Close open trade at end ──
    if in_trade and equity_curve:
        price = float(df['close'].iloc[-1])
        pnl   = (price - entry_price) * position if direction == 'long' else (entry_price - price) * position
        comm  = price * position * commission
        capital = equity_curve[-1]['equity'] - comm
        trades.append({
            'num': len(trades)+1, 'side': direction,
            'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
            'exit_time':  df.index[-1].strftime('%Y-%m-%d %H:%M'),
            'entry': round(entry_price,2), 'exit': round(price,2),
            'sl': round(stop_loss,2), 'tp': round(take_profit,2),
            'pnl': round(pnl-comm,2), 'reason': 'END', 'win': (pnl-comm)>0,
            'duration_h': round((df.index[-1]-entry_time).total_seconds()/3600,1),
        })

    return trades, equity_curve, capital

# ═══════════════════════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════════════════════
def calc_stats(trades, equity_curve, start_capital=10_000.0):
    if not trades:
        return {k: 0 for k in ['total_trades','win_rate','total_pnl','total_return',
                                'sharpe','max_drawdown','profit_factor','avg_win',
                                'avg_loss','expectancy','take_profits','stop_losses',
                                'breakevens','final_capital']}
    pnls   = [t['pnl'] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    final  = equity_curve[-1]['equity'] if equity_curve else start_capital
    ret    = round((final - start_capital) / start_capital * 100, 2)
    eq     = pd.Series([e['equity'] for e in equity_curve])
    rets   = eq.pct_change().dropna()
    sharpe = round(rets.mean() / rets.std() * (8760**0.5), 2) if rets.std() > 0 else 0.0
    peak   = eq.cummax()
    mdd    = round(((eq - peak) / peak * 100).min(), 2)
    gw = sum(wins); gl = abs(sum(losses))
    return {
        'total_trades':  len(trades),
        'win_rate':      round(len(wins)/len(pnls)*100, 1),
        'total_pnl':     round(sum(pnls), 2),
        'total_return':  ret,
        'sharpe':        sharpe,
        'max_drawdown':  mdd,
        'profit_factor': round(gw/gl, 2) if gl > 0 else 0.0,
        'avg_win':       round(sum(wins)/len(wins),   2) if wins   else 0.0,
        'avg_loss':      round(sum(losses)/len(losses),2) if losses else 0.0,
        'expectancy':    round(sum(pnls)/len(pnls), 2),
        'take_profits':  sum(1 for t in trades if 'TAKE'  in t['reason']),
        'stop_losses':   sum(1 for t in trades if 'STOP'  in t['reason']),
        'breakevens':    sum(1 for t in trades if 'BREAK' in t['reason']),
        'final_capital': round(final, 2),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# CSV LOADER
# ═══════════════════════════════════════════════════════════════════════════════
BINANCE_COLS = ['open_time','open','high','low','close','volume',
                'close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']




def load_df_from_binance():
    """Fallback when CSV not found — used on hosted Render server"""
    import asyncio
    from realtime import fetch_historical
    df = asyncio.run(fetch_historical("BTCUSDT", "1h", days=180))
    return df

def load_df():
    path  = 'BTCusd_6month_1hr_data'
    files = sorted(glob.glob(os.path.join(path, '*.csv')))
    if not files:
        print("CSV not found — fetching from Binance API")
        return load_df_from_binance()
    # ... rest of existing load_df code

    
def load_df():
    path  = 'BTCusd_6month_1hr_data'
    files = sorted(glob.glob(os.path.join(path, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV in '{path}/'")
    dfs = []
    for f in files:
        peek = pd.read_csv(f, nrows=1, header=None)
        first = str(peek.iloc[0, 0]).strip()
        if first.isdigit() and int(first) > 1_000_000_000:
            tmp = pd.read_csv(f, header=None, names=BINANCE_COLS[:12])
        else:
            tmp = pd.read_csv(f)
            tmp.columns = BINANCE_COLS[:len(tmp.columns)]
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)
    sample = df['open_time'].iloc[0]
    try:
        s = float(sample)
        if   s > 1e15: df['open_time'] = pd.to_datetime(df['open_time'].astype(float), unit='us')
        elif s > 1e12: df['open_time'] = pd.to_datetime(df['open_time'].astype(float), unit='ms')
        else:          df['open_time'] = pd.to_datetime(df['open_time'].astype(float), unit='s')
    except:
        df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    df.sort_index(inplace=True)
    df = df[['open','high','low','close','volume']].astype(float)
    df = df[~df.index.duplicated(keep='first')]
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/backtest")
async def get_backtest():
    try:
        raw = load_df()
        df  = add_indicators(raw)

        # ── V2 exact notebook params ──────────────────────────────────────────
        trades, equity_curve, final_cap = run_backtest(
            df,
            initial_capital = 10_000.0,
            risk_per_trade  = 0.02,
            atr_sl_mult     = 3.5,
            commission      = 0.001,
            min_ema_spread  = 0.03,   # % — matches notebook exactly
            rsi_min         = 50.0,
            rr_ratio        = 2.0,
            adx_min         = 25.0,
            trail_trigger   = 0.66,
            label           = 'V2 ADX25+Trail'
        )
        stats = calc_stats(trades, equity_curve)

        # Downsample equity for chart (~1 pt/day)
        sampled_eq = equity_curve[::24] if len(equity_curve) > 48 else equity_curve
        if equity_curve:
            if not sampled_eq or sampled_eq[0] != equity_curve[0]:
                sampled_eq = [equity_curve[0]] + sampled_eq
            if sampled_eq[-1] != equity_curve[-1]:
                sampled_eq = sampled_eq + [equity_curve[-1]]

        # ── Chart data (last 150 candles) ─────────────────────────────────────
        subset = df.tail(150)
        ohlc_data, ema50_data, ema200_data = [], [], []
        rsi_data, bb_upper_data, bb_lower_data = [], [], []

        for ts, row in subset.iterrows():
            t = int(ts.timestamp() * 1000)
            ohlc_data.append({"x": t, "y": [round(row['open'],2), round(row['high'],2),
                                              round(row['low'],2),  round(row['close'],2)]})
            ema50_data.append( {"x": t, "y": round(float(row['ema_50']),  2)})
            ema200_data.append({"x": t, "y": round(float(row['ema_200']), 2)})
            rsi_data.append(   {"x": t, "y": round(float(row['rsi']),     2)})
            if not pd.isna(row['bb_upper']):
                bb_upper_data.append({"x": t, "y": round(float(row['bb_upper']), 2)})
                bb_lower_data.append({"x": t, "y": round(float(row['bb_lower']), 2)})

        # Price stats
        last_close = round(float(df['close'].iloc[-1]), 2)
        prev_close = round(float(df['close'].iloc[-2]), 2)

        return {
            "price":         last_close,
            "pct_change":    round((last_close - prev_close) / prev_close * 100, 2),
            "high_24h":      round(float(df.tail(24)['high'].max()), 2),
            "low_24h":       round(float(df.tail(24)['low'].min()),  2),
            "volume_24h":    round(float(df.tail(24)['volume'].sum()), 2),
            "rsi":           round(float(df['rsi'].iloc[-1]), 2),
            "chart_data":    ohlc_data,
            "ema50_data":    ema50_data,
            "ema200_data":   ema200_data,
            "rsi_data":      rsi_data,
            "bb_upper_data": bb_upper_data,
            "bb_lower_data": bb_lower_data,
            "v2_stats":      stats,
            "v2_trades":     trades,
            "v2_equity":     [{"x": e["ts"], "y": e["equity"]} for e in sampled_eq],
            "v2_pnl":        stats['total_pnl'],
            "trades":        trades,
            "debug": {
                "total_candles":  len(df),
                "date_range":     f"{df.index[0]} → {df.index[-1]}",
                "trades_found":   len(trades),
                "params": {
                    "min_ema_spread_pct": 0.03,
                    "adx_min": 25.0,
                    "rsi_min": 50.0,
                    "atr_mult": 3.5,
                    "rr_ratio": 2.0,
                    "trail_trigger": 0.66
                }
            }
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}


@app.get("/api/debug")
async def debug():
    try:
        raw = load_df()
        df  = add_indicators(raw)
        crosses_down = df[df['cross_down']]
        signals = []
        for ts, row in crosses_down.iterrows():
            spread_ok = float(row['ema_spread_abs']) >= 0.03
            adx_ok    = float(row['adx'])            >= 25.0
            signals.append({
                "time":       str(ts),
                "spread_pct": round(float(row['ema_spread_abs']), 4),
                "adx":        round(float(row['adx']), 2),
                "rsi":        round(float(row['rsi']), 2),
                "spread_ok":  spread_ok,
                "adx_ok":     adx_ok,
                "passes":     spread_ok and adx_ok,
            })
        return {
            "total_candles":   len(df),
            "date_range":      f"{df.index[0]} → {df.index[-1]}",
            "death_crosses":   len(crosses_down),
            "signals_passing": sum(1 for s in signals if s['passes']),
            "all_signals":     signals
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}



# ═══════════════════════════════════════════════════════════════════════════════
# AMD STRATEGY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from amd_strategy import run_full_amd_backtest, detect_live_amd_signal

    @app.get("/api/amd/backtest")
    async def amd_backtest(
        symbol:          str   = "ETHUSDT",
        days:            int   = 180,
        range_lookback:  int   = 20,
        max_range_pct:   float = 3.0,
        min_range_pct:   float = 0.5,
        sweep_buffer:    float = 0.001,
        initial_capital: float = 10_000.0,
        risk_per_trade:  float = 0.02,
        rr_ratio:        float = 2.0,
    ):
        """
        Run AMD backtest on Binance historical data.
        Fetches data live — no CSV needed.
        Example: /api/amd/backtest?symbol=ETHUSDT&days=180
        """
        result = await run_full_amd_backtest(
            symbol=symbol, days=days,
            range_lookback=range_lookback,
            max_range_pct=max_range_pct,
            min_range_pct=min_range_pct,
            sweep_buffer=sweep_buffer,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            rr_ratio=rr_ratio,
        )
        return result

    @app.get("/api/amd/signal")
    async def amd_signal(
        symbol:         str   = "ETHUSDT",
        range_lookback: int   = 20,
        max_range_pct:  float = 3.0,
        sweep_buffer:   float = 0.001,
        rr_ratio:       float = 2.0,
    ):
        """
        Live AMD signal: current phase + sweep alert if detected.
        Phase: Accumulation → Manipulation → Distribution
        """
        return await detect_live_amd_signal(
            symbol=symbol,
            range_lookback=range_lookback,
            max_range_pct=max_range_pct,
            sweep_buffer=sweep_buffer,
            rr_ratio=rr_ratio,
        )

    print("[AMD] Strategy endpoints mounted: /api/amd/backtest and /api/amd/signal")

except ImportError as e:
    print(f"[AMD] Not available: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload

# python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload


    

# python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload




# Run with:
#   python -m uvicorn backend:app --reload