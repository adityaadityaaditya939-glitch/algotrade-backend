"""
realtime.py — Real-time market data for AlgoTrade Pro

Features:
  1. Binance WebSocket → live price ticker for all pairs
  2. Binance REST API  → fetch OHLCV candles (last 500 1h candles)
  3. Live signal detector → runs EMA 50/200 strategy on live data
  4. FastAPI WebSocket endpoint → streams live data to frontend

No API key needed — uses Binance public endpoints only.
"""

import asyncio
import json
import time
import threading
from datetime import datetime
from typing import Dict, Set
import websockets
import httpx
import pandas as pd
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

# ─── Binance public endpoints ─────────────────────────────────────────────────
BINANCE_WS_BASE   = "wss://stream.binance.com:9443/stream"
BINANCE_REST_BASE = "https://api.binance.com"

# ─── Pairs to track ───────────────────────────────────────────────────────────
TRACKED_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
INTERVAL      = "1h"    # candle interval for strategy
CANDLE_LIMIT  = 500     # how many candles to fetch for indicators

# ─── Strategy params (same as backtest V2) ───────────────────────────────────
STRATEGY_PARAMS = {
    "min_ema_spread": 0.03,
    "adx_min":        25.0,
    "rsi_min":        50.0,
    "atr_mult":       3.5,
    "rr_ratio":       2.0,
    "trail_trigger":  0.66,
}

# ═══════════════════════════════════════════════════════════════════════════════
# LIVE PRICE STORE  (shared state — updated by WebSocket thread)
# ═══════════════════════════════════════════════════════════════════════════════
class LivePriceStore:
    def __init__(self):
        self.prices: Dict[str, dict] = {}   # symbol → {price, change, high, low, volume}
        self.lock = threading.Lock()

    def update(self, symbol: str, data: dict):
        with self.lock:
            self.prices[symbol] = data

    def get_all(self) -> dict:
        with self.lock:
            return dict(self.prices)

    def get(self, symbol: str) -> dict:
        with self.lock:
            return self.prices.get(symbol, {})

live_prices = LivePriceStore()

# ═══════════════════════════════════════════════════════════════════════════════
# CANDLE CACHE  (updated every 30s by background task)
# ═══════════════════════════════════════════════════════════════════════════════
class CandleCache:
    def __init__(self):
        self.candles: Dict[str, pd.DataFrame] = {}
        self.signals: Dict[str, dict]         = {}
        self.last_update: Dict[str, float]    = {}
        self.lock = threading.Lock()

    def set(self, symbol: str, df: pd.DataFrame, signal: dict):
        with self.lock:
            self.candles[symbol]     = df
            self.signals[symbol]     = signal
            self.last_update[symbol] = time.time()

    def get_df(self, symbol: str) -> pd.DataFrame:
        with self.lock:
            return self.candles.get(symbol, pd.DataFrame())

    def get_signal(self, symbol: str) -> dict:
        with self.lock:
            return self.signals.get(symbol, {})

    def is_stale(self, symbol: str, max_age_secs=60) -> bool:
        with self.lock:
            last = self.last_update.get(symbol, 0)
            return (time.time() - last) > max_age_secs

candle_cache = CandleCache()

# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS  (same as backtest — ensuring consistency)
# ═══════════════════════════════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_50']  = df['close'].ewm(span=50,  adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema_spread']     = (df['ema_50'] - df['ema_200']) / df['ema_200'] * 100
    df['ema_spread_abs'] = df['ema_spread'].abs()

    delta = df['close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    rs    = gain.ewm(span=14, adjust=False).mean() / loss.ewm(span=14, adjust=False).mean().replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    hl  = df['high'] - df['low']
    hcp = (df['high'] - df['close'].shift()).abs()
    lcp = (df['low']  - df['close'].shift()).abs()
    df['atr'] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).ewm(span=14, adjust=False).mean()

    up       = df['high'].diff()
    down     = -df['low'].diff()
    plus_dm  = up.where((up > down)   & (up > 0),   0.0)
    minus_dm = down.where((down > up) & (down > 0),  0.0)
    tr       = pd.concat([df['high']-df['low'],
                           (df['high']-df['close'].shift()).abs(),
                           (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    atr14    = tr.ewm(span=14, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['adx'] = dx.ewm(span=14, adjust=False).mean()

    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20

    df['macd']      = df['close'].ewm(span=12,adjust=False).mean() - df['close'].ewm(span=26,adjust=False).mean()
    df['macd_sig']  = df['macd'].ewm(span=9, adjust=False).mean()

    df['cross_up']   = (df['ema_50'] > df['ema_200']) & (df['ema_50'].shift() <= df['ema_200'].shift())
    df['cross_down'] = (df['ema_50'] < df['ema_200']) & (df['ema_50'].shift() >= df['ema_200'].shift())

    return df.dropna()

# ═══════════════════════════════════════════════════════════════════════════════
# LIVE SIGNAL DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
def detect_live_signal(df: pd.DataFrame, current_price: float) -> dict:
    """
    Run EMA 50/200 strategy on live candles.
    Returns current signal state + alert if new crossover just happened.
    """
    if df.empty or len(df) < 210:
        return {"signal": "NEUTRAL", "alert": False}

    p = STRATEGY_PARAMS
    last = df.iloc[-1]
    prev = df.iloc[-2]

    ema50_now  = float(last['ema_50'])
    ema200_now = float(last['ema_200'])
    rsi_now    = float(last['rsi'])
    adx_now    = float(last['adx'])
    atr_now    = float(last['atr'])
    spread_now = float(last['ema_spread_abs'])

    # Trend direction
    trend = "BEARISH" if ema50_now < ema200_now else "BULLISH"

    # Check for fresh crossover on last candle
    death_cross  = bool(last['cross_down'])
    golden_cross = bool(last['cross_up'])

    spread_ok = spread_now >= p['min_ema_spread']
    adx_ok    = adx_now    >= p['adx_min']
    rsi_ok    = rsi_now    >= p['rsi_min']

    # Live SHORT signal (Death Cross + filters pass)
    if death_cross and spread_ok and adx_ok:
        sl_dist = atr_now * p['atr_mult']
        return {
            "signal":      "SHORT",
            "alert":       True,
            "alert_type":  "DEATH_CROSS",
            "strength":    "STRONG" if adx_now > 35 else "MODERATE",
            "trend":       trend,
            "ema50":       round(ema50_now, 2),
            "ema200":      round(ema200_now, 2),
            "rsi":         round(rsi_now, 2),
            "adx":         round(adx_now, 2),
            "spread_pct":  round(spread_now, 4),
            "entry":       round(current_price, 2),
            "stop_loss":   round(current_price + sl_dist, 2),
            "take_profit": round(current_price - sl_dist * p['rr_ratio'], 2),
            "filters":     {"spread_ok": spread_ok, "adx_ok": adx_ok, "rsi_ok": rsi_ok},
            "timestamp":   datetime.utcnow().isoformat(),
        }

    # Live LONG signal (Golden Cross + filters pass)
    if golden_cross and spread_ok and adx_ok and rsi_ok:
        sl_dist = atr_now * p['atr_mult']
        return {
            "signal":      "LONG",
            "alert":       True,
            "alert_type":  "GOLDEN_CROSS",
            "strength":    "STRONG" if adx_now > 35 else "MODERATE",
            "trend":       trend,
            "ema50":       round(ema50_now, 2),
            "ema200":      round(ema200_now, 2),
            "rsi":         round(rsi_now, 2),
            "adx":         round(adx_now, 2),
            "spread_pct":  round(spread_now, 4),
            "entry":       round(current_price, 2),
            "stop_loss":   round(current_price - sl_dist, 2),
            "take_profit": round(current_price + sl_dist * p['rr_ratio'], 2),
            "filters":     {"spread_ok": spread_ok, "adx_ok": adx_ok, "rsi_ok": rsi_ok},
            "timestamp":   datetime.utcnow().isoformat(),
        }

    # No fresh signal — return current market state
    return {
        "signal":     "SHORT_BIAS" if trend == "BEARISH" else "LONG_BIAS",
        "alert":      False,
        "trend":      trend,
        "ema50":      round(ema50_now, 2),
        "ema200":     round(ema200_now, 2),
        "rsi":        round(rsi_now, 2),
        "adx":        round(adx_now, 2),
        "spread_pct": round(spread_now, 4),
        "filters":    {"spread_ok": spread_ok, "adx_ok": adx_ok, "rsi_ok": rsi_ok},
        "timestamp":  datetime.utcnow().isoformat(),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# BINANCE REST — fetch candles
# ═══════════════════════════════════════════════════════════════════════════════
async def fetch_candles(symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    url = f"{BINANCE_REST_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        df = pd.DataFrame(data, columns=[
            'open_time','open','high','low','close','volume',
            'close_time','quote_vol','trades','taker_base','taker_quote','ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        return df[['open','high','low','close','volume']]
    except Exception as e:
        print(f"[REST] Error fetching {symbol}: {e}")
        return pd.DataFrame()

async def fetch_orderbook_ticker(symbol: str) -> dict:
    """Get best bid/ask for spread display."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{BINANCE_REST_BASE}/api/v3/ticker/bookTicker",
                                  params={"symbol": symbol})
            return r.json()
    except:
        return {}

# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND CANDLE REFRESH TASK
# Runs every 60 seconds — fetches latest candles + runs signal detection
# ═══════════════════════════════════════════════════════════════════════════════
async def candle_refresh_loop():
    print("[Realtime] Candle refresh loop started")
    while True:
        for symbol in TRACKED_PAIRS:
            try:
                df = await fetch_candles(symbol, INTERVAL, CANDLE_LIMIT)
                if not df.empty:
                    df_ind   = add_indicators(df)
                    price    = live_prices.get(symbol).get("price", float(df['close'].iloc[-1]))
                    signal   = detect_live_signal(df_ind, price)
                    candle_cache.set(symbol, df_ind, signal)

                    if signal.get("alert"):
                        print(f"[SIGNAL] {symbol} → {signal['signal']} @ ${signal.get('entry')} | {signal['alert_type']}")

            except Exception as e:
                print(f"[Candle refresh] {symbol} error: {e}")

            await asyncio.sleep(1)   # small gap between symbols

        await asyncio.sleep(55)      # refresh every ~60s

# ═══════════════════════════════════════════════════════════════════════════════
# BINANCE WEBSOCKET — live price ticker
# Subscribes to mini-ticker for all tracked pairs simultaneously
# ═══════════════════════════════════════════════════════════════════════════════
async def binance_ws_price_loop():
    """
    Connects to Binance combined stream for 24h ticker.
    Updates live_prices store in real time.
    Auto-reconnects on disconnect.
    """
    streams = "/".join([f"{s.lower()}@ticker" for s in TRACKED_PAIRS])
    url     = f"{BINANCE_WS_BASE}?streams={streams}"

    while True:
        try:
            print(f"[WS] Connecting to Binance: {url}")
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                print("[WS] Connected ✅")
                async for raw in ws:
                    try:
                        msg  = json.loads(raw)
                        data = msg.get("data", {})
                        sym  = data.get("s", "")
                        if sym in TRACKED_PAIRS:
                            live_prices.update(sym, {
                                "price":      round(float(data.get("c", 0)), 2),
                                "change_pct": round(float(data.get("P", 0)), 2),
                                "high_24h":   round(float(data.get("h", 0)), 2),
                                "low_24h":    round(float(data.get("l", 0)), 2),
                                "volume_24h": round(float(data.get("v", 0)), 2),
                                "open":       round(float(data.get("o", 0)), 2),
                                "ts":         int(data.get("E", 0)),
                            })
                    except Exception as e:
                        print(f"[WS] Parse error: {e}")

        except Exception as e:
            print(f"[WS] Disconnected: {e} — reconnecting in 5s...")
            await asyncio.sleep(5)

# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET CONNECTION MANAGER  (frontend clients)
# ═══════════════════════════════════════════════════════════════════════════════
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        print(f"[WS] Client connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        print(f"[WS] Client disconnected. Total: {len(self.active)}")

    async def broadcast(self, data: dict):
        if not self.active:
            return
        msg  = json.dumps(data)
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except:
                dead.add(ws)
        self.active -= dead

ws_manager = ConnectionManager()

# ═══════════════════════════════════════════════════════════════════════════════
# BROADCAST LOOP  — pushes live data to all connected frontend clients
# Runs every 1 second
# ═══════════════════════════════════════════════════════════════════════════════
async def broadcast_loop():
    print("[Broadcast] Loop started")
    while True:
        if ws_manager.active:
            prices  = live_prices.get_all()
            signals = {sym: candle_cache.get_signal(sym) for sym in TRACKED_PAIRS}

            # Build chart update for BTC (last 5 candles for live chart update)
            btc_df = candle_cache.get_df("BTCUSDT")
            chart_update = []
            if not btc_df.empty:
                last5 = btc_df.tail(5)
                for ts, row in last5.iterrows():
                    t = int(ts.timestamp() * 1000)
                    chart_update.append({
                        "x": t,
                        "y": [round(row['open'],2), round(row['high'],2),
                               round(row['low'],2),  round(row['close'],2)]
                    })

            payload = {
                "type":         "live_update",
                "ts":           int(time.time() * 1000),
                "prices":       prices,
                "signals":      signals,
                "chart_update": chart_update,
            }
            await ws_manager.broadcast(payload)

        await asyncio.sleep(1)

# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP — register all background tasks with FastAPI
# ═══════════════════════════════════════════════════════════════════════════════
async def start_realtime_tasks():
    """Call this from FastAPI startup event."""
    asyncio.create_task(binance_ws_price_loop())
    asyncio.create_task(candle_refresh_loop())
    asyncio.create_task(broadcast_loop())
    print("[Realtime] All tasks started ✅")

# ═══════════════════════════════════════════════════════════════════════════════
# REST HELPER — get formatted live candles for a symbol (used by /api/live/candles)
# ═══════════════════════════════════════════════════════════════════════════════
def get_live_chart_data(symbol: str = "BTCUSDT", candles: int = 150) -> dict:
    df = candle_cache.get_df(symbol)
    if df.empty:
        return {"error": "No data yet — candles loading"}

    subset = df.tail(candles)
    ohlc, ema50, ema200, rsi_d, bbu, bbl = [], [], [], [], [], []

    for ts, row in subset.iterrows():
        t = int(ts.timestamp() * 1000)
        ohlc.append({"x": t, "y": [round(row['open'],2), round(row['high'],2),
                                     round(row['low'],2),  round(row['close'],2)]})
        ema50.append( {"x": t, "y": round(float(row['ema_50']),  2)})
        ema200.append({"x": t, "y": round(float(row['ema_200']), 2)})
        rsi_d.append( {"x": t, "y": round(float(row['rsi']),     2)})
        if not pd.isna(row['bb_upper']):
            bbu.append({"x": t, "y": round(float(row['bb_upper']), 2)})
            bbl.append({"x": t, "y": round(float(row['bb_lower']), 2)})

    last    = df.iloc[-1]
    price   = live_prices.get(symbol).get("price", round(float(last['close']), 2))
    signal  = candle_cache.get_signal(symbol)

    return {
        "symbol":        symbol,
        "price":         price,
        "rsi":           round(float(last['rsi']), 2),
        "ema50":         round(float(last['ema_50']), 2),
        "ema200":        round(float(last['ema_200']), 2),
        "adx":           round(float(last['adx']), 2),
        "signal":        signal,
        "chart_data":    ohlc,
        "ema50_data":    ema50,
        "ema200_data":   ema200,
        "rsi_data":      rsi_d,
        "bb_upper_data": bbu,
        "bb_lower_data": bbl,
        "candle_count":  len(subset),
        "last_updated":  df.index[-1].isoformat(),
    }