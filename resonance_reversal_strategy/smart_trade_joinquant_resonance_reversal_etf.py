from jqdata import *

import hashlib
import json
from enum import Enum


STRATEGY_VERSION = "resonance-v0.1.0"
DEPLOYMENT_BUILD_ID = "20260827.1"
BENCHMARK = "000300.XSHG"


class TurnDirection(Enum):
    BUY_TURN = "BUY_TURN"
    SELL_TURN = "SELL_TURN"
    NEUTRAL = "NEUTRAL"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class ExitReason(Enum):
    ATR_EXIT = "ATR_EXIT"
    SIGNAL_EXIT = "SIGNAL_EXIT"


class Tradability(Enum):
    TRADEABLE = "TRADEABLE"
    PAUSED = "PAUSED"
    UNKNOWN = "UNKNOWN"


class OrderOutcome(Enum):
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    NOT_FILLED = "NOT_FILLED"
    PAUSED = "PAUSED"
    UNKNOWN = "UNKNOWN"


def get_default_etf_pool():
    return [
        "510300.XSHG", "159915.XSHE", "512100.XSHG", "159928.XSHE",
        "510880.XSHG", "513100.XSHG", "513500.XSHG", "159920.XSHE",
        "513880.XSHG", "513050.XSHG", "518880.XSHG", "159985.XSHE",
    ]


def get_default_params():
    return {
        "lookback_days": 120,
        "max_holdings": 3,
        "target_exposure": 0.95,
        "resonance_window": 2,
        "rsi_period": 14,
        "observation_rsi_periods": (6, 12, 24),
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "kdj": (9, 3, 3),
        "kdj_low": 20.0,
        "kdj_high": 80.0,
        "j_low": 0.0,
        "j_high": 100.0,
        "boll": (20, 2.0),
        "atr_period": 14,
        "atr_multiplier": 2.5,
        "stop_floor": 0.05,
        "stop_cap": 0.15,
    }


def business_config_fingerprint(params=None, etf_pool=None):
    payload = {
        "params": params or get_default_params(),
        "etf_pool": etf_pool or get_default_etf_pool(),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def ensure_runtime_state():
    if not hasattr(g, "params"):
        g.params = get_default_params()
    if not hasattr(g, "etf_pool"):
        g.etf_pool = get_default_etf_pool()
    if not hasattr(g, "position_states"):
        g.position_states = {}
    if not hasattr(g, "processed_resonance_ids"):
        g.processed_resonance_ids = {}
    if not hasattr(g, "observation_events"):
        g.observation_events = {}
    if not hasattr(g, "sold_today"):
        g.sold_today = set()
    if not hasattr(g, "daily_attempted_buys"):
        g.daily_attempted_buys = set()


def do_trading(context):
    ensure_runtime_state()


def after_close(context):
    ensure_runtime_state()


def initialize(context):
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_benchmark(BENCHMARK)
    set_slippage(PriceRelatedSlippage(0.001), type="fund")
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0,
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5,
    ), type="fund")
    run_daily(do_trading, time="09:35", reference_security=BENCHMARK)
    run_daily(after_close, time="15:30", reference_security=BENCHMARK)
    ensure_runtime_state()
    log.info("version=%s build=%s fingerprint=%s pool=%s",
             STRATEGY_VERSION, DEPLOYMENT_BUILD_ID,
             business_config_fingerprint(), get_default_etf_pool())


import numpy as np
import pandas as pd


TRADE_INDICATOR_COLUMNS = (
    "rsi14", "k", "d", "j", "kd_diff", "boll_mid",
    "boll_upper", "boll_lower", "atr14",
)
OBSERVATION_COLUMNS = (
    "rsi6", "rsi12", "rsi24", "plus_di", "minus_di", "adx14",
    "volume", "volume_ma5", "volume_ma20", "volume_ratio",
    "boll_width", "boll_mid_slope",
)


def calc_rsi(close, period):
    close = pd.Series(close, dtype=float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False,
                        min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False,
                        min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100.0 - 100.0 / (1.0 + rs)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return result


def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    rolling_high = high.rolling(n, min_periods=n).max()
    rolling_low = low.rolling(n, min_periods=n).min()
    spread = rolling_high - rolling_low
    rsv = 100.0 * (close - rolling_low) / spread.replace(0, np.nan)
    rsv = rsv.mask(spread == 0, 50.0)
    k = rsv.ewm(alpha=1.0 / m1, adjust=False, min_periods=1).mean()
    d = k.ewm(alpha=1.0 / m2, adjust=False, min_periods=1).mean()
    j = 3.0 * k - 2.0 * d
    return k, d, j


def calc_bollinger(close, period=20, std_mult=2.0):
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return mid, mid + std_mult * std, mid - std_mult * std


def true_range(high, low, close):
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def calc_atr(high, low, close, period=14):
    return true_range(high, low, close).rolling(period, min_periods=period).mean()


def calc_dmi_adx(high, low, close, period=14):
    tr = true_range(high, low, close)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where(
        (up_move > down_move) & (up_move > 0), up_move, 0.0,
    ), index=high.index)
    minus_dm = pd.Series(np.where(
        (down_move > up_move) & (down_move > 0), down_move, 0.0,
    ), index=high.index)
    atr_rma = tr.ewm(alpha=1.0 / period, adjust=False,
                     min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period,
    ).mean() / atr_rma
    minus_di = 100.0 * minus_dm.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period,
    ).mean() / atr_rma
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    adx = dx.ewm(alpha=1.0 / period, adjust=False,
                min_periods=period).mean()
    return plus_di, minus_di, adx


def build_indicator_frame(price_frame, params):
    frame = price_frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    frame["rsi14"] = calc_rsi(frame["close"], params["rsi_period"])
    for period in params["observation_rsi_periods"]:
        frame["rsi%s" % period] = calc_rsi(frame["close"], period)
    k, d, j = calc_kdj(frame["high"], frame["low"], frame["close"], *params["kdj"])
    frame["k"], frame["d"], frame["j"] = k, d, j
    frame["kd_diff"] = k - d
    mid, upper, lower = calc_bollinger(frame["close"], *params["boll"])
    frame["boll_mid"], frame["boll_upper"], frame["boll_lower"] = mid, upper, lower
    frame["atr14"] = calc_atr(
        frame["high"], frame["low"], frame["close"], params["atr_period"],
    )
    plus_di, minus_di, adx = calc_dmi_adx(
        frame["high"], frame["low"], frame["close"], 14,
    )
    frame["plus_di"], frame["minus_di"], frame["adx14"] = plus_di, minus_di, adx
    frame["volume_ma5"] = frame["volume"].rolling(5, min_periods=5).mean()
    frame["volume_ma20"] = frame["volume"].rolling(20, min_periods=20).mean()
    frame["volume_ratio"] = frame["volume"] / frame["volume_ma20"].replace(0, np.nan)
    frame["boll_width"] = (upper - lower) / mid.replace(0, np.nan)
    frame["boll_mid_slope"] = mid.diff()
    return frame
