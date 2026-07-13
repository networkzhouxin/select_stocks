# -*- coding: utf-8 -*-
"""Cross-Signal ETF Strategy v0.3.2 for Guojin PTrade.

Business rules are frozen to the JoinQuant v0.3.2 mainline. Only platform,
live-order, restart-recovery, and halted-security handling differ. PTrade
backtests are smoke tests; JoinQuant remains the performance authority.
"""

import numpy as np
import pandas as pd
import builtins as _builtins
import hashlib
import os
import pickle
from datetime import datetime


STRATEGY_VERSION = "cross-v0.3.2"
IOPV_OBSERVE_CODES = frozenset((
    "513100.SS",
    "513500.SS",
    "513880.SS",
    "513050.SS",
))
LIVE_STATE_FILENAME = "cross_signal_v032_live_state_%s.pkl"
DELIVER_RECOVERY_START_DATE = "20100101"
LIVE_STATE_FIELDS = (
    "highest_since_buy",
    "entry_atr",
    "buy_date",
    "last_scores",
    "sold_today",
    "paused_pool_codes",
    "unverified_positions",
    "execution_date",
    "deferred_scores",
    "deferred_signal_date",
)


try:
    log
except NameError:
    class _LocalLog(object):
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    log = _LocalLog()


def get_default_params():
    return {
        "lookback": 120,
        "rebalance_weekdays": [0, 1, 2, 3, 4],
        "max_hold": 3,
        "base_ratio": 0.95,
        "min_signal_hold_days": 5,
        "buy_threshold": 60,
        "strong_buy_threshold": 70,
        "sell_threshold": 30,
        "risk_tighten_threshold": 18,
        "cross_window": 3,
        "rsi_fast": 6,
        "rsi_mid": 12,
        "rsi_slow": 24,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "kdj_n": 9,
        "kdj_m1": 3,
        "kdj_m2": 3,
        "boll_period": 20,
        "boll_std": 2.0,
        "atr_period": 14,
        "adx_period": 14,
        "adx_trend_threshold": 25,
        "trailing_atr_mult": 2.5,
        "stop_floor": 0.05,
        "stop_cap": 0.15,
        "overheat_rsi": 85,
        "a_share_zero_volume_buy_scale": 0.50,
    }


def get_default_etf_pool():
    return [
        "159915.SZ",
        "512100.SS",
        "159928.SZ",
        "513100.SS",
        "513500.SS",
        "513880.SS",
        "513050.SS",
        "518880.SS",
        "159985.SZ",
    ]


def _lock_frozen_business_config():
    """Reassert code-owned strategy configuration after PTrade restores g."""
    g.params = get_default_params()
    g.etf_pool = get_default_etf_pool()
    try:
        set_universe(g.etf_pool)
    except Exception as exc:
        log.warning("[config-lock] set_universe failed: %s" % exc)


def _live_state_path(path=None):
    if path is not None:
        return os.fspath(path)
    try:
        root = get_research_path()
    except Exception as exc:
        log.warning("[state] research path unavailable: %s" % exc)
        return None
    if not root:
        log.warning("[state] research path is empty")
        return None
    identity = []
    for getter_name in ("get_user_name", "get_trade_name"):
        getter = globals().get(getter_name)
        if getter is None:
            continue
        try:
            value = getter()
            if value not in (None, ""):
                identity.append(str(value))
        except Exception as exc:
            log.warning("[state] %s unavailable: %s" % (getter_name, exc))
    if not identity:
        log.error("[state] account/trade identity unavailable; checkpoint disabled")
        return None
    identity_text = "|".join(identity)
    identity_hash = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()[:12]
    return os.path.join(str(root), LIVE_STATE_FILENAME % identity_hash)


def _persist_live_state(path=None):
    state_path = _live_state_path(path)
    if state_path is None:
        return False
    payload = {
        "strategy_version": STRATEGY_VERSION,
        "state": {
            field: getattr(g, field, None)
            for field in LIVE_STATE_FIELDS
        },
    }
    temp_path = state_path + ".tmp"
    try:
        with open(temp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, state_path)
        return True
    except Exception as exc:
        log.error("[state] persist failed: %s" % exc)
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return False


def _validated_live_state(state):
    if not isinstance(state, dict):
        raise ValueError("invalid state body")
    missing = [field for field in LIVE_STATE_FIELDS if field not in state]
    if missing:
        raise ValueError("missing state fields: %s" % ",".join(missing))
    mapping_fields = (
        "highest_since_buy",
        "entry_atr",
        "buy_date",
        "last_scores",
        "sold_today",
    )
    for field in mapping_fields:
        if not isinstance(state[field], dict):
            raise ValueError("invalid mapping field: %s" % field)
    for field in ("paused_pool_codes", "unverified_positions"):
        if not isinstance(state[field], set):
            raise ValueError("invalid set field: %s" % field)
    if not isinstance(state["deferred_scores"], list):
        raise ValueError("invalid deferred scores")

    validated = dict(state)
    for field in ("execution_date", "deferred_signal_date"):
        value = state[field]
        normalized = _as_date(value) if value is not None else None
        if value is not None and normalized is None:
            raise ValueError("invalid date field: %s" % field)
        validated[field] = normalized
    return validated


def _restore_live_state(path=None):
    state_path = _live_state_path(path)
    if state_path is None or not os.path.exists(state_path):
        return False
    try:
        with open(state_path, "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("invalid state payload")
        if payload.get("strategy_version") != STRATEGY_VERSION:
            raise ValueError("strategy version mismatch")
        state = _validated_live_state(payload.get("state"))
        for field in LIVE_STATE_FIELDS:
            setattr(g, field, state[field])
        return True
    except Exception as exc:
        log.error("[state] restore failed: %s" % exc)
        return False


def get_a_share_etf_codes():
    return set([
        "510300",
        "159915",
        "512100",
        "159928",
        "510880",
    ])


def buy_position_scale(score, params=None):
    p = params or get_default_params()
    code = str(score.get("code", "")).split(".")[0]
    if code in get_a_share_etf_codes() and score.get("volume_score", 0) <= 0:
        scale = float(p.get("a_share_zero_volume_buy_scale", 1.0))
        return max(0.0, min(1.0, scale))
    return 1.0


def calc_buy_target_value(total_value, score, params=None):
    p = params or get_default_params()
    base_target = float(total_value) * float(p["base_ratio"]) / int(p["max_hold"])
    return base_target * buy_position_scale(score, p)


def format_indicator_params(params):
    return (
        "RSI(%d,%d,%d) MACD(%d,%d,%d) KDJ(%d,%d,%d) BOLL(%d,%.1f) "
        "ATR(%d) ADX(%d) MA(5,10,20,60)" % (
            params["rsi_fast"], params["rsi_mid"], params["rsi_slow"],
            params["macd_fast"], params["macd_slow"], params["macd_signal"],
            params["kdj_n"], params["kdj_m1"], params["kdj_m2"],
            params["boll_period"], params["boll_std"],
            params["atr_period"], params["adx_period"],
        )
    )


def format_self_check():
    fast = pd.Series([40.0, 41.0, 42.0, 45.9, 48.1])
    slow = pd.Series([42.0, 42.0, 42.0, 50.5, 46.2])
    diff_cross_ok = crossed_above_recent(fast, slow, window=3)
    score = score_buy_snapshot({
        "rsi6": 48.1,
        "rsi6_cross_rsi12_up": False,
        "rsi6_cross_rsi24_up": diff_cross_ok,
        "macd_cross_up": False,
        "kdj_k_cross_up": False,
        "kdj_j_cross_up": False,
    })
    return (
        "[%s] positional-diff-cross enabled | "
        "diff_cross_self_check=%s expected=True | self_rev=%.0f" % (
            STRATEGY_VERSION, diff_cross_ok, score["reversal_score"])
    )


def format_indicator_values(item):
    rsi_diff_12 = item.get("rsi6", np.nan) - item.get("rsi12", np.nan)
    rsi_diff_24 = item.get("rsi6", np.nan) - item.get("rsi24", np.nan)
    rsi_diff_12_prev = item.get("rsi6_prev", np.nan) - item.get("rsi12_prev", np.nan)
    rsi_diff_24_prev = item.get("rsi6_prev", np.nan) - item.get("rsi24_prev", np.nan)
    macd_diff = item.get("dif", np.nan) - item.get("dea", np.nan)
    macd_diff_prev = item.get("dif_prev", np.nan) - item.get("dea_prev", np.nan)
    kdj_diff_k = item.get("k", np.nan) - item.get("d", np.nan)
    kdj_diff_j = item.get("j", np.nan) - item.get("d", np.nan)
    kdj_diff_k_prev = item.get("k_prev", np.nan) - item.get("d_prev", np.nan)
    kdj_diff_j_prev = item.get("j_prev", np.nan) - item.get("d_prev", np.nan)
    return (
        "RSI[6/12/24]=%.1f/%.1f/%.1f "
        "MACD[DIF/DEA/HIST]=%.4f/%.4f/%.4f "
        "KDJ[K/D/J]=%.1f/%.1f/%.1f "
        "BOLL[U/M/L]=%.3f/%.3f/%.3f "
        "MA[5/10/20/60]=%.3f/%.3f/%.3f/%.3f "
        "VOL[5/20]=%.0f/%.0f "
        "ATR14=%.4f "
        "DMI[+DI/-DI/ADX]=%.1f/%.1f/%.1f "
        "RSI_DIFF[6-12/6-24]=%.1f/%.1f(prev %.1f/%.1f) "
        "MACD_DIFF[DIF-DEA]=%.4f(prev %.4f) "
        "KDJ_DIFF[K-D/J-D]=%.1f/%.1f(prev %.1f/%.1f)" % (
            item.get("rsi6", np.nan), item.get("rsi12", np.nan), item.get("rsi24", np.nan),
            item.get("dif", np.nan), item.get("dea", np.nan), item.get("macd_hist", np.nan),
            item.get("k", np.nan), item.get("d", np.nan), item.get("j", np.nan),
            item.get("boll_upper", np.nan), item.get("boll_mid", np.nan), item.get("boll_lower", np.nan),
            item.get("ma5", np.nan), item.get("ma10", np.nan),
            item.get("ma20", np.nan), item.get("ma60", np.nan),
            item.get("vol5", np.nan), item.get("vol20", np.nan),
            item.get("atr", np.nan),
            item.get("plus_di", np.nan), item.get("minus_di", np.nan), item.get("adx", np.nan),
            rsi_diff_12, rsi_diff_24, rsi_diff_12_prev, rsi_diff_24_prev,
            macd_diff, macd_diff_prev,
            kdj_diff_k, kdj_diff_j, kdj_diff_k_prev, kdj_diff_j_prev,
        )
    )


def format_cross_flags(item):
    return (
        "RSI12_UP=%s RSI24_UP=%s MACD_UP=%s KDJ_K_UP=%s KDJ_J_UP=%s "
        "RSI12_DOWN=%s RSI24_DOWN=%s MACD_DOWN=%s KDJ_K_DOWN=%s KDJ_J_DOWN=%s" % (
            item.get("rsi6_cross_rsi12_up"),
            item.get("rsi6_cross_rsi24_up"),
            item.get("macd_cross_up"),
            item.get("kdj_k_cross_up"),
            item.get("kdj_j_cross_up"),
            item.get("rsi6_cross_rsi12_down"),
            item.get("rsi6_cross_rsi24_down"),
            item.get("macd_cross_down"),
            item.get("kdj_k_cross_down"),
            item.get("kdj_j_cross_down"),
        )
    )


def initialize(context):
    set_benchmark("000300.SS")
    try:
        set_parameters(
            receive_cancel_response="1",
            not_restart_trade="0",
            server_restart_not_do_before="0",
        )
    except Exception as exc:
        log.warning("[initialize] platform parameter setup failed: %s" % exc)
    try:
        set_commission(commission_ratio=0.0003, min_commission=5.0, type="ETF")
        set_slippage(slippage=0.001)
    except Exception as exc:
        log.warning("[initialize] commission/slippage setup failed: %s" % exc)

    g.params = get_default_params()
    g.etf_pool = get_default_etf_pool()
    g.highest_since_buy = {}
    g.entry_atr = {}
    g.buy_date = {}
    g.last_scores = {}
    g.sold_today = {}
    g.paused_pool_codes = set()
    g.unverified_positions = set()
    g.execution_date = None
    g.deferred_scores = []
    g.deferred_signal_date = None
    g.__last_snapshot = {}
    g.__pending_orders = {}
    g.__pending_sells = {}
    g.__order_state_unknown = False
    g.__data = None

    try:
        set_universe(g.etf_pool)
    except Exception as exc:
        log.warning("[initialize] set_universe failed: %s" % exc)

    try:
        g.__is_live = bool(is_trade())
    except Exception:
        g.__is_live = False

    if g.__is_live:
        run_daily(context, _do_trading_wrapper, time="09:35")
        run_daily(context, _halt_recover_wrapper, time="10:35")
        run_daily(context, _after_close_wrapper, time="15:30")

    log.info("[%s] initialized: max_hold=%d base_ratio=%.2f min_signal_hold=%d" % (
        STRATEGY_VERSION,
        g.params["max_hold"],
        g.params["base_ratio"],
        g.params["min_signal_hold_days"]))
    log.info(format_self_check())
    log.info("[indicator params] %s" % format_indicator_params(g.params))


def handle_data(context, data):
    """PTrade backtest entry; daily backtests execute at the platform close."""
    g.__data = data
    if g.__is_live:
        return
    do_trading(context)


def before_trading_start(context, data):
    g.__data = data
    g.__last_snapshot = {}
    if g.__is_live:
        _restore_live_state()
    _lock_frozen_business_config()
    today = _as_date(get_context_datetime(context))
    if today is None:
        g.__order_state_unknown = True
        log.error("[day-reset] current trading date unavailable; trading blocked")
        return
    if g.execution_date != today:
        g.execution_date = today
        g.sold_today = {}
        g.paused_pool_codes = set()
        g.deferred_scores = []
        g.deferred_signal_date = None
    if g.__is_live:
        _reconcile_open_orders(context)
        _recover_live_state_with_available_sources(context, allow_deliver=True)
    else:
        g.__pending_orders = {}
        g.__pending_sells = {}
        g.__order_state_unknown = False
    if g.__is_live:
        _persist_live_state()


def after_trading_end(context, data):
    if not g.__is_live:
        after_close(context)
    g.sold_today = {}
    if g.__is_live:
        _persist_live_state()


def _do_trading_wrapper(context):
    do_trading(context)
    _persist_live_state()


def _halt_recover_wrapper(context):
    halt_recover(context)
    _persist_live_state()


def _after_close_wrapper(context):
    _recover_live_state_with_available_sources(context, allow_deliver=False)
    after_close(context)
    g.sold_today = {}
    _persist_live_state()


def calc_rsi(close, period):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    rsi[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return rsi


def calc_macd(close, fast, slow, signal):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, 2 * (dif - dea)


def calc_kdj(high, low, close, n, m1, m2):
    lowest = low.rolling(n).min()
    highest = high.rolling(n).max()
    rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def calc_bollinger(close, period, std_mult):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + std_mult * std, mid, mid - std_mult * std


def calc_atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_dmi_adx(high, low, close, period):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    atr = calc_atr(high, low, close, period)
    plus_di = 100 * plus_dm.rolling(period).sum() / atr.rolling(period).sum()
    minus_di = 100 * minus_dm.rolling(period).sum() / atr.rolling(period).sum()
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return plus_di, minus_di, adx


def _valid_pair(a_prev, a_cur, b_prev, b_cur):
    return not _builtins.any(pd.isna(v) for v in [a_prev, a_cur, b_prev, b_cur])


def _as_float_array(values):
    if hasattr(values, "values"):
        values = values.values
    return np.asarray(values, dtype=float)


def latest_cross_direction_by_diff_recent(fast, slow, window=3):
    fast_values = _as_float_array(fast)
    slow_values = _as_float_array(slow)
    if len(fast_values) < window + 1 or len(slow_values) < window + 1:
        return None
    diff = fast_values - slow_values
    latest_direction = None
    for offset in range(window, 0, -1):
        prev_idx = -offset - 1
        cur_idx = -offset
        prev_diff, cur_diff = diff[prev_idx], diff[cur_idx]
        if not _builtins.any(pd.isna(v) for v in [prev_diff, cur_diff]) and prev_diff <= 0 and cur_diff > 0:
            latest_direction = "above"
        elif not _builtins.any(pd.isna(v) for v in [prev_diff, cur_diff]) and prev_diff >= 0 and cur_diff < 0:
            latest_direction = "below"
    return latest_direction


def crossed_above_by_diff_recent(fast, slow, window=3):
    return latest_cross_direction_by_diff_recent(fast, slow, window) == "above"


def crossed_below_by_diff_recent(fast, slow, window=3):
    return latest_cross_direction_by_diff_recent(fast, slow, window) == "below"


def crossed_above_recent(fast, slow, window=3):
    return crossed_above_by_diff_recent(fast, slow, window)


def crossed_below_recent(fast, slow, window=3):
    return crossed_below_by_diff_recent(fast, slow, window)


def rsi_group_direction(snapshot):
    rsi_up = (
        snapshot.get("rsi6_cross_rsi12_up") or
        snapshot.get("rsi6_cross_rsi24_up")
    )
    rsi_down = (
        snapshot.get("rsi6_cross_rsi12_down") or
        snapshot.get("rsi6_cross_rsi24_down")
    )
    if rsi_up and not rsi_down:
        return "up"
    if rsi_down and not rsi_up:
        return "down"
    return None


def score_buy_snapshot(snapshot, params=None):
    p = params or get_default_params()
    reversal = 0
    if rsi_group_direction(snapshot) == "up":
        reversal += 12 if snapshot.get("rsi6_cross_rsi12_up") else 0
        reversal += 12 if snapshot.get("rsi6_cross_rsi24_up") else 0
    reversal += 10 if snapshot.get("macd_cross_up") else 0
    reversal += 6 if snapshot.get("kdj_k_cross_up") else 0
    reversal += 5 if snapshot.get("kdj_j_cross_up") else 0

    location = 0
    location += 10 if snapshot.get("close_between_boll_lower_mid") else 0
    location += 8 if snapshot.get("close_cross_boll_mid_up") else 0
    location += 7 if snapshot.get("close_near_ma20") else 0
    location -= 10 if snapshot.get("close_far_above_ma20") else 0

    trend = 0
    trend += 6 if snapshot.get("ma5_gt_ma10") else 0
    trend += 6 if snapshot.get("ma10_gt_ma20") else 0
    trend += 5 if snapshot.get("ma20_slope_non_negative") else 0
    trend += 3 if snapshot.get("close_gt_ma60") else 0
    trend -= 15 if snapshot.get("downside_continuation") else 0

    volume = 0
    volume += 6 if snapshot.get("volume_above_vol20_and_up") else 0
    volume += 4 if snapshot.get("vol5_gt_vol20") else 0

    total = max(0, reversal + location + trend + volume)
    rsi6 = snapshot.get("rsi6")
    buy_allowed = rsi6 is None or pd.isna(rsi6) or rsi6 < p["overheat_rsi"]
    return {
        "buy_score": total,
        "reversal_score": reversal,
        "location_score": location,
        "trend_score": trend,
        "volume_score": volume,
        "buy_allowed": buy_allowed,
    }


def score_sell_snapshot(snapshot):
    reversal = 0
    if rsi_group_direction(snapshot) == "down":
        reversal += 12 if snapshot.get("rsi6_cross_rsi12_down") else 0
        reversal += 12 if snapshot.get("rsi6_cross_rsi24_down") else 0
    reversal += 10 if snapshot.get("macd_cross_down") else 0
    reversal += 6 if snapshot.get("kdj_k_cross_down") else 0
    reversal += 5 if snapshot.get("kdj_j_cross_down") else 0

    risk = 0
    risk += 8 if snapshot.get("far_above_ma20_and_rsi6_down") else 0
    risk += 10 if snapshot.get("close_below_falling_ma10") else 0
    risk += 6 if snapshot.get("fell_back_inside_boll") else 0

    return {
        "sell_score": max(0, reversal + risk),
        "sell_reversal_score": reversal,
        "sell_risk_score": risk,
    }


def should_force_sell(sell_score_result, atr_stop_triggered=False, params=None):
    p = params or get_default_params()
    if atr_stop_triggered:
        return True
    return (
        sell_score_result.get("sell_score", 0) >= p["sell_threshold"] and
        has_signal_sell_confirmation(sell_score_result) and
        not is_protected_by_strong_adx_uptrend(sell_score_result, p)
    )


def is_protected_by_strong_adx_uptrend(snapshot, params=None):
    p = params or get_default_params()
    severe_break = (
        snapshot.get("close_below_ma20") or
        snapshot.get("close_below_falling_ma10") or
        snapshot.get("downside_continuation")
    )
    if severe_break:
        return False
    return is_strong_adx_uptrend(snapshot, p)


def is_strong_adx_uptrend(snapshot, params=None):
    p = params or get_default_params()
    adx = snapshot.get("adx")
    plus_di = snapshot.get("plus_di")
    minus_di = snapshot.get("minus_di")
    if _builtins.any(pd.isna(v) for v in [adx, plus_di, minus_di]):
        return False
    return (
        adx >= p["adx_trend_threshold"] and
        plus_di > minus_di and
        snapshot.get("ma20_slope_non_negative")
    )


def has_signal_sell_confirmation(snapshot):
    return (
        snapshot.get("close_below_ma20") or
        snapshot.get("close_below_boll_mid") or
        snapshot.get("close_below_falling_ma10") or
        snapshot.get("downside_continuation") or
        snapshot.get("far_above_ma20_and_rsi6_down")
    )


def _date_key(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def can_sell_by_signal(buy_date, today, min_hold_days=1, trade_days=None):
    if buy_date is None:
        return True
    if int(min_hold_days) <= 1:
        return _date_key(buy_date) != _date_key(today)
    buy_key = _date_key(buy_date)
    today_key = _date_key(today)
    if trade_days is not None:
        keys = [_date_key(day) for day in trade_days]
        if buy_key in keys and today_key in keys:
            return keys.index(today_key) - keys.index(buy_key) >= int(min_hold_days)
    return (pd.Timestamp(today_key) - pd.Timestamp(buy_key)).days >= int(min_hold_days)


def can_sell_with_verified_calendar(buy_date, today, min_hold_days=1, trade_days=None):
    if int(min_hold_days) > 1 and trade_days is None:
        return False
    return can_sell_by_signal(buy_date, today, min_hold_days, trade_days)


def sort_candidates(candidates):
    return sorted(candidates, key=lambda x: (
        -x.get("buy_score", 0),
        -x.get("reversal_score", 0),
        x.get("code", "")
    ))


def has_new_buy_position(snapshot, params=None):
    if snapshot.get("close_far_above_ma20"):
        return False
    return (
        snapshot.get("close_between_boll_lower_mid") or
        snapshot.get("close_cross_boll_mid_up") or
        snapshot.get("close_near_ma20")
    )


def filter_buy_candidates(scores, held_codes, params=None):
    p = params or get_default_params()
    held = set(held_codes)
    return [
        s for s in scores
        if s.get("buy_allowed")
        and s.get("buy_score", 0) >= p["buy_threshold"]
        and s.get("sell_score", 0) < p["sell_threshold"]
        and has_new_buy_position(s, p)
        and not is_blocked_entry_combo(s)
        and s.get("code") not in held
    ]


def is_blocked_entry_combo(score):
    rsi_up = score.get("rsi6_cross_rsi12_up") or score.get("rsi6_cross_rsi24_up")
    kdj_up = score.get("kdj_k_cross_up") or score.get("kdj_j_cross_up")
    return (
        bool(rsi_up)
        and bool(score.get("macd_cross_up"))
        and not bool(kdj_up)
        and _numeric_score(score.get("volume_score")) > 0
        and 0 < _numeric_score(score.get("trend_score")) < 20
    )


def _numeric_score(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_cross_signal_candidates(scores, limit=5):
    items = [s for s in scores if s.get("reversal_score", 0) > 0]
    items = sorted(items, key=lambda x: (
        -x.get("reversal_score", 0),
        -x.get("buy_score", 0),
        x.get("code", "")
    ))
    return {
        "count": len(items),
        "items": items[:limit],
    }


def summarize_loose_reversal_candidates(scores, limit=5):
    items = []
    for score in scores:
        rsi6_delta = score.get("rsi6", np.nan) - score.get("rsi6_prev", np.nan)
        dif_delta = score.get("dif", np.nan) - score.get("dif_prev", np.nan)
        k_delta = score.get("k", np.nan) - score.get("k_prev", np.nan)
        j_delta = score.get("j", np.nan) - score.get("j_prev", np.nan)
        rsi_turn_up = rsi6_delta > 0
        macd_turn_up = dif_delta > 0
        kdj_turn_up = k_delta > 0 or j_delta > 0
        loose_count = int(rsi_turn_up) + int(macd_turn_up) + int(kdj_turn_up)
        if loose_count <= 0:
            continue
        item = dict(score)
        item.update({
            "loose_reversal_count": loose_count,
            "rsi_turn_up": rsi_turn_up,
            "macd_turn_up": macd_turn_up,
            "kdj_turn_up": kdj_turn_up,
            "rsi6_delta": round(float(rsi6_delta), 4),
            "dif_delta": round(float(dif_delta), 4),
            "k_delta": round(float(k_delta), 4),
            "j_delta": round(float(j_delta), 4),
        })
        items.append(item)

    items = sorted(items, key=lambda x: (
        -x.get("loose_reversal_count", 0),
        -x.get("buy_score", 0),
        x.get("code", "")
    ))
    return {
        "count": len(items),
        "items": items[:limit],
    }


def score_skip_reason(df, snapshot, required_fields, min_len):
    if df is None:
        return "no_data"
    if len(df) < min_len:
        return "short_data:%d<%d" % (len(df), min_len)
    if "close" in df and df["close"].iloc[-1] <= 0:
        return "invalid_close:%.4f" % df["close"].iloc[-1]
    if "volume" in df and df["volume"].iloc[-5:].sum() == 0:
        return "zero_recent_volume"
    if snapshot is None:
        return None
    nan_fields = [k for k in required_fields if pd.isna(snapshot.get(k))]
    if nan_fields:
        return "nan_fields:%s" % ",".join(nan_fields)
    return None


def normalize_code(code):
    """Normalize PTrade callbacks and JoinQuant symbols to the PTrade universe."""
    text = str(code or "").strip().upper()
    if not text:
        return ""
    base = text.split(".")[0]
    if text.endswith((".XSHG", ".SH", ".SS")):
        return base + ".SS"
    if text.endswith((".XSHE", ".SZ")):
        return base + ".SZ"
    return base + (".SS" if base.startswith(("5", "6", "9")) else ".SZ")


def get_context_datetime(context):
    blotter = getattr(context, "blotter", None)
    value = getattr(blotter, "current_dt", None)
    if value is None:
        value = getattr(context, "current_dt", None)
    return value


def _as_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "date") and not isinstance(value, str):
        try:
            return value.date()
        except Exception:
            pass
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _previous_day_from_result(result, today):
    if result is None:
        return None
    if isinstance(result, tuple):
        values = []
        for item in result:
            if isinstance(item, (list, tuple, np.ndarray, pd.Index)):
                values.extend(list(item))
            else:
                values.append(item)
    elif isinstance(result, (list, tuple, np.ndarray, pd.Index, pd.Series)):
        values = list(result)
    else:
        values = [result]
    dates = [_as_date(item) for item in values]
    dates = sorted(set(item for item in dates if item is not None and item < today))
    return dates[-1] if dates else None


def get_prev_trade_date(context):
    now = get_context_datetime(context)
    today = _as_date(now)
    if today is None:
        log.error("[trade-date] context current_dt unavailable; trading aborted")
        return None
    try:
        result = get_trade_days(end_date=today, count=2)
        prev = _previous_day_from_result(result, today)
        if prev is not None:
            return prev
    except Exception as exc:
        log.warning("[trade-date] get_trade_days failed: %s" % exc)
    try:
        result = get_all_trades_days(date=today.strftime("%Y%m%d"))
        prev = _previous_day_from_result(result, today)
        if prev is not None:
            return prev
    except Exception as exc:
        log.warning("[trade-date] get_all_trades_days failed: %s" % exc)
    log.error("[trade-date] cannot prove T-1 trading day; trading aborted")
    return None


def get_price_data(code, end_date, count):
    """Load pre-adjusted daily bars ending at the proven T-1 date."""
    end_date_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)
    fields = ["open", "close", "high", "low", "volume"]
    try:
        frame = get_price(
            code,
            end_date=end_date_str,
            count=count,
            frequency="1d",
            fields=fields,
            fq="pre",
        )
        if frame is not None and len(frame) > 0:
            frame = pd.DataFrame(frame).copy()
            if "code" in frame.columns:
                frame = frame[frame["code"].map(normalize_code) == normalize_code(code)]
            if all(field in frame.columns for field in fields):
                return frame[frame["volume"] > 0][fields]
    except Exception as exc:
        log.warning("[daily-data] get_price failed %s: %s" % (code, exc))

    try:
        series = {}
        for field in fields:
            raw = get_history(
                count,
                "1d",
                field,
                [code],
                fq="pre",
                include=False,
            )
            if isinstance(raw, pd.DataFrame):
                series[field] = raw[code]
            elif isinstance(raw, dict):
                series[field] = pd.Series(raw[code])
            else:
                raise ValueError("unsupported get_history result")
        frame = pd.DataFrame(series)
        try:
            frame = frame[frame.index <= pd.Timestamp(end_date_str)]
        except Exception:
            pass
        return frame[frame["volume"] > 0]
    except Exception as exc:
        log.error("[daily-data] unavailable %s: %s" % (code, exc))
        return None


def build_signal_snapshot(df, params):
    C, H, L, V = df["close"], df["high"], df["low"], df["volume"]
    rsi6 = calc_rsi(C, params["rsi_fast"])
    rsi12 = calc_rsi(C, params["rsi_mid"])
    rsi24 = calc_rsi(C, params["rsi_slow"])
    dif, dea, macd_hist = calc_macd(C, params["macd_fast"], params["macd_slow"], params["macd_signal"])
    k, d, j = calc_kdj(H, L, C, params["kdj_n"], params["kdj_m1"], params["kdj_m2"])
    boll_upper, boll_mid, boll_lower = calc_bollinger(C, params["boll_period"], params["boll_std"])
    atr = calc_atr(H, L, C, params["atr_period"])
    plus_di, minus_di, adx = calc_dmi_adx(H, L, C, params["adx_period"])

    ma5 = C.rolling(5).mean()
    ma10 = C.rolling(10).mean()
    ma20 = C.rolling(20).mean()
    ma60 = C.rolling(60).mean()
    vol5 = V.rolling(5).mean()
    vol20 = V.rolling(20).mean()

    latest = C.iloc[-1]
    prev = C.iloc[-2]
    ma20_slope = ma20.iloc[-1] - ma20.iloc[-6] if len(ma20) >= 6 else np.nan
    rsi6_down = rsi6.iloc[-1] < rsi6.iloc[-2] if len(rsi6) >= 2 else False

    snapshot = {
        "close": latest,
        "rsi6": rsi6.iloc[-1],
        "rsi6_prev": rsi6.iloc[-2],
        "rsi12": rsi12.iloc[-1],
        "rsi12_prev": rsi12.iloc[-2],
        "rsi24": rsi24.iloc[-1],
        "rsi24_prev": rsi24.iloc[-2],
        "dif": dif.iloc[-1],
        "dif_prev": dif.iloc[-2],
        "dea": dea.iloc[-1],
        "dea_prev": dea.iloc[-2],
        "macd_hist": macd_hist.iloc[-1],
        "macd_hist_prev": macd_hist.iloc[-2],
        "k": k.iloc[-1],
        "k_prev": k.iloc[-2],
        "d": d.iloc[-1],
        "d_prev": d.iloc[-2],
        "j": j.iloc[-1],
        "j_prev": j.iloc[-2],
        "ma20": ma20.iloc[-1],
        "ma5": ma5.iloc[-1],
        "ma10": ma10.iloc[-1],
        "ma60": ma60.iloc[-1],
        "vol5": vol5.iloc[-1],
        "vol20": vol20.iloc[-1],
        "boll_upper": boll_upper.iloc[-1],
        "boll_mid": boll_mid.iloc[-1],
        "boll_lower": boll_lower.iloc[-1],
        "atr": atr.iloc[-1],
        "plus_di": plus_di.iloc[-1],
        "minus_di": minus_di.iloc[-1],
        "adx": adx.iloc[-1],
        "rsi6_cross_rsi12_up": crossed_above_recent(rsi6, rsi12, params["cross_window"]),
        "rsi6_cross_rsi24_up": crossed_above_recent(rsi6, rsi24, params["cross_window"]),
        "rsi6_cross_rsi12_down": crossed_below_recent(rsi6, rsi12, params["cross_window"]),
        "rsi6_cross_rsi24_down": crossed_below_recent(rsi6, rsi24, params["cross_window"]),
        "macd_cross_up": crossed_above_recent(dif, dea, params["cross_window"]),
        "macd_cross_down": crossed_below_recent(dif, dea, params["cross_window"]),
        "kdj_k_cross_up": crossed_above_recent(k, d, params["cross_window"]),
        "kdj_j_cross_up": crossed_above_recent(j, d, params["cross_window"]),
        "kdj_k_cross_down": crossed_below_recent(k, d, params["cross_window"]),
        "kdj_j_cross_down": crossed_below_recent(j, d, params["cross_window"]),
        "close_between_boll_lower_mid": boll_lower.iloc[-1] <= latest <= boll_mid.iloc[-1],
        "close_cross_boll_mid_up": crossed_above_recent(C, boll_mid, params["cross_window"]),
        "close_near_ma20": abs(latest / ma20.iloc[-1] - 1) <= 0.05 if ma20.iloc[-1] > 0 else False,
        "close_far_above_ma20": latest / ma20.iloc[-1] - 1 > 0.12 if ma20.iloc[-1] > 0 else False,
        "close_below_ma20": latest < ma20.iloc[-1],
        "close_below_boll_mid": latest < boll_mid.iloc[-1],
        "ma5_gt_ma10": ma5.iloc[-1] > ma10.iloc[-1],
        "ma10_gt_ma20": ma10.iloc[-1] > ma20.iloc[-1],
        "ma20_slope_non_negative": ma20_slope >= 0,
        "close_gt_ma60": latest > ma60.iloc[-1],
        "downside_continuation": latest < ma60.iloc[-1] and ma20_slope < 0,
        "volume_above_vol20_and_up": V.iloc[-1] > vol20.iloc[-1] and latest > prev,
        "vol5_gt_vol20": vol5.iloc[-1] > vol20.iloc[-1],
        "far_above_ma20_and_rsi6_down": latest / ma20.iloc[-1] - 1 > 0.10 and rsi6_down if ma20.iloc[-1] > 0 else False,
        "close_below_falling_ma10": latest < ma10.iloc[-1] and ma10.iloc[-1] < ma10.iloc[-2],
        "fell_back_inside_boll": C.iloc[-2] > boll_upper.iloc[-2] and latest <= boll_upper.iloc[-1],
    }
    return snapshot


def calc_cross_signal_score(code, end_date, return_reason=False):
    p = g.params
    min_len = p["lookback"] - 10
    required = ["rsi6", "rsi12", "rsi24", "dif", "dea", "k", "d", "j", "ma20", "atr", "adx"]
    try:
        df = get_price_data(code, end_date, p["lookback"])
        reason = score_skip_reason(df, None, required, min_len)
        if reason is not None:
            return (None, reason) if return_reason else None

        snapshot = build_signal_snapshot(df, p)
    except Exception as exc:
        reason = "exception:%s" % exc.__class__.__name__
        return (None, reason) if return_reason else None

    reason = score_skip_reason(df, snapshot, required, min_len)
    if reason is not None:
        return (None, reason) if return_reason else None

    buy_score = score_buy_snapshot(snapshot, p)
    sell_score = score_sell_snapshot(snapshot)
    result = {}
    result.update(snapshot)
    result.update(buy_score)
    result.update(sell_score)
    result["code"] = code
    return (result, None) if return_reason else result


def calc_stop_price(highest, atr_val, cost, params=None):
    p = params or get_default_params()
    if highest <= 0 or atr_val <= 0:
        return cost * (1 - p["stop_cap"])
    pct_stop = p["trailing_atr_mult"] * atr_val / highest
    pct_stop = max(p["stop_floor"], min(p["stop_cap"], pct_stop))
    return highest * (1 - pct_stop)


def current_hold_codes(context):
    return [
        normalize_code(code) for code, pos in _positions(context).items()
        if _pos_amount(pos) > 0
    ]


def has_position(context, code):
    code = normalize_code(code)
    for held_code, pos in _positions(context).items():
        if normalize_code(held_code) == code and _pos_amount(pos) > 0:
            return True
    return False


def _total_value(context):
    return float(context.portfolio.portfolio_value)


def _available_cash(context):
    return float(context.portfolio.cash)


def _positions(context):
    return context.portfolio.positions


def _pos_amount(pos):
    return float(getattr(pos, "amount", 0) or 0)


def _pos_cost(pos):
    return float(getattr(pos, "cost_basis", 0) or 0)


def _pos_price(pos):
    return float(getattr(pos, "last_sale_price", 0) or 0)


def _get_position(context, code):
    code = normalize_code(code)
    for held_code, pos in _positions(context).items():
        if normalize_code(held_code) == code:
            return pos
    return None


def _order_field(order_obj, name, default=None):
    if isinstance(order_obj, dict):
        return order_obj.get(name, default)
    return getattr(order_obj, name, default)


def _reconcile_open_orders(context):
    g.__pending_orders = {}
    g.__pending_sells = {}
    g.__order_state_unknown = False
    try:
        open_orders = get_open_orders()
    except Exception as exc:
        g.__order_state_unknown = True
        log.error("[order-recovery] get_open_orders failed; trading blocked: %s" % exc)
        return False

    if not isinstance(open_orders, (list, tuple)):
        g.__order_state_unknown = True
        log.error("[order-recovery] invalid get_open_orders response; trading blocked")
        return False

    today = _as_date(get_context_datetime(context))
    pending_buys = {}
    pending_sells = {}
    sold_guards = {}
    for order_obj in open_orders:
        code = normalize_code(_order_field(order_obj, "symbol", ""))
        order_id = str(_order_field(order_obj, "id", "") or "")
        amount = _safe_float(_order_field(order_obj, "amount", 0))
        filled = abs(_safe_float(_order_field(order_obj, "filled", 0)))
        requested = abs(amount)
        quantities_valid = (
            np.isfinite(amount) and
            np.isfinite(filled) and
            requested > 0 and
            0 <= filled <= requested
        )
        if not code or not order_id or not quantities_valid:
            g.__order_state_unknown = True
            log.error("[order-recovery] malformed open order; trading blocked")
            return False
        if code in pending_buys or code in pending_sells:
            g.__order_state_unknown = True
            log.error("[order-recovery] multiple open orders for %s; trading blocked" % code)
            return False
        if amount > 0:
            score = g.last_scores.get(code, {})
            pos = _get_position(context, code)
            filled_price = _pos_cost(pos) if pos is not None else 0.0
            fill_value_complete = filled == 0 or _is_positive_finite(filled_price)
            pending_buys[code] = {
                "requested_qty": requested,
                "filled_qty": filled,
                "filled_value": filled * filled_price if fill_value_complete else 0.0,
                "fill_value_complete": fill_value_complete,
                "atr": g.entry_atr.get(code, score.get("atr")),
                "buy_date": g.buy_date.get(code, today),
                "order_id": order_id,
                "recovered_guard": True,
            }
        else:
            pending_sells[code] = {
                "requested_qty": requested,
                "filled_qty": filled,
                "reason": "recovered_open_order",
                "order_id": order_id,
                "recovered_guard": True,
            }
            sold_guards[code] = True
    g.__pending_orders = pending_buys
    g.__pending_sells = pending_sells
    g.sold_today.update(sold_guards)
    if g.__pending_orders or g.__pending_sells:
        log.warning("[order-recovery] open buys=%d open sells=%d" % (
            len(g.__pending_orders), len(g.__pending_sells)))
    return True


def _clear_position_state(code):
    code = normalize_code(code)
    g.highest_since_buy.pop(code, None)
    g.entry_atr.pop(code, None)
    g.buy_date.pop(code, None)
    g.last_scores.pop(code, None)
    g.unverified_positions.discard(code)


def _snapshot_record(raw, code):
    if not isinstance(raw, dict):
        return None
    if "last_px" in raw or "trade_status" in raw:
        return raw
    for key in (code, code.split(".")[0]):
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    for key, value in raw.items():
        if normalize_code(key) == code and isinstance(value, dict):
            return value
    return None


def _positive_float_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def _snapshot_age_seconds(raw_timestamp, observed_at):
    if raw_timestamp in (None, "") or not isinstance(observed_at, datetime):
        return None
    digits = "".join(ch for ch in str(raw_timestamp) if ch.isdigit())
    if len(digits) < 14:
        return None
    try:
        snapshot_dt = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
        return (observed_at - snapshot_dt).total_seconds()
    except (TypeError, ValueError):
        return None


def build_iopv_observation(
    code,
    snapshot,
    execution_price,
    observed_at=None,
):
    code = normalize_code(code)
    if code not in IOPV_OBSERVE_CODES:
        return None
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    snapshot_price = _positive_float_or_none(snapshot.get("last_px"))
    fallback_price = _positive_float_or_none(execution_price)
    market_price = snapshot_price if snapshot_price is not None else fallback_price
    iopv = _positive_float_or_none(snapshot.get("iopv"))
    premium = (
        market_price / iopv - 1.0
        if market_price is not None and iopv is not None
        else None
    )
    timestamp = snapshot.get("hsTimeStamp")
    return {
        "code": code,
        "valid": bool(market_price is not None and iopv is not None),
        "market_price": market_price,
        "iopv": iopv,
        "premium": premium,
        "snapshot_timestamp": timestamp,
        "snapshot_age_seconds": _snapshot_age_seconds(timestamp, observed_at),
    }


def log_iopv_buy_observation(context, code, execution_price):
    if not getattr(g, "__is_live", False):
        return
    try:
        normalized = normalize_code(code)
        snapshot = getattr(g, "__last_snapshot", {}).get(normalized, {})
        observation = build_iopv_observation(
            normalized,
            snapshot,
            execution_price,
            observed_at=get_context_datetime(context),
        )
        if observation is None:
            return
        premium_pct = (
            observation["premium"] * 100.0
            if observation["premium"] is not None
            else None
        )
        log.info(
            "[iopv-observe] event=buy dt=%s code=%s valid=%s price=%s "
            "iopv=%s premium_pct=%s hsTimeStamp=%s age_seconds=%s"
            % (
                get_context_datetime(context),
                observation["code"],
                observation["valid"],
                observation["market_price"],
                observation["iopv"],
                premium_pct,
                observation["snapshot_timestamp"],
                observation["snapshot_age_seconds"],
            )
        )
    except Exception as exc:
        try:
            log.warning("[iopv-observe] code=%s unavailable: %s" % (code, exc))
        except Exception:
            pass


def get_current_price(code):
    code = normalize_code(code)
    if getattr(g, "__is_live", False):
        try:
            snapshot = _snapshot_record(get_snapshot(code), code)
            price = float(snapshot.get("last_px", 0)) if snapshot else 0.0
            if price > 0:
                g.__last_snapshot[code] = snapshot
                return price
        except Exception as exc:
            log.warning("[snapshot] price unavailable %s: %s" % (code, exc))
        return None

    data = getattr(g, "__data", None)
    if data is not None:
        try:
            price = float(data[code].price)
            if price > 0:
                return price
        except Exception:
            pass
    try:
        raw = get_history(1, "1d", "close", [code], fq="pre", include=True)
        price = float(raw[code].iloc[-1])
        return price if price > 0 else None
    except Exception:
        return None


def is_paused(code):
    code = normalize_code(code)
    if getattr(g, "__is_live", False):
        try:
            result = get_stock_status([code], "HALT")
            if isinstance(result, dict):
                for key, value in result.items():
                    if normalize_code(key) == code:
                        if isinstance(value, (bool, np.bool_)):
                            return bool(value)
                        log.warning("[status] unknown halt value %s=%r" % (code, value))
                        return True
        except Exception as exc:
            log.warning("[status] halt query failed %s: %s" % (code, exc))
        snapshot = getattr(g, "__last_snapshot", {}).get(code)
        if snapshot:
            status = str(snapshot.get("trade_status", "")).upper()
            if status in ("HALT", "SUSP", "STOPT", "DELISTED"):
                return True
            if status in ("TRADE", "OCALL", "BREAK", "ENDTR", "POSTR"):
                return False
        return True

    data = getattr(g, "__data", None)
    if data is not None:
        try:
            return int(data[code].is_open) == 0
        except Exception:
            pass
    try:
        result = get_stock_status([code], "HALT")
        if isinstance(result, dict):
            for key, value in result.items():
                if normalize_code(key) == code:
                    return bool(value)
    except Exception:
        pass
    return False


def _find_paused_pool_codes(pool, pause_check):
    return set(code for code in pool if pause_check(code))


def get_sell_limit_price(code, current):
    code = normalize_code(code)
    price = round(float(current), 3)
    snapshot = getattr(g, "__last_snapshot", {}).get(code, {})
    if getattr(g, "__is_live", False):
        try:
            down_price = float(snapshot.get("down_px", 0))
            if down_price > 0:
                return round(down_price, 3)
        except (TypeError, ValueError):
            pass
    return price


def execute_sell(code, context, reason):
    code = normalize_code(code)
    if g.sold_today.get(code) or code in getattr(g, "__pending_sells", {}):
        return False
    pos = _get_position(context, code)
    amount = int(_pos_amount(pos)) if pos is not None else 0
    if amount <= 0:
        return False
    price = get_current_price(code)
    if price is None or price <= 0:
        log.warning("[sell] %s price unavailable; order skipped" % code)
        return False
    limit_price = get_sell_limit_price(code, price)
    log.info("[sell] %s reason=%s amount=%s limit=%.3f" % (
        code, reason, amount, limit_price))
    try:
        order_id = order_target(code, 0, limit_price=limit_price)
    except Exception as exc:
        log.error("[sell] %s submission failed: %s" % (code, exc))
        return False
    if order_id is None:
        log.error("[sell] %s submission returned no order id" % code)
        return False
    if getattr(g, "__is_live", False):
        g.sold_today[code] = True
        g.__pending_sells[code] = {
            "requested_qty": amount,
            "filled_qty": 0.0,
            "reason": reason,
            "order_id": str(order_id),
        }
    else:
        _clear_position_state(code)
    return True


def check_atr_stops(context):
    triggered = []
    today = _as_date(get_context_datetime(context))
    for code in current_hold_codes(context):
        if code in g.unverified_positions:
            continue
        if g.buy_date.get(code) == today:
            continue
        if g.sold_today.get(code) or code in getattr(g, "__pending_sells", {}):
            continue
        if is_paused(code):
            continue
        pos = _get_position(context, code)
        price = get_current_price(code)
        if price is None or price <= 0:
            continue
        if code not in g.highest_since_buy or code not in g.entry_atr:
            continue
        stop_price = calc_stop_price(
            g.highest_since_buy[code], g.entry_atr[code], _pos_cost(pos), g.params)
        if price <= stop_price:
            triggered.append((code, stop_price, price))
    return triggered


def execute_buy_candidates(context, all_scores, today):
    """Submit buys only against broker-confirmed holdings and cash."""
    if getattr(g, "__order_state_unknown", False):
        log.error("[buy] broker order state unknown; deferred buys blocked")
        return 0
    if getattr(g, "__pending_sells", {}):
        log.info("[buy deferred] waiting for %d sell order(s)" % len(g.__pending_sells))
        return 0

    held = set(current_hold_codes(context))
    pending_buys = set(getattr(g, "__pending_orders", {}).keys())
    slots = g.params["max_hold"] - len(held | pending_buys)
    if slots <= 0:
        return 0
    candidates = filter_buy_candidates(all_scores, held | pending_buys, g.params)
    if not candidates:
        log.info("[%s] no buy candidates above threshold" % STRATEGY_VERSION)
        return 0

    available = _available_cash(context)
    bought = 0
    for score in candidates:
        if bought >= slots:
            break
        code = score["code"]
        if code in getattr(g, "__pending_orders", {}):
            continue
        if is_paused(code):
            continue
        price = get_current_price(code)
        if price is None or price <= 0:
            log.warning("[buy skip] %s current price unavailable" % code)
            continue
        target_value = min(calc_buy_target_value(_total_value(context), score, g.params), available)
        shares = int(target_value / price / 100) * 100
        if shares < 100:
            log.info("[buy skip] %s insufficient cash %.0f" % (code, available))
            continue
        log_iopv_buy_observation(context, code, price)
        log.info(
            "[buy] %s buy=%.0f rev=%.0f loc=%.0f trend=%.0f vol=%.0f "
            "target=%.0f shares=%d" % (
                code, score["buy_score"], score["reversal_score"],
                score["location_score"], score["trend_score"],
                score["volume_score"], target_value, shares))
        try:
            order_id = order(code, shares, limit_price=round(price, 3))
        except Exception as exc:
            log.error("[buy] %s submission failed: %s" % (code, exc))
            continue
        if order_id is None:
            log.error("[buy] %s submission returned no order id" % code)
            continue
        if getattr(g, "__is_live", False):
            g.__pending_orders[code] = {
                "requested_qty": shares,
                "filled_qty": 0.0,
                "filled_value": 0.0,
                "fill_value_complete": True,
                "atr": score["atr"],
                "buy_date": today,
                "order_id": str(order_id),
            }
        else:
            g.buy_date[code] = today
            g.highest_since_buy[code] = price
            g.entry_atr[code] = score["atr"]
        available -= shares * price
        bought += 1
    return bought


def do_trading(context):
    p = g.params
    if getattr(g, "__is_live", False) and getattr(g, "__order_state_unknown", False):
        log.error("[trade] broker order state unknown; no orders submitted")
        return
    today = _as_date(get_context_datetime(context))
    prev_date = get_prev_trade_date(context)
    if today is None or prev_date is None:
        log.error("[trade] date boundary unavailable; no orders submitted")
        return
    g.execution_date = today
    g.deferred_signal_date = prev_date
    g.deferred_scores = []
    is_rebalance = today.weekday() in p["rebalance_weekdays"]
    g.paused_pool_codes = _find_paused_pool_codes(g.etf_pool, is_paused)

    log.info("[%s] date=%s signal_date=%s rebalance=%s" % (
        STRATEGY_VERSION, today, prev_date, is_rebalance))

    stop_hits = check_atr_stops(context)
    for code, stop_price, price in stop_hits:
        execute_sell(code, context, "atr_stop %.3f<=%.3f" % (price, stop_price))

    if not is_rebalance:
        if not stop_hits:
            log.info("[cross-v0.1] non-rebalance day: stop check passed")
        return

    all_scores = []
    skip_reasons = {}
    for code in g.etf_pool:
        if code in g.paused_pool_codes:
            skip_reasons[code] = "paused"
            continue
        score, reason = calc_cross_signal_score(code, prev_date, return_reason=True)
        if score is not None:
            all_scores.append(score)
        else:
            skip_reasons[code] = reason or "unknown"

    if not all_scores:
        reason_counts = {}
        for reason in skip_reasons.values():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        summary = " | ".join("%s=%d" % (k, v) for k, v in sorted(reason_counts.items()))
        samples = " | ".join("%s:%s" % (c, r) for c, r in sorted(skip_reasons.items())[:6])
        log.info("[cross-v0.1] no valid scores")
        log.info("[score skip summary] %s" % summary)
        log.info("[score skip samples] %s" % samples)
        return

    if skip_reasons:
        reason_counts = {}
        for reason in skip_reasons.values():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        summary = " | ".join("%s=%d" % (k, v) for k, v in sorted(reason_counts.items()))
        log.info("[score skip summary] %s" % summary)

    all_scores = sort_candidates(all_scores)
    score_map = {s["code"]: s for s in all_scores}
    g.last_scores = score_map

    log.info("[top candidates]")
    for item in all_scores[:5]:
        log.info(
            "  %s buy=%.0f rev=%.0f loc=%.0f trend=%.0f vol=%.0f sell=%.0f "
            "close=%.3f %s" % (
                item["code"], item["buy_score"], item["reversal_score"],
                item["location_score"], item["trend_score"], item["volume_score"],
                item["sell_score"], item["close"], format_indicator_values(item)))

    cross_summary = summarize_cross_signal_candidates(all_scores)
    if cross_summary["count"] == 0:
        log.info("[cross signals] none in full pool")
    else:
        log.info("[cross signals] count=%d" % cross_summary["count"])
        for item in cross_summary["items"]:
            log.info(
                "  %s rev=%.0f buy=%.0f sell=%.0f %s %s" % (
                    item["code"], item["reversal_score"], item["buy_score"], item["sell_score"],
                    format_cross_flags(item),
                    format_indicator_values(item)))

    loose_summary = summarize_loose_reversal_candidates(all_scores)
    if loose_summary["count"] == 0:
        log.info("[loose reversal] none in full pool")
    else:
        log.info("[loose reversal] count=%d" % loose_summary["count"])
        for item in loose_summary["items"]:
            log.info(
                "  %s loose=%d buy=%.0f rev=%.0f "
                "RSI_UP=%s dRSI=%.2f MACD_UP=%s dDIF=%.4f KDJ_UP=%s dK=%.2f dJ=%.2f %s" % (
                    item["code"], item["loose_reversal_count"],
                    item["buy_score"], item["reversal_score"],
                    item["rsi_turn_up"], item["rsi6_delta"],
                    item["macd_turn_up"], item["dif_delta"],
                    item["kdj_turn_up"], item["k_delta"], item["j_delta"],
                    format_indicator_values(item)))

    held = current_hold_codes(context)
    try:
        signal_hold_days = get_trade_days(
            end_date=today,
            count=max(2, int(p.get("min_signal_hold_days", 1)) + 1),
        )
    except Exception as exc:
        signal_hold_days = None
        log.warning("[min-hold] trade-day query failed; signal sells blocked: %s" % exc)
    for code in list(held):
        if code not in score_map:
            continue
        if code in g.unverified_positions:
            log.error("[hold] %s risk state unverified; automatic signal sell blocked" % code)
            continue
        if is_paused(code):
            log.info("[hold] %s paused, skip signal sell" % code)
            continue
        score = score_map[code]
        if not can_sell_with_verified_calendar(
            g.buy_date.get(code),
            today,
            min_hold_days=p.get("min_signal_hold_days", 1),
            trade_days=signal_hold_days,
        ):
            log.info("[hold] %s min-hold, skip signal sell" % code)
            continue
        if score["buy_score"] >= p["strong_buy_threshold"] and score["sell_score"] < p["sell_threshold"]:
            log.info("[hold] %s strong buy_score %.0f sell_score %.0f" % (
                code, score["buy_score"], score["sell_score"]))
            continue
        if should_force_sell(score, False, p):
            execute_sell(code, context, "sell_score %.0f" % score["sell_score"])
        elif score["sell_score"] >= p["risk_tighten_threshold"]:
            log.info("[risk-tighten] %s sell_score %.0f" % (code, score["sell_score"]))

    g.deferred_scores = list(all_scores)
    execute_buy_candidates(context, all_scores, today)


def after_close(context):
    total = _total_value(context)
    cash = _available_cash(context)
    holds = current_hold_codes(context)
    log.info("=" * 60)
    log.info("[%s close] total=%.2f cash=%.2f holdings=%d/%d" % (
        STRATEGY_VERSION, total, cash, len(holds), g.params["max_hold"]))
    for code in holds:
        if code in g.unverified_positions:
            log.error("  %s risk state unverified; close/ATR state not updated" % code)
            continue
        price = get_current_price(code)
        pos = _get_position(context, code)
        if price is None or price <= 0:
            continue
        prev_high = g.highest_since_buy.get(code, price)
        g.highest_since_buy[code] = max(prev_high, price)
        if code not in g.entry_atr:
            score = g.last_scores.get(code)
            if score is not None and score.get("atr") and not pd.isna(score.get("atr")):
                g.entry_atr[code] = score["atr"]
        atr_val = g.entry_atr.get(code, np.nan)
        cost = _pos_cost(pos)
        stop_price = calc_stop_price(g.highest_since_buy[code], atr_val, cost, g.params) \
            if not pd.isna(atr_val) else np.nan
        pnl = (price - cost) / cost if cost > 0 else 0
        score = g.last_scores.get(code, {})
        log.info("  %s cost=%.3f price=%.3f high=%.3f pnl=%.1f%% buy=%.0f sell=%.0f stop=%.3f" % (
            code, cost, price, g.highest_since_buy[code], pnl * 100,
            score.get("buy_score", 0), score.get("sell_score", 0), stop_price))
    log.info("=" * 60)


def halt_recover(context):
    if getattr(g, "__order_state_unknown", False):
        log.error("[halt-recover] broker order state unknown; no orders submitted")
        return
    today = _as_date(get_context_datetime(context))
    prev_date = get_prev_trade_date(context) if today is not None else None
    if (
        today is None or prev_date is None or
        g.execution_date != today or
        g.deferred_signal_date != prev_date
    ):
        log.error("[halt-recover] deferred score date mismatch; no orders submitted")
        return
    if not _reconcile_open_orders(context):
        return
    _recover_live_state_with_available_sources(context, allow_deliver=False)
    previous = set(getattr(g, "paused_pool_codes", set()))
    still_paused = set(code for code in previous if is_paused(code))
    recovered = sorted(previous - still_paused)
    g.paused_pool_codes = still_paused
    scores = list(getattr(g, "deferred_scores", []))
    if recovered:
        by_code = {item["code"]: item for item in scores}
        for code in recovered:
            score, reason = calc_cross_signal_score(code, prev_date, return_reason=True)
            if score is not None:
                by_code[code] = score
                g.last_scores[code] = score
            else:
                log.warning("[halt-recover] %s score unavailable: %s" % (code, reason))
        scores = sort_candidates(list(by_code.values()))
        g.deferred_scores = scores
        log.info("[halt-recover] resumed=%s; evaluate deferred buys only" % ",".join(recovered))
    elif previous:
        log.info("[halt-recover] no tracked ETF resumed")
    if scores:
        execute_buy_candidates(context, scores, today)


def _has_incomplete_position_state(context):
    for code in current_hold_codes(context):
        if (
            _as_date(g.buy_date.get(code)) is None or
            not _is_positive_finite(g.entry_atr.get(code)) or
            not _is_positive_finite(g.highest_since_buy.get(code))
        ):
            return True
    return False


def _recover_live_state_with_available_sources(context, allow_deliver):
    if not _has_incomplete_position_state(context):
        recover_live_state(context)
        return
    prev_date = get_prev_trade_date(context)
    current_records = _fetch_current_strategy_trades()
    deliver_records = (
        _fetch_deliver_records(prev_date)
        if allow_deliver and prev_date is not None else None
    )
    recover_live_state(
        context,
        deliver_records=deliver_records,
        current_trade_records=current_records,
        prev_date=prev_date,
    )


def _fetch_deliver_records(prev_date):
    """Fetch broker delivery records once, only from the supported pre-open phase."""
    end_date = _as_date(prev_date)
    if end_date is None:
        return []
    getter = globals().get("get_deliver")
    if getter is None:
        log.error("[recovery] get_deliver unavailable")
        return []
    end_text = end_date.strftime("%Y%m%d")
    try:
        records = getter(DELIVER_RECOVERY_START_DATE, end_text)
    except Exception as exc:
        log.error("[recovery] get_deliver failed: %s" % exc)
        return []
    if not isinstance(records, (list, tuple)):
        log.error("[recovery] invalid get_deliver response")
        return []
    log.info("[recovery] delivery records=%d range=%s~%s" % (
        len(records), DELIVER_RECOVERY_START_DATE, end_text))
    return list(records)


def _fetch_current_strategy_trades():
    """Normalize PTrade's strategy-only current-day fills into delivery rows."""
    getter = globals().get("get_trades")
    if getter is None:
        log.error("[recovery] get_trades unavailable")
        return []
    try:
        payload = getter()
    except Exception as exc:
        log.error("[recovery] get_trades failed: %s" % exc)
        return []
    if not isinstance(payload, dict):
        log.error("[recovery] invalid get_trades response")
        return []

    records = []
    for order_id, fills in payload.items():
        if not isinstance(fills, (list, tuple)):
            continue
        for fill in fills:
            if not isinstance(fill, (list, tuple)) or len(fill) < 8:
                continue
            side = str(fill[3] or "").strip()
            if "\u4e70" in side:
                entrust_bs = "1"
            elif "\u5356" in side:
                entrust_bs = "2"
            else:
                continue
            try:
                trade_time = pd.Timestamp(fill[7])
            except Exception:
                continue
            records.append({
                "stock_code": fill[2],
                "entrust_bs": entrust_bs,
                "business_amount": fill[4],
                "business_price": fill[5],
                "init_date": trade_time.strftime("%Y%m%d"),
                "business_time": trade_time.strftime("%H%M%S"),
                "order_id": str(order_id),
            })
    log.info("[recovery] current strategy trades=%d" % len(records))
    return records


def _delivery_trade_date(record):
    for field in ("init_date", "entrust_date", "date_back", "business_date"):
        raw = record.get(field)
        digits = "".join(ch for ch in str(raw or "") if ch.isdigit())
        if len(digits) < 8:
            continue
        try:
            return datetime.strptime(digits[:8], "%Y%m%d").date()
        except Exception:
            continue
    return None


def _delivery_direction(record):
    side = str(record.get("entrust_bs", "") or "").strip()
    business_name = str(record.get("business_name", "") or "")
    if side == "1" or "\u4e70\u5165" in business_name:
        return 1
    if side == "2" or "\u5356\u51fa" in business_name:
        return -1
    return 0


def _delivery_quantity(record):
    for field in ("business_amount", "occur_amount"):
        value = _safe_float(record.get(field), np.nan)
        if np.isfinite(value) and abs(value) > 0:
            return abs(value)
    return 0.0


def _delivery_sort_key(record):
    trade_date = _delivery_trade_date(record)
    if trade_date is None:
        return (datetime.max.date(), 0, 0)
    trade_time = int(abs(_safe_float(
        record.get("business_time", record.get("report_time", 0)), 0
    )))
    serial = int(abs(_safe_float(
        record.get("serial_no", record.get("business_no", 0)), 0
    )))
    return (trade_date, trade_time, serial)


def _reconstruct_open_position(records, code, broker_amount):
    """Rebuild the current open episode and require exact broker-quantity parity."""
    target = normalize_code(code)
    expected = _safe_float(broker_amount, np.nan)
    if not _is_positive_finite(expected):
        return None
    matched = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        if normalize_code(record.get("stock_code")) != target:
            continue
        direction = _delivery_direction(record)
        quantity = _delivery_quantity(record)
        trade_date = _delivery_trade_date(record)
        if direction == 0 or quantity <= 0 or trade_date is None:
            continue
        matched.append(record)
    if not matched:
        return None

    amount = 0.0
    buy_date = None
    entry_quantity = 0.0
    entry_value = 0.0
    tolerance = max(1e-6, expected * 1e-8)
    for record in sorted(matched, key=_delivery_sort_key):
        direction = _delivery_direction(record)
        quantity = _delivery_quantity(record)
        if direction > 0:
            if amount <= tolerance:
                buy_date = _delivery_trade_date(record)
                entry_quantity = 0.0
                entry_value = 0.0
            price = _safe_float(record.get("business_price"), np.nan)
            if _is_positive_finite(price):
                entry_quantity += quantity
                entry_value += quantity * price
            amount += quantity
        else:
            amount -= quantity
            if amount < -tolerance:
                return None
            if abs(amount) <= tolerance:
                amount = 0.0
                buy_date = None
                entry_quantity = 0.0
                entry_value = 0.0

    if buy_date is None or abs(amount - expected) > tolerance:
        return None
    entry_price = entry_value / entry_quantity if entry_quantity > 0 else None
    return {
        "buy_date": buy_date,
        "amount": amount,
        "entry_price": entry_price,
    }


def _previous_trade_date_before(value):
    trade_date = _as_date(value)
    if trade_date is None:
        return None
    try:
        result = get_trade_days(end_date=trade_date, count=2)
        previous = _previous_day_from_result(result, trade_date)
        if previous is not None:
            return previous
    except Exception as exc:
        log.warning("[recovery] get_trade_days failed: %s" % exc)
    try:
        result = get_all_trades_days(date=trade_date.strftime("%Y%m%d"))
        previous = _previous_day_from_result(result, trade_date)
        if previous is not None:
            return previous
    except Exception as exc:
        log.warning("[recovery] get_all_trades_days failed: %s" % exc)
    return None


def _get_recovery_close_data(code, start_date, end_date):
    """Load comparable pre-adjusted closes for disaster recovery only."""
    start = _as_date(start_date)
    end = _as_date(end_date)
    if start is None or end is None or start > end:
        return None
    try:
        frame = get_price(
            code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            frequency="1d",
            fields=["close", "volume"],
            fq="pre",
        )
        frame = pd.DataFrame(frame).copy()
    except Exception as exc:
        log.error("[recovery] close history unavailable %s: %s" % (code, exc))
        return None
    if "code" in frame.columns:
        frame = frame[frame["code"].map(normalize_code) == normalize_code(code)]
    if "close" not in frame.columns or "volume" not in frame.columns:
        return None
    frame = frame[frame["volume"] > 0]
    try:
        index_dates = pd.to_datetime(frame.index).date
        frame = frame[(index_dates >= start) & (index_dates <= end)]
    except Exception:
        return None
    return frame


def _is_proven_strategy_entry(score):
    if not isinstance(score, dict):
        return False
    return bool(filter_buy_candidates([score], [], g.params))


def _recover_position_from_broker(code, pos, records, prev_date):
    if code not in set(g.etf_pool):
        return False
    amount = _pos_amount(pos)
    cost = _pos_cost(pos)
    if not _is_positive_finite(amount) or not _is_positive_finite(cost):
        return False
    open_position = _reconstruct_open_position(records, code, amount)
    if open_position is None:
        return False
    buy_date = open_position["buy_date"]
    signal_date = _previous_trade_date_before(buy_date)
    if signal_date is None:
        return False
    score = calc_cross_signal_score(code, signal_date)
    if not _is_proven_strategy_entry(score):
        return False
    atr = score.get("atr")
    if not _is_positive_finite(atr):
        return False
    entry_price = open_position.get("entry_price")
    if not _is_positive_finite(entry_price):
        return False
    prev_date = _as_date(prev_date)
    if prev_date is None:
        return False
    if buy_date <= prev_date:
        closes = _get_recovery_close_data(code, buy_date, prev_date)
        if closes is None or len(closes) == 0:
            return False
        valid_closes = pd.to_numeric(closes["close"], errors="coerce")
        valid_closes = valid_closes[np.isfinite(valid_closes) & (valid_closes > 0)]
        if len(valid_closes) == 0:
            return False
        highest = max(float(entry_price), float(valid_closes.max()))
    else:
        if signal_date != prev_date:
            return False
        highest = float(entry_price)

    g.buy_date[code] = buy_date
    g.entry_atr[code] = float(atr)
    g.highest_since_buy[code] = highest
    log.warning(
        "[recovery] %s rebuilt from broker delivery: buy_date=%s "
        "signal_date=%s ATR=%.6f highest_close=%.6f cost=%.6f" % (
            code, buy_date, signal_date, atr, highest, cost)
    )
    return True


def _prune_closed_position_state(held):
    for field in ("highest_since_buy", "entry_atr", "buy_date"):
        mapping = getattr(g, field, {})
        for code in list(mapping.keys()):
            if normalize_code(code) not in held:
                mapping.pop(code, None)
    g.unverified_positions.intersection_update(held)


def recover_live_state(
        context, deliver_records=None, current_trade_records=None, prev_date=None):
    """Verify persisted state, then rebuild only facts provable from broker data."""
    held = set(current_hold_codes(context))
    _prune_closed_position_state(held)
    recovery_records = None
    if deliver_records is not None or current_trade_records is not None:
        recovery_records = list(deliver_records or []) + list(current_trade_records or [])
    for code in held:
        pos = _get_position(context, code)
        buy_date = _as_date(g.buy_date.get(code))
        complete = (
            pos is not None and
            _is_positive_finite(_pos_cost(pos)) and
            buy_date is not None and
            _is_positive_finite(g.highest_since_buy.get(code)) and
            _is_positive_finite(g.entry_atr.get(code))
        )
        if not complete and recovery_records is not None and prev_date is not None:
            complete = _recover_position_from_broker(
                code, pos, recovery_records, prev_date
            )
        if complete:
            g.unverified_positions.discard(code)
        else:
            g.unverified_positions.add(code)
            log.error(
                "[recovery] %s historical buy date/ATR/high or broker cost "
                "cannot be proved; automatic exits blocked" % code
            )


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_positive_finite(value):
    number = _safe_float(value, np.nan)
    return bool(np.isfinite(number) and number > 0)


def _complete_buy(code, pending):
    quantity = pending.get("filled_qty", 0.0)
    if quantity <= 0:
        return
    _apply_buy_fill_state(code, pending)
    g.__pending_orders.pop(code, None)


def _apply_buy_fill_state(code, pending):
    quantity = pending.get("filled_qty", 0.0)
    value_complete = bool(pending.get("fill_value_complete", True))
    average = pending.get("filled_value", 0.0) / quantity if quantity > 0 else np.nan
    buy_date = _as_date(pending.get("buy_date"))
    atr = pending.get("atr")
    verified = (
        quantity > 0 and
        value_complete and
        _is_positive_finite(average) and
        buy_date is not None and
        _is_positive_finite(atr)
    )
    if buy_date is not None:
        g.buy_date[code] = buy_date
    if _is_positive_finite(atr):
        g.entry_atr[code] = atr
    if verified:
        g.highest_since_buy[code] = average
        g.unverified_positions.discard(code)
    else:
        g.unverified_positions.add(code)
        log.error("[fill] %s entry fill baseline unverified; automatic exits blocked" % code)


def _pending_completion_qty(pending):
    return float(pending.get("terminal_qty", pending.get("requested_qty", 0.0)))


def _response_matches_pending(response, pending):
    response_id = str(response.get("order_id", "") or "")
    pending_id = str(pending.get("order_id", "") or "")
    return bool(response_id and pending_id and response_id == pending_id)


def _finish_terminal_sell(code, pending):
    if pending.get("filled_qty", 0.0) < _pending_completion_qty(pending):
        return False
    g.__pending_sells.pop(code, None)
    if pending.get("filled_qty", 0.0) >= pending.get("requested_qty", 0.0):
        _clear_position_state(code)
    else:
        g.sold_today.pop(code, None)
        log.warning("[sell residual] %s partial fill retained; risk state preserved" % code)
    return True


def on_order_response(context, order_list):
    if not getattr(g, "__is_live", False):
        return
    orders = order_list if isinstance(order_list, list) else [order_list]
    for response in orders:
        code = normalize_code(response.get("stock_code"))
        status = str(response.get("status", ""))
        if not code or status not in ("5", "6", "9"):
            continue
        filled = _safe_float(response.get("business_amount", 0))
        error = response.get("error_info", "")
        buy_pending = getattr(g, "__pending_orders", {}).get(code)
        sell_pending = getattr(g, "__pending_sells", {}).get(code)

        if buy_pending is not None:
            if not _response_matches_pending(response, buy_pending):
                log.warning("[order callback] ignored unmatched buy order %s" % code)
                continue
            if filled > 0 and status in ("5", "6"):
                buy_pending["terminal_qty"] = filled
                if buy_pending.get("filled_qty", 0.0) >= filled:
                    _complete_buy(code, buy_pending)
                log.warning("[buy partial/cancel] %s filled=%.0f reason=%s" % (
                    code, filled, error))
            else:
                g.__pending_orders.pop(code, None)
                log.warning("[buy rejected/cancelled] %s status=%s reason=%s" % (
                    code, status, error))
            continue

        if sell_pending is not None:
            if not _response_matches_pending(response, sell_pending):
                log.warning("[order callback] ignored unmatched sell order %s" % code)
                continue
            if filled > 0 and status in ("5", "6"):
                sell_pending["terminal_qty"] = filled
                _finish_terminal_sell(code, sell_pending)
                log.warning("[sell partial/cancel] %s filled=%.0f reason=%s" % (
                    code, filled, error))
            else:
                g.__pending_sells.pop(code, None)
                g.sold_today.pop(code, None)
                log.error("[sell rejected/cancelled] %s status=%s reason=%s" % (
                    code, status, error))
    _persist_live_state()


def on_trade_response(context, trade_list):
    if not getattr(g, "__is_live", False):
        return
    trades = trade_list if isinstance(trade_list, list) else [trade_list]
    for trade in trades:
        if str(trade.get("real_type", "")) == "2":
            log.info("[fill] cancellation push ignored")
            continue
        code = normalize_code(trade.get("stock_code"))
        direction = str(trade.get("entrust_bs", ""))
        quantity = _safe_float(trade.get("business_amount", 0))
        price = _safe_float(trade.get("business_price", 0))
        if not code or quantity <= 0:
            continue

        if direction == "1":
            pending = getattr(g, "__pending_orders", {}).get(code)
            if pending is None:
                log.warning("[fill] unmatched buy %s qty=%.0f" % (code, quantity))
                continue
            if not _response_matches_pending(trade, pending):
                log.warning("[fill] ignored old/unmatched buy order %s" % code)
                continue
            pending["filled_qty"] = pending.get("filled_qty", 0.0) + quantity
            if _is_positive_finite(price):
                pending["filled_value"] = pending.get("filled_value", 0.0) + quantity * price
            else:
                pending["fill_value_complete"] = False
            _apply_buy_fill_state(code, pending)
            if pending["filled_qty"] >= _pending_completion_qty(pending):
                g.__pending_orders.pop(code, None)
            log.info("[fill] buy %s qty=%.0f @%.3f cumulative=%.0f" % (
                code, quantity, price, pending["filled_qty"]))

        elif direction == "2":
            pending = getattr(g, "__pending_sells", {}).get(code)
            if pending is None:
                log.warning("[fill] unmatched sell %s qty=%.0f" % (code, quantity))
                continue
            if not _response_matches_pending(trade, pending):
                log.warning("[fill] ignored old/unmatched sell order %s" % code)
                continue
            pending["filled_qty"] = pending.get("filled_qty", 0.0) + quantity
            _finish_terminal_sell(code, pending)
            log.info("[fill] sell %s qty=%.0f @%.3f cumulative=%.0f" % (
                code, quantity, price, pending["filled_qty"]))
    _persist_live_state()
