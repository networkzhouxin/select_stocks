# -*- coding: utf-8 -*-
"""
Cross-Signal ETF Strategy v0.1 for JoinQuant.

Research protocol:
- Develop first on 2019-01-01 to 2021-12-31 only.
- Use T-1 daily bars for all signals.
- Keep this file isolated from the production multi-factor strategy.
"""

import numpy as np
import pandas as pd
import builtins as _builtins
from jqdata import *


STRATEGY_VERSION = "cross-v0.3.2-sell35-candidate"


try:
    log
except NameError:
    class _LocalLog(object):
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
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
        "sell_threshold": 35,
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
        "159915.XSHE",
        "512100.XSHG",
        "159928.XSHE",
        "513100.XSHG",
        "513500.XSHG",
        "513880.XSHG",
        "513050.XSHG",
        "518880.XSHG",
        "159985.XSHE",
    ]


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
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)

    set_slippage(PriceRelatedSlippage(0.001))
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0,
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5
    ), type="stock")

    g.params = get_default_params()
    g.etf_pool = get_default_etf_pool()
    g.highest_since_buy = {}
    g.entry_atr = {}
    g.buy_date = {}
    g.last_scores = {}

    run_daily(do_trading, time="09:35")
    run_daily(after_close, time="15:30")

    log.info("[%s] initialized: max_hold=%d base_ratio=%.2f min_signal_hold=%d" % (
        STRATEGY_VERSION,
        g.params["max_hold"],
        g.params["base_ratio"],
        g.params["min_signal_hold_days"]))
    log.info(format_self_check())
    log.info("[indicator params] %s" % format_indicator_params(g.params))


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


def get_prev_trade_date(context):
    days = get_trade_days(end_date=context.current_dt.date(), count=2)
    if len(days) >= 2:
        return days[-2]
    return context.previous_date


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
        df = get_price(
            code,
            end_date=end_date,
            count=p["lookback"],
            frequency="daily",
            fields=["open", "close", "high", "low", "volume"],
            skip_paused=True,
            fq="pre",
        )
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
        code for code, pos in context.portfolio.positions.items()
        if getattr(pos, "total_amount", 0) > 0
    ]


def has_position(context, code):
    for held_code, pos in context.portfolio.positions.items():
        if held_code == code and getattr(pos, "total_amount", 0) > 0:
            return True
    return False


def sync_sell_state_after_order(code, context):
    if has_position(context, code):
        return
    g.highest_since_buy.pop(code, None)
    g.entry_atr.pop(code, None)
    g.buy_date.pop(code, None)
    g.last_scores.pop(code, None)


def sync_buy_state_after_order(code, context, today, price, atr):
    if not has_position(context, code):
        return
    g.buy_date[code] = today
    g.highest_since_buy[code] = price
    g.entry_atr[code] = atr


def is_paused(current_data, code):
    try:
        return current_data[code].paused
    except Exception:
        return False


def current_price(current_data, code):
    try:
        return current_data[code].last_price
    except Exception:
        return None


def execute_sell(code, context, reason):
    pos = context.portfolio.positions[code]
    log.info("[sell] %s reason=%s amount=%s" % (code, reason, pos.total_amount))
    order_target(code, 0)
    sync_sell_state_after_order(code, context)


def check_atr_stops(context, current_data):
    triggered = []
    for code in current_hold_codes(context):
        if is_paused(current_data, code):
            continue
        pos = context.portfolio.positions[code]
        price = current_price(current_data, code)
        if price is None or price <= 0:
            continue
        if code not in g.highest_since_buy or code not in g.entry_atr:
            continue
        stop_price = calc_stop_price(g.highest_since_buy[code], g.entry_atr[code], pos.avg_cost, g.params)
        if price <= stop_price:
            triggered.append((code, stop_price, price))
    return triggered


def do_trading(context):
    p = g.params
    today = context.current_dt.date()
    prev_date = get_prev_trade_date(context)
    current_data = get_current_data()
    is_rebalance = today.weekday() in p["rebalance_weekdays"]

    log.info("[cross-v0.1] date=%s signal_date=%s rebalance=%s" % (
        today, prev_date, is_rebalance))

    stop_hits = check_atr_stops(context, current_data)
    for code, stop_price, price in stop_hits:
        execute_sell(code, context, "atr_stop %.3f<=%.3f" % (price, stop_price))

    if not is_rebalance:
        if not stop_hits:
            log.info("[cross-v0.1] non-rebalance day: stop check passed")
        return

    all_scores = []
    skip_reasons = {}
    for code in g.etf_pool:
        if is_paused(current_data, code):
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
    signal_hold_days = get_trade_days(
        end_date=today,
        count=max(2, int(p.get("min_signal_hold_days", 1)) + 1),
    )
    for code in list(held):
        if code not in score_map:
            continue
        if is_paused(current_data, code):
            log.info("[hold] %s paused, skip signal sell" % code)
            continue
        score = score_map[code]
        if not can_sell_by_signal(
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

    held_after_sell = current_hold_codes(context)
    slots = p["max_hold"] - len(held_after_sell)
    if slots <= 0:
        return

    candidates = filter_buy_candidates(all_scores, held_after_sell, p)
    if not candidates:
        log.info("[cross-v0.1] no buy candidates above threshold")
        return

    bought = 0
    for score in candidates:
        if bought >= slots:
            break
        code = score["code"]
        if is_paused(current_data, code):
            continue
        price = current_price(current_data, code)
        if price is None or price <= 0:
            continue
        target_value = calc_buy_target_value(context.portfolio.total_value, score, p)
        log.info("[buy] %s buy=%.0f rev=%.0f loc=%.0f trend=%.0f vol=%.0f target=%.0f" % (
            code, score["buy_score"], score["reversal_score"], score["location_score"],
            score["trend_score"], score["volume_score"], target_value))
        order_target_value(code, target_value)
        sync_buy_state_after_order(code, context, today, price, score["atr"])
        bought += 1


def after_close(context):
    current_data = get_current_data()
    total = context.portfolio.total_value
    cash = context.portfolio.available_cash
    holds = current_hold_codes(context)
    log.info("=" * 60)
    log.info("[cross-v0.1 close] total=%.2f cash=%.2f holdings=%d/%d" % (
        total, cash, len(holds), g.params["max_hold"]))
    for code in holds:
        price = current_price(current_data, code)
        pos = context.portfolio.positions[code]
        if price is None or price <= 0:
            continue
        prev_high = g.highest_since_buy.get(code, price)
        g.highest_since_buy[code] = max(prev_high, price)
        if code not in g.entry_atr:
            score = g.last_scores.get(code)
            if score is not None and score.get("atr") and not pd.isna(score.get("atr")):
                g.entry_atr[code] = score["atr"]
        atr_val = g.entry_atr.get(code, np.nan)
        stop_price = calc_stop_price(g.highest_since_buy[code], atr_val, pos.avg_cost, g.params) \
            if not pd.isna(atr_val) else np.nan
        pnl = (price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
        score = g.last_scores.get(code, {})
        log.info("  %s cost=%.3f price=%.3f high=%.3f pnl=%.1f%% buy=%.0f sell=%.0f stop=%.3f" % (
            code, pos.avg_cost, price, g.highest_since_buy[code], pnl * 100,
            score.get("buy_score", 0), score.get("sell_score", 0), stop_price))
    log.info("=" * 60)
