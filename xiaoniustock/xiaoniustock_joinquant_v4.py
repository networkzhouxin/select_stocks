# -*- coding: utf-8 -*-
"""Xiaoniu V4: convergence, volume breakout, pullback confirmation.

The pure helpers in this module are intentionally independent from JoinQuant so
their signal timing and risk calculations can be regression tested locally.
"""

import numpy as np

try:
    from jqdata import *
except ImportError:
    # Pure helpers remain importable for local tests outside JoinQuant.
    pass


MA_PERIODS = (10, 20, 60)
BREAKOUT_LOOKBACK = 20
CONVERGENCE_THRESHOLD = 0.03
CONVERGENCE_WINDOW = 5
CONVERGENCE_MIN_DAYS = 3
BREAKOUT_VOLUME_RATIO = 1.5
CONFIRMATION_MIN_DAYS = 1
CONFIRMATION_MAX_DAYS = 5
SUPPORT_LOW_TOLERANCE = 0.01
STOP_BUFFER = 0.01
RISK_FRACTION = 0.01
MAX_POSITION_FRACTION = 0.30
LOT_SIZE = 100
MARKET_MA_PERIOD = 60
MARKET_MA_SLOPE_SESSIONS = 5
MAX_HOLD_DAYS = 20
MAX_HOLDINGS = 3
MAX_EXECUTION_GAP = 0.03
HISTORY_COUNT = 90
HISTORY_BATCH_SIZE = 50
MARKET_INDEX = "000300.XSHG"
UNIVERSE_INDEXES = ("000300.XSHG", "000905.XSHG")


def _log_info(message):
    logger = globals().get("log")
    if logger is not None:
        logger.info(message)


def mapping_get(mapping, key, default=None):
    """Read dict-like JoinQuant containers that may not implement ``get``."""
    try:
        return mapping[key]
    except (KeyError, TypeError):
        getter = getattr(mapping, "get", None)
        return getter(key, default) if getter is not None else default


def initialize(context):
    """Configure the isolated V4 JoinQuant strategy."""
    set_benchmark(MARKET_INDEX)
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_slippage(PriceRelatedSlippage(0.002))
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5,
    ), type="stock")

    g.prepared_for = None
    g.signal_date = None
    g.market_open = False
    g.candidates = []
    g.exit_reasons = {}
    g.last_scan_stats = {}
    g.managed_codes = set()
    g.stop_prices = {}
    g.entry_dates = {}
    g.sold_today = set()
    g.last_close_position_count = 0

    run_daily(prepare_daily_state, time="09:30")
    run_daily(execute_trades, time="09:35")
    run_daily(record_close_state, time="15:00")
    run_daily(after_close, time="15:30")


def get_prev_trade_date(context):
    """Resolve T-1 explicitly at the start of each trading day."""
    today = context.current_dt.date()
    trade_days = get_trade_days(end_date=today, count=2)
    return trade_days[0]


def _frame_to_bars(frame):
    if frame is None:
        return None
    required = ("open", "high", "low", "close", "volume")
    try:
        bars = {
            name: np.asarray(frame[name], dtype=float)
            for name in required
        }
    except (KeyError, TypeError, ValueError):
        return None

    if len({len(values) for values in bars.values()}) != 1:
        return None

    columns = getattr(frame, "columns", ())
    try:
        if "time" in columns:
            final_time = frame["time"].iloc[-1]
        else:
            final_time = frame.index[-1]
        bars["last_date"] = (
            final_time.date() if hasattr(final_time, "date") else final_time
        )
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    return bars


def load_daily_bars(code, prev_date, count=HISTORY_COUNT):
    """Load one security's completed daily bars, explicitly ending at T-1."""
    frame = get_price(
        code,
        end_date=prev_date,
        count=count,
        frequency="daily",
        fields=["open", "high", "low", "close", "volume"],
        skip_paused=True,
        fq="pre",
    )
    return _frame_to_bars(frame)


def get_market_closes(prev_date):
    """Load only completed CSI 300 closes for the market gate."""
    frame = get_price(
        MARKET_INDEX,
        end_date=prev_date,
        count=MARKET_MA_PERIOD + MARKET_MA_SLOPE_SESSIONS,
        frequency="daily",
        fields=["close"],
        skip_paused=True,
        fq="pre",
    )
    if frame is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(frame["close"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return np.array([], dtype=float)


def get_point_in_time_universe(prev_date):
    """Return T-1 HS300+CSI500 main-board members excluding T-1 ST names."""
    pool = set()
    for index_code in UNIVERSE_INDEXES:
        pool.update(get_index_stocks(index_code, date=prev_date))
    pool = sorted(
        code for code in pool
        if code.split(".", 1)[0].startswith(("60", "00"))
    )
    if not pool:
        return []

    st_frame = get_extras(
        "is_st",
        pool,
        start_date=prev_date,
        end_date=prev_date,
        df=True,
    )
    if st_frame is None or getattr(st_frame, "empty", True):
        return pool

    try:
        row = st_frame.iloc[-1]
        return [
            code for code in pool
            if code not in row.index or not bool(row[code])
        ]
    except (AttributeError, IndexError, KeyError, TypeError):
        return pool


def _load_daily_bars_batch(codes, prev_date):
    """Load a small batch while retaining one causal series per code."""
    frame = get_price(
        codes,
        end_date=prev_date,
        count=HISTORY_COUNT,
        frequency="daily",
        fields=["open", "high", "low", "close", "volume", "paused"],
        skip_paused=False,
        fq="pre",
        panel=False,
    )
    if frame is None or getattr(frame, "empty", False):
        return {}

    columns = getattr(frame, "columns", ())
    if "code" not in columns:
        return {codes[0]: _frame_to_bars(frame)} if len(codes) == 1 else {}

    result = {}
    try:
        grouped = frame.groupby("code", sort=False)
        for code, code_frame in grouped:
            if "time" in code_frame.columns:
                code_frame = code_frame.sort_values("time")
            volume = np.asarray(code_frame["volume"], dtype=float)
            tradable = np.isfinite(volume) & (volume > 0)
            if "paused" in code_frame.columns:
                paused = np.asarray(code_frame["paused"], dtype=bool)
                tradable &= ~paused
            code_frame = code_frame.loc[tradable]
            if code_frame.empty:
                continue
            result[code] = _frame_to_bars(code_frame)
    except (AttributeError, KeyError, TypeError, ValueError):
        return {}
    return result


def scan_candidates(universe, prev_date):
    """Scan the point-in-time universe using daily bars ending at T-1."""
    candidates = []
    stats = {
        "universe": len(universe),
        "loaded": 0,
        "stale": 0,
        "evaluated": 0,
        "signals": 0,
    }
    for start in range(0, len(universe), HISTORY_BATCH_SIZE):
        batch = universe[start:start + HISTORY_BATCH_SIZE]
        histories = _load_daily_bars_batch(batch, prev_date)
        stats["loaded"] += len(histories)
        for code in batch:
            bars = histories.get(code)
            if bars is None:
                continue
            last_date = bars.get("last_date")
            if last_date is not None and last_date != prev_date:
                stats["stale"] += 1
                continue
            stats["evaluated"] += 1
            signal = detect_convergence_breakout(bars)
            if signal is None:
                continue
            stats["signals"] += 1
            candidate = dict(signal)
            candidate["code"] = code
            candidates.append(candidate)
    g.last_scan_stats = stats
    _log_info(
        "[V4 scan] universe=%d loaded=%d stale=%d evaluated=%d signals=%d" % (
            stats["universe"], stats["loaded"], stats["stale"],
            stats["evaluated"], stats["signals"],
        )
    )
    return rank_candidates(candidates)


def _holding_days(entry_date, prev_date):
    try:
        return len(get_trade_days(start_date=entry_date, end_date=prev_date))
    except (TypeError, ValueError):
        return 0


def prepare_exit_reasons(context, prev_date):
    """Freeze exit decisions from T-1 bars for positions opened by V4."""
    reasons = {}
    positions = context.portfolio.positions
    managed_codes = set(getattr(g, "managed_codes", set()))
    for code in sorted(managed_codes):
        position = mapping_get(positions, code)
        amount = getattr(position, "total_amount", 0) if position is not None else 0
        if amount <= 0:
            continue
        if code not in g.stop_prices or code not in g.entry_dates:
            _log_info("[V4 state warning] skip unmanaged state for %s" % code)
            continue

        bars = load_daily_bars(code, prev_date, count=MARKET_MA_PERIOD + 5)
        if bars is None or len(bars["close"]) < 20:
            continue
        last_date = bars.get("last_date")
        if last_date is not None and last_date != prev_date:
            continue
        previous_close = float(bars["close"][-1])
        ma20 = float(np.mean(bars["close"][-20:]))
        holding_days = _holding_days(g.entry_dates[code], prev_date)
        reason = should_exit_position(
            previous_close,
            ma20,
            g.stop_prices[code],
            holding_days,
        )
        if reason is not None:
            reasons[code] = reason
    return reasons


def prepare_daily_state(context):
    """At 09:30, freeze all decisions from information available by T-1."""
    today = context.current_dt.date()
    prev_date = get_prev_trade_date(context)
    closes = get_market_closes(prev_date)

    g.prepared_for = today
    g.signal_date = prev_date
    g.sold_today = set()
    g.market_open = market_gate_is_open(closes)
    g.exit_reasons = prepare_exit_reasons(context, prev_date)
    if g.market_open:
        universe = get_point_in_time_universe(prev_date)
        g.candidates = scan_candidates(universe, prev_date)
    else:
        g.candidates = []
        g.last_scan_stats = {
            "universe": 0, "loaded": 0, "stale": 0,
            "evaluated": 0, "signals": 0,
        }

    _log_info(
        "[V4 snapshot] signal_date=%s market=%s candidates=%d exits=%d" % (
            prev_date,
            "open" if g.market_open else "closed",
            len(g.candidates),
            len(g.exit_reasons),
        )
    )


def execution_quote_is_buyable(quote, confirmation_close):
    """Apply execution-only guards without feeding T quotes into signals."""
    if quote is None or getattr(quote, "paused", True):
        return False
    name = str(getattr(quote, "name", ""))
    if getattr(quote, "is_st", False) or "ST" in name.upper() or "退" in name:
        return False

    price = getattr(quote, "last_price", np.nan)
    high_limit = getattr(quote, "high_limit", np.nan)
    numeric = (price, high_limit, confirmation_close)
    if any(not np.isfinite(value) or value <= 0 for value in numeric):
        return False
    if price >= high_limit or np.isclose(price, high_limit):
        return False
    if price > confirmation_close * (1.0 + MAX_EXECUTION_GAP):
        return False
    return True


def can_open_candidate(code, held_codes, sold_today):
    return code not in held_codes and code not in sold_today


def _quote_is_sellable(quote):
    if quote is None or getattr(quote, "paused", True):
        return False
    price = getattr(quote, "last_price", np.nan)
    return bool(np.isfinite(price) and price > 0)


def execute_trades(context):
    """At 09:35, execute the already frozen T-1 decisions."""
    today = context.current_dt.date()
    if getattr(g, "prepared_for", None) != today:
        _log_info("[V4 execution skipped] no snapshot for %s" % today)
        return

    current_data = get_current_data()
    positions = context.portfolio.positions
    sold_submitted = set()

    for code in sorted(g.exit_reasons):
        position = mapping_get(positions, code)
        if position is None or getattr(position, "total_amount", 0) <= 0:
            continue
        quote = mapping_get(current_data, code)
        if not _quote_is_sellable(quote):
            continue
        result = order_target(code, 0)
        if result is not None:
            sold_submitted.add(code)
            g.sold_today.add(code)
            _log_info("[V4 sell] %s reason=%s" % (code, g.exit_reasons[code]))

    held_codes = {
        code for code, position in positions.items()
        if getattr(position, "total_amount", 0) > 0 and code not in sold_submitted
    }
    slots = max(0, MAX_HOLDINGS - len(held_codes))
    if slots == 0 or not g.market_open:
        return

    cash_left = float(context.portfolio.available_cash)
    total_value = float(context.portfolio.total_value)
    opened = 0
    for candidate in g.candidates:
        if opened >= slots:
            break
        code = candidate["code"]
        if not can_open_candidate(code, held_codes, g.sold_today):
            continue
        quote = mapping_get(current_data, code)
        if not execution_quote_is_buyable(quote, candidate["confirmation_close"]):
            continue

        price = float(quote.last_price)
        shares = calculate_order_shares(
            total_value,
            cash_left,
            price,
            candidate["stop_price"],
        )
        if shares < LOT_SIZE:
            continue
        result = order(code, shares)
        if result is None:
            continue

        g.managed_codes.add(code)
        g.stop_prices[code] = float(candidate["stop_price"])
        g.entry_dates[code] = today
        held_codes.add(code)
        cash_left -= shares * price
        opened += 1
        _log_info(
            "[V4 buy] %s shares=%d price=%.3f stop=%.3f volume_ratio=%.2f" % (
                code, shares, price, candidate["stop_price"],
                candidate["volume_ratio"],
            )
        )


def record_close_state(context):
    """Clean completed position state without calculating a new signal."""
    positions = context.portfolio.positions
    active = {
        code for code, position in positions.items()
        if getattr(position, "total_amount", 0) > 0
    }
    closed = set(getattr(g, "managed_codes", set())) - active
    for code in closed:
        g.managed_codes.discard(code)
        g.stop_prices.pop(code, None)
        g.entry_dates.pop(code, None)
    g.last_close_position_count = len(active)


def after_close(context):
    _log_info(
        "[V4 close] positions=%d sold_today=%d" % (
            getattr(g, "last_close_position_count", 0),
            len(getattr(g, "sold_today", set())),
        )
    )


def _normalise_bars(bars):
    """Return validated one-dimensional OHLCV numpy arrays."""
    required = ("open", "high", "low", "close", "volume")
    try:
        values = {
            name: np.asarray(bars[name], dtype=float)
            for name in required
        }
    except (KeyError, TypeError, ValueError):
        return None

    if any(array.ndim != 1 for array in values.values()):
        return None
    lengths = {len(array) for array in values.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 66:
        return None
    if any(not np.all(np.isfinite(array)) for array in values.values()):
        return None
    if any(np.any(array <= 0) for array in values.values()):
        return None
    return values


def _rolling_mean(values, period):
    result = np.full(len(values), np.nan, dtype=float)
    if len(values) >= period:
        result[period - 1:] = np.convolve(
            values, np.ones(period, dtype=float) / period, mode="valid"
        )
    return result


def _has_convergence_setup(moving_averages, breakout_index):
    start = breakout_index - CONVERGENCE_WINDOW
    if start < 0:
        return False

    converged_days = 0
    for index in range(start, breakout_index):
        values = np.array(
            [moving_averages[period][index] for period in MA_PERIODS],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.min(values) <= 0:
            continue
        spread = (np.max(values) - np.min(values)) / np.min(values)
        if spread <= CONVERGENCE_THRESHOLD:
            converged_days += 1
    return converged_days >= CONVERGENCE_MIN_DAYS


def detect_convergence_breakout(bars):
    """Return a candidate only when the final completed bar confirms it."""
    values = _normalise_bars(bars)
    if values is None:
        return None

    confirmation_index = len(values["close"]) - 1
    moving_averages = {
        period: _rolling_mean(values["close"], period)
        for period in MA_PERIODS
    }

    earliest = max(
        max(MA_PERIODS) - 1 + CONVERGENCE_WINDOW,
        confirmation_index - CONFIRMATION_MAX_DAYS,
    )
    latest = confirmation_index - CONFIRMATION_MIN_DAYS

    for breakout_index in range(latest, earliest - 1, -1):
        if breakout_index < BREAKOUT_LOOKBACK:
            continue
        if not _has_convergence_setup(moving_averages, breakout_index):
            continue

        prior_close = values["close"][
            breakout_index - BREAKOUT_LOOKBACK:breakout_index
        ]
        prior_volume = values["volume"][
            breakout_index - BREAKOUT_LOOKBACK:breakout_index
        ]
        breakout_level = float(np.max(prior_close))
        mean_volume = float(np.mean(prior_volume))
        volume_ratio = float(values["volume"][breakout_index] / mean_volume)

        if values["close"][breakout_index] <= breakout_level:
            continue
        if volume_ratio < BREAKOUT_VOLUME_RATIO:
            continue
        if values["close"][breakout_index] <= moving_averages[20][breakout_index]:
            continue
        if moving_averages[20][breakout_index] < moving_averages[60][breakout_index]:
            continue

        if values["close"][confirmation_index] < breakout_level:
            continue
        if values["low"][confirmation_index] < (
                breakout_level * (1.0 - SUPPORT_LOW_TOLERANCE)):
            continue
        if values["volume"][confirmation_index] >= values["volume"][breakout_index]:
            continue
        if values["close"][confirmation_index] < values["open"][confirmation_index]:
            continue

        stop_price = float(
            min(breakout_level, values["low"][confirmation_index])
            * (1.0 - STOP_BUFFER)
        )
        return {
            "breakout_index": breakout_index,
            "confirmation_index": confirmation_index,
            "breakout_level": breakout_level,
            "confirmation_close": float(values["close"][confirmation_index]),
            "volume_ratio": volume_ratio,
            "stop_price": stop_price,
        }

    return None


def rank_candidates(candidates):
    """Rank by support strength, breakout volume, then security code."""
    def key(candidate):
        support_strength = (
            float(candidate["confirmation_close"])
            / float(candidate["breakout_level"])
        )
        return (
            -support_strength,
            -float(candidate["volume_ratio"]),
            str(candidate["code"]),
        )

    return sorted(candidates, key=key)


def calculate_order_shares(
        total_value,
        available_cash,
        execution_price,
        stop_price,
        risk_fraction=RISK_FRACTION,
        max_position_fraction=MAX_POSITION_FRACTION,
        lot_size=LOT_SIZE):
    """Return a risk-sized A-share order rounded down to board lots."""
    values = (
        total_value, available_cash, execution_price, stop_price,
        risk_fraction, max_position_fraction, lot_size,
    )
    if any(not np.isfinite(value) for value in values):
        return 0
    if (total_value <= 0 or available_cash <= 0 or execution_price <= 0
            or stop_price <= 0 or risk_fraction <= 0
            or max_position_fraction <= 0 or lot_size <= 0):
        return 0

    risk_per_share = execution_price - stop_price
    if risk_per_share <= 0:
        return 0

    risk_cap = (total_value * risk_fraction) / risk_per_share
    position_cap = (total_value * max_position_fraction) / execution_price
    cash_cap = available_cash / execution_price
    raw_shares = min(risk_cap, position_cap, cash_cap)
    return int(raw_shares // int(lot_size)) * int(lot_size)


def market_gate_is_open(
        closes,
        ma_period=MARKET_MA_PERIOD,
        slope_sessions=MARKET_MA_SLOPE_SESSIONS):
    """Return whether the final completed index bar passes the market gate."""
    closes = np.asarray(closes, dtype=float)
    required = ma_period + slope_sessions
    if (closes.ndim != 1 or len(closes) < required
            or not np.all(np.isfinite(closes)) or np.any(closes <= 0)):
        return False

    moving_average = _rolling_mean(closes, ma_period)
    current_ma = moving_average[-1]
    prior_ma = moving_average[-1 - slope_sessions]
    price_ok = closes[-1] >= current_ma or np.isclose(closes[-1], current_ma)
    slope_ok = current_ma >= prior_ma or np.isclose(current_ma, prior_ma)
    return bool(price_ok and slope_ok)


def should_exit_position(
        previous_close,
        ma20,
        stop_price,
        holding_days,
        max_hold_days=MAX_HOLD_DAYS):
    """Return the first frozen T-1 exit reason, or ``None`` to hold."""
    numeric = (previous_close, ma20, stop_price, holding_days, max_hold_days)
    if any(not np.isfinite(value) for value in numeric):
        return None
    if previous_close < stop_price:
        return "shape_stop"
    if previous_close < ma20:
        return "ma20"
    if holding_days >= max_hold_days:
        return "time"
    return None
