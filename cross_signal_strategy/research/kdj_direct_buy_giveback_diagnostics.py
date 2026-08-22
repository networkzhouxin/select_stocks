# -*- coding: utf-8 -*-
"""Observation-only diagnosis of profit giveback after direct KDJ-bonus buys.

Every decision state is built from the score that was causally available at
09:35 (T-1 daily data). Holding-period closes and the peak are retrospective
labels only; they never enter scoring, ranking, order planning, or execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from cross_signal_strategy.local.local_data_loader import (
    APPROVED_WARMUP_ROOT,
    CrossSignalTrainingDataLoader,
)
from cross_signal_strategy.local.local_order_planner import strategy
from cross_signal_strategy.research.extreme_zone_score_candidate import (
    _assert_approved_loader,
    _assert_training_dates,
    _run_replay,
)
from cross_signal_strategy.research.kdj_tiered_moderate_points_candidate import (
    KdjTieredModeratePointsScoreAdapter,
)
from cross_signal_strategy.research.trade_diagnostics import (
    ClosedTradeDiagnostic,
    DiagnosticOrderPlanner,
    build_closed_trade_diagnostics,
)


TARGET_TRADES = (
    ("512100", "2019-09-30", "2019-10-21"),
    ("513880", "2021-03-04", "2021-03-23"),
)


@dataclass(frozen=True)
class SellDecisionState:
    decision_date: str
    signal_date: str
    max_data_date: str
    execution_price: float
    execution_available: bool
    unrealized_return: float
    sell_score: float
    sell_reversal_score: float
    sell_risk_score: float
    sell_extreme_zone_score: float
    k_value: float | None
    rsi_down_cross: bool
    macd_down_cross: bool
    kdj_k_down_cross: bool
    kdj_j_down_cross: bool
    price_confirmed: bool
    adx_protected: bool
    minimum_hold_eligible: bool
    official_signal_sell: bool
    close_below_ma20: bool
    close_below_boll_mid: bool
    close_below_falling_ma10: bool
    downside_continuation: bool
    far_above_ma20_and_rsi6_down: bool
    adx: float | None
    plus_di: float | None
    minus_di: float | None


@dataclass(frozen=True)
class TradeGivebackDiagnostic:
    code: str
    buy_date: str
    sell_date: str
    sell_reason: str
    buy_price: float
    sell_price: float
    peak_date: str
    peak_close: float
    peak_return: float
    realized_return: float
    giveback_from_peak: float
    best_decision_date: str
    best_decision_price: float
    best_decision_return: float
    giveback_from_best_decision: float
    first_score_30_date: str | None
    first_eligible_score_30_date: str | None
    first_confirmation_date: str | None
    first_score_30_confirmation_date: str | None
    first_adx_unprotected_date: str | None
    first_official_sell_date: str | None
    states: tuple[SellDecisionState, ...]


def build_trade_giveback_diagnostic(
    trade: ClosedTradeDiagnostic,
    score_source: object,
    loader: object,
    trade_dates: Sequence[str],
    params: Mapping[str, object] | None = None,
) -> TradeGivebackDiagnostic:
    """Reconstruct causal daily sell states and attach ex-post peak labels."""

    dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in trade_dates]
    buy_date = pd.Timestamp(trade.buy_date).strftime("%Y-%m-%d")
    sell_date = pd.Timestamp(trade.sell_date).strftime("%Y-%m-%d")
    _assert_training_dates((buy_date, sell_date))
    try:
        buy_index = dates.index(buy_date)
        sell_index = dates.index(sell_date)
    except ValueError as exc:
        raise ValueError("filled buy/sell dates must exist in training calendar") from exc
    if sell_index <= buy_index:
        raise ValueError("sell date must follow buy date")

    p = dict(params or strategy.get_default_params())
    code = str(trade.code).split(".")[0]
    buy_price = float(trade.buy_price)
    sell_price = float(trade.sell_price)
    if buy_price <= 0 or sell_price <= 0:
        raise ValueError("filled buy and sell prices must be positive")

    states = []
    for decision_date in dates[buy_index + 1:sell_index + 1]:
        score = score_source.score(code, decision_date)
        if score is None:
            raise ValueError(
                "missing daily score for held position %s on %s"
                % (code, decision_date)
            )
        score = dict(score)
        signal_date = _causal_date(score, "signal_date", decision_date)
        max_data_date = _causal_date(score, "max_data_date", decision_date)
        execution_bar = loader.get_minute_bar(code, decision_date, "09:35")
        execution_price = float(execution_bar["close"])
        if execution_price <= 0:
            raise ValueError("09:35 execution price must be positive")
        execution_available = _bar_has_executable_trade(execution_bar)
        eligible = bool(strategy.can_sell_by_signal(
            buy_date,
            decision_date,
            min_hold_days=int(p.get("min_signal_hold_days", 5)),
            trade_days=dates,
        ))
        confirmation = bool(strategy.has_signal_sell_confirmation(score))
        protected = bool(strategy.is_protected_by_strong_adx_uptrend(score, p))
        official_sell = bool(
            eligible
            and strategy.should_force_sell(
                score,
                atr_stop_triggered=False,
                params=p,
            )
        )
        states.append(SellDecisionState(
            decision_date=decision_date,
            signal_date=signal_date,
            max_data_date=max_data_date,
            execution_price=execution_price,
            execution_available=execution_available,
            unrealized_return=execution_price / buy_price - 1.0,
            sell_score=_number(score.get("sell_score")),
            sell_reversal_score=_number(score.get("sell_reversal_score")),
            sell_risk_score=_number(score.get("sell_risk_score")),
            sell_extreme_zone_score=_number(score.get("sell_extreme_zone_score")),
            k_value=_optional_float(score.get("k")),
            rsi_down_cross=bool(
                score.get("rsi6_cross_rsi12_down")
                or score.get("rsi6_cross_rsi24_down")
            ),
            macd_down_cross=bool(score.get("macd_cross_down")),
            kdj_k_down_cross=bool(score.get("kdj_k_cross_down")),
            kdj_j_down_cross=bool(score.get("kdj_j_cross_down")),
            price_confirmed=confirmation,
            adx_protected=protected,
            minimum_hold_eligible=eligible,
            official_signal_sell=official_sell,
            close_below_ma20=bool(score.get("close_below_ma20")),
            close_below_boll_mid=bool(score.get("close_below_boll_mid")),
            close_below_falling_ma10=bool(score.get("close_below_falling_ma10")),
            downside_continuation=bool(score.get("downside_continuation")),
            far_above_ma20_and_rsi6_down=bool(
                score.get("far_above_ma20_and_rsi6_down")
            ),
            adx=_optional_float(score.get("adx")),
            plus_di=_optional_float(score.get("plus_di")),
            minus_di=_optional_float(score.get("minus_di")),
        ))

    peak_date, peak_close = _pre_exit_peak_close(
        loader,
        code,
        dates[buy_index:sell_index],
    )
    peak_return = peak_close / buy_price - 1.0
    realized_return = sell_price / buy_price - 1.0
    executable_states = [item for item in states if item.execution_available]
    if not executable_states:
        raise ValueError("holding period has no executable 09:35 decision bar")
    best_decision = max(executable_states, key=lambda item: item.execution_price)
    best_decision_return = best_decision.execution_price / buy_price - 1.0
    return TradeGivebackDiagnostic(
        code=code,
        buy_date=buy_date,
        sell_date=sell_date,
        sell_reason=str(trade.sell_reason),
        buy_price=buy_price,
        sell_price=sell_price,
        peak_date=peak_date,
        peak_close=peak_close,
        peak_return=peak_return,
        realized_return=realized_return,
        giveback_from_peak=peak_return - realized_return,
        best_decision_date=best_decision.decision_date,
        best_decision_price=best_decision.execution_price,
        best_decision_return=best_decision_return,
        giveback_from_best_decision=best_decision_return - realized_return,
        first_score_30_date=_first_date(states, lambda item: item.sell_score >= 30),
        first_eligible_score_30_date=_first_date(
            states,
            lambda item: item.minimum_hold_eligible and item.sell_score >= 30,
        ),
        first_confirmation_date=_first_date(
            states, lambda item: item.price_confirmed
        ),
        first_score_30_confirmation_date=_first_date(
            states,
            lambda item: item.sell_score >= 30 and item.price_confirmed,
        ),
        first_adx_unprotected_date=_first_date(
            states, lambda item: not item.adx_protected
        ),
        first_official_sell_date=_first_date(
            states, lambda item: item.official_signal_sell
        ),
        states=tuple(states),
    )


def run_direct_kdj_buy_giveback_diagnostics(
    loader: object | None = None,
    initial_cash: float = 20000.0,
    warmup_root: Path | str = APPROVED_WARMUP_ROOT,
) -> tuple[TradeGivebackDiagnostic, ...]:
    """Replay the frozen moderate candidate and diagnose two predeclared trades."""

    from cross_signal_strategy.local_training_run import (
        build_training_signal_adapter,
        get_training_trade_dates,
    )
    from cross_signal_strategy.research.friction_diagnostics import (
        PrecomputedSignalAdapter,
    )

    loader = loader or CrossSignalTrainingDataLoader()
    _assert_approved_loader(loader)
    warmup = Path(warmup_root).expanduser().resolve()
    if warmup != Path(APPROVED_WARMUP_ROOT).expanduser().resolve():
        raise ValueError("Use approved warm-up data root only: %s" % APPROVED_WARMUP_ROOT)
    trade_dates = get_training_trade_dates(loader)
    _assert_training_dates(trade_dates)
    params = strategy.get_default_params()
    pool = [code.split(".")[0] for code in strategy.get_default_etf_pool()]
    official_source = build_training_signal_adapter(loader, warmup_root=warmup)
    cached = PrecomputedSignalAdapter.from_source(
        official_source,
        trade_dates=trade_dates,
        codes=pool,
    )
    candidate_source = KdjTieredModeratePointsScoreAdapter(cached, trade_dates)
    planner = DiagnosticOrderPlanner(
        candidate_source,
        etf_pool=pool,
        params=dict(params),
        trade_dates=trade_dates,
    )
    days = _run_replay(loader, planner, trade_dates, initial_cash)
    trades = build_closed_trade_diagnostics(
        days,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )
    indexed = {
        (
            str(trade.code).split(".")[0],
            str(trade.buy_date),
            str(trade.sell_date),
        ): trade
        for trade in trades
    }
    missing = [key for key in TARGET_TRADES if key not in indexed]
    if missing:
        raise ValueError("predeclared candidate trades missing from replay: %s" % missing)
    return tuple(
        build_trade_giveback_diagnostic(
            indexed[key],
            candidate_source,
            loader,
            trade_dates,
            params=params,
        )
        for key in TARGET_TRADES
    )


def format_direct_kdj_buy_giveback_diagnostics(
    diagnostics: Sequence[TradeGivebackDiagnostic],
) -> str:
    lines = [
        "Direct KDJ-bonus buy giveback diagnosis (2019-2021; observation only)",
        "peak=maximum daily close from filled buy date through session before exit",
        "decision_state=09:35 execution price plus causal T-1 score only",
    ]
    for item in diagnostics:
        lines.append(
            "TRADE code=%s buy=%s@%.4f sell=%s@%.4f reason=%s "
            "peak=%s@%.4f peak_return=%.4f realized=%.4f giveback=%.4f "
            "best_0935=%s@%.4f best_0935_return=%.4f giveback_0935=%.4f "
            "first_score30=%s first_eligible_score30=%s first_confirmation=%s "
            "first_score30_confirmation=%s first_adx_unprotected=%s "
            "first_official_sell=%s"
            % (
                item.code,
                item.buy_date,
                item.buy_price,
                item.sell_date,
                item.sell_price,
                item.sell_reason,
                item.peak_date,
                item.peak_close,
                item.peak_return,
                item.realized_return,
                item.giveback_from_peak,
                item.best_decision_date,
                item.best_decision_price,
                item.best_decision_return,
                item.giveback_from_best_decision,
                _optional_text(item.first_score_30_date),
                _optional_text(item.first_eligible_score_30_date),
                _optional_text(item.first_confirmation_date),
                _optional_text(item.first_score_30_confirmation_date),
                _optional_text(item.first_adx_unprotected_date),
                _optional_text(item.first_official_sell_date),
            )
        )
        for state in item.states:
            lines.append(
                "DAY code=%s decision=%s signal=%s price=%.4f unrealized=%.4f "
                "executable=%s sell=%.1f reversal=%.1f risk=%.1f extreme=%.1f k=%s "
                "rsi_down=%s macd_down=%s kdj_k_down=%s kdj_j_down=%s "
                "eligible=%s confirmation=%s adx_protected=%s official_sell=%s "
                "below_ma20=%s below_boll_mid=%s below_falling_ma10=%s "
                "downside=%s far_above_rsi_down=%s adx=%s plus_di=%s minus_di=%s"
                % (
                    item.code,
                    state.decision_date,
                    state.signal_date,
                    state.execution_price,
                    state.unrealized_return,
                    state.execution_available,
                    state.sell_score,
                    state.sell_reversal_score,
                    state.sell_risk_score,
                    state.sell_extreme_zone_score,
                    _optional_number(state.k_value),
                    state.rsi_down_cross,
                    state.macd_down_cross,
                    state.kdj_k_down_cross,
                    state.kdj_j_down_cross,
                    state.minimum_hold_eligible,
                    state.price_confirmed,
                    state.adx_protected,
                    state.official_signal_sell,
                    state.close_below_ma20,
                    state.close_below_boll_mid,
                    state.close_below_falling_ma10,
                    state.downside_continuation,
                    state.far_above_ma20_and_rsi6_down,
                    _optional_number(state.adx),
                    _optional_number(state.plus_di),
                    _optional_number(state.minus_di),
                )
            )
    lines.append("future_peak=ex_post_label_only; never used by strategy")
    lines.append("authority=local_diagnostic_only; JoinQuant remains authoritative")
    return "\n".join(lines)


def _causal_date(score: Mapping[str, object], field: str, decision_date: str) -> str:
    raw = score.get(field)
    if raw in (None, ""):
        raise ValueError("daily score is missing %s" % field)
    value = pd.Timestamp(raw).strftime("%Y-%m-%d")
    if value >= decision_date:
        raise ValueError("%s must precede decision date" % field)
    return value


def _pre_exit_peak_close(
    loader: object,
    code: str,
    eligible_dates: Sequence[str],
) -> tuple[str, float]:
    if not eligible_dates:
        raise ValueError("pre-exit peak requires at least one held session")
    frames = {}
    closes = []
    for date in eligible_dates:
        year = int(str(date)[:4])
        if year not in frames:
            frame = loader.load_daily_frame(code, "%04d-12-31" % year).copy()
            frame["_date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
            frames[year] = frame
        rows = frames[year].loc[frames[year]["_date"] == str(date)]
        if rows.empty:
            raise ValueError("missing daily close for %s on %s" % (code, date))
        close = float(rows.iloc[-1]["close"])
        if close <= 0:
            raise ValueError("daily close must be positive")
        closes.append((str(date), close))
    return max(closes, key=lambda item: item[1])


def _first_date(states, predicate) -> str | None:
    return next((item.decision_date for item in states if predicate(item)), None)


def _number(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _optional_text(value: str | None) -> str:
    return value if value is not None else "n/a"


def _optional_number(value: float | None) -> str:
    return "n/a" if value is None else "%.2f" % value


def _bar_has_executable_trade(bar: Mapping[str, object]) -> bool:
    def numeric(field: str) -> float:
        try:
            return float(bar.get(field, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return numeric("volume") > 0.0 or numeric("num_trades") > 0.0


def main() -> None:
    diagnostics = run_direct_kdj_buy_giveback_diagnostics()
    print(format_direct_kdj_buy_giveback_diagnostics(diagnostics))


if __name__ == "__main__":
    main()
