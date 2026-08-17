# -*- coding: utf-8 -*-
"""Observation-only extreme-lag attribution for the 2019-2021 training path.

Signal-derived fields in this module are restricted to T-1 and earlier data.
Forward path values are retrospective labels and never enter score or order code.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

from cross_signal_strategy.local.local_data_loader import (
    TRAIN_END,
    TRAIN_START,
    assert_not_training_write_path,
)
from cross_signal_strategy.research.order_path_diagnostics import OrderPathEvent


@dataclass(frozen=True)
class OfficialPathEvidence:
    status: str
    expected_count: int
    actual_count: int


@dataclass(frozen=True)
class ContributingCross:
    name: str
    age: int
    weight: float
    cross_date: str
    cross_close: float


@dataclass(frozen=True)
class EntryLagObservation:
    code: str
    buy_date: str
    signal_date: str
    buy_price: float
    entry_atr: float | None
    contributing_crosses: tuple[ContributingCross, ...]
    reversal_contribution_by_age: tuple[float, float, float]
    age_two_reversal_share: float | None
    earliest_cross_date: str | None
    earliest_cross_to_fill_sessions: int | None
    extension_from_earliest_cross_atr: float | None
    execution_gap_atr: float | None
    evaluation_mae_5: float | None
    evaluation_mfe_5: float | None
    missing_fields: tuple[str, ...]


@dataclass(frozen=True)
class ExitSignalDay:
    execution_date: str
    signal_date: str
    sell_score: float
    has_confirmation: bool
    is_protected: bool
    execution_price: float


@dataclass(frozen=True)
class ExitLagObservation:
    code: str
    buy_date: str
    sell_date: str
    exit_type: str
    first_high_score_date: str | None
    first_high_score_signal_date: str | None
    first_high_score_state: str | None
    first_high_score_to_exit_sessions: int | None
    profit_at_first_high_score: float | None
    peak_close_profit: float | None
    exit_profit: float
    giveback_from_peak: float | None
    incremental_giveback_after_first_high_score: float | None
    evaluation_post_exit_return_3: float | None
    evaluation_post_exit_return_5: float | None
    missing_fields: tuple[str, ...]


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    usable_count: int
    missing_count: int
    median: float | None
    q1: float | None
    q3: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class Step0Decision:
    status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExtremeLagReport:
    path_evidence: OfficialPathEvidence
    entries: tuple[EntryLagObservation, ...]
    exits: tuple[ExitLagObservation, ...]
    entry_distributions: Mapping[str, Mapping[str, DistributionSummary]]
    exit_distributions: Mapping[str, Mapping[str, DistributionSummary]]
    top_entry_etf_share: float | None
    top_exit_etf_share: float | None
    decision: Step0Decision


@dataclass(frozen=True)
class FilledEntryEpisode:
    code: str
    buy_date: str
    buy_price: float
    entry_score: Mapping[str, object]


_BULLISH_CROSS_WEIGHTS = (
    ("rsi6_cross_rsi12_up", 12.0),
    ("rsi6_cross_rsi24_up", 12.0),
    ("macd_cross_up", 10.0),
    ("kdj_k_cross_up", 6.0),
    ("kdj_j_cross_up", 5.0),
)


def assert_official_fill_path(
    expected_events: Sequence[OrderPathEvent],
    actual_events: Sequence[OrderPathEvent],
) -> OfficialPathEvidence:
    """Require exact official/local filled-event identity before attribution."""
    if not expected_events:
        raise ValueError("official JoinQuant filled path is required")
    if len(expected_events) != len(actual_events):
        raise ValueError(
            "order path mismatch: expected %d events, actual %d"
            % (len(expected_events), len(actual_events))
        )
    for index, (expected, actual) in enumerate(zip(expected_events, actual_events)):
        for event in (expected, actual):
            event_date = pd.Timestamp(event.date)
            if event_date < TRAIN_START or event_date > TRAIN_END:
                raise ValueError("filled event date must stay inside the 2019-2021 training window")
        if expected.as_key() != actual.as_key():
            raise ValueError(
                "order path mismatch at index %d: expected %r, actual %r"
                % (index, expected.as_key(), actual.as_key())
            )
        if expected.amount is None or actual.amount is None:
            raise ValueError("filled path amount evidence is required at index %d" % index)
        if int(expected.amount) != int(actual.amount):
            raise ValueError(
                "amount mismatch at index %d: expected %d, actual %d"
                % (index, int(expected.amount), int(actual.amount))
            )
    return OfficialPathEvidence(
        status="aligned",
        expected_count=len(expected_events),
        actual_count=len(actual_events),
    )


def assert_training_episode_dates(buy_date: str, sell_date: str) -> None:
    buy = pd.Timestamp(buy_date)
    sell = pd.Timestamp(sell_date)
    if buy < TRAIN_START or buy > TRAIN_END or sell < TRAIN_START or sell > TRAIN_END:
        raise ValueError("episode dates must stay inside the 2019-2021 training window")
    if sell < buy:
        raise ValueError("sell date precedes buy date")


def assert_report_path(path: str | Path) -> Path:
    assert_not_training_write_path(path)
    return Path(path).expanduser().resolve()


def build_entry_lag_observation(
    trade: object,
    signal_frame: pd.DataFrame,
    forward_closes: Sequence[float],
) -> EntryLagObservation:
    """Build one entry observation without exposing forward labels to scoring."""
    buy_date = pd.Timestamp(trade.buy_date)
    if buy_date < TRAIN_START or buy_date > TRAIN_END:
        raise ValueError("buy date must stay inside the 2019-2021 training window")
    sell_date = getattr(trade, "sell_date", None)
    if sell_date is not None:
        assert_training_episode_dates(str(trade.buy_date), str(sell_date))
    score: Mapping[str, object] = dict(getattr(trade, "entry_score", {}) or {})
    signal_date = str(score.get("signal_date", ""))
    if not signal_date:
        raise ValueError("entry score requires signal_date")
    signal_ts = pd.Timestamp(signal_date)
    buy_ts = pd.Timestamp(trade.buy_date)
    if signal_ts >= buy_ts:
        raise ValueError("signal_date must precede the filled buy date")
    frame = _validated_signal_frame(signal_frame, signal_ts)

    active = []
    age_contributions = [0.0, 0.0, 0.0]
    rsi_up = bool(score.get("rsi6_cross_rsi12_up") or score.get("rsi6_cross_rsi24_up"))
    rsi_down = bool(score.get("rsi6_cross_rsi12_down") or score.get("rsi6_cross_rsi24_down"))
    for name, weight in _BULLISH_CROSS_WEIGHTS:
        if not bool(score.get(name)):
            continue
        if name.startswith("rsi6_") and not (rsi_up and not rsi_down):
            continue
        raw_age = score.get(name + "_age")
        if raw_age is None:
            raise ValueError("active cross age is missing for %s" % name)
        try:
            age = int(raw_age)
        except (TypeError, ValueError) as exc:
            raise ValueError("active cross age is invalid for %s" % name) from exc
        if age not in (0, 1, 2):
            raise ValueError("active cross age must be 0, 1, or 2 for %s" % name)
        row_index = len(frame) - 1 - age
        if row_index < 0:
            raise ValueError("signal frame is too short for active cross age")
        row = frame.iloc[row_index]
        active.append(ContributingCross(
            name=name,
            age=age,
            weight=weight,
            cross_date=pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            cross_close=float(row["close"]),
        ))
        age_contributions[age] += weight

    contribution_total = sum(age_contributions)
    expected_reversal = _optional_float(score.get("reversal_score"))
    if expected_reversal is not None and abs(expected_reversal - contribution_total) > 1e-9:
        raise ValueError(
            "entry reversal score does not match contributing bullish crosses: %.6f != %.6f"
            % (expected_reversal, contribution_total)
        )

    earliest = max(active, key=lambda item: item.age) if active else None
    atr = _positive_float(score.get("atr"))
    buy_price = float(trade.buy_price)
    if buy_price <= 0:
        raise ValueError("filled buy price must be positive")
    signal_close = float(frame.iloc[-1]["close"])
    missing = []
    if atr is None:
        missing.append("entry_atr")
    if earliest is None:
        missing.append("contributing_crosses")
    price_metrics_allowed = (
        signal_ts >= TRAIN_START
        and earliest is not None
        and pd.Timestamp(earliest.cross_date) >= TRAIN_START
    )
    if not price_metrics_allowed and earliest is not None:
        missing.append("warmup_price_metrics_excluded")
    evaluation_mae, evaluation_mfe = _evaluation_excursions(
        forward_closes, buy_price, horizon=5
    )
    if evaluation_mae is None:
        missing.append("evaluation_path_5")

    return EntryLagObservation(
        code=str(trade.code).split(".")[0],
        buy_date=str(trade.buy_date),
        signal_date=signal_ts.strftime("%Y-%m-%d"),
        buy_price=buy_price,
        entry_atr=atr,
        contributing_crosses=tuple(active),
        reversal_contribution_by_age=tuple(age_contributions),
        age_two_reversal_share=(age_contributions[2] / contribution_total if contribution_total else None),
        earliest_cross_date=earliest.cross_date if earliest else None,
        earliest_cross_to_fill_sessions=earliest.age + 1 if earliest else None,
        extension_from_earliest_cross_atr=(
            (buy_price - earliest.cross_close) / atr
            if price_metrics_allowed and atr is not None else None
        ),
        execution_gap_atr=(
            (buy_price - signal_close) / atr
            if price_metrics_allowed and atr is not None else None
        ),
        evaluation_mae_5=evaluation_mae,
        evaluation_mfe_5=evaluation_mfe,
        missing_fields=tuple(missing),
    )


def build_exit_lag_observation(
    trade: object,
    trade_dates: Sequence[str],
    signal_days: Sequence[ExitSignalDay],
    peak_close: float | None,
    post_exit_closes: Sequence[float],
    min_hold_days: int = 5,
) -> ExitLagObservation:
    """Describe exit delay while keeping forward returns as output-only labels."""
    assert_training_episode_dates(str(trade.buy_date), str(trade.sell_date))
    dates = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in trade_dates]
    buy_date = pd.Timestamp(trade.buy_date).strftime("%Y-%m-%d")
    sell_date = pd.Timestamp(trade.sell_date).strftime("%Y-%m-%d")
    try:
        buy_index = dates.index(buy_date)
        sell_index = dates.index(sell_date)
    except ValueError as exc:
        raise ValueError("filled buy/sell dates must exist in the training calendar") from exc
    if sell_index < buy_index:
        raise ValueError("sell date precedes buy date in the training calendar")

    normalized_days = []
    for day in signal_days:
        execution_date = pd.Timestamp(day.execution_date).strftime("%Y-%m-%d")
        signal_date = pd.Timestamp(day.signal_date).strftime("%Y-%m-%d")
        if signal_date >= execution_date:
            raise ValueError("exit signal_date must precede execution date")
        if execution_date not in dates:
            raise ValueError("exit signal execution date is absent from the training calendar")
        normalized_days.append((dates.index(execution_date), day))
    normalized_days.sort(key=lambda pair: pair[0])

    eligible_index = buy_index + int(min_hold_days)
    first_pair = next(
        (
            (index, day)
            for index, day in normalized_days
            if eligible_index <= index <= sell_index and float(day.sell_score) >= 30.0
        ),
        None,
    )
    first_index = first_pair[0] if first_pair else None
    first_day = first_pair[1] if first_pair else None
    first_state = None
    first_profit = None
    if first_day is not None:
        if not bool(first_day.has_confirmation):
            first_state = "confirmation_absent"
        elif bool(first_day.is_protected):
            first_state = "protected"
        else:
            first_state = "confirmation_present"
        if float(trade.buy_price) <= 0 or float(first_day.execution_price) <= 0:
            raise ValueError("entry and high-score execution prices must be positive")
        first_profit = float(first_day.execution_price) / float(trade.buy_price) - 1.0

    buy_price = float(trade.buy_price)
    sell_price = float(trade.sell_price)
    if buy_price <= 0 or sell_price <= 0:
        raise ValueError("filled buy/sell prices must be positive")
    exit_profit = sell_price / buy_price - 1.0
    peak_number = _positive_float(peak_close)
    peak_profit = peak_number / buy_price - 1.0 if peak_number is not None else None
    post_3 = _forward_return_at(post_exit_closes, sell_price, 3)
    post_5 = _forward_return_at(post_exit_closes, sell_price, 5)
    missing = []
    if first_day is None:
        missing.append("first_high_score")
    if peak_profit is None:
        missing.append("peak_close")
    if post_3 is None:
        missing.append("evaluation_post_exit_3")
    if post_5 is None:
        missing.append("evaluation_post_exit_5")

    return ExitLagObservation(
        code=str(trade.code).split(".")[0],
        buy_date=buy_date,
        sell_date=sell_date,
        exit_type=str(trade.sell_reason),
        first_high_score_date=(
            pd.Timestamp(first_day.execution_date).strftime("%Y-%m-%d")
            if first_day is not None else None
        ),
        first_high_score_signal_date=(
            pd.Timestamp(first_day.signal_date).strftime("%Y-%m-%d")
            if first_day is not None else None
        ),
        first_high_score_state=first_state,
        first_high_score_to_exit_sessions=(
            sell_index - first_index if first_index is not None else None
        ),
        profit_at_first_high_score=first_profit,
        peak_close_profit=peak_profit,
        exit_profit=exit_profit,
        giveback_from_peak=(peak_profit - exit_profit if peak_profit is not None else None),
        incremental_giveback_after_first_high_score=(
            first_profit - exit_profit if first_profit is not None else None
        ),
        evaluation_post_exit_return_3=post_3,
        evaluation_post_exit_return_5=post_5,
        missing_fields=tuple(missing),
    )


def summarize_distribution(values: Sequence[float | None]) -> DistributionSummary:
    usable = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not usable:
        return DistributionSummary(
            count=len(values),
            usable_count=0,
            missing_count=len(values),
            median=None,
            q1=None,
            q3=None,
            minimum=None,
            maximum=None,
        )
    series = pd.Series(usable, dtype=float)
    return DistributionSummary(
        count=len(values),
        usable_count=len(usable),
        missing_count=len(values) - len(usable),
        median=float(series.median()),
        q1=float(series.quantile(0.25)),
        q3=float(series.quantile(0.75)),
        minimum=float(series.min()),
        maximum=float(series.max()),
    )


def run_training_extreme_lag_attribution(
    joinquant_events: Sequence[OrderPathEvent],
    loader=None,
    initial_cash: float = 20000.0,
) -> ExtremeLagReport:
    """Run the frozen training path only after official fill evidence is supplied."""
    if not joinquant_events:
        raise ValueError("official JoinQuant filled path is required")
    return _run_aligned_training_attribution(
        joinquant_events=joinquant_events,
        loader=loader,
        initial_cash=initial_cash,
    )


def assert_repository_report_dir(path: str | Path) -> Path:
    resolved = assert_report_path(path)
    repository_root = Path(__file__).resolve().parents[2]
    approved = (repository_root / "cross_signal_strategy" / "reports").resolve()
    if resolved != approved:
        raise ValueError("report directory must be cross_signal_strategy/reports")
    return resolved


def write_extreme_lag_artifacts(
    report: ExtremeLagReport,
    report_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = assert_report_path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = assert_report_path(
        output_dir / "extreme_lag_attribution_2019_2021.md"
    )
    json_path = assert_report_path(
        output_dir / "extreme_lag_attribution_2019_2021.json"
    )
    _atomic_write_text(markdown_path, format_extreme_lag_report(report))
    _atomic_write_text(
        json_path,
        json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return markdown_path, json_path


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _run_aligned_training_attribution(
    joinquant_events: Sequence[OrderPathEvent],
    loader,
    initial_cash: float,
) -> ExtremeLagReport:
    from cross_signal_strategy.local.local_backtester import LocalBacktestEngine
    from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_training_run import (
        build_training_signal_adapter,
        get_training_trade_dates,
    )
    from cross_signal_strategy.research.trade_diagnostics import DiagnosticOrderPlanner

    class _ExtremeLagPlanner(DiagnosticOrderPlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.daily_score_snapshots = {}
            self.daily_execution_prices = {}

        def plan_orders(self, current_date, previous_date, broker, current_prices=None):
            orders = super().plan_orders(
                current_date,
                previous_date,
                broker,
                current_prices=current_prices,
            )
            for code, score in self.last_scores.items():
                self.daily_score_snapshots[(str(current_date), str(code))] = dict(score)
            for code, price in (current_prices or {}).items():
                self.daily_execution_prices[(str(current_date), str(code))] = float(price)
            return orders

    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = _ExtremeLagPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    return build_extreme_lag_from_capture(
        joinquant_events,
        results,
        planner,
        adapter,
        loader,
        trade_dates,
    )


def build_extreme_lag_from_capture(
    joinquant_events: Sequence[OrderPathEvent],
    results: Sequence[object],
    planner: object,
    adapter: object,
    loader: object,
    trade_dates: Sequence[str],
) -> ExtremeLagReport:
    """Bind an already-captured local replay to official fills, then attribute."""
    from cross_signal_strategy.local.local_order_planner import strategy
    from cross_signal_strategy.research.order_path_diagnostics import extract_local_order_events
    from cross_signal_strategy.research.trade_diagnostics import build_closed_trade_diagnostics

    actual_events = extract_local_order_events(results)
    path_evidence = assert_official_fill_path(joinquant_events, actual_events)
    entry_snapshots = getattr(planner, "entry_score_snapshots", {})
    exit_snapshots = getattr(planner, "exit_score_snapshots", {})
    trades = build_closed_trade_diagnostics(results, entry_snapshots, exit_snapshots)
    history_cache: dict[str, pd.DataFrame] = {}

    entries = []
    for episode in _filled_entry_episodes(results, entry_snapshots):
        code = episode.code
        history = _history_for_code(loader, code, history_cache)
        buy_index = _history_date_index(history, episode.buy_date)
        forward_closes = history.loc[
            buy_index:buy_index + 4, "close"
        ].astype(float).tolist()
        signal_frame, loaded_signal_date = adapter.load_signal_frame(code, episode.buy_date)
        score_signal_date = str(episode.entry_score.get("signal_date", ""))
        if str(loaded_signal_date) != score_signal_date:
            raise ValueError("entry signal frame date does not match frozen entry score")
        entries.append(
            build_entry_lag_observation(
                episode,
                signal_frame,
                forward_closes=forward_closes,
            )
        )

    exits = []
    daily_scores = getattr(planner, "daily_score_snapshots", {})
    daily_prices = getattr(planner, "daily_execution_prices", {})
    min_hold_days = int(getattr(planner, "params", {}).get("min_signal_hold_days", 5))
    ordered_dates = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in trade_dates]
    for trade in trades:
        code = str(trade.code).split(".")[0]
        history = _history_for_code(loader, code, history_cache)
        buy_index = _history_date_index(history, str(trade.buy_date))
        sell_index = _history_date_index(history, str(trade.sell_date))
        peak_rows = history.loc[buy_index:max(buy_index, sell_index - 1), "close"]
        peak_close = float(peak_rows.astype(float).max()) if not peak_rows.empty else None
        post_exit_closes = history.loc[
            sell_index + 1:sell_index + 5, "close"
        ].astype(float).tolist()
        signal_days = []
        for current_date in ordered_dates:
            if current_date < str(trade.buy_date) or current_date > str(trade.sell_date):
                continue
            score = daily_scores.get((current_date, code))
            execution_price = daily_prices.get((current_date, code))
            if score is None or execution_price is None:
                continue
            signal_date = str(score.get("signal_date", ""))
            if not signal_date:
                raise ValueError("daily exit score is missing signal_date")
            confirmation = bool(strategy.has_signal_sell_confirmation(score))
            protected = bool(
                confirmation and strategy.is_protected_by_strong_adx_uptrend(score)
            )
            signal_days.append(ExitSignalDay(
                execution_date=current_date,
                signal_date=signal_date,
                sell_score=float(score.get("sell_score", 0.0) or 0.0),
                has_confirmation=confirmation,
                is_protected=protected,
                execution_price=float(execution_price),
            ))
        exits.append(build_exit_lag_observation(
            trade,
            trade_dates=ordered_dates,
            signal_days=signal_days,
            peak_close=peak_close,
            post_exit_closes=post_exit_closes,
            min_hold_days=min_hold_days,
        ))
    return summarize_extreme_lag(entries, exits, path_evidence)


def _filled_entry_episodes(
    results: Sequence[object],
    entry_score_snapshots: Mapping[tuple[str, str], Mapping[str, object]],
) -> list[FilledEntryEpisode]:
    episodes = []
    for day in results:
        date = str(day.date)
        for order in getattr(day, "orders", []):
            if not getattr(order, "filled", False) or int(order.amount_delta) <= 0:
                continue
            code = str(order.code).split(".")[0]
            score = dict(entry_score_snapshots.get((date, code), {}))
            if not score:
                raise ValueError("filled buy is missing its frozen entry score")
            episodes.append(FilledEntryEpisode(
                code=code,
                buy_date=date,
                buy_price=float(order.exec_price),
                entry_score=score,
            ))
    return episodes


def _history_for_code(
    loader: object,
    code: str,
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if code in cache:
        return cache[code]
    frames = []
    for year in (2019, 2020, 2021):
        try:
            frame = loader.load_daily_frame(code, "%d-12-31" % year)
        except FileNotFoundError:
            continue
        rows = frame[["date", "close"]].copy()
        rows["date"] = pd.to_datetime(rows["date"], errors="raise")
        if (rows["date"] < TRAIN_START).any() or (rows["date"] > TRAIN_END).any():
            raise ValueError("daily history contains dates outside 2019-2021 training window")
        frames.append(rows)
    if not frames:
        raise FileNotFoundError("no training daily history for %s" % code)
    cache[code] = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return cache[code]


def _history_date_index(history: pd.DataFrame, date: str) -> int:
    target = pd.Timestamp(date)
    matches = history.index[history["date"] == target].tolist()
    if not matches:
        raise KeyError("daily history is missing %s" % target.strftime("%Y-%m-%d"))
    return int(matches[0])


def summarize_extreme_lag(
    entry_rows: Sequence[EntryLagObservation],
    exit_rows: Sequence[ExitLagObservation],
    path_evidence: OfficialPathEvidence,
) -> ExtremeLagReport:
    entries = tuple(entry_rows)
    exits = tuple(exit_rows)
    entry_groups = _entry_groups(entries)
    exit_groups = _exit_groups(exits)
    entry_distributions = {
        key: _entry_distribution(items) for key, items in sorted(entry_groups.items())
    }
    exit_distributions = {
        key: _exit_distribution(items) for key, items in sorted(exit_groups.items())
    }
    top_entry_share = _top_code_share(entries)
    top_exit_share = _top_code_share(exits)
    decision = _evaluate_step0(
        entries,
        exits,
        path_evidence,
        top_entry_share,
        top_exit_share,
    )
    return ExtremeLagReport(
        path_evidence=path_evidence,
        entries=entries,
        exits=exits,
        entry_distributions=entry_distributions,
        exit_distributions=exit_distributions,
        top_entry_etf_share=top_entry_share,
        top_exit_etf_share=top_exit_share,
        decision=decision,
    )


def _entry_groups(
    rows: Sequence[EntryLagObservation],
) -> dict[str, list[EntryLagObservation]]:
    groups: dict[str, list[EntryLagObservation]] = {"full": list(rows)}
    for year in (2019, 2020, 2021):
        groups["year:%d" % year] = [
            row for row in rows if pd.Timestamp(row.buy_date).year == year
        ]
    for code in sorted({row.code for row in rows}):
        groups["etf:%s" % code] = [row for row in rows if row.code == code]
    return groups


def _exit_groups(
    rows: Sequence[ExitLagObservation],
) -> dict[str, list[ExitLagObservation]]:
    groups: dict[str, list[ExitLagObservation]] = {"full": list(rows)}
    for year in (2019, 2020, 2021):
        groups["year:%d" % year] = [
            row for row in rows if pd.Timestamp(row.sell_date).year == year
        ]
    for code in sorted({row.code for row in rows}):
        groups["etf:%s" % code] = [row for row in rows if row.code == code]
    for exit_type in sorted({row.exit_type for row in rows}):
        groups["exit_type:%s" % exit_type] = [
            row for row in rows if row.exit_type == exit_type
        ]
    return groups


def _entry_distribution(
    rows: Sequence[EntryLagObservation],
) -> Mapping[str, DistributionSummary]:
    return {
        "earliest_delay_sessions": summarize_distribution(
            [row.earliest_cross_to_fill_sessions for row in rows]
        ),
        "age_two_share": summarize_distribution([row.age_two_reversal_share for row in rows]),
        "extension_atr": summarize_distribution(
            [row.extension_from_earliest_cross_atr for row in rows]
        ),
        "execution_gap_atr": summarize_distribution([row.execution_gap_atr for row in rows]),
        "evaluation_mae_5": summarize_distribution([row.evaluation_mae_5 for row in rows]),
        "evaluation_mfe_5": summarize_distribution([row.evaluation_mfe_5 for row in rows]),
    }


def _exit_distribution(
    rows: Sequence[ExitLagObservation],
) -> Mapping[str, DistributionSummary]:
    return {
        "delay_sessions": summarize_distribution(
            [row.first_high_score_to_exit_sessions for row in rows]
        ),
        "profit_at_first_high_score": summarize_distribution(
            [row.profit_at_first_high_score for row in rows]
        ),
        "peak_close_profit": summarize_distribution([row.peak_close_profit for row in rows]),
        "exit_profit": summarize_distribution([row.exit_profit for row in rows]),
        "giveback_from_peak": summarize_distribution([row.giveback_from_peak for row in rows]),
        "incremental_giveback": summarize_distribution(
            [row.incremental_giveback_after_first_high_score for row in rows]
        ),
        "evaluation_post_exit_return_3": summarize_distribution(
            [row.evaluation_post_exit_return_3 for row in rows]
        ),
        "evaluation_post_exit_return_5": summarize_distribution(
            [row.evaluation_post_exit_return_5 for row in rows]
        ),
    }


def _evaluate_step0(
    entries: Sequence[EntryLagObservation],
    exits: Sequence[ExitLagObservation],
    path_evidence: OfficialPathEvidence,
    top_entry_share: float | None,
    top_exit_share: float | None,
) -> Step0Decision:
    if path_evidence.status != "aligned":
        return Step0Decision(
            status="stop",
            reasons=("official filled path is not aligned",),
        )
    reasons = []

    entry_consistent, entry_reason = _consistent_annual_relationship(
        entries,
        date_getter=lambda row: row.buy_date,
        x_getter=lambda row: row.extension_from_earliest_cross_atr,
        y_getter=lambda row: row.evaluation_mfe_5,
        adverse_sign=-1,
        label="entry extension/MFE",
    )
    exit_consistent, exit_reason = _consistent_annual_relationship(
        exits,
        date_getter=lambda row: row.sell_date,
        x_getter=lambda row: row.first_high_score_to_exit_sessions,
        y_getter=lambda row: row.incremental_giveback_after_first_high_score,
        adverse_sign=1,
        label="exit delay/giveback",
    )
    if not entry_consistent and not exit_consistent:
        reasons.extend(reason for reason in (entry_reason, exit_reason) if reason)
        reasons.append("no adverse timing relationship has a consistent annual direction")

    if entry_consistent and top_entry_share is not None and top_entry_share >= 0.5:
        reasons.append("entry evidence fails ETF concentration check (top ETF is at least 50%)")
    if exit_consistent and top_exit_share is not None and top_exit_share >= 0.5:
        reasons.append("exit evidence fails ETF concentration check (top ETF is at least 50%)")
    return Step0Decision(
        status="stop" if reasons else "eligible_for_separate_design",
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _consistent_annual_relationship(
    rows: Sequence[object],
    date_getter: Callable[[object], str],
    x_getter: Callable[[object], float | None],
    y_getter: Callable[[object], float | None],
    adverse_sign: int,
    label: str,
) -> tuple[bool, str | None]:
    for year in (2019, 2020, 2021):
        pairs = [
            (float(x_getter(row)), float(y_getter(row)))
            for row in rows
            if pd.Timestamp(date_getter(row)).year == year
            and x_getter(row) is not None
            and y_getter(row) is not None
        ]
        if len(pairs) < 2:
            return False, "%s has insufficient paired evidence in %d" % (label, year)
        x_mean = sum(pair[0] for pair in pairs) / len(pairs)
        y_mean = sum(pair[1] for pair in pairs) / len(pairs)
        covariance = sum(
            (x - x_mean) * (y - y_mean) for x, y in pairs
        )
        if covariance == 0 or (1 if covariance > 0 else -1) != int(adverse_sign):
            return False, "%s adverse direction is inconsistent in %d" % (label, year)
    return True, None


def _top_code_share(rows: Sequence[object]) -> float | None:
    if not rows:
        return None
    counts: dict[str, int] = {}
    for row in rows:
        code = str(getattr(row, "code"))
        counts[code] = counts.get(code, 0) + 1
    return max(counts.values()) / len(rows)


def format_extreme_lag_report(report: ExtremeLagReport) -> str:
    lines = [
        "# Cross-Signal Extreme-Lag Attribution (2019-2021)",
        "",
        "Step 0 status: %s" % (
            "STOP" if report.decision.status == "stop" else "ELIGIBLE FOR SEPARATE DESIGN"
        ),
        "",
        "Official path status: `%s` (%d expected / %d actual)"
        % (
            report.path_evidence.status,
            report.path_evidence.expected_count,
            report.path_evidence.actual_count,
        ),
        "",
        "All MAE/MFE and post-exit returns below are forward labels only; they are not signal inputs.",
        "",
        "## Decision reasons",
        "",
    ]
    lines.extend("- %s" % reason for reason in report.decision.reasons)
    if not report.decision.reasons:
        lines.append("- The pre-registered consistency and concentration gates passed.")
    lines.extend(["", "## Entry distributions", ""])
    lines.extend(_format_distribution_groups(report.entry_distributions))
    lines.extend(["", "## Exit distributions", ""])
    lines.extend(_format_distribution_groups(report.exit_distributions))
    lines.extend(["", "## Tail observations (examples only)", ""])
    entry_tails = sorted(
        (
            row for row in report.entries
            if row.extension_from_earliest_cross_atr is not None
        ),
        key=lambda row: float(row.extension_from_earliest_cross_atr),
        reverse=True,
    )[:5]
    exit_tails = sorted(
        (
            row for row in report.exits
            if row.first_high_score_to_exit_sessions is not None
        ),
        key=lambda row: int(row.first_high_score_to_exit_sessions),
        reverse=True,
    )[:5]
    lines.extend(
        "- Entry example `%s` `%s`: extension_atr=%.6f, age2_share=%s"
        % (
            row.code,
            row.buy_date,
            float(row.extension_from_earliest_cross_atr),
            _format_optional(row.age_two_reversal_share),
        )
        for row in entry_tails
    )
    lines.extend(
        "- Exit example `%s` `%s`: type=%s, delay_sessions=%d, giveback=%s"
        % (
            row.code,
            row.sell_date,
            row.exit_type,
            int(row.first_high_score_to_exit_sessions),
            _format_optional(row.incremental_giveback_after_first_high_score),
        )
        for row in exit_tails
    )
    if not entry_tails and not exit_tails:
        lines.append("- No usable tail examples; the evidence gate is blocked or empty.")
    return "\n".join(lines) + "\n"


def _format_distribution_groups(
    groups: Mapping[str, Mapping[str, DistributionSummary]],
) -> list[str]:
    lines = []
    for group, metrics in groups.items():
        for metric, stats in metrics.items():
            lines.append(
                "- `%s` / `%s`: count=%d usable=%d missing=%d median=%s q1=%s q3=%s min=%s max=%s"
                % (
                    group,
                    metric,
                    stats.count,
                    stats.usable_count,
                    stats.missing_count,
                    _format_optional(stats.median),
                    _format_optional(stats.q1),
                    _format_optional(stats.q3),
                    _format_optional(stats.minimum),
                    _format_optional(stats.maximum),
                )
            )
    return lines


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else "%.6f" % value


def blocked_extreme_lag_report() -> ExtremeLagReport:
    return summarize_extreme_lag(
        [],
        [],
        OfficialPathEvidence(
            status="blocked_missing_official_path",
            expected_count=0,
            actual_count=0,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered 2019-2021 extreme-lag attribution."
    )
    parser.add_argument(
        "--joinquant-transactions",
        help="JoinQuant GBK transaction export for cross-v0.3.3 2019-2021",
    )
    parser.add_argument(
        "--report-dir",
        default="cross_signal_strategy/reports",
        help="Must resolve to cross_signal_strategy/reports",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report_dir = assert_repository_report_dir(args.report_dir)
    if not args.joinquant_transactions:
        report = blocked_extreme_lag_report()
        markdown_path, json_path = write_extreme_lag_artifacts(report, report_dir)
        print("BLOCKED_MISSING_OFFICIAL_FILL_PATH")
        print(markdown_path)
        print(json_path)
        return 2

    from cross_signal_strategy.research.order_path_diagnostics import (
        parse_joinquant_transaction_csv,
    )

    official_events = parse_joinquant_transaction_csv(args.joinquant_transactions)
    report = run_training_extreme_lag_attribution(official_events)
    markdown_path, json_path = write_extreme_lag_artifacts(report, report_dir)
    print("STEP0_%s" % report.decision.status.upper())
    print(markdown_path)
    print(json_path)
    return 0


def _validated_signal_frame(frame: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    required = {"date", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("signal frame is missing columns: %s" % ",".join(sorted(missing)))
    result = frame[["date", "close"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values("date").reset_index(drop=True)
    if result.empty:
        raise ValueError("signal frame is empty")
    max_date = result["date"].max()
    if max_date > signal_date:
        raise ValueError("signal frame ends after signal_date")
    if max_date != signal_date:
        raise ValueError("signal frame does not end on signal_date")
    return result


def _evaluation_excursions(
    closes: Sequence[float], entry_price: float, horizon: int
) -> tuple[float | None, float | None]:
    if len(closes) < int(horizon):
        return None, None
    returns = [float(close) / entry_price - 1.0 for close in closes[:horizon]]
    return min(returns), max(returns)


def _forward_return_at(
    closes: Sequence[float], reference_price: float, horizon: int
) -> float | None:
    if len(closes) < int(horizon) or reference_price <= 0:
        return None
    return float(closes[int(horizon) - 1]) / reference_price - 1.0


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_float(value: object) -> float | None:
    number = _optional_float(value)
    return number if number is not None and number > 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
