# -*- coding: utf-8 -*-
"""Observation-only attribution for changed moderate-KDJ candidate days.

Future returns in this module are ex-post labels. They never enter the signal
adapter, order planner, broker, ranking, or execution path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

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


DEFAULT_HORIZONS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class ChangedOrderAttribution:
    date: str
    code: str
    side: str
    path: str
    reason: str
    origin: str
    amount: int
    exec_price: float
    k_value: float | None
    official_buy_score: float
    candidate_buy_score: float
    official_sell_score: float
    candidate_sell_score: float
    buy_bonus: float
    sell_bonus: float
    buy_threshold_crossed: bool
    sell_threshold_crossed: bool
    price_confirmed: bool
    adx_protected: bool
    forward_returns: Mapping[int, float | None] = field(default_factory=dict)
    baseline_total_value: float = 0.0
    candidate_total_value: float = 0.0
    value_delta: float = 0.0


@dataclass(frozen=True)
class ChangedTradePath:
    path: str
    code: str
    buy_date: str
    sell_date: str
    sell_reason: str
    baseline_pnl: float | None
    candidate_pnl: float | None
    pnl_delta: float
    baseline_return_pct: float | None
    candidate_return_pct: float | None
    amount_delta: int


@dataclass(frozen=True)
class ChangedDayAttributionReport:
    changed_days: int
    origin_counts: Mapping[str, int]
    order_rows: tuple[ChangedOrderAttribution, ...]
    trade_rows: tuple[ChangedTradePath, ...]
    baseline_terminal_value: float
    candidate_terminal_value: float
    terminal_value_delta: float


def build_changed_order_attributions(
    baseline_days: Sequence[object],
    candidate_days: Sequence[object],
    official_source: object,
    candidate_source: object,
    loader: object,
    trade_dates: Sequence[str],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> list[ChangedOrderAttribution]:
    """Describe the filled-order symmetric difference on each changed day."""

    dates = tuple(str(value) for value in trade_dates)
    date_index = {date: index for index, date in enumerate(dates)}
    baseline_map = {str(day.date): day for day in baseline_days}
    candidate_map = {str(day.date): day for day in candidate_days}
    if tuple(baseline_map) != tuple(candidate_map):
        raise ValueError("A/B attribution requires identical ordered dates")

    rows: list[ChangedOrderAttribution] = []
    for date in baseline_map:
        baseline_day = baseline_map[date]
        candidate_day = candidate_map[date]
        baseline_events = _filled_events(baseline_day)
        candidate_events = _filled_events(candidate_day)
        baseline_only = sorted(set(baseline_events) - set(candidate_events))
        candidate_only = sorted(set(candidate_events) - set(baseline_events))
        if not baseline_only and not candidate_only:
            continue

        candidate_direct: Dict[tuple[str, str, str], str] = {}
        for key in candidate_only:
            side, code, _reason = key
            official_score = official_source.score(code, date) or {}
            candidate_score = candidate_source.score(code, date) or {}
            origin = _direct_origin(side, official_score, candidate_score)
            if origin is not None:
                candidate_direct[key] = origin
        has_direct = bool(candidate_direct)

        for path, keys, events in (
            ("baseline_only", baseline_only, baseline_events),
            ("candidate_only", candidate_only, candidate_events),
        ):
            for key in keys:
                side, code, reason = key
                official_score = official_source.score(code, date) or {}
                candidate_score = candidate_source.score(code, date) or {}
                origin = candidate_direct.get(key)
                if origin is None:
                    origin = "same_day_portfolio_chain" if has_direct else "portfolio_chain"
                rows.append(
                    _changed_order_row(
                        date=date,
                        code=code,
                        side=side,
                        path=path,
                        reason=reason,
                        origin=origin,
                        order=events[key],
                        official_score=official_score,
                        candidate_score=candidate_score,
                        loader=loader,
                        dates=dates,
                        date_index=date_index,
                        horizons=horizons,
                        baseline_total_value=float(baseline_day.total_value),
                        candidate_total_value=float(candidate_day.total_value),
                    )
                )
    return rows


def compare_closed_trade_paths(
    baseline_trades: Iterable[ClosedTradeDiagnostic],
    candidate_trades: Iterable[ClosedTradeDiagnostic],
) -> list[ChangedTradePath]:
    """Compare path-only trades and material differences on matched trades."""

    baseline = {_trade_key(trade): trade for trade in baseline_trades}
    candidate = {_trade_key(trade): trade for trade in candidate_trades}
    rows = []
    for key in sorted(set(baseline) | set(candidate), key=lambda item: (item[1], item[0], item[2])):
        base = baseline.get(key)
        cand = candidate.get(key)
        if base is not None and cand is not None:
            if (
                int(base.amount) == int(cand.amount)
                and abs(float(base.pnl) - float(cand.pnl)) < 1e-9
                and abs(float(base.return_pct) - float(cand.return_pct)) < 1e-9
            ):
                continue
            path = "matched_changed"
        elif base is not None:
            path = "baseline_only"
        else:
            path = "candidate_only"
        baseline_pnl = float(base.pnl) if base is not None else None
        candidate_pnl = float(cand.pnl) if cand is not None else None
        rows.append(
            ChangedTradePath(
                path=path,
                code=key[0],
                buy_date=key[1],
                sell_date=key[2],
                sell_reason=key[3],
                baseline_pnl=baseline_pnl,
                candidate_pnl=candidate_pnl,
                pnl_delta=(candidate_pnl or 0.0) - (baseline_pnl or 0.0),
                baseline_return_pct=(
                    float(base.return_pct) if base is not None else None
                ),
                candidate_return_pct=(
                    float(cand.return_pct) if cand is not None else None
                ),
                amount_delta=(
                    (int(cand.amount) if cand is not None else 0)
                    - (int(base.amount) if base is not None else 0)
                ),
            )
        )
    return rows


def run_kdj_tiered_changed_day_attribution(
    loader: object | None = None,
    initial_cash: float = 20000.0,
    warmup_root: Path | str = APPROVED_WARMUP_ROOT,
) -> ChangedDayAttributionReport:
    """Run baseline and the frozen moderate candidate for attribution only."""

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
    baseline_planner = DiagnosticOrderPlanner(
        cached,
        etf_pool=pool,
        params=dict(params),
        trade_dates=trade_dates,
    )
    candidate_planner = DiagnosticOrderPlanner(
        candidate_source,
        etf_pool=pool,
        params=dict(params),
        trade_dates=trade_dates,
    )
    baseline_days = _run_replay(
        loader,
        baseline_planner,
        trade_dates,
        initial_cash,
    )
    candidate_days = _run_replay(
        loader,
        candidate_planner,
        trade_dates,
        initial_cash,
    )
    order_rows = build_changed_order_attributions(
        baseline_days,
        candidate_days,
        cached,
        candidate_source,
        loader,
        trade_dates,
    )
    baseline_trades = build_closed_trade_diagnostics(
        baseline_days,
        baseline_planner.entry_score_snapshots,
        baseline_planner.exit_score_snapshots,
    )
    candidate_trades = build_closed_trade_diagnostics(
        candidate_days,
        candidate_planner.entry_score_snapshots,
        candidate_planner.exit_score_snapshots,
    )
    trade_rows = compare_closed_trade_paths(baseline_trades, candidate_trades)
    changed_dates = {row.date for row in order_rows}
    origin_counts = Counter(row.origin for row in order_rows)
    baseline_terminal = float(baseline_days[-1].total_value)
    candidate_terminal = float(candidate_days[-1].total_value)
    return ChangedDayAttributionReport(
        changed_days=len(changed_dates),
        origin_counts=dict(sorted(origin_counts.items())),
        order_rows=tuple(order_rows),
        trade_rows=tuple(trade_rows),
        baseline_terminal_value=baseline_terminal,
        candidate_terminal_value=candidate_terminal,
        terminal_value_delta=candidate_terminal - baseline_terminal,
    )


def format_kdj_tiered_changed_day_attribution(
    report: ChangedDayAttributionReport,
) -> str:
    lines = [
        "KDJ moderate-points changed-day attribution (2019-2021; observation only)",
        "changed_days=%d origin_counts=%s terminal_baseline=%.2f terminal_candidate=%.2f delta=%.2f"
        % (
            report.changed_days,
            dict(report.origin_counts),
            report.baseline_terminal_value,
            report.candidate_terminal_value,
            report.terminal_value_delta,
        ),
    ]
    for row in report.order_rows:
        lines.append(
            "ORDER date=%s path=%s side=%s code=%s reason=%s origin=%s "
            "k=%s official_buy=%.1f candidate_buy=%.1f buy_bonus=%.1f "
            "official_sell=%.1f candidate_sell=%.1f sell_bonus=%.1f "
            "buy_cross=%s sell_cross=%s price_confirmed=%s adx_protected=%s "
            "forward=%s value_delta=%.2f"
            % (
                row.date,
                row.path,
                row.side,
                row.code,
                row.reason,
                row.origin,
                "n/a" if row.k_value is None else "%.2f" % row.k_value,
                row.official_buy_score,
                row.candidate_buy_score,
                row.buy_bonus,
                row.official_sell_score,
                row.candidate_sell_score,
                row.sell_bonus,
                row.buy_threshold_crossed,
                row.sell_threshold_crossed,
                row.price_confirmed,
                row.adx_protected,
                {
                    horizon: None if value is None else round(value, 6)
                    for horizon, value in row.forward_returns.items()
                },
                row.value_delta,
            )
        )
    for row in report.trade_rows:
        lines.append(
            "TRADE path=%s code=%s buy=%s sell=%s reason=%s baseline_pnl=%s "
            "candidate_pnl=%s pnl_delta=%.2f baseline_return=%s candidate_return=%s "
            "amount_delta=%d"
            % (
                row.path,
                row.code,
                row.buy_date,
                row.sell_date,
                row.sell_reason,
                _optional_number(row.baseline_pnl),
                _optional_number(row.candidate_pnl),
                row.pnl_delta,
                _optional_number(row.baseline_return_pct),
                _optional_number(row.candidate_return_pct),
                row.amount_delta,
            )
        )
    lines.append("future_returns=ex_post_labels_only; never used by strategy")
    lines.append("authority=local_attribution_only; JoinQuant remains authoritative")
    return "\n".join(lines)


def _filled_events(day: object) -> Dict[tuple[str, str, str], object]:
    events = {}
    for order in getattr(day, "orders", []):
        if not getattr(order, "filled", False):
            continue
        amount = int(getattr(order, "amount_delta", 0))
        if amount == 0:
            continue
        side = "BUY" if amount > 0 else "SELL"
        code = str(getattr(order, "code")).split(".")[0]
        reason = str(getattr(order, "reason", ""))
        events[(side, code, reason)] = order
    return events


def _direct_origin(
    side: str,
    official_score: Mapping[str, object],
    candidate_score: Mapping[str, object],
) -> str | None:
    if side == "BUY" and _number(candidate_score.get("buy_extreme_zone_score")) > 0:
        return "buy_bonus_direct"
    if (
        side == "SELL"
        and _number(candidate_score.get("sell_extreme_zone_score")) > 0
        and _number(official_score.get("sell_score")) < 30.0
        <= _number(candidate_score.get("sell_score"))
    ):
        return "sell_bonus_direct"
    return None


def _changed_order_row(
    *,
    date: str,
    code: str,
    side: str,
    path: str,
    reason: str,
    origin: str,
    order: object,
    official_score: Mapping[str, object],
    candidate_score: Mapping[str, object],
    loader: object,
    dates: Sequence[str],
    date_index: Mapping[str, int],
    horizons: Sequence[int],
    baseline_total_value: float,
    candidate_total_value: float,
) -> ChangedOrderAttribution:
    official_buy = _number(official_score.get("buy_score"))
    candidate_buy = _number(candidate_score.get("buy_score"))
    official_sell = _number(official_score.get("sell_score"))
    candidate_sell = _number(candidate_score.get("sell_score"))
    return ChangedOrderAttribution(
        date=date,
        code=code,
        side=side,
        path=path,
        reason=reason,
        origin=origin,
        amount=abs(int(getattr(order, "amount_delta", 0))),
        exec_price=float(getattr(order, "exec_price", 0.0)),
        k_value=_optional_float(candidate_score.get("k")),
        official_buy_score=official_buy,
        candidate_buy_score=candidate_buy,
        official_sell_score=official_sell,
        candidate_sell_score=candidate_sell,
        buy_bonus=_number(candidate_score.get("buy_extreme_zone_score")),
        sell_bonus=_number(candidate_score.get("sell_extreme_zone_score")),
        buy_threshold_crossed=official_buy < 60.0 <= candidate_buy,
        sell_threshold_crossed=official_sell < 30.0 <= candidate_sell,
        price_confirmed=bool(strategy.has_signal_sell_confirmation(candidate_score)),
        adx_protected=bool(
            strategy.is_protected_by_strong_adx_uptrend(candidate_score)
        ),
        forward_returns=_forward_returns(
            loader,
            code,
            date,
            dates,
            date_index,
            horizons,
        ),
        baseline_total_value=baseline_total_value,
        candidate_total_value=candidate_total_value,
        value_delta=candidate_total_value - baseline_total_value,
    )


def _forward_returns(
    loader: object,
    code: str,
    date: str,
    dates: Sequence[str],
    date_index: Mapping[str, int],
    horizons: Sequence[int],
) -> Dict[int, float | None]:
    try:
        anchor = float(loader.get_minute_bar(code, date, "09:35")["close"])
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        anchor = 0.0
    start = date_index.get(date)
    result: Dict[int, float | None] = {}
    for raw_horizon in horizons:
        horizon = int(raw_horizon)
        target_index = start + horizon if start is not None else len(dates)
        if anchor <= 0 or target_index >= len(dates):
            result[horizon] = None
            continue
        target_date = dates[target_index]
        try:
            frame = loader.load_daily_frame(code, target_date)
            rows = frame.loc[frame["date"].astype(str) == target_date]
            close = float(rows.iloc[0]["close"])
        except (FileNotFoundError, KeyError, IndexError, TypeError, ValueError):
            result[horizon] = None
            continue
        result[horizon] = close / anchor - 1.0
    return result


def _trade_key(trade: ClosedTradeDiagnostic) -> tuple[str, str, str, str]:
    return (
        str(trade.code).split(".")[0],
        str(trade.buy_date),
        str(trade.sell_date),
        str(trade.sell_reason),
    )


def _number(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_number(value: float | None) -> str:
    return "n/a" if value is None else "%.4f" % float(value)


def main() -> None:
    report = run_kdj_tiered_changed_day_attribution()
    print(format_kdj_tiered_changed_day_attribution(report))


if __name__ == "__main__":
    main()
