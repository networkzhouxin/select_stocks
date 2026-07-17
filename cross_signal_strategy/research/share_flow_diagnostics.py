# -*- coding: utf-8 -*-
"""Training-only, observation-only ETF shares-outstanding diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Union

import pandas as pd

from cross_signal_strategy.local.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)
from cross_signal_strategy.research.trade_diagnostics import (
    DiagnosticOrderPlanner,
    build_closed_trade_diagnostics,
)


APPROVED_SHARE_FLOW_ROOT = Path(
    r"G:\financial\history_data\cross_signal_flow_train_2018_2021"
)
FLOW_WARMUP_START = pd.Timestamp("2018-01-01")
TRAINING_START = pd.Timestamp("2019-01-01")
TRAINING_END = pd.Timestamp("2021-12-31")
FLOW_LOOKBACK = 5
REQUIRED_COLUMNS = ("code", "trade_date", "total_share_wan")
ELIGIBLE_DOMESTIC_CODES = (
    "159915",
    "512100",
    "159928",
    "518880",
    "159985",
)
BLOCKED_QDII_CODES = (
    "513100",
    "513500",
    "513880",
    "513050",
)
KNOWN_CODES = frozenset(ELIGIBLE_DOMESTIC_CODES + BLOCKED_QDII_CODES)


PathLike = Union[str, Path]


def _resolve(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def _code_text(code: object) -> str:
    return str(code).split(".")[0]


def _date_text(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class CorporateAction:
    code: str
    trade_date: str
    event: str


@dataclass(frozen=True)
class ShareFlowObservation:
    code: str
    decision_date: str
    signal_date: str
    baseline_date: str | None
    value: float | None
    raw_state: str
    comparison_group: str


def validate_share_frame(frame: pd.DataFrame, expected_code: str) -> pd.DataFrame:
    """Validate and normalize one isolated share-history frame."""
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError("share-flow frame missing columns: %s" % ", ".join(missing))

    normalized = frame.loc[:, REQUIRED_COLUMNS].copy()
    normalized["code"] = normalized["code"].map(_code_text)
    expected = _code_text(expected_code)
    if not normalized.empty and not normalized["code"].eq(expected).all():
        raise ValueError("share-flow frame code mismatch for %s" % expected)

    parsed_dates = pd.to_datetime(normalized["trade_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("share-flow frame contains invalid trade_date")
    if (
        (parsed_dates < FLOW_WARMUP_START).any()
        or (parsed_dates > TRAINING_END).any()
    ):
        raise ValueError("share-flow frame contains dates outside approved share-flow dates")
    normalized["trade_date"] = parsed_dates

    shares = pd.to_numeric(normalized["total_share_wan"], errors="coerce")
    if shares.isna().any() or (shares <= 0).any():
        raise ValueError("share-flow frame requires positive total_share_wan")
    normalized["total_share_wan"] = shares.astype(float)
    if normalized["trade_date"].duplicated().any():
        raise ValueError("share-flow frame contains duplicate trade_date")
    return normalized.sort_values("trade_date").reset_index(drop=True)


@dataclass(frozen=True)
class ShareFlowDataLoader:
    """Exact-root reader for isolated 2018-2021 ETF share histories."""

    root: PathLike = APPROVED_SHARE_FLOW_ROOT

    def __post_init__(self) -> None:
        resolved = _resolve(self.root)
        approved = _resolve(APPROVED_SHARE_FLOW_ROOT)
        if resolved != approved:
            raise ValueError(
                "Use approved share-flow data root only: %s"
                % APPROVED_SHARE_FLOW_ROOT
            )
        object.__setattr__(self, "root", resolved)
        object.__setattr__(self, "_frame_cache", {})
        object.__setattr__(self, "_corporate_actions_cache", None)

    def load_history(self, code: str, signal_date: object) -> pd.DataFrame:
        code_text = _code_text(code)
        if code_text not in KNOWN_CODES:
            raise ValueError("Unsupported share-flow code: %s" % code_text)
        signal_ts = pd.Timestamp(signal_date)
        if signal_ts < FLOW_WARMUP_START or signal_ts > TRAINING_END:
            raise ValueError("Requested date is outside approved share-flow dates")

        frames = []
        for path in self._history_paths(code_text, int(signal_ts.year)):
            if not path.exists():
                continue
            cache = getattr(self, "_frame_cache")
            key = str(path)
            if key not in cache:
                raw = pd.read_csv(path, dtype={"code": str})
                cache[key] = validate_share_frame(raw, expected_code=code_text)
            frames.append(cache[key])

        if not frames:
            return pd.DataFrame(columns=REQUIRED_COLUMNS).assign(
                trade_date=pd.Series(dtype="datetime64[ns]"),
                total_share_wan=pd.Series(dtype="float64"),
            )
        combined = pd.concat(frames, ignore_index=True)
        combined = validate_share_frame(combined, expected_code=code_text)
        visible = combined.loc[combined["trade_date"] <= signal_ts]
        return visible.reset_index(drop=True).copy()

    def load_corporate_actions(self) -> tuple[CorporateAction, ...]:
        cached = getattr(self, "_corporate_actions_cache")
        if cached is not None:
            return cached
        path = self.root / "meta" / "corporate_actions.csv"
        frame = pd.read_csv(path, dtype={"code": str})
        required = {"code", "trade_date", "event"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                "corporate-action file missing columns: %s" % ", ".join(missing)
            )
        normalized = frame.loc[:, ["code", "trade_date", "event"]].copy()
        normalized["code"] = normalized["code"].map(_code_text)
        if not normalized["code"].isin(KNOWN_CODES).all():
            raise ValueError("corporate-action file contains unsupported code")
        dates = pd.to_datetime(normalized["trade_date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("corporate-action file contains invalid trade_date")
        if (dates < FLOW_WARMUP_START).any() or (dates > TRAINING_END).any():
            raise ValueError("corporate-action date is outside approved share-flow dates")
        normalized["trade_date"] = dates.dt.strftime("%Y-%m-%d")
        if normalized.duplicated(["code", "trade_date", "event"]).any():
            raise ValueError("corporate-action file contains duplicate event")
        actions = tuple(
            CorporateAction(
                code=str(row.code),
                trade_date=str(row.trade_date),
                event=str(row.event),
            )
            for row in normalized.itertuples(index=False)
        )
        object.__setattr__(self, "_corporate_actions_cache", actions)
        return actions

    def _history_paths(self, code: str, signal_year: int) -> tuple[Path, ...]:
        if signal_year == 2018:
            return (self.root / "warmup" / "2018" / (code + ".csv"),)
        previous_year = signal_year - 1
        previous_partition = "warmup" if previous_year == 2018 else "training"
        return (
            self.root
            / previous_partition
            / str(previous_year)
            / (code + ".csv"),
            self.root / "training" / str(signal_year) / (code + ".csv"),
        )


def calculate_share_flow(
    frame: pd.DataFrame,
    code: str,
    decision_date: object,
    signal_date: object,
    corporate_actions: Iterable[CorporateAction],
) -> ShareFlowObservation:
    """Calculate the fixed T-1/T-6 log share change or an excluded state."""
    code_text = _code_text(code)
    if code_text not in KNOWN_CODES:
        raise ValueError("Unsupported share-flow code: %s" % code_text)
    decision_ts = pd.Timestamp(decision_date)
    signal_ts = pd.Timestamp(signal_date)
    if decision_ts < TRAINING_START or decision_ts > TRAINING_END:
        raise ValueError("decision_date is outside 2019-2021 training window")
    if signal_ts >= decision_ts:
        raise ValueError("signal_date must be strictly before decision_date")

    common = {
        "code": code_text,
        "decision_date": _date_text(decision_ts),
        "signal_date": _date_text(signal_ts),
    }
    if code_text in BLOCKED_QDII_CODES:
        return ShareFlowObservation(
            **common,
            baseline_date=None,
            value=None,
            raw_state="blocked_qdii",
            comparison_group="excluded",
        )
    if frame.empty:
        return ShareFlowObservation(
            **common,
            baseline_date=None,
            value=None,
            raw_state="insufficient_history",
            comparison_group="excluded",
        )

    visible = validate_share_frame(frame, expected_code=code_text)
    if not visible.empty and visible["trade_date"].max() > signal_ts:
        raise ValueError("share-flow frame contains data after signal_date")
    signal_rows = visible.loc[visible["trade_date"] == signal_ts]
    if signal_rows.empty or len(visible) < FLOW_LOOKBACK + 1:
        return ShareFlowObservation(
            **common,
            baseline_date=None,
            value=None,
            raw_state="insufficient_history",
            comparison_group="excluded",
        )

    window = visible.tail(FLOW_LOOKBACK + 1)
    baseline = window.iloc[0]
    endpoint = window.iloc[-1]
    baseline_ts = pd.Timestamp(baseline["trade_date"])
    endpoint_ts = pd.Timestamp(endpoint["trade_date"])
    if endpoint_ts != signal_ts:
        return ShareFlowObservation(
            **common,
            baseline_date=None,
            value=None,
            raw_state="insufficient_history",
            comparison_group="excluded",
        )

    crosses_action = any(
        _code_text(action.code) == code_text
        and baseline_ts < pd.Timestamp(action.trade_date) <= signal_ts
        for action in corporate_actions
    )
    baseline_date = _date_text(baseline_ts)
    if crosses_action:
        return ShareFlowObservation(
            **common,
            baseline_date=baseline_date,
            value=None,
            raw_state="corporate_action",
            comparison_group="excluded",
        )

    value = math.log(
        float(endpoint["total_share_wan"])
        / float(baseline["total_share_wan"])
    )
    if value > 0:
        raw_state = "net_creation"
        comparison_group = "positive"
    elif value < 0:
        raw_state = "net_redemption"
        comparison_group = "non_positive"
    else:
        raw_state = "flat"
        comparison_group = "non_positive"
    return ShareFlowObservation(
        **common,
        baseline_date=baseline_date,
        value=value,
        raw_state=raw_state,
        comparison_group=comparison_group,
    )


@dataclass
class ShareFlowSignalAdapter:
    """Add share-flow metadata to copied official scores without changing orders."""

    source: object
    flow_loader: ShareFlowDataLoader
    _cache: dict[tuple[str, str], tuple[dict | None, str | None]] = field(
        default_factory=dict
    )

    def score(self, code, current_date, return_reason=False):
        code_text = _code_text(code)
        date_text = _date_text(current_date)
        key = (code_text, date_text)
        if key not in self._cache:
            base_score, reason = self.source.score(
                code_text,
                date_text,
                return_reason=True,
            )
            if base_score is None:
                self._cache[key] = (None, reason)
            else:
                _, frame_signal_date = self.source.load_signal_frame(
                    code_text,
                    date_text,
                )
                score_signal_date = base_score.get("signal_date")
                if score_signal_date is None:
                    raise ValueError("share-flow base score is missing signal_date")
                if _date_text(frame_signal_date) != _date_text(score_signal_date):
                    raise ValueError(
                        "share-flow signal_date does not match base price frame"
                    )
                signal_text = _date_text(score_signal_date)
                if code_text in BLOCKED_QDII_CODES:
                    history = pd.DataFrame()
                    actions: tuple[CorporateAction, ...] = ()
                else:
                    history = self.flow_loader.load_history(code_text, signal_text)
                    actions = self.flow_loader.load_corporate_actions()
                observation = calculate_share_flow(
                    frame=history,
                    code=code_text,
                    decision_date=date_text,
                    signal_date=signal_text,
                    corporate_actions=actions,
                )
                enriched = dict(base_score)
                enriched.update({
                    "share_flow_value_5": observation.value,
                    "share_flow_raw_state": observation.raw_state,
                    "share_flow_comparison_group": observation.comparison_group,
                    "share_flow_signal_date": observation.signal_date,
                    "share_flow_baseline_date": observation.baseline_date,
                    "share_flow_blocked": (
                        observation.comparison_group == "excluded"
                    ),
                })
                self._cache[key] = (enriched, None)

        cached_score, cached_reason = self._cache[key]
        result = dict(cached_score) if cached_score is not None else None
        return (result, cached_reason) if return_reason else result


@dataclass(frozen=True)
class ShareFlowCoverage:
    total_closed_buys: int = 0
    eligible_domestic_closed_buys: int = 0
    comparable_closed_buys: int = 0

    @property
    def coverage_rate_all(self) -> float:
        return (
            self.comparable_closed_buys / self.total_closed_buys
            if self.total_closed_buys
            else 0.0
        )

    @property
    def coverage_rate_eligible(self) -> float:
        return (
            self.comparable_closed_buys / self.eligible_domestic_closed_buys
            if self.eligible_domestic_closed_buys
            else 0.0
        )


@dataclass(frozen=True)
class ShareFlowStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_return: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else None


@dataclass(frozen=True)
class ShareFlowGateDecision:
    passed: bool
    dominant_group: str | None = None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShareFlowReport:
    coverage: ShareFlowCoverage
    raw_state_counts: Dict[str, int]
    by_group: Dict[str, ShareFlowStats]
    by_year_group: Dict[str, ShareFlowStats]
    gate: ShareFlowGateDecision


def evaluate_share_flow_gate(
    positive_by_year: Mapping[int, ShareFlowStats],
    non_positive_by_year: Mapping[int, ShareFlowStats],
) -> ShareFlowGateDecision:
    """Apply the locked sample and annual-direction observation gate."""
    years = (2019, 2020, 2021)
    reasons = []
    groups = {
        "positive": positive_by_year,
        "non_positive": non_positive_by_year,
    }
    for name, values in groups.items():
        total = sum(
            values.get(year, ShareFlowStats()).closed_trades for year in years
        )
        if total < 6:
            reasons.append("%s group has fewer than 6 closed trades" % name)
        for year in years:
            if values.get(year, ShareFlowStats()).closed_trades < 2:
                reasons.append(
                    "%d %s group has fewer than 2 closed trades" % (year, name)
                )

    positive_better = all(
        positive_by_year.get(year, ShareFlowStats()).average_return
        > non_positive_by_year.get(year, ShareFlowStats()).average_return
        and positive_by_year.get(year, ShareFlowStats()).win_rate
        > non_positive_by_year.get(year, ShareFlowStats()).win_rate
        for year in years
    )
    non_positive_better = all(
        non_positive_by_year.get(year, ShareFlowStats()).average_return
        > positive_by_year.get(year, ShareFlowStats()).average_return
        and non_positive_by_year.get(year, ShareFlowStats()).win_rate
        > positive_by_year.get(year, ShareFlowStats()).win_rate
        for year in years
    )
    if positive_better:
        dominant_group = "positive"
    elif non_positive_better:
        dominant_group = "non_positive"
    else:
        dominant_group = None
        reasons.append(
            "neither group has strictly higher average return and win rate in every year"
        )
    return ShareFlowGateDecision(
        passed=not reasons,
        dominant_group=dominant_group,
        reasons=tuple(reasons),
    )


def build_share_flow_report(trades: Iterable[object]) -> ShareFlowReport:
    """Aggregate closed buys by pre-registered share-flow state and year."""
    items = list(trades)
    _assert_training_trade_dates(items)
    eligible = [
        trade for trade in items if _code_text(trade.code) in ELIGIBLE_DOMESTIC_CODES
    ]
    comparable = [
        trade
        for trade in eligible
        if _comparison_group(trade) in {"positive", "non_positive"}
    ]
    raw_counts = dict(sorted(Counter(
        _raw_state(trade) for trade in items
    ).items()))
    by_group = {
        group: _share_flow_stats([
            trade for trade in comparable if _comparison_group(trade) == group
        ])
        for group in ("positive", "non_positive")
    }
    by_year_group = {
        "%d:%s" % (year, group): _share_flow_stats([
            trade
            for trade in comparable
            if str(trade.buy_date).startswith(str(year))
            and _comparison_group(trade) == group
        ])
        for year in (2019, 2020, 2021)
        for group in ("positive", "non_positive")
    }
    positive_by_year = {
        year: by_year_group["%d:positive" % year]
        for year in (2019, 2020, 2021)
    }
    non_positive_by_year = {
        year: by_year_group["%d:non_positive" % year]
        for year in (2019, 2020, 2021)
    }
    return ShareFlowReport(
        coverage=ShareFlowCoverage(
            total_closed_buys=len(items),
            eligible_domestic_closed_buys=len(eligible),
            comparable_closed_buys=len(comparable),
        ),
        raw_state_counts=raw_counts,
        by_group=by_group,
        by_year_group=by_year_group,
        gate=evaluate_share_flow_gate(positive_by_year, non_positive_by_year),
    )


def _comparison_group(trade: object) -> str:
    return str(trade.entry_score.get("share_flow_comparison_group", "excluded"))


def _raw_state(trade: object) -> str:
    return str(trade.entry_score.get("share_flow_raw_state", "insufficient_history"))


def _share_flow_stats(trades: Iterable[object]) -> ShareFlowStats:
    items = list(trades)
    returns = [float(trade.return_pct) / 100.0 for trade in items]
    return ShareFlowStats(
        closed_trades=len(items),
        wins=sum(1 for trade in items if float(trade.pnl) > 0),
        losses=sum(1 for trade in items if float(trade.pnl) < 0),
        realized_pnl=sum(float(trade.pnl) for trade in items),
        gross_profit=sum(float(trade.pnl) for trade in items if float(trade.pnl) > 0),
        gross_loss=sum(abs(float(trade.pnl)) for trade in items if float(trade.pnl) < 0),
        average_return=(sum(returns) / len(returns) if returns else 0.0),
    )


def _assert_training_trade_dates(trades: Iterable[object]) -> None:
    for trade in trades:
        for value in (trade.buy_date, trade.sell_date):
            date = pd.Timestamp(value)
            if date < TRAINING_START or date > TRAINING_END:
                raise ValueError(
                    "share-flow attribution contains dates outside 2019-2021 training window"
                )


def run_training_share_flow_observation(
    loader=None,
    flow_loader=None,
    initial_cash: float = 20000.0,
) -> ShareFlowReport:
    """Run one shadow-only attribution through the frozen training replay."""
    loader = loader or CrossSignalTrainingDataLoader()
    flow_loader = flow_loader or ShareFlowDataLoader()
    trade_dates = get_training_trade_dates(loader)
    source = build_training_signal_adapter(loader)
    adapter = ShareFlowSignalAdapter(source=source, flow_loader=flow_loader)
    planner = DiagnosticOrderPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    trades = build_closed_trade_diagnostics(
        results,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )
    return build_share_flow_report(trades)


def format_share_flow_report(report: ShareFlowReport) -> str:
    lines = [
        "Cross-signal ETF share-flow shadow diagnostic (2019-2021, observation-only)",
        "RULE log(shares[T-1]/shares[T-6]) over exactly five observations",
        "ELIGIBLE %s" % ",".join(ELIGIBLE_DOMESTIC_CODES),
        "BLOCKED_QDII %s" % ",".join(BLOCKED_QDII_CODES),
        (
            "COVERAGE all={}/{} ({:.2%}) eligible={}/{} ({:.2%})"
        ).format(
            report.coverage.comparable_closed_buys,
            report.coverage.total_closed_buys,
            report.coverage.coverage_rate_all,
            report.coverage.comparable_closed_buys,
            report.coverage.eligible_domestic_closed_buys,
            report.coverage.coverage_rate_eligible,
        ),
    ]
    for state, count in report.raw_state_counts.items():
        lines.append("RAW_STATE %s count=%d" % (state, count))
    for label, section in (
        ("GROUP", report.by_group),
        ("YEAR", report.by_year_group),
    ):
        for key, stats in section.items():
            ratio = stats.profit_loss_ratio
            lines.append(
                (
                    "%s %s trades=%d wins=%d losses=%d pnl=%.2f "
                    "avg_ret=%.2f%% win=%.2f%% pl=%s"
                )
                % (
                    label,
                    key,
                    stats.closed_trades,
                    stats.wins,
                    stats.losses,
                    stats.realized_pnl,
                    stats.average_return * 100.0,
                    stats.win_rate * 100.0,
                    "n/a" if ratio is None else "%.3f" % ratio,
                )
            )
    lines.append(
        "OBSERVATION_GATE passed=%s dominant_group=%s"
        % (report.gate.passed, report.gate.dominant_group or "none")
    )
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def main() -> None:
    report = run_training_share_flow_observation()
    print(format_share_flow_report(report))


if __name__ == "__main__":
    main()
