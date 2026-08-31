"""Read-only performance analysis primitives for resonance candidates."""

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import html
import importlib.util
import json
import math
import os
import pathlib
import re
import statistics
import sys
import tempfile


INITIAL_CAPITAL = 20000.0
FINAL_TOTAL_RETURN_GATE = 1.2925
FINAL_WIN_RATE_GATE = 0.558
FINAL_MAX_DRAWDOWN_GATE = 0.0628
FINAL_MIN_CLOSED_TRADES = 80
FINAL_MAX_TOP_PROFIT_SHARE = 0.50
BENCHMARK_RETURN = 0.6410
EVALUATION_START = date(2019, 1, 1)
EVALUATION_END = date(2021, 12, 31)


FILL_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"security=(?P<code>\S+).*action=(?P<action>open|close).*"
    r"trade price:\s*(?P<price>[0-9.]+),\s*"
    r"amount:\s*(?P<amount>\d+),\s*"
    r"commission:\s*(?P<commission>[0-9.]+)"
)


@dataclass(frozen=True)
class Fill:
    timestamp: datetime
    trade_date: date
    code: str
    side: str
    price: float
    amount: int
    commission: float


@dataclass(frozen=True)
class PortfolioPoint:
    closing_date: date
    total_value: float
    available_cash: float
    positions: tuple


@dataclass(frozen=True)
class ParsedLog:
    fills: tuple
    portfolio_points: tuple
    strategy_builds: tuple


@dataclass(frozen=True)
class CompletedTrade:
    code: str
    entry_date: date
    exit_date: date
    buy_price: float
    buy_amount: int
    buy_commission: float
    sell_price: float
    sell_amount: int
    sell_commission: float
    pnl: float
    return_rate: float
    amount_ratio: float


@dataclass(frozen=True)
class TradeLedger:
    completed_trades: tuple
    open_positions: tuple


def _reject_duplicate_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _parse_fill(line):
    match = FILL_RE.search(html.unescape(line))
    if match is None:
        return None
    price = float(match.group("price"))
    amount = int(match.group("amount"))
    commission = float(match.group("commission"))
    if not math.isfinite(price) or price <= 0:
        raise ValueError("fill price must be finite and positive")
    if amount <= 0:
        raise ValueError("fill amount must be positive")
    if not math.isfinite(commission) or commission < 0:
        raise ValueError("fill commission must be finite and nonnegative")
    timestamp = datetime.strptime(
        match.group("timestamp"), "%Y-%m-%d %H:%M:%S",
    )
    return Fill(
        timestamp=timestamp,
        trade_date=timestamp.date(),
        code=match.group("code"),
        side="BUY" if match.group("action") == "open" else "SELL",
        price=price,
        amount=amount,
        commission=commission,
    )


def _required_finite_number(payload, field):
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("portfolio %s must be finite" % field)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("portfolio %s must be finite" % field)
    return value


def _parse_positions(value):
    if not isinstance(value, dict):
        raise ValueError("portfolio positions must be an object")
    positions = []
    for code, amount in value.items():
        if not isinstance(code, str) or not code:
            raise ValueError("portfolio positions code must be nonempty")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError("portfolio positions amount must be a positive integer")
        positions.append((code, amount))
    return tuple(sorted(positions))


def _parse_portfolio_point(payload):
    closing_date = payload.get("closing_date")
    if not isinstance(closing_date, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", closing_date):
        raise ValueError("portfolio closing_date must be an ISO date")
    try:
        parsed_date = date.fromisoformat(closing_date)
    except ValueError as exc:
        raise ValueError("portfolio closing_date must be an ISO date") from exc
    total_value = _required_finite_number(payload, "total_value")
    if total_value <= 0:
        raise ValueError("portfolio total_value must be positive")
    available_cash = _required_finite_number(payload, "available_cash")
    return PortfolioPoint(
        closing_date=parsed_date,
        total_value=total_value,
        available_cash=available_cash,
        positions=_parse_positions(payload.get("positions")),
    )


def _parse_structured_payload(line):
    text = html.unescape(line.strip())
    payload_start = text.find("{")
    if payload_start < 0 or '"event"' not in text[payload_start:]:
        return None
    try:
        payload = json.loads(
            text[payload_start:],
            object_pairs_hook=_reject_duplicate_json_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid structured JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("structured event must be an object")
    return payload


def parse_joinquant_log(paths):
    fills = []
    portfolio_points = []
    strategy_builds = []
    previous_portfolio_date = None
    normalized_paths = sorted(
        pathlib.Path(path_value).expanduser().resolve(strict=True)
        for path_value in paths
    )
    for path_value in normalized_paths:
        with path_value.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                fill_record = _parse_fill(line)
                if fill_record is not None:
                    fills.append(fill_record)
                payload = _parse_structured_payload(line)
                if payload is None:
                    continue
                if payload.get("event") == "strategy_initialized":
                    strategy_builds.append(payload.get("build"))
                if payload.get("event") != "portfolio_summary":
                    continue
                point = _parse_portfolio_point(payload)
                if (previous_portfolio_date is not None
                        and point.closing_date == previous_portfolio_date):
                    raise ValueError("duplicate portfolio date")
                if (previous_portfolio_date is not None
                        and point.closing_date < previous_portfolio_date):
                    raise ValueError(
                        "portfolio dates must be strictly increasing"
                    )
                portfolio_points.append(point)
                previous_portfolio_date = point.closing_date
    return ParsedLog(
        tuple(fills), tuple(portfolio_points), tuple(strategy_builds),
    )


def _close_trade(buy_fill, sell_fill):
    buy_cost = buy_fill.price * buy_fill.amount + buy_fill.commission
    sell_proceeds = (
        sell_fill.price * sell_fill.amount - sell_fill.commission
    )
    pnl = sell_proceeds - buy_cost
    return CompletedTrade(
        code=buy_fill.code,
        entry_date=buy_fill.trade_date,
        exit_date=sell_fill.trade_date,
        buy_price=buy_fill.price,
        buy_amount=buy_fill.amount,
        buy_commission=buy_fill.commission,
        sell_price=sell_fill.price,
        sell_amount=sell_fill.amount,
        sell_commission=sell_fill.commission,
        pnl=pnl,
        return_rate=pnl / buy_cost,
        amount_ratio=float(sell_fill.amount) / buy_fill.amount,
    )


def pair_completed_trades(fills):
    open_by_code = {}
    completed = []
    previous_timestamp = None
    for fill_record in fills:
        if (previous_timestamp is not None
                and fill_record.timestamp < previous_timestamp):
            raise ValueError("fill timestamps must be nondecreasing")
        previous_timestamp = fill_record.timestamp
        if fill_record.side == "BUY":
            if fill_record.code in open_by_code:
                raise ValueError(
                    "duplicate open for %s" % fill_record.code
                )
            open_by_code[fill_record.code] = fill_record
            continue
        if fill_record.side != "SELL":
            raise ValueError("unknown fill side")
        buy_fill = open_by_code.pop(fill_record.code, None)
        if buy_fill is None:
            raise ValueError(
                "sell without open for %s" % fill_record.code
            )
        completed.append(_close_trade(buy_fill, fill_record))
    return TradeLedger(
        completed_trades=tuple(completed),
        open_positions=tuple(
            open_by_code[code] for code in sorted(open_by_code)
        ),
    )


def wilson_lower_bound(wins, total, z=1.96):
    if total <= 0:
        return None
    proportion = float(wins) / total
    denominator = 1.0 + z * z / total
    center = (
        proportion + z * z / (2.0 * total)
    ) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return center - half


def max_drawdown(values):
    peak = None
    worst = 0.0
    for value in values:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("equity values must be finite and positive")
        peak = value if peak is None else max(peak, value)
        worst = max(worst, (peak - value) / peak)
    return worst


def summarize_trades(completed_trades):
    trades = tuple(completed_trades)
    wins = tuple(trade for trade in trades if trade.pnl > 0)
    losses = tuple(trade for trade in trades if trade.pnl < 0)
    breakeven_count = len(trades) - len(wins) - len(losses)
    gross_profit = sum(trade.pnl for trade in wins)
    gross_loss = sum(trade.pnl for trade in losses)
    top_trade_count = math.ceil(len(trades) * 0.10)
    positive_pnls = sorted(
        (trade.pnl for trade in wins), reverse=True,
    )
    top_profit = sum(positive_pnls[:top_trade_count])
    return {
        "closed_trade_count": len(trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "breakeven_count": breakeven_count,
        "win_rate": (
            float(len(wins)) / len(trades) if trades else None
        ),
        "wilson_lower_95": wilson_lower_bound(len(wins), len(trades)),
        "median_trade_return": (
            statistics.median(trade.return_rate for trade in trades)
            if trades else None
        ),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": (
            gross_profit / abs(gross_loss) if gross_loss < 0 else None
        ),
        "top_10pct_trade_count": top_trade_count,
        "top_10pct_gross_profit_share": (
            top_profit / gross_profit if gross_profit > 0 else None
        ),
    }


def _annual_returns(portfolio_points, initial_capital):
    year_end_values = {}
    for point in portfolio_points:
        year_end_values[point.closing_date.year] = point.total_value
    returns = {}
    prior_value = initial_capital
    for year in sorted(year_end_values):
        end_value = year_end_values[year]
        returns[str(year)] = end_value / prior_value - 1.0
        prior_value = end_value
    return returns


def summarize_performance_from_records(
        portfolio_points, completed_trades,
        initial_capital=INITIAL_CAPITAL):
    if (not isinstance(initial_capital, (int, float))
            or isinstance(initial_capital, bool)
            or not math.isfinite(float(initial_capital))
            or initial_capital <= 0):
        raise ValueError("initial capital must be finite and positive")
    points = tuple(portfolio_points)
    if not points:
        raise ValueError("portfolio summary records are required")
    previous_date = None
    for point in points:
        if (previous_date is not None
                and point.closing_date <= previous_date):
            raise ValueError("portfolio dates must be strictly increasing")
        if (not math.isfinite(point.total_value)
                or point.total_value <= 0):
            raise ValueError("portfolio total_value must be finite and positive")
        previous_date = point.closing_date
    report = {
        "initial_capital": float(initial_capital),
        "final_total_value": points[-1].total_value,
        "first_portfolio_date": points[0].closing_date.isoformat(),
        "last_portfolio_date": points[-1].closing_date.isoformat(),
        "portfolio_point_count": len(points),
        "total_return": points[-1].total_value / initial_capital - 1.0,
        "annual_returns": _annual_returns(points, initial_capital),
        "max_drawdown": max_drawdown(
            point.total_value for point in points
        ),
    }
    report.update(summarize_trades(completed_trades))
    return report


def summarize_performance(parsed_log, initial_capital=INITIAL_CAPITAL):
    ledger = pair_completed_trades(parsed_log.fills)
    report = summarize_performance_from_records(
        parsed_log.portfolio_points,
        ledger.completed_trades,
        initial_capital,
    )
    report["open_position_count"] = len(ledger.open_positions)
    report["amount_change_trade_count"] = sum(
        trade.buy_amount != trade.sell_amount
        for trade in ledger.completed_trades
    )
    return report


def _finite_metric(metrics, key):
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        return None
    return value


def evaluate_final_gates(candidate, double_friction):
    total_return = _finite_metric(candidate, "total_return")
    win_rate = _finite_metric(candidate, "win_rate")
    wilson = _finite_metric(candidate, "wilson_lower_95")
    drawdown = _finite_metric(candidate, "max_drawdown")
    trade_count = _finite_metric(candidate, "closed_trade_count")
    median_return = _finite_metric(candidate, "median_trade_return")
    top_share = _finite_metric(
        candidate, "top_10pct_gross_profit_share",
    )
    friction_return = _finite_metric(double_friction, "total_return")
    gates = {
        "total_return_above_cross": (
            total_return is not None
            and total_return > FINAL_TOTAL_RETURN_GATE
        ),
        "win_rate_above_cross": (
            win_rate is not None and win_rate > FINAL_WIN_RATE_GATE
        ),
        "wilson_lower_above_half": (
            wilson is not None and wilson > 0.50
        ),
        "max_drawdown_below_cross": (
            drawdown is not None
            and drawdown < FINAL_MAX_DRAWDOWN_GATE
        ),
        "closed_trades_at_least_80": (
            trade_count is not None
            and trade_count >= FINAL_MIN_CLOSED_TRADES
        ),
        "median_trade_return_positive": (
            median_return is not None and median_return > 0
        ),
        "top_profit_share_at_most_half": (
            top_share is not None
            and top_share <= FINAL_MAX_TOP_PROFIT_SHARE
        ),
        "double_friction_beats_benchmark": (
            friction_return is not None
            and friction_return > BENCHMARK_RETURN
        ),
    }
    gates["all_passed"] = all(gates.values())
    return gates


def _load_manifest_api():
    module_name = "_resonance_relative_turn_manifest_api"
    module = sys.modules.get(module_name)
    if module is None:
        module_path = (
            pathlib.Path(__file__).resolve().with_name(
                "analyze_relative_turn_observations.py"
            )
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError("unable to load session manifest validator")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return (
        module.read_session_calendar_manifest_bytes,
        module.validate_session_calendar_manifest,
    )


(
    read_session_calendar_manifest_bytes,
    validate_session_calendar_manifest,
) = _load_manifest_api()


def require_expected_build(parsed_log, expected_build, role):
    builds = parsed_log.strategy_builds
    if not builds:
        raise ValueError("missing %s strategy initialization" % role)
    if len(builds) != 1:
        raise ValueError("duplicate %s strategy initialization" % role)
    actual_build = builds[0]
    if (not isinstance(actual_build, str)
            or actual_build != expected_build):
        raise ValueError(
            "%s build mismatch: expected %r, got %r"
            % (role, expected_build, actual_build)
        )
    return actual_build


def validate_training_boundary(parsed_log, manifest, role):
    evaluation_sessions = tuple(
        session for session in manifest.sessions
        if EVALUATION_START <= session <= EVALUATION_END
    )
    if not evaluation_sessions:
        raise ValueError("manifest has no 2019-2021 evaluation sessions")
    portfolio_dates = tuple(
        point.closing_date for point in parsed_log.portfolio_points
    )
    for closing_date in portfolio_dates:
        if not EVALUATION_START <= closing_date <= EVALUATION_END:
            raise ValueError(
                "%s portfolio date outside 2019-2021: %s"
                % (role, closing_date)
            )
    if portfolio_dates != evaluation_sessions:
        raise ValueError(
            "%s portfolio sessions do not exactly match manifest"
            % role
        )
    evaluation_session_set = frozenset(evaluation_sessions)
    for fill_record in parsed_log.fills:
        if not EVALUATION_START <= fill_record.trade_date <= EVALUATION_END:
            raise ValueError(
                "%s fill date outside 2019-2021: %s"
                % (role, fill_record.trade_date)
            )
        if fill_record.trade_date not in evaluation_session_set:
            raise ValueError(
                "%s fill date absent from manifest: %s"
                % (role, fill_record.trade_date)
            )


def _path_identity(path_value, must_exist):
    path = pathlib.Path(path_value).expanduser().resolve(strict=must_exist)
    normalized = os.path.normcase(str(path))
    inode = None
    if path.exists():
        stat_result = path.stat()
        inode = (stat_result.st_dev, stat_result.st_ino)
    return path, normalized, inode


def validate_distinct_paths(args):
    labelled = []
    for role, path_values in (
            ("baseline log", args.baseline_log),
            ("candidate log", args.candidate_log),
            ("double friction log", args.double_friction_log)):
        for path_value in path_values:
            labelled.append((role, path_value, True))
    labelled.append(("session calendar", args.session_calendar, True))
    labelled.append(("output", args.output, False))
    seen_names = {}
    seen_inodes = {}
    resolved = {}
    for label, path_value, must_exist in labelled:
        path, normalized, inode = _path_identity(path_value, must_exist)
        if normalized in seen_names:
            raise ValueError(
                "%s and %s must be distinct physical files"
                % (seen_names[normalized], label)
            )
        if inode is not None and inode in seen_inodes:
            raise ValueError(
                "%s and %s must be distinct physical files"
                % (seen_inodes[inode], label)
            )
        seen_names[normalized] = label
        if inode is not None:
            seen_inodes[inode] = label
        resolved.setdefault(label, []).append(path)
    if not resolved["output"][0].parent.is_dir():
        raise ValueError("output parent directory must exist")


def _manifest_report(manifest):
    if isinstance(manifest, dict):
        return manifest
    metadata = manifest.metadata
    return {
        "schema_version": metadata.schema_version,
        "market": metadata.market,
        "calendar_coverage_start": metadata.calendar_coverage_start,
        "calendar_coverage_end": metadata.calendar_coverage_end,
        "evaluation_start": metadata.evaluation_start,
        "evaluation_end": metadata.evaluation_end,
        "source": metadata.source,
        "session_count": metadata.session_count,
        "first_session": manifest.sessions[0].isoformat(),
        "last_session": manifest.sessions[-1].isoformat(),
        "sha256": manifest.sha256,
    }


def analyze_paths(args, manifest):
    baseline_log = parse_joinquant_log(args.baseline_log)
    candidate_log = parse_joinquant_log(args.candidate_log)
    double_friction_log = parse_joinquant_log(args.double_friction_log)
    baseline_build = require_expected_build(
        baseline_log, args.expected_baseline_build, "baseline",
    )
    candidate_build = require_expected_build(
        candidate_log, args.expected_candidate_build, "candidate",
    )
    double_friction_build = require_expected_build(
        double_friction_log,
        args.expected_candidate_build,
        "double friction",
    )
    validate_training_boundary(baseline_log, manifest, "baseline")
    validate_training_boundary(candidate_log, manifest, "candidate")
    validate_training_boundary(
        double_friction_log, manifest, "double friction",
    )
    baseline = summarize_performance(baseline_log)
    candidate = summarize_performance(candidate_log)
    double_friction = summarize_performance(double_friction_log)
    return {
        "session_calendar": _manifest_report(manifest),
        "run_identity": {
            "baseline_build": baseline_build,
            "candidate_build": candidate_build,
            "double_friction_build": double_friction_build,
        },
        "baseline": baseline,
        "candidate": candidate,
        "double_friction": double_friction,
        "gates": evaluate_final_gates(candidate, double_friction),
    }


def write_json_atomically(path_value, payload):
    output = pathlib.Path(path_value).expanduser().resolve(strict=False)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % output.name,
        suffix=".tmp",
        dir=str(output.parent),
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload, stream, ensure_ascii=False,
                sort_keys=True, indent=2, allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(output))
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", action="append", required=True)
    parser.add_argument("--candidate-log", action="append", required=True)
    parser.add_argument(
        "--double-friction-log", action="append", required=True,
    )
    parser.add_argument(
        "--expected-baseline-build", default="20260827.4",
    )
    parser.add_argument("--expected-candidate-build", required=True)
    parser.add_argument("--session-calendar", required=True)
    parser.add_argument("--session-calendar-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = _argument_parser().parse_args(argv)
    validate_distinct_paths(args)
    manifest = validate_session_calendar_manifest(
        read_session_calendar_manifest_bytes(args.session_calendar),
        args.session_calendar_sha256,
    )
    report = analyze_paths(args, manifest)
    write_json_atomically(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
