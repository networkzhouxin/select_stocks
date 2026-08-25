"""Order-free CLI for the blocked RSI low-turn prospective observer."""

import argparse
from dataclasses import asdict
from datetime import date, datetime
import json
from pathlib import Path
import sys
import tempfile
from typing import Mapping
from zoneinfo import ZoneInfo


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from cross_signal_strategy.research.rsi_low_turn_outcomes import (
    EventOutcomeRecord,
    MaturedLabel,
    RoundTripResult,
    build_summary,
    mature_event_labels,
)
from cross_signal_strategy.research.rsi_low_turn_shadow import detect_rsi_low_turn
from cross_signal_strategy.research.rsi_low_turn_source import (
    ApprovedFuturePriceSource,
    MIN_COLLECTION_START,
    SHANGHAI,
    SourceContractError,
    file_sha256,
    load_arrival_input,
    load_manifest,
)
from cross_signal_strategy.research.rsi_low_turn_store import (
    ShadowStore,
    SourceRewriteError,
)


FROZEN_ETF_CODES = (
    "159915", "512100", "159928", "513100", "513500", "513880", "513050",
    "518880", "159985",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--data-root", type=Path, required=True)
    collect.add_argument("--approved-root", type=Path, required=True)
    collect.add_argument("--state-dir", type=Path, required=True)
    collect.add_argument("--as-of", required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--state-dir", type=Path, required=True)
    summarize.add_argument("--generated-at", required=True)
    return parser


def collect(data_root: Path, approved_root: Path, state_dir: Path, as_of: str) -> None:
    observed_at = _parse_shanghai_datetime(as_of, "as_of")
    manifest = load_manifest(data_root, approved_root)
    state_path = Path(state_dir).resolve()
    if state_path == manifest.root or state_path.is_relative_to(manifest.root):
        raise SourceContractError("state dir must be outside the approved data root")

    inputs = tuple(
        load_arrival_input(manifest.root, manifest.root, code, observed_at)
        for code in FROZEN_ETF_CODES
    )
    store = ShadowStore(state_path)
    existing_events = store.load_events()
    labels = _matured_existing_labels(existing_events, manifest.root, observed_at)

    _record_source_hashes(store, manifest.root, manifest.daily_subdir,
                          manifest.minute_subdir, observed_at)
    evaluation_results = [
        store.record_evaluation(detect_rsi_low_turn(item), observed_at)
        for item in inputs
    ]
    label_results = [
        store.append_label(event_id, horizon, payload)
        for event_id, horizon, payload in labels
    ]
    print(
        "收集完成："
        f"evaluations={sum(result.written for result in evaluation_results)} "
        f"matured_labels={sum(result.written for result in label_results)} "
        "orders_disabled=True"
    )


def summarize(state_dir: Path, generated_at: str) -> None:
    timestamp = _parse_aware_datetime(generated_at, "generated_at")
    state_path = Path(state_dir).resolve()
    if not state_path.is_dir():
        raise SourceRewriteError("state directory is required")
    store = ShadowStore(state_path)
    summary = build_summary(
        _event_outcome_records(store.load_events(), store.load_labels()),
        MIN_COLLECTION_START,
        timestamp,
    )
    _atomic_write_summary(state_path, summary)
    print(
        "汇总完成："
        f"matured_five_day_events={summary['counts']['matured_five_day_events']} "
        f"status={summary['status']} orders_disabled=True"
    )


def _record_source_hashes(
    store: ShadowStore, root: Path, daily_subdir: str, minute_subdir: str,
    observed_at: datetime,
) -> None:
    paths = [root / "manifest.json"]
    paths.extend(root / daily_subdir / f"{code}.csv" for code in FROZEN_ETF_CODES)
    paths.extend(root / minute_subdir / f"{code}.csv" for code in FROZEN_ETF_CODES)
    for path in paths:
        store.record_source_hash(path.relative_to(root).as_posix(), file_sha256(path), observed_at)


def _matured_existing_labels(
    events: tuple[Mapping[str, object], ...], root: Path, as_of: datetime,
) -> tuple[tuple[str, int, Mapping[str, object]], ...]:
    source = ApprovedFuturePriceSource(root, root)
    labels = []
    for event in events:
        for label in mature_event_labels(event, source, as_of):
            if label.status == "matured":
                labels.append((label.event_id, label.horizon, asdict(label)))
    return tuple(labels)


def _event_outcome_records(
    events: tuple[Mapping[str, object], ...], labels: tuple[Mapping[str, object], ...],
) -> tuple[EventOutcomeRecord, ...]:
    labels_by_event: dict[str, dict[int, MaturedLabel]] = {}
    for record in labels:
        event_id = record.get("event_id")
        horizon = record.get("horizon")
        payload = record.get("payload")
        if not isinstance(event_id, str) or not isinstance(horizon, int) or not isinstance(payload, Mapping):
            raise SourceRewriteError("stored label is invalid")
        labels_by_event.setdefault(event_id, {})[horizon] = _matured_label(event_id, horizon, payload)

    records = []
    for event in events:
        event_id = event.get("event_id")
        code = event.get("code")
        arrival_date = event.get("arrival_date")
        if not isinstance(event_id, str) or not isinstance(code, str) or not isinstance(arrival_date, str):
            raise SourceRewriteError("stored event is invalid")
        try:
            arrival = date.fromisoformat(arrival_date)
        except ValueError as exc:
            raise SourceRewriteError("stored event arrival date is invalid") from exc
        records.append(EventOutcomeRecord(event_id, code, arrival, labels_by_event.get(event_id, {})))
    return tuple(records)


def _matured_label(event_id: str, horizon: int, payload: Mapping[str, object]) -> MaturedLabel:
    if payload.get("event_id") != event_id or payload.get("horizon") != horizon:
        raise SourceRewriteError("stored label identity is invalid")
    return MaturedLabel(
        event_id=event_id,
        horizon=horizon,
        status=_required_str(payload, "status"),
        exit_price=_optional_number(payload.get("exit_price")),
        nominal=_round_trip(payload.get("nominal")),
        doubled=_round_trip(payload.get("doubled")),
        mfe=_optional_number(payload.get("mfe")),
        mae=_optional_number(payload.get("mae")),
    )


def _round_trip(value: object) -> RoundTripResult | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SourceRewriteError("stored round trip is invalid")
    try:
        return RoundTripResult(
            int(value["amount"]), float(value["buy_exec_price"]),
            float(value["sell_exec_price"]), float(value["buy_commission"]),
            float(value["sell_commission"]), float(value["net_pnl"]),
            float(value["net_return"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceRewriteError("stored round trip is invalid") from exc


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise SourceRewriteError(f"stored label {key} is invalid")
    return value


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SourceRewriteError("stored label number is invalid") from exc


def _atomic_write_summary(state_dir: Path, summary: Mapping[str, object]) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=state_dir,
            prefix=".summary-", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(summary, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary_path.replace(state_dir / "summary.json")
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _parse_shanghai_datetime(value: str, label: str) -> datetime:
    parsed = _parse_aware_datetime(value, label)
    return parsed.astimezone(SHANGHAI)


def _parse_aware_datetime(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise SourceContractError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SourceContractError(f"{label} must be timezone-aware")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "collect":
            collect(args.data_root, args.approved_root, args.state_dir, args.as_of)
        else:
            summarize(args.state_dir, args.generated_at)
    except (SourceContractError, SourceRewriteError) as exc:
        print(f"错误：{exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
