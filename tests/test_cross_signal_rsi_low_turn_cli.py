import ast
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from cross_signal_strategy.research.rsi_low_turn_store import (
    ShadowStore,
    STATE_MARKER,
    STATE_MARKER_FILE,
    SourceRewriteError,
)
from cross_signal_strategy.research.rsi_low_turn_outcomes import (
    DOUBLED_FRICTION,
    NOMINAL_FRICTION,
    calculate_round_trip,
)
from cross_signal_strategy.research.rsi_low_turn_shadow import (
    RsiTurnInput,
    detect_rsi_low_turn,
)
from cross_signal_strategy.research.rsi_low_turn_source import SourceContractError
from cross_signal_strategy.tools import run_rsi_low_turn_shadow as cli_module


WORKTREE = Path(__file__).resolve().parents[1]
CLI_PATH = WORKTREE / "cross_signal_strategy" / "tools" / "run_rsi_low_turn_shadow.py"
OBSERVER_MODULE_PATHS = (
    WORKTREE / "cross_signal_strategy" / "research" / "rsi_low_turn_shadow.py",
    WORKTREE / "cross_signal_strategy" / "research" / "rsi_low_turn_source.py",
    WORKTREE / "cross_signal_strategy" / "research" / "rsi_low_turn_store.py",
    WORKTREE / "cross_signal_strategy" / "research" / "rsi_low_turn_outcomes.py",
    CLI_PATH,
)
FROZEN_CODES = (
    "159915", "512100", "159928", "513100", "513500", "513880", "513050",
    "518880", "159985",
)
VALID_MANIFEST = {
    "purpose": "rsi_low_turn_prospective_shadow",
    "version": "rsi-low-turn-shadow-v0.1",
    "collection_start": "2026-08-26",
    "timezone": "Asia/Shanghai",
    "append_only": True,
    "daily_subdir": "daily",
    "minute_subdir": "minute_0935",
}


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args], text=True, capture_output=True,
    )


def hash_tree(root):
    return tuple(sorted(
        (path.relative_to(root).as_posix(), _sha256(path))
        for path in root.rglob("*") if path.is_file()
    ))


def imported_module_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module + "." if node.module else "")
            names.update(f"{prefix}{alias.name}" for alias in node.names)
    return names


def called_function_names(tree):
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                names.append(node.func.id)
    return names


def imported_module_aliases(tree, module):
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(
                alias.asname or alias.name
                for alias in node.names if alias.name == module
            )
    return aliases


def local_broker_order_calls(tree):
    local_broker_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id == "LocalBroker":
                local_broker_names.update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr.startswith("order"):
                calls.append((node.func.attr, node.func.value, node.lineno))
    return local_broker_names, calls


def dynamic_order_lookups(tree):
    aliases = {"getattr"}
    builtins_aliases = imported_module_aliases(tree, "builtins")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "builtins":
            aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "getattr")
    lookups = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        direct = node.func.id if isinstance(node.func, ast.Name) else None
        member = (
            node.func.attr if isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in builtins_aliases else None
        )
        if (direct not in aliases and member != "getattr") or len(node.args) < 2:
            continue
        method = node.args[1]
        if isinstance(method, ast.Constant) and isinstance(method.value, str):
            if method.value.startswith("order") or method.value == "execute_sell":
                lookups.append(node.lineno)
    return lookups


def forbidden_dynamic_imports(tree):
    aliases = {"__import__"}
    importlib_aliases = {"importlib"}
    builtins_aliases = imported_module_aliases(tree, "builtins")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            importlib_aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "importlib")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "importlib":
                aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
            if node.module == "builtins":
                aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "__import__")
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        direct = node.func.id if isinstance(node.func, ast.Name) else None
        member = (
            node.func.attr if isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in importlib_aliases | builtins_aliases else None
        )
        if (
            direct not in aliases
            and not (member == "import_module" and node.func.value.id in importlib_aliases)
            and not (member == "__import__" and node.func.value.id in builtins_aliases)
        ) or not node.args:
            continue
        value = node.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            names.append(value.value)
    return names


def is_forbidden_import(name):
    name = name.lstrip(".")
    return (
        name == "jqdata" or name.startswith("jqdata.")
        or name == "smart_trade_joinquant_cross_signal_etf"
        or name.startswith("smart_trade_joinquant_cross_signal_etf.")
        or name == "smart_trade_ptrade_cross_signal_etf"
        or name.startswith("smart_trade_ptrade_cross_signal_etf.")
        or name == "cross_signal_strategy.smart_trade_joinquant_cross_signal_etf"
        or name.startswith("cross_signal_strategy.smart_trade_joinquant_cross_signal_etf.")
        or name == "cross_signal_strategy.smart_trade_ptrade_cross_signal_etf"
        or name.startswith("cross_signal_strategy.smart_trade_ptrade_cross_signal_etf.")
    )


def build_valid_source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(VALID_MANIFEST), encoding="utf-8")
    (root / "daily").mkdir()
    (root / "minute_0935").mkdir()
    dates = pd.bdate_range(end="2026-08-25", periods=30)
    for offset, code in enumerate(FROZEN_CODES):
        daily = pd.DataFrame({
            "code": code,
            "date": dates.date,
            "open": np.linspace(2.30 + offset, 2.01 + offset, 30),
            "high": np.linspace(2.32 + offset, 2.03 + offset, 30),
            "low": np.linspace(2.28 + offset, 1.99 + offset, 30),
            "close": np.linspace(2.30 + offset, 2.01 + offset, 30),
            "volume": 100000,
            "available_at": [f"{day}T15:01:00+08:00" for day in dates.date],
            "source": "pytest_fixture",
        })
        daily.to_csv(root / "daily" / f"{code}.csv", index=False)
        pd.DataFrame([{
            "code": code, "timestamp": "2026-08-26T09:35:00+08:00",
            "open": 2.035 + offset, "close": 2.035 + offset,
            "volume": 1000, "num_trades": 10,
            "available_at": "2026-08-26T09:35:00+08:00",
            "source": "pytest_fixture",
        }]).to_csv(root / "minute_0935" / f"{code}.csv", index=False)
    return root


def append_second_day(root):
    for offset, code in enumerate(FROZEN_CODES):
        daily_path = root / "daily" / f"{code}.csv"
        with daily_path.open("ab") as handle:
            handle.write(
                f"{code},2026-08-26,{2.00 + offset},{2.10 + offset},{1.95 + offset},"
                f"{2.05 + offset},100000,2026-08-26T15:01:00+08:00,pytest_fixture\n".encode("utf-8")
            )
        minute_path = root / "minute_0935" / f"{code}.csv"
        with minute_path.open("ab") as handle:
            handle.write(
                f"{code},2026-08-27T09:35:00+08:00,{2.04 + offset},{2.04 + offset},"
                "1000,10,2026-08-27T09:35:00+08:00,pytest_fixture\n".encode("utf-8")
            )


def append_second_day_with_nonfinite_close(root, bad_code):
    for offset, code in enumerate(FROZEN_CODES):
        close = "nan" if code == bad_code else str(2.05 + offset)
        daily_path = root / "daily" / f"{code}.csv"
        with daily_path.open("ab") as handle:
            handle.write(
                f"{code},2026-08-26,{2.00 + offset},{2.10 + offset},{1.95 + offset},"
                f"{close},100000,2026-08-26T15:01:00+08:00,pytest_fixture\n".encode("utf-8")
            )
        minute_path = root / "minute_0935" / f"{code}.csv"
        with minute_path.open("ab") as handle:
            handle.write(
                f"{code},2026-08-27T09:35:00+08:00,{2.04 + offset},{2.04 + offset},"
                "1000,10,2026-08-27T09:35:00+08:00,pytest_fixture\n".encode("utf-8")
            )


def append_third_day(root):
    for offset, code in enumerate(FROZEN_CODES):
        daily_path = root / "daily" / f"{code}.csv"
        with daily_path.open("ab") as handle:
            handle.write(
                f"{code},2026-08-27,{2.04 + offset},{2.10 + offset},{1.95 + offset},"
                f"{2.06 + offset},100000,2026-08-27T15:01:00+08:00,pytest_fixture\n".encode("utf-8")
            )
        minute_path = root / "minute_0935" / f"{code}.csv"
        with minute_path.open("ab") as handle:
            handle.write(
                f"{code},2026-08-28T09:35:00+08:00,{2.08 + offset},{2.08 + offset},"
                "1000,10,2026-08-28T09:35:00+08:00,pytest_fixture\n".encode("utf-8")
            )


def make_first_code_signal(root):
    path = root / "daily" / f"{FROZEN_CODES[0]}.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.index[-4:], "close"] = [2.00, 2.02, 1.98, 2.00]
    frame.to_csv(path, index=False)


def run_collect_and_summarize(root, state):
    collect = run_cli(
        "collect", "--data-root", str(root), "--approved-root", str(root),
        "--state-dir", str(state), "--as-of", "2026-08-26T09:35:00+08:00",
    )
    if collect.returncode:
        return collect
    return run_cli("summarize", "--state-dir", str(state),
                   "--generated-at", "2026-08-26T09:35:00+08:00")


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _integrity_decision(code, arrival, signal=True):
    source_hashes = tuple(
        hashlib.sha256((relative_path + "\n").encode()).hexdigest()
        for relative_path in (
            "manifest.json", f"daily/{code}.csv", f"minute_0935/{code}.csv",
        )
    )
    return detect_rsi_low_turn(RsiTurnInput(
        code=code,
        arrival_dt=arrival,
        signal_date=arrival.date() - timedelta(days=1),
        r2=24.0 if signal else 18.0,
        r1=18.0,
        r0=21.0,
        c1=2.00,
        c0=2.01,
        entry_open=2.035,
        price_proved=True,
        background={"rsi12": 25.0},
        source_hashes=source_hashes,
    ))


def _seed_complete_integrity_state(tmp_path, event_count=1, forge_passing_labels=False):
    state = tmp_path / "state"
    store = ShadowStore(state)
    observed = datetime(2026, 8, 26, 9, 35, tzinfo=ZoneInfo("Asia/Shanghai"))
    for relative_path in cli_module._required_snapshot_paths():
        store.record_source_snapshot(relative_path, (relative_path + "\n").encode(), observed)

    event_ids = []
    prior_by_code = {}
    for index in range(event_count):
        code = FROZEN_CODES[index % 6]
        arrival = observed + timedelta(days=index * 4)
        if code in prior_by_code:
            reset = arrival - timedelta(days=1)
            store.record_evaluation(_integrity_decision(code, reset, False), reset)
        decision = _integrity_decision(code, arrival, True)
        store.record_evaluation(decision, arrival)
        event_ids.append(decision.event_id)
        prior_by_code[code] = arrival

    for code in FROZEN_CODES:
        if code in prior_by_code:
            continue
        arrival = observed + timedelta(minutes=FROZEN_CODES.index(code) + 1)
        store.record_evaluation(_integrity_decision(code, arrival, False), arrival)

    if forge_passing_labels:
        for event_id in event_ids:
            for horizon in (5, 10):
                nominal = asdict(calculate_round_trip("513100", 2.035, 2.10, NOMINAL_FRICTION))
                doubled = asdict(calculate_round_trip("513100", 2.035, 2.10, DOUBLED_FRICTION))
                nominal.update(net_pnl=63.33333333333333, net_return=0.01)
                doubled.update(net_pnl=63.33333333333333, net_return=0.01)
                store.append_label(event_id, horizon, {
                    "event_id": event_id,
                    "horizon": horizon,
                    "status": "matured",
                    "exit_price": 2.10,
                    "nominal": nominal,
                    "doubled": doubled,
                    "mfe": None,
                    "mae": None,
                })
    store.write_state_marker()
    return state


def _rewrite_first_record(path, mutate):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(records[0])
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


def _assert_summarize_fails_without_replacing_summary(state, message):
    summary_path = state / "summary.json"
    summary_path.write_bytes(b'{"preserve":true}\n')
    before = summary_path.read_bytes()
    with pytest.raises(SourceRewriteError, match=message):
        cli_module.summarize(state, "2027-05-01T09:35:00+08:00")
    assert summary_path.read_bytes() == before


def _append_provenance_label(state, mutate=None, include_provenance=True):
    store = ShadowStore(state, create=False)
    event = store.load_events()[0]
    exit_price = 2.10
    payload = {
        "event_id": event["event_id"],
        "horizon": 5,
        "status": "matured",
        "exit_price": exit_price,
        "nominal": asdict(calculate_round_trip(
            event["code"], event["entry_open"], exit_price, NOMINAL_FRICTION,
        )),
        "doubled": asdict(calculate_round_trip(
            event["code"], event["entry_open"], exit_price, DOUBLED_FRICTION,
        )),
        "mfe": None,
        "mae": None,
    }
    if include_provenance:
        relative_path = f"minute_0935/{event['code']}.csv"
        snapshot = next(
            item for item in store.load_source_snapshots()
            if item["relative_path"] == relative_path
        )
        target = datetime.fromisoformat(event["arrival_dt"]) + timedelta(days=1)
        payload.update({
            "target_timestamp": target.isoformat(),
            "available_at": target.isoformat(),
            "collected_at": target.isoformat(),
            "source_relative_path": relative_path,
            "source_sha256": snapshot["sha256"],
            "source_byte_length": snapshot["byte_length"],
        })
    if mutate is not None:
        mutate(payload)
    store.append_label(event["event_id"], 5, payload)


def test_collect_refuses_nonmatching_approved_root(tmp_path):
    result = run_cli(
        "collect", "--data-root", str(tmp_path / "source"),
        "--approved-root", str(tmp_path / "other"),
        "--state-dir", str(tmp_path / "state"),
        "--as-of", "2026-08-26T09:35:00+08:00",
    )

    assert result.returncode == 2
    assert "approved root" in result.stdout


def test_observer_modules_have_no_platform_or_order_dependency():
    for path in OBSERVER_MODULE_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = imported_module_names(tree)
        assert not any(is_forbidden_import(name) for name in imported)
        assert not any(is_forbidden_import(name) for name in forbidden_dynamic_imports(tree))
        assert not any(name == "execute_sell" for name in called_function_names(tree))
        assert not dynamic_order_lookups(tree)
        local_broker_names, order_calls = local_broker_order_calls(tree)
        for method, receiver, _ in order_calls:
            assert path.name == "rsi_low_turn_outcomes.py"
            assert method == "order_target_value"
            assert isinstance(receiver, ast.Name)
            assert receiver.id in local_broker_names

    outcomes = ast.parse(OBSERVER_MODULE_PATHS[3].read_text(encoding="utf-8"))
    _, order_calls = local_broker_order_calls(outcomes)
    assert len(order_calls) == 2


@pytest.mark.parametrize("source", [
    "import jqdata.submodule",
    "from jqdata.submodule import price",
    "import cross_signal_strategy.smart_trade_joinquant_cross_signal_etf",
    "from smart_trade_ptrade_cross_signal_etf import initialize",
    "__import__('jqdata.market')",
    "importlib.import_module('jqdata.market')",
    "from importlib import import_module as load\nload('jqdata.market')",
    "import importlib as il\nil.import_module('jqdata.market')",
    "from builtins import __import__ as load\nload('jqdata.market')",
    "import builtins as bi\nbi.__import__('jqdata')",
    "from .. import smart_trade_joinquant_cross_signal_etf",
    "getattr(broker, 'order_target_value')",
    "from builtins import getattr as load\nload(broker, 'order_target_value')",
    "import builtins as bi\nbi.getattr(broker, 'order_target_value')(broker)",
])
def test_ast_guard_rejects_import_and_dynamic_order_bypasses(source):
    tree = ast.parse(source)
    assert (
        any(is_forbidden_import(name) for name in imported_module_names(tree))
        or any(is_forbidden_import(name) for name in forbidden_dynamic_imports(tree))
        or bool(dynamic_order_lookups(tree))
    )


def test_collect_and_summarize_do_not_modify_source(tmp_path):
    root = build_valid_source(tmp_path)
    before = hash_tree(root)
    state = tmp_path / "state"

    assert run_collect_and_summarize(root, state).returncode == 0
    assert hash_tree(root) == before
    assert (state / "summary.json").exists()


def test_collect_accepts_second_day_prefix_append_only(tmp_path):
    root = build_valid_source(tmp_path)
    state = tmp_path / "state"
    first = run_cli(
        "collect", "--data-root", str(root), "--approved-root", str(root),
        "--state-dir", str(state), "--as-of", "2026-08-26T09:35:00+08:00",
    )
    append_second_day(root)
    second = run_cli(
        "collect", "--data-root", str(root), "--approved-root", str(root),
        "--state-dir", str(state), "--as-of", "2026-08-27T09:35:00+08:00",
    )

    assert first.returncode == 0
    assert second.returncode == 0
    snapshots = [json.loads(line) for line in (state / "source_hashes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(snapshots) == 37
    assert all("byte_length" in item for item in snapshots)


def test_collect_persists_matured_label_from_same_stable_snapshot(tmp_path):
    root = build_valid_source(tmp_path)
    make_first_code_signal(root)
    state = tmp_path / "state"
    cli_module.collect(root, root, state, "2026-08-26T09:35:00+08:00")
    append_second_day(root)
    cli_module.collect(root, root, state, "2026-08-27T09:35:00+08:00")
    append_third_day(root)

    cli_module.collect(root, root, state, "2026-08-28T09:35:00+08:00")

    labels = [json.loads(line) for line in (state / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    horizon_one = next(record["payload"] for record in labels if record["horizon"] == 1)
    minute_path = root / "minute_0935" / f"{FROZEN_CODES[0]}.csv"
    assert horizon_one["available_at"] == "2026-08-27T09:35:00+08:00"
    assert horizon_one["collected_at"] == "2026-08-28T09:35:00+08:00"
    assert horizon_one["source_relative_path"] == f"minute_0935/{FROZEN_CODES[0]}.csv"
    assert horizon_one["source_sha256"] == _sha256(minute_path)
    assert horizon_one["source_byte_length"] == len(minute_path.read_bytes())
    assert horizon_one["mfe"] is None and horizon_one["mae"] is None
    cli_module.summarize(state, "2026-08-28T09:35:00+08:00")


def test_collect_rejects_corrupt_existing_label_provenance_before_writes(tmp_path):
    root = build_valid_source(tmp_path)
    make_first_code_signal(root)
    state = tmp_path / "state"
    cli_module.collect(root, root, state, "2026-08-26T09:35:00+08:00")
    append_second_day(root)
    cli_module.collect(root, root, state, "2026-08-27T09:35:00+08:00")
    append_third_day(root)
    cli_module.collect(root, root, state, "2026-08-28T09:35:00+08:00")
    _rewrite_first_record(
        state / "labels.jsonl",
        lambda record: record["payload"].update(source_sha256="f" * 64),
    )
    before = hash_tree(state)

    with pytest.raises(SourceRewriteError, match="stored label provenance"):
        cli_module.collect(root, root, state, "2026-08-28T09:35:00+08:00")

    assert hash_tree(state) == before


def test_collect_rejects_rewritten_old_prefix_before_any_state_write(tmp_path):
    root = build_valid_source(tmp_path)
    state = tmp_path / "state"
    assert run_cli(
        "collect", "--data-root", str(root), "--approved-root", str(root),
        "--state-dir", str(state), "--as-of", "2026-08-26T09:35:00+08:00",
    ).returncode == 0
    before_state = hash_tree(state)
    path = root / "daily" / "513100.csv"
    path.write_bytes(path.read_bytes().replace(b"5.3", b"5.4", 1))
    append_second_day(root)

    result = run_cli(
        "collect", "--data-root", str(root), "--approved-root", str(root),
        "--state-dir", str(state), "--as-of", "2026-08-27T09:35:00+08:00",
    )

    assert result.returncode == 2
    assert "source hash changed" in result.stdout
    assert hash_tree(state) == before_state


def test_collect_rejects_concurrent_source_drift_before_creating_state(tmp_path, monkeypatch):
    root = build_valid_source(tmp_path)
    state = tmp_path / "state"
    original = cli_module.load_arrival_input

    def drift_after_last_input(data_root, approved_root, code, arrival_dt, **kwargs):
        item = original(data_root, approved_root, code, arrival_dt, **kwargs)
        if code == FROZEN_CODES[-1]:
            with (root / "daily" / "513100.csv").open("ab") as handle:
                handle.write(b"\n")
        return item

    monkeypatch.setattr(cli_module, "load_arrival_input", drift_after_last_input)
    with pytest.raises(SourceRewriteError, match="source changed during collection"):
        cli_module.collect(root, root, state, "2026-08-26T09:35:00+08:00")

    assert not state.exists()


def test_collect_requires_one_consistent_prior_source_session_before_creating_state(tmp_path):
    root = build_valid_source(tmp_path)
    path = root / "daily" / f"{FROZEN_CODES[-1]}.csv"
    frame = pd.read_csv(path)
    frame.iloc[:-1].to_csv(path, index=False)
    state = tmp_path / "state"

    with pytest.raises(SourceContractError, match="consistent prior source session"):
        cli_module.collect(root, root, state, "2026-08-26T09:35:00+08:00")

    assert not state.exists()


def test_collect_insufficient_indicator_history_creates_no_state(tmp_path):
    root = build_valid_source(tmp_path)
    for code in FROZEN_CODES:
        path = root / "daily" / f"{code}.csv"
        pd.read_csv(path).iloc[-10:].to_csv(path, index=False)
    state = tmp_path / "state"

    with pytest.raises(SourceContractError, match="insufficient indicator history"):
        cli_module.collect(root, root, state, "2026-08-26T09:35:00+08:00")

    assert not state.exists()


def test_collect_nonfinite_new_session_preserves_existing_state_bytes(tmp_path):
    root = build_valid_source(tmp_path)
    state = tmp_path / "state"
    cli_module.collect(root, root, state, "2026-08-26T09:35:00+08:00")
    before = hash_tree(state)
    append_second_day_with_nonfinite_close(root, FROZEN_CODES[-1])

    with pytest.raises(SourceContractError, match="insufficient indicator history"):
        cli_module.collect(root, root, state, "2026-08-27T09:35:00+08:00")

    assert hash_tree(state) == before


def test_summarize_rejects_source_and_uninitialized_dirs_without_modifying_summary(tmp_path):
    root = build_valid_source(tmp_path)
    root_before = hash_tree(root)
    source_result = run_cli(
        "summarize", "--state-dir", str(root),
        "--generated-at", "2026-08-26T09:35:00+08:00",
    )
    uninitialized = tmp_path / "uninitialized"
    uninitialized.mkdir()
    summary_path = uninitialized / "summary.json"
    summary_path.write_text('{"preserve":true}\n', encoding="utf-8")
    before = hash_tree(uninitialized)
    uninitialized_result = run_cli(
        "summarize", "--state-dir", str(uninitialized),
        "--generated-at", "2026-08-26T09:35:00+08:00",
    )

    assert source_result.returncode == 2
    assert uninitialized_result.returncode == 2
    assert hash_tree(root) == root_before
    assert hash_tree(uninitialized) == before


def test_summarize_requires_marker_and_complete_initial_collection(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / STATE_MARKER_FILE).write_text(json.dumps(STATE_MARKER), encoding="utf-8")
    before = hash_tree(state)

    result = run_cli(
        "summarize", "--state-dir", str(state),
        "--generated-at", "2026-08-26T09:35:00+08:00",
    )

    assert result.returncode == 2
    assert hash_tree(state) == before


def test_collect_resumes_markerless_interrupted_initial_collection(tmp_path, monkeypatch):
    root = build_valid_source(tmp_path)
    state = tmp_path / "state"
    original = cli_module.ShadowStore.record_evaluation
    calls = 0

    def interrupt_once(self, decision, observed_at):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected interruption")
        return original(self, decision, observed_at)

    monkeypatch.setattr(cli_module.ShadowStore, "record_evaluation", interrupt_once)
    with pytest.raises(RuntimeError, match="injected interruption"):
        cli_module.collect(root, root, state, "2026-08-26T09:35:00+08:00")
    assert not (state / STATE_MARKER_FILE).exists()

    monkeypatch.setattr(cli_module.ShadowStore, "record_evaluation", original)
    cli_module.collect(root, root, state, "2026-08-26T09:35:00+08:00")

    assert (state / STATE_MARKER_FILE).is_file()


def test_corrupt_evaluation_refuses_collect_before_any_state_write(tmp_path):
    root = build_valid_source(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "evaluations.jsonl").write_text('{"bad":true}\n', encoding="utf-8")
    before = hash_tree(state)

    with pytest.raises(SourceRewriteError, match="stored evaluation"):
        cli_module.collect(root, root, state, "2026-08-26T09:35:00+08:00")

    assert hash_tree(state) == before


def test_nested_evaluation_provenance_refuses_collect_before_any_state_write(tmp_path):
    root = build_valid_source(tmp_path)
    state = tmp_path / "state"
    assert run_cli(
        "collect", "--data-root", str(root), "--approved-root", str(root),
        "--state-dir", str(state), "--as-of", "2026-08-26T09:35:00+08:00",
    ).returncode == 0
    path = state / "evaluations.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["item"]["source_hashes"] = ["a" * 64, "b" * 64]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    before = hash_tree(state)

    with pytest.raises(SourceRewriteError, match="stored evaluation source hashes"):
        cli_module.collect(root, root, state, "2026-08-26T09:35:00+08:00")

    assert hash_tree(state) == before


@pytest.mark.parametrize(("events", "labels", "message"), [
    (({"event_id": "e1", "code": "513100", "arrival_date": "2026-08-26"},) * 2, (), "duplicate event"),
    (({"event_id": "e1", "code": "513100", "arrival_date": "2026-08-26"},), (
        {"event_id": "e1", "horizon": 1, "payload": {"event_id": "e1", "horizon": 1, "status": "pending_horizon_not_arrived", "exit_price": None, "nominal": None, "doubled": None, "mfe": None, "mae": None}},
        {"event_id": "e1", "horizon": 1, "payload": {"event_id": "e1", "horizon": 1, "status": "pending_horizon_not_arrived", "exit_price": None, "nominal": None, "doubled": None, "mfe": None, "mae": None}},
    ), "duplicate label"),
    (({"event_id": "e1", "code": "513100", "arrival_date": "2026-08-26"},), (
        {"event_id": "missing", "horizon": 1, "payload": {"event_id": "missing", "horizon": 1, "status": "pending_horizon_not_arrived", "exit_price": None, "nominal": None, "doubled": None, "mfe": None, "mae": None}},
    ), "dangling label"),
    (({"event_id": "e1", "code": "513100", "arrival_date": "2026-08-26"},), (
        {"event_id": "e1", "horizon": 2, "payload": {"event_id": "e1", "horizon": 2, "status": "pending_horizon_not_arrived", "exit_price": None, "nominal": None, "doubled": None, "mfe": None, "mae": None}},
    ), "unsupported horizon"),
    (({"event_id": "e1", "code": "513100", "arrival_date": "2026-08-26"},), (
        {"event_id": "e1", "horizon": 1, "payload": {"event_id": "other", "horizon": 1, "status": "pending_horizon_not_arrived", "exit_price": None, "nominal": None, "doubled": None, "mfe": None, "mae": None}},
    ), "stored label identity"),
    (({"event_id": "e1", "code": "513100", "arrival_date": "2026-08-26"},), (
        {"event_id": "e1", "horizon": 1, "payload": {"event_id": "e1", "horizon": 1, "status": "pending_horizon_not_arrived", "exit_price": None, "nominal": None, "doubled": None, "mfe": None, "mae": None}},
    ), "stored label status"),
    (({"event_id": "e1", "code": "513100", "arrival_date": "2026-08-26"},), (
        {"event_id": "e1", "horizon": 1, "payload": {"event_id": "e1", "horizon": 1, "status": "matured", "exit_price": None, "nominal": None, "doubled": None, "mfe": None, "mae": None}},
    ), "stored label numeric type"),
    (({"event_id": "e1", "code": "513100", "arrival_date": "2026-08-26"},), (
        {"event_id": "e1", "horizon": 1, "payload": {"event_id": "e1", "horizon": 1, "status": "unknown", "exit_price": None, "nominal": None, "doubled": None, "mfe": None, "mae": None}},
    ), "stored label status"),
    (({"event_id": "e1", "code": "513100", "arrival_date": "2026-08-26"},), (
        {"event_id": "e1", "horizon": 1, "payload": {"event_id": "e1", "horizon": 1, "status": "matured", "exit_price": float("nan"), "nominal": None, "doubled": None, "mfe": None, "mae": None}},
    ), "stored label exit price"),
    (({"event_id": "e1", "code": "513100", "arrival_date": "2026-08-26"},), (
        {"event_id": "e1", "horizon": 1, "payload": {"event_id": "e1", "horizon": 1, "status": "matured", "exit_price": 2.0, "nominal": None, "doubled": None, "mfe": None, "mae": None}},
    ), "stored label round trip"),
])
def test_evidence_reconstruction_refuses_ambiguous_state(events, labels, message):
    with pytest.raises(SourceRewriteError, match=message):
        cli_module._event_outcome_records(events, labels)


def test_forged_events_and_one_percent_labels_cannot_manufacture_a_pass(tmp_path):
    state = _seed_complete_integrity_state(tmp_path, event_count=60, forge_passing_labels=True)

    _assert_summarize_fails_without_replacing_summary(state, "stored label round trip")


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda event: event.update(code="000001"), "stored events do not match evaluation replay"),
    (lambda event: event.update(entry_open=9.99), "stored events do not match evaluation replay"),
    (lambda event: event["decision"].update(reasons=["forged"]), "stored events do not match evaluation replay"),
    (lambda event: event.update(source_hashes=["f" * 64] * 3), "stored events do not match evaluation replay"),
])
def test_summarize_rejects_tampered_event_fields_before_replacing_summary(tmp_path, mutate, message):
    state = _seed_complete_integrity_state(tmp_path)
    _rewrite_first_record(state / "events.jsonl", mutate)

    _assert_summarize_fails_without_replacing_summary(state, message)


@pytest.mark.parametrize("mutation", ["signal_boolean", "reason"])
def test_summarize_recomputes_stored_decisions_before_replacing_summary(tmp_path, mutation):
    state = _seed_complete_integrity_state(tmp_path)
    path = state / "evaluations.jsonl"

    def mutate(record):
        if mutation == "signal_boolean":
            record["signal_detected"] = False
            record["valid_event"] = False
        else:
            record["reasons"] = ["forged"]

    _rewrite_first_record(path, mutate)
    _assert_summarize_fails_without_replacing_summary(state, "stored evaluation decision")


def test_summarize_recomputes_label_round_trip_before_replacing_summary(tmp_path):
    state = _seed_complete_integrity_state(tmp_path, forge_passing_labels=True)

    _assert_summarize_fails_without_replacing_summary(state, "stored label round trip")


def test_summarize_requires_matured_label_source_provenance(tmp_path):
    state = _seed_complete_integrity_state(tmp_path)
    _append_provenance_label(state, include_provenance=False)

    _assert_summarize_fails_without_replacing_summary(state, "stored label provenance")


@pytest.mark.parametrize("mutate", [
    lambda payload: payload.update(source_relative_path="minute_0935/000001.csv"),
    lambda payload: payload.update(source_sha256="f" * 64),
    lambda payload: payload.update(source_byte_length=999999),
    lambda payload: payload.update(available_at="2026-08-27T09:36:00+08:00"),
    lambda payload: payload.update(collected_at="2026-08-25T09:35:00+08:00"),
])
def test_summarize_cross_checks_matured_label_provenance_history(tmp_path, mutate):
    state = _seed_complete_integrity_state(tmp_path)
    _append_provenance_label(state, mutate=mutate)

    _assert_summarize_fails_without_replacing_summary(state, "stored label provenance")


def test_summarize_accepts_valid_replayed_event_and_label_provenance(tmp_path):
    state = _seed_complete_integrity_state(tmp_path)
    _append_provenance_label(state)

    cli_module.summarize(state, "2027-05-01T09:35:00+08:00")

    summary = json.loads((state / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["matured_five_day_events"] == 1


def test_summarize_rejects_evaluation_and_event_hashes_tampered_together(tmp_path):
    state = _seed_complete_integrity_state(tmp_path)
    forged = ["d" * 64, "e" * 64, "f" * 64]
    _rewrite_first_record(
        state / "evaluations.jsonl",
        lambda record: record["item"].update(source_hashes=forged),
    )

    def mutate_event(record):
        record["source_hashes"] = forged
        record["input"]["source_hashes"] = forged

    _rewrite_first_record(state / "events.jsonl", mutate_event)

    _assert_summarize_fails_without_replacing_summary(
        state, "stored evaluation source hashes do not match source history",
    )


@pytest.mark.parametrize("mutate", [
    lambda payload: payload.update(exit_price="2.10"),
    lambda payload: payload["nominal"].update(net_return=str(payload["nominal"]["net_return"])),
])
def test_summarize_rejects_non_numeric_matured_label_fields(tmp_path, mutate):
    state = _seed_complete_integrity_state(tmp_path)
    _append_provenance_label(state, mutate=mutate)

    _assert_summarize_fails_without_replacing_summary(state, "stored label numeric type")


def test_summarize_rejects_stored_non_none_mfe_end_to_end(tmp_path):
    state = _seed_complete_integrity_state(tmp_path)
    _append_provenance_label(state, mutate=lambda payload: payload.update(mfe=0.1))

    _assert_summarize_fails_without_replacing_summary(state, "MFE/MAE")


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_summarize_rejects_missing_or_extra_event_records(tmp_path, mutation):
    state = _seed_complete_integrity_state(tmp_path)
    path = state / "events.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if mutation == "missing":
        records = []
    else:
        extra = dict(records[0])
        extra["event_id"] = "f" * 64
        records.append(extra)
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )

    _assert_summarize_fails_without_replacing_summary(
        state, "stored events do not match evaluation replay",
    )
