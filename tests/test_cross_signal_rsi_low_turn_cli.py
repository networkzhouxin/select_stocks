import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cross_signal_strategy.research.rsi_low_turn_store import (
    STATE_MARKER,
    STATE_MARKER_FILE,
    SourceRewriteError,
)
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
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
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
    lookups = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "getattr" or len(node.args) < 2:
            continue
        method = node.args[1]
        if isinstance(method, ast.Constant) and isinstance(method.value, str):
            if method.value.startswith("order") or method.value == "execute_sell":
                lookups.append(node.lineno)
    return lookups


def forbidden_dynamic_imports(tree):
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.id if isinstance(node.func, ast.Name)
            else node.func.attr if isinstance(node.func, ast.Attribute) else None
        )
        if name not in {"__import__", "import_module"} or not node.args:
            continue
        value = node.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            names.append(value.value)
    return names


def is_forbidden_import(name):
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
    "getattr(broker, 'order_target_value')",
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

    def drift_after_last_input(data_root, approved_root, code, arrival_dt):
        item = original(data_root, approved_root, code, arrival_dt)
        if code == FROZEN_CODES[-1]:
            with (root / "daily" / "513100.csv").open("ab") as handle:
                handle.write(b"\n")
        return item

    monkeypatch.setattr(cli_module, "load_arrival_input", drift_after_last_input)
    with pytest.raises(SourceRewriteError, match="source changed during collection"):
        cli_module.collect(root, root, state, "2026-08-26T09:35:00+08:00")

    assert not state.exists()


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
    ), "stored label exit price"),
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
