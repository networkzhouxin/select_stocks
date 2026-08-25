import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
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
    forbidden = {
        "jqdata", "smart_trade_joinquant_cross_signal_etf",
        "smart_trade_ptrade_cross_signal_etf",
    }
    for path in OBSERVER_MODULE_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = imported_module_names(tree)
        called = called_function_names(tree)
        assert not (imported & forbidden)
        assert not any(name.startswith("order") or name == "execute_sell" for name in called)


def test_collect_and_summarize_do_not_modify_source(tmp_path):
    root = build_valid_source(tmp_path)
    before = hash_tree(root)
    state = tmp_path / "state"

    assert run_collect_and_summarize(root, state).returncode == 0
    assert hash_tree(root) == before
    assert (state / "summary.json").exists()
