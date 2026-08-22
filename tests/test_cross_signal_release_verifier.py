from pathlib import Path
import importlib.util
import types


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "cross_signal_strategy" / "tools" / "verify_release.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        "cross_signal_release_verifier", TOOL_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_verifier_passes_static_formal_checks():
    verifier = load_verifier()

    report = verifier.verify_release(ROOT, run_tests=False)

    assert report["status"] == "通过"
    assert report["strategy_version"] == "cross-v0.3.3"
    assert report["deployment_build"] == "20260822.2"
    assert report["business_fingerprint"] == "77e44d93d255"
    assert all(item["status"] == "通过" for item in report["checks"])


def test_release_verifier_requires_high_risk_execution_contract_tests():
    verifier = load_verifier()

    report = verifier.verify_release(ROOT, run_tests=False)
    contract_check = next(
        item
        for item in report["checks"]
        if item["key"] == "execution_contract_tests"
    )

    assert contract_check["status"] == "通过"
    assert contract_check["label"] == "跨平台高风险执行合同"
    assert "10" in contract_check["detail"]
    ptrade_contracts = verifier.EXECUTION_CONTRACT_TESTS[
        "test_cross_signal_ptrade_strategy.py"
    ]
    assert {
        "test_eight_percent_premium_executes_blocked_qdii_sell",
        "test_below_eight_percent_keeps_original_blocked_sell",
        "test_iopv_sell_override_rejects_unsafe_snapshot",
        "test_iopv_sell_override_exception_cannot_interrupt_sell_evaluation",
    } <= ptrade_contracts


def test_release_verifier_disables_pytest_cache_for_full_gate(monkeypatch):
    verifier = load_verifier()
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return types.SimpleNamespace(returncode=0, stdout="1 passed\n", stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", fake_run)

    report = verifier.verify_release(ROOT, run_tests=True)

    assert report["status"] == "通过"
    assert "-p" in commands[0]
    assert "no:cacheprovider" in commands[0]


def test_release_verifier_reports_temp_acl_as_environment_failure(monkeypatch):
    verifier = load_verifier()
    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda command, **kwargs: types.SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="PermissionError: [WinError 5] access denied\n",
        ),
    )

    report = verifier.verify_release(ROOT, run_tests=True)
    test_check = next(item for item in report["checks"] if item["key"] == "tests")

    assert report["status"] == "失败"
    assert test_check["detail"].startswith("测试环境不可用:")


def test_release_verifier_surfaces_failed_test_node_ids(monkeypatch):
    verifier = load_verifier()
    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda command, **kwargs: types.SimpleNamespace(
            returncode=1,
            stdout=(
                "FAILED tests/test_example.py::test_broken - AssertionError\n"
                "1 failed, 2 passed\n"
            ),
            stderr="",
        ),
    )

    report = verifier.verify_release(ROOT, run_tests=True)
    test_check = next(item for item in report["checks"] if item["key"] == "tests")

    assert report["status"] == "失败"
    assert "tests/test_example.py::test_broken" in test_check["detail"]
    assert "1 failed, 2 passed" in test_check["detail"]


def test_release_verifier_registers_cross_age_helper_for_ast_parity():
    verifier = load_verifier()

    assert (
        "_latest_cross_age_by_diff_recent"
        in verifier.PURE_BUSINESS_FUNCTIONS
    )


def test_release_verifier_registers_atr_stress_functions_for_ast_parity():
    verifier = load_verifier()

    assert "trading_days_between" in verifier.PURE_BUSINESS_FUNCTIONS
    assert "portfolio_atr_stress_buy_scale" in verifier.PURE_BUSINESS_FUNCTIONS
    assert (
        "calc_stress_adjusted_buy_target_value"
        in verifier.PURE_BUSINESS_FUNCTIONS
    )


def test_release_verifier_rejects_stale_formal_label(tmp_path):
    verifier = load_verifier()
    strategy_root = tmp_path / "cross_signal_strategy"
    strategy_root.mkdir()
    source_root = ROOT / "cross_signal_strategy"
    for name in (
        "smart_trade_joinquant_cross_signal_etf.py",
        "smart_trade_ptrade_cross_signal_etf.py",
        "local_training_run.py",
    ):
        text = (source_root / name).read_text(encoding="utf-8")
        if name.startswith("smart_trade_joinquant"):
            text += '\nlog.info("[cross-v0.1] stale")\n'
        (strategy_root / name).write_text(text, encoding="utf-8")

    report = verifier.verify_release(tmp_path, run_tests=False)

    assert report["status"] == "失败"
    stale_check = next(
        item for item in report["checks"] if item["key"] == "release_labels"
    )
    assert stale_check["status"] == "失败"
