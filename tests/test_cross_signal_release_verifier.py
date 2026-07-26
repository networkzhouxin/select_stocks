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
    assert report["strategy_version"] == "cross-v0.3.2"
    assert report["deployment_build"] == "20260726.9"
    assert len(report["business_fingerprint"]) == 12
    assert all(item["status"] == "通过" for item in report["checks"])


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
