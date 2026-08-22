from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_ROOT = ROOT / "cross_signal_strategy"


def test_cross_signal_root_keeps_three_formal_entries_and_labels_candidates():
    root_python_files = sorted(path.name for path in STRATEGY_ROOT.glob("*.py"))

    formal_entries = {
        "local_training_run.py",
        "smart_trade_joinquant_cross_signal_etf.py",
        "smart_trade_ptrade_cross_signal_etf.py",
    }
    candidate_entries = set(root_python_files) - formal_entries

    assert formal_entries.issubset(root_python_files)
    assert candidate_entries == {
        "smart_trade_joinquant_cross_signal_etf_late_macd_boll_filter_candidate.py",
        "smart_trade_joinquant_cross_signal_etf_late_veto_early_pre_macd_candidate.py",
    }
    assert all(name.endswith("_candidate.py") for name in candidate_entries)


def test_cross_signal_archives_are_separated_by_role():
    expected_directories = [
        STRATEGY_ROOT / "archive" / "candidates",
        STRATEGY_ROOT / "archive" / "probes",
        STRATEGY_ROOT / "local",
        STRATEGY_ROOT / "research",
        STRATEGY_ROOT / "tools",
    ]

    for directory in expected_directories:
        assert directory.is_dir(), directory
        assert (directory / "__init__.py").is_file(), directory

    assert (
        STRATEGY_ROOT
        / "archive"
        / "candidates"
        / "smart_trade_joinquant_cross_signal_etf_combo_candidate.py"
    ).is_file()
    assert (
        STRATEGY_ROOT
        / "archive"
        / "probes"
        / "smart_trade_ptrade_cross_signal_iopv_probe.py"
    ).is_file()
    assert (STRATEGY_ROOT / "local" / "local_backtester.py").is_file()
    assert (STRATEGY_ROOT / "research" / "trade_diagnostics.py").is_file()
    assert (STRATEGY_ROOT / "tools" / "trade_chart.py").is_file()


def test_cross_signal_readme_documents_the_archived_layout():
    readme = (STRATEGY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "archive/candidates" in readme
    assert "archive/probes" in readme
    assert "local/" in readme
    assert "research/" in readme
    assert "tools/" in readme


def test_current_spec_documents_daily_signal_evaluation():
    spec = (STRATEGY_ROOT / "docs" / "strategy_spec.md").read_text(
        encoding="utf-8"
    )

    assert "Current formal schedule: Monday through Friday" in spec
    assert "Current formal rotation: 09:35 every trading day" in spec


def test_cross_signal_readme_documents_release_verification_command():
    readme = (STRATEGY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "tools/verify_release.py" in readme
    assert "--run-tests" in readme
