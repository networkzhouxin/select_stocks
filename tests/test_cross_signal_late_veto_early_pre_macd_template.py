# -*- coding: utf-8 -*-
"""Executable contract for the paired JoinQuant evidence template."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "cross_signal_strategy"
    / "reports"
    / "templates"
    / "late_veto_early_pre_macd_pair.json"
)


def test_example_pair_template_is_loadable_and_passes_its_declared_gate():
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        evaluate_pair,
        load_paired_run,
    )

    raw = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    pair = load_paired_run(TEMPLATE)
    decision = evaluate_pair(pair)

    assert raw["example_only"] is True
    assert pair.kind.value == "training_nominal"
    assert pair.baseline.config.initial_cash == 20000
    assert pair.baseline.config.execution_time == "09:35"
    assert pair.baseline.config.fingerprint == "77e44d93d255"
    assert pair.candidate.config.fingerprint == "f6b08195dd3d"
    assert decision.passed is True


def test_example_template_cannot_be_used_as_authoritative_cli_evidence():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cross_signal_strategy.research.late_veto_early_pre_macd_gate",
            str(TEMPLATE),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "INVALID" in completed.stdout
    assert "example_only" in completed.stdout
