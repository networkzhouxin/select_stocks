# -*- coding: utf-8 -*-

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from cross_signal_strategy.research.prospective_log_archive import (
    PROTOCOL_START,
    LogIdentityError,
    LogProtocolError,
    archive_log_bundle,
    inspect_log_bytes,
)


EXPECTED_VERSION = "cross-v0.3.2"
EXPECTED_BUILD = "20260718.1"
EXPECTED_FINGERPRINT = "1506a0e834fe"
REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DOC = (
    REPO_ROOT
    / "cross_signal_strategy"
    / "docs"
    / "prospective_live_log_protocol.md"
)
DOC_INDEX = REPO_ROOT / "cross_signal_strategy" / "docs" / "README.md"
STRATEGY_README = REPO_ROOT / "cross_signal_strategy" / "README.md"
DECISIONS = REPO_ROOT / "cross_signal_strategy" / "docs" / "decisions.md"
RESEARCH_BUDGET = (
    REPO_ROOT / "cross_signal_strategy" / "docs" / "research_budget.md"
)


def _valid_log(day="2026-07-18", build=EXPECTED_BUILD, fingerprint=EXPECTED_FINGERPRINT):
    return (
        "%s 09:30:00 - INFO - [发布指纹] 构建=%s 业务配置=%s 状态结构=2\n"
        "%s 09:35:00 - INFO - [%s] 执行日期=%s 信号日期=2026-07-17 是否调仓=是\n"
        "%s 09:35:01 - INFO - [IOPV观察] 事件=买入 时间=%s 09:35:01 "
        "代码=513100.SS 有效=True 市价=1.234 IOPV=1.230 溢价率百分比=0.32\n"
        "%s 09:35:02 - INFO - [买入] 513100.SS 买入评分=70 目标金额=6000 股数=4800\n"
        "%s 09:35:03 - INFO - [成交回报] 买入 513100.SS 数量=4800 价格=1.234 累计成交=4800\n"
        "%s 10:35:00 - INFO - [复牌补偿] 已复牌=513500.SS，已执行延后卖出与买入评估\n"
        % (
            day,
            build,
            fingerprint,
            day,
            EXPECTED_VERSION,
            day,
            day,
            day,
            day,
            day,
            day,
        )
    ).encode("utf-8")


def test_inspect_log_extracts_only_forward_evidence_metadata():
    raw = _valid_log()

    record = inspect_log_bytes(
        "live-20260718.log",
        raw,
        expected_version=EXPECTED_VERSION,
        expected_build=EXPECTED_BUILD,
        expected_fingerprint=EXPECTED_FINGERPRINT,
    )

    assert record.sha256 == hashlib.sha256(raw).hexdigest()
    assert record.first_timestamp == "2026-07-18 09:30:00"
    assert record.last_timestamp == "2026-07-18 10:35:00"
    assert record.execution_dates == ("2026-07-18",)
    assert record.signal_dates == ("2026-07-17",)
    assert record.release_identity_count == 1
    assert record.execution_count == 1
    assert record.buy_order_count == 1
    assert record.sell_order_count == 0
    assert record.buy_fill_count == 1
    assert record.sell_fill_count == 0
    assert record.iopv_observation_count == 1
    assert record.halt_recovery_count == 1
    assert not hasattr(record, "return_pct")
    assert not hasattr(record, "execution_price")


@pytest.mark.parametrize(
    ("raw", "error_type", "message"),
    [
        (
            _valid_log(day="2026-07-17"),
            LogProtocolError,
            "protocol start",
        ),
        (
            _valid_log(build="20260718.0"),
            LogIdentityError,
            "deployment build",
        ),
        (
            _valid_log(fingerprint="deadbeef0000"),
            LogIdentityError,
            "business fingerprint",
        ),
        (
            (
                "2026-07-18 09:35:00 - INFO - [cross-v0.3.2] "
                "执行日期=2026-07-18 信号日期=2026-07-17 是否调仓=是\n"
            ).encode("utf-8"),
            LogIdentityError,
            "release identity",
        ),
    ],
)
def test_inspect_log_fails_closed_on_nonprospective_or_unidentified_input(
    raw, error_type, message
):
    with pytest.raises(error_type, match=message):
        inspect_log_bytes(
            "candidate.log",
            raw,
            expected_version=EXPECTED_VERSION,
            expected_build=EXPECTED_BUILD,
            expected_fingerprint=EXPECTED_FINGERPRINT,
        )


def test_inspect_log_rejects_a_preprotocol_line_anywhere_in_file():
    raw = _valid_log() + (
        "2026-07-17 15:00:00 - INFO - [平台] 旧记录不应混入前瞻归档\n"
    ).encode("utf-8")

    with pytest.raises(LogProtocolError, match="protocol start"):
        inspect_log_bytes(
            "mixed-date.log",
            raw,
            expected_version=EXPECTED_VERSION,
            expected_build=EXPECTED_BUILD,
            expected_fingerprint=EXPECTED_FINGERPRINT,
        )


def test_archive_is_content_addressed_idempotent_and_source_immutable(tmp_path):
    source = tmp_path / "exported-ptrade.log"
    archive_root = tmp_path / "archive"
    raw = _valid_log()
    source.write_bytes(raw)

    first = archive_log_bundle(
        [source],
        archive_root,
        expected_version=EXPECTED_VERSION,
        expected_build=EXPECTED_BUILD,
        expected_fingerprint=EXPECTED_FINGERPRINT,
    )
    second = archive_log_bundle(
        [source],
        archive_root,
        expected_version=EXPECTED_VERSION,
        expected_build=EXPECTED_BUILD,
        expected_fingerprint=EXPECTED_FINGERPRINT,
    )

    digest = hashlib.sha256(raw).hexdigest()
    archived = archive_root / "raw" / (digest + ".log")
    manifest_path = archive_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert source.read_bytes() == raw
    assert archived.read_bytes() == raw
    assert first == second
    assert first["protocol_start"] == PROTOCOL_START.isoformat()
    assert first["release"] == {
        "strategy_version": EXPECTED_VERSION,
        "deployment_build": EXPECTED_BUILD,
        "business_fingerprint": EXPECTED_FINGERPRINT,
    }
    assert len(first["files"]) == 1
    assert first["files"][0]["sha256"] == digest
    assert manifest == first
    assert "price" not in json.dumps(manifest, ensure_ascii=False).lower()
    assert "return" not in json.dumps(manifest, ensure_ascii=False).lower()


def test_archive_rejects_source_inside_archive_root(tmp_path):
    archive_root = tmp_path / "archive"
    source = archive_root / "incoming.log"
    source.parent.mkdir(parents=True)
    source.write_bytes(_valid_log())

    with pytest.raises(LogProtocolError, match="outside the archive root"):
        archive_log_bundle(
            [source],
            archive_root,
            expected_version=EXPECTED_VERSION,
            expected_build=EXPECTED_BUILD,
            expected_fingerprint=EXPECTED_FINGERPRINT,
        )


def test_archive_refuses_to_append_when_existing_raw_evidence_is_missing(tmp_path):
    first_source = tmp_path / "first.log"
    second_source = tmp_path / "second.log"
    archive_root = tmp_path / "archive"
    first_raw = _valid_log()
    second_raw = _valid_log(day="2026-07-19").replace(
        b"2026-07-17", b"2026-07-18"
    )
    first_source.write_bytes(first_raw)
    second_source.write_bytes(second_raw)
    archive_log_bundle(
        [first_source],
        archive_root,
        expected_version=EXPECTED_VERSION,
        expected_build=EXPECTED_BUILD,
        expected_fingerprint=EXPECTED_FINGERPRINT,
    )
    digest = hashlib.sha256(first_raw).hexdigest()
    (archive_root / "raw" / (digest + ".log")).unlink()

    with pytest.raises(LogProtocolError, match="existing raw evidence is missing"):
        archive_log_bundle(
            [second_source],
            archive_root,
            expected_version=EXPECTED_VERSION,
            expected_build=EXPECTED_BUILD,
            expected_fingerprint=EXPECTED_FINGERPRINT,
        )


def test_protocol_start_is_fixed_before_any_forward_log_is_archived():
    assert PROTOCOL_START == date(2026, 7, 18)


def test_cli_uses_current_formal_release_identity(tmp_path, capsys):
    from cross_signal_strategy.tools.archive_ptrade_forward_logs import main

    source = tmp_path / "future-live.log"
    archive_root = tmp_path / "forward-archive"
    source.write_bytes(_valid_log())

    exit_code = main([
        "--archive-root",
        str(archive_root),
        str(source),
    ])
    manifest = json.loads(
        (archive_root / "manifest.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert manifest["release"]["strategy_version"] == EXPECTED_VERSION
    assert manifest["release"]["deployment_build"] == EXPECTED_BUILD
    assert manifest["release"]["business_fingerprint"] == EXPECTED_FINGERPRINT
    assert "归档完成" in capsys.readouterr().out


def test_cli_fails_closed_before_writing_mismatched_release(tmp_path, capsys):
    from cross_signal_strategy.tools.archive_ptrade_forward_logs import main

    source = tmp_path / "wrong-build.log"
    archive_root = tmp_path / "forward-archive"
    source.write_bytes(_valid_log(build="20260718.0"))

    exit_code = main([
        "--archive-root",
        str(archive_root),
        str(source),
    ])

    assert exit_code == 2
    assert not (archive_root / "manifest.json").exists()
    assert "归档拒绝" in capsys.readouterr().err


def test_forward_log_protocol_freezes_collection_without_opening_research():
    protocol = PROTOCOL_DOC.read_text(encoding="utf-8")
    index = DOC_INDEX.read_text(encoding="utf-8")
    strategy_readme = STRATEGY_README.read_text(encoding="utf-8")
    decisions = DECISIONS.read_text(encoding="utf-8")
    research_budget = RESEARCH_BUDGET.read_text(encoding="utf-8")

    required_terms = (
        "2026-07-18",
        "20260718.1",
        "1506a0e834fe",
        "不改变交易逻辑",
        "不新增定时任务",
        "不新增行情请求",
        "只增不改",
        "先登记假设",
        "独立确认样本",
        "不得用于验证期调参",
        "不计算收益",
    )
    for term in required_terms:
        assert term in protocol
    assert "prospective_live_log_protocol.md" in index
    assert "archive_ptrade_forward_logs.py" in strategy_readme
    assert "prospective_live_log_protocol.md" in strategy_readme
    assert "Adopt A Prospective PTrade Log Archive Without Opening Research" in decisions
    assert "Prospective log collection does not reopen a research family" in research_budget
