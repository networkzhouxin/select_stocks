from datetime import date
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = (
    ROOT / "cross_signal_strategy" / "tools" / "audit_ptrade_runtime_log.py"
)
README_PATH = ROOT / "cross_signal_strategy" / "README.md"
DEPLOYMENT_PATH = (
    ROOT / "cross_signal_strategy" / "docs" / "ptrade_deployment.md"
)


@pytest.fixture
def audit():
    assert TOOL_PATH.exists(), "PTrade日志审计工具尚未实现"
    spec = importlib.util.spec_from_file_location("ptrade_runtime_log_audit", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def line(timestamp, level, message):
    return "%s - %s - %s" % (timestamp, level, message)


def normal_day_lines(include_conditional_events=True):
    lines = [
        line(
            "2026-07-20 08:00:00",
            "INFO",
            "[cross-v0.3.3] 初始化完成: 最大持仓=3 基础仓位比例=0.95 普通信号最短持有=5",
        ),
        line(
            "2026-07-20 08:55:00",
            "INFO",
            "[PTrade框架g] 状态=未提供 代次=不适用 原因=未发现持久状态元数据",
        ),
        line(
            "2026-07-20 08:55:00",
            "INFO",
            "[连续状态恢复] 来源=状态台账 代次=3",
        ),
        line(
            "2026-07-20 08:55:00",
            "INFO",
            "[持仓风险恢复] 来源=账户接管:交割单",
        ),
        line(
            "2026-07-20 08:55:00",
            "INFO",
            "[持仓风险恢复] 代码=513100.SS 数量=100 成本=1.000000 买入日期=2026-07-17 ATR=0.020000 持仓最高收盘价=1.050000 状态=已验证 来源=账户接管:交割单",
        ),
        line(
            "2026-07-20 09:35:00",
            "INFO",
            "[cross-v0.3.3] 执行日期=2026-07-20 信号日期=2026-07-17 是否调仓=是",
        ),
    ]
    if include_conditional_events:
        lines.extend(
            [
                line(
                    "2026-07-20 09:35:02",
                    "INFO",
                    "[IOPV观察] 事件=买入 时间=2026-07-20 09:35:02 代码=513500.SS 有效=True 市价=1.2 IOPV=1.19 溢价率百分比=0.84 行情时间戳=20260720093501 行情延迟秒数=1",
                ),
                line(
                    "2026-07-20 09:35:03",
                    "INFO",
                    "[买入] 513500.SS 买入评分=70 反转评分=35 位置评分=17 趋势评分=20 量能评分=4 目标金额=6000 股数=5000",
                ),
                line(
                    "2026-07-20 09:35:05",
                    "INFO",
                    "[成交回报] 买入 513500.SS 数量=5000 价格=1.200 累计成交=5000",
                ),
                line(
                    "2026-07-20 10:35:00",
                    "INFO",
                    "[复牌补偿] 已复牌=513050.SS，已执行延后卖出与买入评估",
                ),
            ]
        )
    else:
        lines.append(
            line(
                "2026-07-20 09:35:04",
                "INFO",
                "[cross-v0.3.3] 没有达到阈值的买入候选",
            )
        )
    lines.append(
        line(
            "2026-07-20 15:30:00",
            "INFO",
            "[cross-v0.3.3 收盘] 总资产=21000.00 可用资金=5000.00 持仓数=2/3",
        )
    )
    return lines


def test_complete_runtime_day_passes_all_triggered_checks(audit):
    report = audit.audit_runtime_log("\n".join(normal_day_lines()))

    assert report.overall_status == audit.STATUS_PASS
    assert report.get_check("initialization").status == audit.STATUS_PASS
    assert report.get_check("recovery").status == audit.STATUS_PASS
    assert report.get_check("main_0935").status == audit.STATUS_PASS
    assert report.get_check("halt_1035").status == audit.STATUS_PASS
    assert report.get_check("after_close").status == audit.STATUS_PASS
    assert report.get_check("orders").status == audit.STATUS_PASS
    assert report.get_check("iopv").status == audit.STATUS_PASS
    assert report.get_check("errors").status == audit.STATUS_PASS


def test_current_trading_day_start_marker_passes_main_0935_check(audit):
    lines = normal_day_lines()
    lines[5] = line(
        "2026-07-20 09:35:00",
        "INFO",
        "[交易日开始] 执行日期=2026-07-20 信号日期=2026-07-17 "
        "策略=cross-v0.3.3 是否调仓=是",
    )

    report = audit.audit_runtime_log("\n".join(lines))

    assert report.get_check("main_0935").status == audit.STATUS_PASS


def test_no_halt_and_no_order_are_conditional_not_failures(audit):
    report = audit.audit_runtime_log(
        "\n".join(normal_day_lines(include_conditional_events=False))
    )

    assert report.overall_status == audit.STATUS_PASS
    assert report.get_check("halt_1035").status == audit.STATUS_NOT_TRIGGERED
    assert report.get_check("orders").status == audit.STATUS_NOT_TRIGGERED
    assert report.get_check("iopv").status == audit.STATUS_NOT_TRIGGERED


def test_unverified_position_fails_recovery_gate(audit):
    lines = normal_day_lines(include_conditional_events=False)
    lines[4] = lines[4].replace("状态=已验证", "状态=未验证")

    report = audit.audit_runtime_log("\n".join(lines))

    assert report.overall_status == audit.STATUS_FAIL
    recovery = report.get_check("recovery")
    assert recovery.status == audit.STATUS_FAIL
    assert "513100.SS" in recovery.evidence


def test_any_error_level_event_fails_runtime_gate(audit):
    lines = normal_day_lines(include_conditional_events=False)
    lines.append(
        line(
            "2026-07-20 09:35:06",
            "ERROR",
            "[交易] 券商委托状态无法确认，本次不提交委托",
        )
    )

    report = audit.audit_runtime_log("\n".join(lines))

    assert report.overall_status == audit.STATUS_FAIL
    assert report.get_check("errors").status == audit.STATUS_FAIL


@pytest.mark.parametrize("iopv_position", [None, "after"])
def test_qdii_buy_requires_preceding_iopv_observation(audit, iopv_position):
    lines = normal_day_lines(include_conditional_events=False)[:-1]
    buy = line(
        "2026-07-20 09:35:03",
        "INFO",
        "[买入] 513500.SS 买入评分=70 反转评分=35 位置评分=17 趋势评分=20 量能评分=4 目标金额=6000 股数=5000",
    )
    lines.append(buy)
    if iopv_position == "after":
        lines.append(
            line(
                "2026-07-20 09:35:04",
                "INFO",
                "[IOPV观察] 事件=买入 时间=2026-07-20 09:35:04 代码=513500.SS 有效=False 市价=1.2 IOPV=None 溢价率百分比=None 行情时间戳=None 行情延迟秒数=None",
            )
        )
    lines.append(
        line(
            "2026-07-20 15:30:00",
            "INFO",
            "[cross-v0.3.3 收盘] 总资产=21000.00 可用资金=5000.00 持仓数=2/3",
        )
    )

    report = audit.audit_runtime_log("\n".join(lines))

    assert report.overall_status == audit.STATUS_FAIL
    assert report.get_check("iopv").status == audit.STATUS_FAIL


def test_submitted_order_without_later_callback_requires_review(audit):
    lines = normal_day_lines(include_conditional_events=False)[:-1]
    lines.extend(
        [
            line(
                "2026-07-20 09:35:03",
                "INFO",
                "[买入] 159915.SZ 买入评分=70 反转评分=35 位置评分=17 趋势评分=20 量能评分=4 目标金额=6000 股数=5000",
            ),
            line(
                "2026-07-20 15:30:00",
                "INFO",
                "[cross-v0.3.3 收盘] 总资产=21000.00 可用资金=5000.00 持仓数=2/3",
            ),
        ]
    )

    report = audit.audit_runtime_log("\n".join(lines))

    assert report.overall_status == audit.STATUS_REVIEW
    assert report.get_check("orders").status == audit.STATUS_REVIEW


def test_target_date_excludes_other_day_errors(audit):
    lines = normal_day_lines(include_conditional_events=False)
    lines.append(
        line(
            "2026-07-19 09:35:00",
            "ERROR",
            "[交易] 旧日期错误，不属于本次审计",
        )
    )

    report = audit.audit_runtime_log("\n".join(lines), target_date=date(2026, 7, 20))

    assert report.overall_status == audit.STATUS_PASS
    assert report.target_date == "2026-07-20"


def test_multiple_dates_without_explicit_date_require_review(audit):
    lines = normal_day_lines(include_conditional_events=False)
    lines.append(
        line(
            "2026-07-19 15:30:00",
            "INFO",
            "[cross-v0.3.3 收盘] 总资产=20000.00 可用资金=5000.00 持仓数=2/3",
        )
    )

    report = audit.audit_runtime_log("\n".join(lines))

    assert report.overall_status == audit.STATUS_REVIEW
    assert report.target_date == "多日期或未知"
    assert report.get_check("date_scope").status == audit.STATUS_INSUFFICIENT


def test_cli_prints_chinese_report_and_returns_nonzero_for_failure(
    audit, tmp_path, capsys
):
    path = tmp_path / "ptrade.log"
    path.write_text(
        line("2026-07-20 09:35:00", "ERROR", "[交易] 测试错误"),
        encoding="utf-8",
    )

    exit_code = audit.main([str(path)])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "PTrade运行日志审计" in output
    assert "总体状态：失败" in output


def test_tool_documents_conditional_evidence_boundaries_in_chinese_comments():
    assert TOOL_PATH.exists(), "PTrade日志审计工具尚未实现"
    text = TOOL_PATH.read_text(encoding="utf-8")

    assert "证据边界" in text
    assert "# 10:35" in text
    assert "# IOPV" in text
    assert "不读取行情数据" in text


def test_readme_links_runtime_log_audit_command():
    text = README_PATH.read_text(encoding="utf-8")

    assert "audit_ptrade_runtime_log.py" in text
    assert "--date YYYY-MM-DD" in text
    assert "只读" in text


def test_ptrade_deployment_documents_audit_status_and_evidence_boundary():
    text = DEPLOYMENT_PATH.read_text(encoding="utf-8")

    assert "PTrade 运行日志审计" in text
    assert "条件未触发" in text
    assert "需复核" in text
    assert "证据不足" in text
    assert "不读取行情数据" in text
    assert "不能替代" in text
