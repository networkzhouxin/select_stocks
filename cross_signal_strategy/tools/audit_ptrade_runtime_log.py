# -*- coding: utf-8 -*-
"""审计 cross-v0.3.2 的 PTrade 运行日志。

证据边界：本工具只检查日志中已经出现的运行事实，不调用 PTrade API，
不读取行情数据，也不推断没有记录在日志里的成交、停牌或检查点写入结果。
因此“条件未触发”不是失败，“证据不足”也不能被解释为已经通过。
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path
import re


STATUS_PASS = "通过"
STATUS_FAIL = "失败"
STATUS_REVIEW = "需复核"
STATUS_NOT_TRIGGERED = "条件未触发"
STATUS_INSUFFICIENT = "证据不足"

QDII_CODES = {"513050", "513100", "513500", "513880"}

LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"\s+-\s+(?P<level>DEBUG|INFO|WARNING|ERROR)\s+-\s+(?P<message>.*)$"
)
BUY_SUBMIT_RE = re.compile(r"^\[买入\]\s+(?P<code>\S+)\s+买入评分=")
SELL_SUBMIT_RE = re.compile(r"^\[卖出\]\s+(?P<code>\S+)\s+原因=")


@dataclass(frozen=True)
class LogEvent:
    line_number: int
    timestamp: datetime
    level: str
    message: str


@dataclass(frozen=True)
class CheckResult:
    key: str
    label: str
    status: str
    evidence: str


@dataclass(frozen=True)
class RuntimeAuditReport:
    overall_status: str
    target_date: str
    parsed_event_count: int
    ignored_line_count: int
    checks: tuple

    def get_check(self, key):
        for item in self.checks:
            if item.key == key:
                return item
        raise KeyError(key)

    def to_dict(self):
        result = asdict(self)
        result["checks"] = [asdict(item) for item in self.checks]
        return result


def _normalize_target_date(value):
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _parse_events(text, target_date=None):
    selected_date = _normalize_target_date(target_date)
    events = []
    ignored = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        match = LOG_LINE_RE.match(stripped)
        if match is None:
            ignored += 1
            continue
        timestamp = datetime.fromisoformat(match.group("timestamp"))
        if selected_date is not None and timestamp.date() != selected_date:
            continue
        events.append(
            LogEvent(
                line_number=line_number,
                timestamp=timestamp,
                level=match.group("level"),
                message=match.group("message"),
            )
        )
    return events, ignored


def _result(key, label, status, evidence):
    return CheckResult(key=key, label=label, status=status, evidence=evidence)


def _check_format(events, ignored):
    if not events:
        return _result("format", "日志格式", STATUS_FAIL, "没有解析到标准PTrade日志行")
    if ignored:
        return _result(
            "format",
            "日志格式",
            STATUS_REVIEW,
            "解析到%d条事件，另有%d条非空行无法识别" % (len(events), ignored),
        )
    return _result("format", "日志格式", STATUS_PASS, "解析到%d条事件" % len(events))


def _check_date_scope(events, selected_date):
    """阻止把多日运行事实拼成一个貌似完整的交易日。"""
    event_dates = sorted({event.timestamp.date() for event in events})
    if selected_date is not None:
        return _result(
            "date_scope",
            "审计日期范围",
            STATUS_PASS,
            "已按指定日期%s过滤" % selected_date.isoformat(),
        )
    if len(event_dates) == 1:
        return _result(
            "date_scope",
            "审计日期范围",
            STATUS_PASS,
            "日志只包含%s" % event_dates[0].isoformat(),
        )
    if not event_dates:
        return _result(
            "date_scope", "审计日期范围", STATUS_INSUFFICIENT, "没有可识别的日志日期"
        )
    return _result(
        "date_scope",
        "审计日期范围",
        STATUS_INSUFFICIENT,
        "日志包含多个日期=%s；请使用--date逐日审计"
        % ",".join(item.isoformat() for item in event_dates),
    )


def _check_initialization(events):
    matched = [event for event in events if "[cross-v0.3.2] 初始化完成:" in event.message]
    if not matched:
        return _result(
            "initialization", "策略初始化", STATUS_INSUFFICIENT, "未找到cross-v0.3.2初始化完成日志"
        )
    return _result("initialization", "策略初始化", STATUS_PASS, "初始化日志=%d条" % len(matched))


def _check_recovery(events):
    ptrade_g = [
        event
        for event in events
        if event.message.startswith("[PTrade框架g] 状态=")
    ]
    continuity = [
        event
        for event in events
        if event.message.startswith("[连续状态恢复] 来源=")
    ]
    risk_headers = [
        event
        for event in events
        if event.message.startswith("[持仓风险恢复] 来源=")
    ]
    positions = [
        event
        for event in events
        if event.message.startswith("[持仓风险恢复] 代码=")
    ]
    unverified = [event for event in positions if "状态=未验证" in event.message]
    if unverified:
        codes = []
        for event in unverified:
            match = re.search(r"代码=(\S+)", event.message)
            if match:
                codes.append(match.group(1))
        return _result(
            "recovery",
            "状态恢复",
            STATUS_FAIL,
            "存在未验证持仓=%s" % (",".join(codes) or "未知"),
        )
    missing = []
    if not ptrade_g:
        missing.append("PTrade框架g")
    if not continuity:
        missing.append("连续状态恢复")
    if not risk_headers:
        missing.append("持仓风险恢复")
    if missing:
        return _result(
            "recovery",
            "状态恢复",
            STATUS_INSUFFICIENT,
            "缺少恢复诊断=%s" % ",".join(missing),
        )
    return _result(
        "recovery",
        "状态恢复",
        STATUS_PASS,
        "PTrade框架g=%d条，连续状态=%d条，持仓来源=%d条，已验证持仓=%d条" % (
            len(ptrade_g), len(continuity), len(risk_headers), len(positions)
        ),
    )


def _check_main_0935(events):
    matched = [
        event
        for event in events
        if event.message.startswith((
            "[cross-v0.3.2] 执行日期=",
            "[交易日开始] 执行日期=",
        ))
    ]
    if not matched:
        return _result("main_0935", "09:35主流程", STATUS_INSUFFICIENT, "未找到执行日期日志")
    at_expected_time = [
        event for event in matched if (event.timestamp.hour, event.timestamp.minute) == (9, 35)
    ]
    if not at_expected_time:
        times = ",".join(event.timestamp.strftime("%H:%M:%S") for event in matched)
        return _result(
            "main_0935", "09:35主流程", STATUS_FAIL, "执行日志时间不符合实盘计划=%s" % times
        )
    return _result(
        "main_0935", "09:35主流程", STATUS_PASS, "09:35执行日志=%d条" % len(at_expected_time)
    )


def _check_halt_1035(events):
    # 10:35 只有存在停牌跟踪、复牌或异常时才输出日志；无日志不能伪装成已验证。
    matched = [event for event in events if event.message.startswith("[复牌补偿]")]
    if not matched:
        return _result(
            "halt_1035", "10:35复牌补偿", STATUS_NOT_TRIGGERED, "当天没有可审计的复牌补偿事件"
        )
    failed = [
        event
        for event in matched
        if event.level == "ERROR" or "不提交委托" in event.message
    ]
    if failed:
        return _result(
            "halt_1035", "10:35复牌补偿", STATUS_FAIL, failed[0].message
        )
    expected_time = [
        event for event in matched if (event.timestamp.hour, event.timestamp.minute) == (10, 35)
    ]
    if not expected_time:
        return _result(
            "halt_1035", "10:35复牌补偿", STATUS_REVIEW, "存在补偿日志，但时间不是10:35"
        )
    return _result(
        "halt_1035", "10:35复牌补偿", STATUS_PASS, "补偿日志=%d条" % len(matched)
    )


def _check_after_close(events):
    matched = [
        event
        for event in events
        if event.message.startswith("[cross-v0.3.2 收盘]")
    ]
    if not matched:
        return _result("after_close", "收盘流程", STATUS_INSUFFICIENT, "未找到收盘汇总")
    return _result("after_close", "收盘流程", STATUS_PASS, "收盘汇总=%d条" % len(matched))


def _submitted_orders(events):
    submissions = []
    for event in events:
        buy_match = BUY_SUBMIT_RE.match(event.message)
        if buy_match:
            submissions.append(("买入", buy_match.group("code"), event))
            continue
        sell_match = SELL_SUBMIT_RE.match(event.message)
        if sell_match:
            submissions.append(("卖出", sell_match.group("code"), event))
    return submissions


def _has_later_order_response(events, side, code, submit_line):
    prefixes = (
        "[成交回报] %s %s" % (side, code),
        "[%s拒绝或撤单] %s" % (side, code),
        "[%s部分成交或撤单] %s" % (side, code),
    )
    return any(
        event.line_number > submit_line
        and any(event.message.startswith(prefix) for prefix in prefixes)
        for event in events
    )


def _check_orders(events):
    submissions = _submitted_orders(events)
    if not submissions:
        return _result("orders", "委托生命周期", STATUS_NOT_TRIGGERED, "当天没有买卖委托")
    unmatched = [
        "%s:%s" % (side, code)
        for side, code, event in submissions
        if not _has_later_order_response(events, side, code, event.line_number)
    ]
    if unmatched:
        return _result(
            "orders",
            "委托生命周期",
            STATUS_REVIEW,
            "提交后未看到后续回报=%s" % ",".join(unmatched),
        )
    return _result(
        "orders", "委托生命周期", STATUS_PASS, "已观察委托及后续回报=%d笔" % len(submissions)
    )


def _check_iopv(events):
    # IOPV 是观察性证据。每一笔 QDII 买单前必须恰好出现一次同代码观察日志，
    # 但有效或无效都不得改变委托数量，也不能据此重新开放溢价调参。
    qdii_buys = [
        (code, event)
        for side, code, event in _submitted_orders(events)
        if side == "买入" and code.split(".")[0] in QDII_CODES
    ]
    if not qdii_buys:
        return _result("iopv", "QDII买前IOPV", STATUS_NOT_TRIGGERED, "当天没有QDII买入委托")

    missing = []
    duplicates = []
    previous_buy_line = {}
    for code, buy_event in qdii_buys:
        lower_bound = previous_buy_line.get(code, 0)
        observations = [
            event
            for event in events
            if lower_bound < event.line_number < buy_event.line_number
            and event.message.startswith("[IOPV观察]")
            and ("代码=%s" % code) in event.message
        ]
        if not observations:
            missing.append(code)
        elif len(observations) > 1:
            duplicates.append(code)
        previous_buy_line[code] = buy_event.line_number

    if missing:
        return _result(
            "iopv", "QDII买前IOPV", STATUS_FAIL, "买单前缺少IOPV观察=%s" % ",".join(missing)
        )
    if duplicates:
        return _result(
            "iopv", "QDII买前IOPV", STATUS_REVIEW, "单笔买单前出现重复IOPV观察=%s" % ",".join(duplicates)
        )
    return _result(
        "iopv", "QDII买前IOPV", STATUS_PASS, "QDII买单与前置IOPV观察一一对应=%d笔" % len(qdii_buys)
    )


def _check_errors(events):
    errors = [event for event in events if event.level == "ERROR"]
    if not errors:
        return _result("errors", "错误日志", STATUS_PASS, "未发现ERROR级别日志")
    evidence = " | ".join(event.message for event in errors[:5])
    return _result("errors", "错误日志", STATUS_FAIL, "ERROR=%d: %s" % (len(errors), evidence))


def _check_warnings(events):
    warnings = [event for event in events if event.level == "WARNING"]
    if not warnings:
        return _result("warnings", "警告日志", STATUS_PASS, "未发现WARNING级别日志")
    evidence = " | ".join(event.message for event in warnings[:5])
    return _result(
        "warnings", "警告日志", STATUS_REVIEW, "WARNING=%d: %s" % (len(warnings), evidence)
    )


def _overall_status(checks):
    statuses = {item.status for item in checks}
    if STATUS_FAIL in statuses:
        return STATUS_FAIL
    if STATUS_REVIEW in statuses or STATUS_INSUFFICIENT in statuses:
        return STATUS_REVIEW
    return STATUS_PASS


def audit_runtime_log(text, target_date=None):
    """返回单个日志文本的结构化验收结果，不修改原文件。"""
    normalized_date = _normalize_target_date(target_date)
    events, ignored = _parse_events(text, normalized_date)
    checks = (
        _check_format(events, ignored),
        _check_date_scope(events, normalized_date),
        _check_initialization(events),
        _check_recovery(events),
        _check_main_0935(events),
        _check_halt_1035(events),
        _check_after_close(events),
        _check_orders(events),
        _check_iopv(events),
        _check_errors(events),
        _check_warnings(events),
    )
    inferred_dates = sorted({event.timestamp.date() for event in events})
    date_text = (
        normalized_date.isoformat()
        if normalized_date is not None
        else inferred_dates[0].isoformat()
        if len(inferred_dates) == 1
        else "多日期或未知"
    )
    return RuntimeAuditReport(
        overall_status=_overall_status(checks),
        target_date=date_text,
        parsed_event_count=len(events),
        ignored_line_count=ignored,
        checks=checks,
    )


def _read_log_text(path):
    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", data, 0, 1, "日志不是UTF-8或GB18030编码")


def render_text_report(report):
    lines = [
        "PTrade运行日志审计",
        "总体状态：%s" % report.overall_status,
        "审计日期：%s" % report.target_date,
        "解析事件：%d，忽略行：%d" % (
            report.parsed_event_count,
            report.ignored_line_count,
        ),
    ]
    for item in report.checks:
        lines.append("- [%s] %s：%s" % (item.status, item.label, item.evidence))
    return "\n".join(lines)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="审计cross-v0.3.2的PTrade运行日志")
    parser.add_argument("log_file", help="PTrade导出的日志文本文件")
    parser.add_argument("--date", dest="target_date", help="只审计YYYY-MM-DD指定日期")
    parser.add_argument("--json", action="store_true", help="输出JSON而不是中文文本报告")
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    report = audit_runtime_log(_read_log_text(args.log_file), args.target_date)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_text_report(report))
    if report.overall_status == STATUS_FAIL:
        return 1
    if report.overall_status == STATUS_REVIEW:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
