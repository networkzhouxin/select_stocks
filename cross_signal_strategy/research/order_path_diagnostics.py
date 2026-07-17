# -*- coding: utf-8 -*-
"""Order-path diagnostics for JoinQuant logs versus local replay results."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, List, Sequence


BUY_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) 09:35:00 .*?\[buy\]\s+([0-9]{6})\.(?:XSHG|XSHE).*?target=([0-9.]+)"
)
SELL_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) 09:35:00 .*?\[sell\]\s+([0-9]{6})\.(?:XSHG|XSHE)\s+"
    r"reason=(.*?)\s+amount=([0-9]+)"
)
FILLED_ORDER_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) 09:35:00 .*?order StockOrder\(.*?"
    r"security=([0-9]{6})\.(?:XSHG|XSHE).*?action=(open|close).*?"
    r"trade price:\s*([0-9.]+),\s*amount:([0-9]+),\s*commission:\s*([0-9.]+)",
)


@dataclass(frozen=True)
class OrderPathEvent:
    date: str
    side: str
    code: str
    amount: int | None = None
    target_value: float | None = None
    price: float | None = None
    trade_value: float | None = None
    commission: float | None = None
    status: str = ""
    reason: str = ""

    def as_key(self) -> tuple[str, str, str]:
        return (self.date, self.side, self.code)


@dataclass(frozen=True)
class OrderPathDivergence:
    index: int
    expected: OrderPathEvent | None
    actual: OrderPathEvent | None
    message: str


@dataclass(frozen=True)
class OrderExecutionDiff:
    index: int
    key: tuple[str, str, str]
    amount_diff: int | None
    price_diff: float | None
    commission_diff: float | None
    trade_value_diff: float | None


def parse_joinquant_order_events(text: str) -> List[OrderPathEvent]:
    events: List[OrderPathEvent] = []
    for line in str(text).splitlines():
        buy = BUY_RE.search(line)
        if buy:
            date, code, target = buy.groups()
            events.append(
                OrderPathEvent(
                    date=date,
                    side="BUY",
                    code=code,
                    target_value=float(target),
                )
            )
            continue

        sell = SELL_RE.search(line)
        if sell:
            date, code, reason, amount = sell.groups()
            events.append(
                OrderPathEvent(
                    date=date,
                    side="SELL",
                    code=code,
                    amount=int(amount),
                    reason=reason,
                )
            )
    return events


def parse_joinquant_filled_order_events(text: str) -> List[OrderPathEvent]:
    events: List[OrderPathEvent] = []
    for line in str(text).splitlines():
        match = FILLED_ORDER_RE.search(line)
        if not match:
            continue
        date, code, action, price, amount, commission = match.groups()
        side = "BUY" if action == "open" else "SELL"
        signed_value = int(amount) * float(price) * (1 if side == "BUY" else -1)
        events.append(
            OrderPathEvent(
                date=date,
                side=side,
                code=code,
                amount=int(amount),
                price=float(price),
                trade_value=signed_value,
                commission=float(commission),
            )
        )
    return events


def parse_joinquant_transaction_csv(path: str | Path, filled_only: bool = True) -> List[OrderPathEvent]:
    """Parse JoinQuant exported transaction details as filled order events."""
    events: List[OrderPathEvent] = []
    with Path(path).open("r", encoding="gbk", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = str(row.get("状态", "")).strip()
            amount = _parse_share_amount(row.get("成交数量", ""))
            if filled_only and (status != "全部成交" or amount <= 0):
                continue

            date = str(row.get("日期", "")).strip()
            code = _extract_code(str(row.get("标的", "")))
            side = _normalize_side(str(row.get("交易类型", "")))
            price = _parse_optional_float(row.get("成交价", ""))
            trade_value = _parse_optional_float(row.get("成交额", ""))
            commission = _parse_optional_float(row.get("手续费", ""))
            events.append(
                OrderPathEvent(
                    date=date,
                    side=side,
                    code=code,
                    amount=amount,
                    price=price,
                    trade_value=trade_value,
                    commission=commission,
                    status=status,
                )
            )
    return events


def extract_local_order_events(results: Iterable[object]) -> List[OrderPathEvent]:
    events: List[OrderPathEvent] = []
    for day in results:
        date = str(getattr(day, "date"))
        for order in getattr(day, "orders"):
            if not getattr(order, "filled", False):
                continue
            amount_delta = int(getattr(order, "amount_delta"))
            if amount_delta == 0:
                continue
            side = "BUY" if amount_delta > 0 else "SELL"
            events.append(
                OrderPathEvent(
                    date=date,
                    side=side,
                    code=str(getattr(order, "code")).split(".")[0],
                    amount=abs(amount_delta),
                    price=float(getattr(order, "exec_price")),
                    trade_value=amount_delta * float(getattr(order, "exec_price")),
                    commission=float(getattr(order, "commission", 0.0)),
                    reason=str(getattr(order, "reason", "")),
                )
            )
    return events


def find_first_order_divergence(
    expected_events: Sequence[OrderPathEvent],
    actual_events: Sequence[OrderPathEvent],
) -> OrderPathDivergence | None:
    max_len = max(len(expected_events), len(actual_events))
    for index in range(max_len):
        expected = expected_events[index] if index < len(expected_events) else None
        actual = actual_events[index] if index < len(actual_events) else None
        if _event_key(expected) == _event_key(actual):
            continue
        return OrderPathDivergence(
            index=index,
            expected=expected,
            actual=actual,
            message=_format_divergence(index, expected, actual),
        )
    return None


def compare_order_execution_fields(
    expected_events: Sequence[OrderPathEvent],
    actual_events: Sequence[OrderPathEvent],
) -> List[OrderExecutionDiff]:
    diffs: List[OrderExecutionDiff] = []
    for index, (expected, actual) in enumerate(zip(expected_events, actual_events)):
        if expected.as_key() != actual.as_key():
            raise ValueError("Order path must be aligned before comparing execution fields")
        diff = OrderExecutionDiff(
            index=index,
            key=expected.as_key(),
            amount_diff=_optional_number_diff(actual.amount, expected.amount),
            price_diff=_optional_number_diff(actual.price, expected.price),
            commission_diff=_optional_number_diff(actual.commission, expected.commission),
            trade_value_diff=_optional_number_diff(actual.trade_value, expected.trade_value),
        )
        if any(
            value not in (None, 0, 0.0)
            for value in [diff.amount_diff, diff.price_diff, diff.commission_diff, diff.trade_value_diff]
        ):
            diffs.append(diff)
    return diffs


def _event_key(event: OrderPathEvent | None) -> tuple[str, str, str] | None:
    return event.as_key() if event is not None else None


def _format_divergence(
    index: int,
    expected: OrderPathEvent | None,
    actual: OrderPathEvent | None,
) -> str:
    return (
        "first mismatch at order index %d: expected %s, actual %s"
        % (index, _describe_event(expected), _describe_event(actual))
    )


def _describe_event(event: OrderPathEvent | None) -> str:
    if event is None:
        return "<missing>"
    return "%s %s %s" % (event.date, event.side, event.code)


def _extract_code(target: str) -> str:
    match = re.search(r"\(([0-9]{6})\.(?:XSHG|XSHE)\)", target)
    if not match:
        raise ValueError(f"Cannot parse JoinQuant target code: {target}")
    return match.group(1)


def _normalize_side(side: str) -> str:
    side = side.strip()
    if side == "买":
        return "BUY"
    if side == "卖":
        return "SELL"
    raise ValueError(f"Unknown JoinQuant side: {side}")


def _parse_share_amount(value: object) -> int:
    match = re.search(r"[-0-9]+", str(value).replace(",", ""))
    return int(match.group(0)) if match else 0


def _parse_optional_float(value: object) -> float | None:
    text = str(value).replace(",", "").strip()
    if not text or text == "--":
        return None
    return float(text)


def _optional_number_diff(actual, expected):
    if actual is None or expected is None:
        return None
    return actual - expected
