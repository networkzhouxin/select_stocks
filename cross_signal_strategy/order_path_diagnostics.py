# -*- coding: utf-8 -*-
"""Order-path diagnostics for JoinQuant logs versus local replay results."""

from __future__ import annotations

from dataclasses import dataclass
import re
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
    r"trade price:\s*([0-9.]+),\s*amount:([0-9]+),",
)


@dataclass(frozen=True)
class OrderPathEvent:
    date: str
    side: str
    code: str
    amount: int | None = None
    target_value: float | None = None
    price: float | None = None
    reason: str = ""

    def as_key(self) -> tuple[str, str, str]:
        return (self.date, self.side, self.code)


@dataclass(frozen=True)
class OrderPathDivergence:
    index: int
    expected: OrderPathEvent | None
    actual: OrderPathEvent | None
    message: str


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
        date, code, action, price, amount = match.groups()
        events.append(
            OrderPathEvent(
                date=date,
                side="BUY" if action == "open" else "SELL",
                code=code,
                amount=int(amount),
                price=float(price),
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
