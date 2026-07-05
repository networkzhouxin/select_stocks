# -*- coding: utf-8 -*-
"""JoinQuant log versus local-data diagnostics for cross_signal_strategy."""

from __future__ import annotations

import re
from typing import Iterable, List, Mapping


RICH_INDICATOR_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) 09:35:00 .*? -\s+([0-9]{6})\.(?:XSHG|XSHE) "
    r"buy=(-?\d+) rev=(-?\d+) loc=(-?\d+) trend=(-?\d+) vol=(-?\d+) sell=(-?\d+) "
    r"close=([0-9.]+)"
)


def parse_joinquant_rich_indicator_rows(text: str) -> List[dict]:
    rows = []
    seen = set()
    for line in str(text).splitlines():
        match = RICH_INDICATOR_RE.search(line)
        if not match:
            continue
        date, code, buy, rev, _loc, _trend, _vol, sell, close = match.groups()
        key = (date, code)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "date": date,
            "code": code,
            "buy": int(buy),
            "rev": int(rev),
            "sell": int(sell),
            "close": float(close),
        })
    return rows


def find_close_mismatches(rows: Iterable[Mapping[str, object]], adapter, tolerance: float = 0.002) -> List[dict]:
    mismatches = []
    for row in rows:
        date = str(row["date"])
        code = str(row["code"])
        score, reason = adapter.score(code, date, return_reason=True)
        if score is None:
            continue
        jq_close = float(row["close"])
        local_close = float(score["close"])
        diff = abs(jq_close - local_close)
        if diff > tolerance:
            mismatches.append({
                "date": date,
                "code": code,
                "jq_close": jq_close,
                "local_close": local_close,
                "diff": round(diff, 6),
            })
    return mismatches
