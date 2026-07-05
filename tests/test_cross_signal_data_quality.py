# -*- coding: utf-8 -*-
"""Tests for cross-signal JoinQuant/local data-quality diagnostics."""

import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def test_parse_joinquant_rich_indicator_rows_extracts_close():
    from cross_signal_strategy.local_data_quality import parse_joinquant_rich_indicator_rows

    text = (
        "2020-09-22 09:35:00 - INFO  -   512100.XSHG buy=65 rev=35 loc=17 "
        "trend=9 vol=4 sell=0 close=0.963 RSI[6/12/24]=56.4/50.6/52.6 "
        "MACD[DIF/DEA/HIST]=-0.0041/-0.0021/-0.0041 KDJ[K/D/J]=60.1/46.5/87.3"
    )

    rows = parse_joinquant_rich_indicator_rows(text)

    assert rows == [
        {
            "date": "2020-09-22",
            "code": "512100",
            "buy": 65,
            "rev": 35,
            "sell": 0,
            "close": 0.963,
        }
    ]


def test_find_close_mismatches_uses_adapter_scores():
    from cross_signal_strategy.local_data_quality import find_close_mismatches

    rows = [
        {"date": "2020-01-17", "code": "510880", "buy": 31, "rev": 0, "sell": 35, "close": 2.803},
        {"date": "2020-09-22", "code": "512100", "buy": 65, "rev": 35, "sell": 0, "close": 0.963},
    ]

    class Adapter(object):
        def score(self, code, date, return_reason=False):
            values = {"510880": 2.947, "512100": 0.963}
            score = {"close": values[code]}
            return (score, None) if return_reason else score

    mismatches = find_close_mismatches(rows, Adapter(), tolerance=0.002)

    assert mismatches == [
        {
            "date": "2020-01-17",
            "code": "510880",
            "jq_close": 2.803,
            "local_close": 2.947,
            "diff": 0.144,
        }
    ]
