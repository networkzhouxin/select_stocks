# -*- coding: utf-8 -*-
"""Temporary JoinQuant underlying-index availability probe.

This file places no trades and is not a strategy candidate.  It tests whether
JoinQuant's historical backtest can read the tracked underlying index at China
09:35 with ``avoid_future_data`` enabled.  Passing this probe cannot prove the index publisher's original release timestamp or reconstruct later revisions.

JoinQuant run settings:
- Backtest window: 2019-01-01 through 2021-12-31.
- Frequency: daily; the scheduled callback runs at 09:35.
- Initial capital: any value; no trading API is called.
- Return all log lines prefixed with ``[underlying-availability-``.
"""

import datetime

import pandas as pd
from jqdata import *
from jqdata import finance


PROBE_ETFS = (
    "513500.XSHG",  # S&P 500 ETF
    "513050.XSHG",  # CSI Overseas China Internet 50 ETF
)
TARGET_DATES = {
    datetime.date(2019, 1, 2),
    datetime.date(2020, 2, 7),
    datetime.date(2020, 9, 21),
    datetime.date(2021, 12, 27),
}


def _format_frame(frame):
    if frame is None:
        return "None"
    try:
        return "index=%s rows=%s" % (
            [str(value) for value in frame.index],
            frame.to_dict("records"),
        )
    except Exception:
        return repr(frame)


def _clean_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _active_tracking_metadata(context, etf_code):
    table = finance.FUND_INVEST_TARGET
    query_object = query(
        table.code,
        table.name,
        table.pub_date,
        table.start_date,
        table.end_date,
        table.traced_index_name,
        table.traced_index_code,
    ).filter(
        table.code == etf_code,
        finance.FUND_INVEST_TARGET.pub_date <= context.current_dt.date(),
    )
    frame = finance.run_query(query_object)
    if frame is None or frame.empty:
        return None, frame

    as_of = pd.Timestamp(context.current_dt.date())
    normalized = frame.copy()
    for column in ("pub_date", "start_date", "end_date"):
        normalized[column] = pd.to_datetime(normalized[column], errors="coerce")
    active = normalized[
        normalized["pub_date"].le(as_of)
        & normalized["start_date"].le(as_of)
        & (normalized["end_date"].isna() | normalized["end_date"].ge(as_of))
    ].sort_values(["start_date", "pub_date"])
    if active.empty:
        return None, normalized
    return active.iloc[-1], normalized


def _probe_t1_history(context, etf_code, index_code):
    try:
        frame = get_price(
            index_code,
            end_date=context.previous_date,
            count=2,
            frequency="daily",
            fields=["close"],
            skip_paused=True,
            fq=None,
        )
        log.info(
            "[underlying-availability-t1] dt=%s etf=%s index=%s "
            "end_date=%s readable=True data=%s",
            context.current_dt,
            etf_code,
            index_code,
            context.previous_date,
            _format_frame(frame),
        )
    except Exception as exc:
        log.info(
            "[underlying-availability-t1] dt=%s etf=%s index=%s "
            "end_date=%s readable=False type=%s error=%s",
            context.current_dt,
            etf_code,
            index_code,
            context.previous_date,
            exc.__class__.__name__,
            exc,
        )


def _probe_same_day_negative_control(context, etf_code, index_code):
    try:
        frame = get_price(
            index_code,
            end_date=context.current_dt.date(),
            count=1,
            frequency="daily",
            fields=["close"],
            skip_paused=True,
            fq=None,
        )
        log.info(
            "[underlying-availability-same-day] dt=%s etf=%s index=%s "
            "blocked=False data=%s",
            context.current_dt,
            etf_code,
            index_code,
            _format_frame(frame),
        )
    except Exception as exc:
        log.info(
            "[underlying-availability-same-day] dt=%s etf=%s index=%s "
            "blocked=True type=%s error=%s",
            context.current_dt,
            etf_code,
            index_code,
            exc.__class__.__name__,
            exc,
        )


def probe_underlying_availability(context):
    if context.current_dt.date() not in TARGET_DATES:
        return

    for etf_code in PROBE_ETFS:
        try:
            active, metadata = _active_tracking_metadata(context, etf_code)
        except Exception as exc:
            log.info(
                "[underlying-availability-metadata] dt=%s etf=%s "
                "readable=False type=%s error=%s",
                context.current_dt,
                etf_code,
                exc.__class__.__name__,
                exc,
            )
            continue

        if active is None:
            log.info(
                "[underlying-availability-metadata] dt=%s etf=%s "
                "readable=True active=False data=%s",
                context.current_dt,
                etf_code,
                _format_frame(metadata),
            )
            continue

        index_name = _clean_text(active.get("traced_index_name"))
        index_code = _clean_text(active.get("traced_index_code"))
        log.info(
            "[underlying-availability-metadata] dt=%s etf=%s readable=True "
            "active=True traced_index_name=%s traced_index_code=%s "
            "pub_date=%s start_date=%s end_date=%s",
            context.current_dt,
            etf_code,
            index_name,
            index_code,
            active.get("pub_date"),
            active.get("start_date"),
            active.get("end_date"),
        )
        if not index_code:
            log.info(
                "[underlying-availability-t1] dt=%s etf=%s index= "
                "readable=False reason=tracked-index-code-empty",
                context.current_dt,
                etf_code,
            )
            continue

        _probe_t1_history(context, etf_code, index_code)
        _probe_same_day_negative_control(context, etf_code, index_code)


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    run_daily(probe_underlying_availability, time="09:35")

    log.info(
        "[underlying-availability-init] etfs=%s target_dates=%s "
        "trades_disabled=True platform_readability_only=True "
        "publisher_timestamp_proved=False",
        list(PROBE_ETFS),
        sorted(str(value) for value in TARGET_DATES),
    )
