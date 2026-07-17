# -*- coding: utf-8 -*-
"""Temporary IOPV capability probe for JoinQuant.

This file places no orders and must not be treated as a strategy candidate.
It checks whether point-in-time IOPV or same-day NAV is exposed at 09:35.

JoinQuant run settings:
- Backtest window: 2020-02-07 through 2020-09-21.
- Frequency: daily (the scheduled callbacks still run intraday).
- Initial capital: any value; there are no orders.
- Return only log lines prefixed with ``[iopv-probe-`` for analysis.
"""

import datetime

from jqdata import *


PROBE_CODE = "513100.XSHG"
TARGET_DATES = {
    datetime.date(2020, 2, 7),
    datetime.date(2020, 9, 21),
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


def probe_iopv_capability(context):
    if context.current_dt.date() not in TARGET_DATES:
        return

    current = get_current_data()[PROBE_CODE]
    log.info(
        "[iopv-probe-current] dt=%s code=%s paused=%s last_price=%s "
        "day_open=%s has_iopv_attr=%s iopv=%s",
        context.current_dt,
        PROBE_CODE,
        getattr(current, "paused", None),
        getattr(current, "last_price", None),
        getattr(current, "day_open", None),
        hasattr(current, "iopv"),
        getattr(current, "iopv", None),
    )

    try:
        minute = get_price(
            PROBE_CODE,
            end_date=context.current_dt,
            count=3,
            frequency="1m",
            fields=["close", "volume", "money"],
            skip_paused=False,
            fq=None,
        )
        log.info(
            "[iopv-probe-minute] dt=%s code=%s data=%s",
            context.current_dt,
            PROBE_CODE,
            _format_frame(minute),
        )
    except Exception as exc:
        log.info(
            "[iopv-probe-minute-error] dt=%s code=%s type=%s error=%s",
            context.current_dt,
            PROBE_CODE,
            exc.__class__.__name__,
            exc,
        )

    try:
        trade_days = get_trade_days(end_date=context.current_dt.date(), count=2)
        previous_date = trade_days[-2] if len(trade_days) >= 2 else context.previous_date
        previous_nav = get_extras(
            "unit_net_value",
            [PROBE_CODE],
            start_date=previous_date,
            end_date=previous_date,
            df=True,
        )
        log.info(
            "[iopv-probe-prev-nav] dt=%s code=%s previous_date=%s data=%s",
            context.current_dt,
            PROBE_CODE,
            previous_date,
            _format_frame(previous_nav),
        )
    except Exception as exc:
        log.info(
            "[iopv-probe-prev-nav-error] dt=%s code=%s type=%s error=%s",
            context.current_dt,
            PROBE_CODE,
            exc.__class__.__name__,
            exc,
        )

    try:
        same_day_nav = get_extras(
            "unit_net_value",
            [PROBE_CODE],
            start_date=context.current_dt.date(),
            end_date=context.current_dt.date(),
            df=True,
        )
        log.info(
            "[iopv-probe-same-day-nav] dt=%s code=%s data=%s",
            context.current_dt,
            PROBE_CODE,
            _format_frame(same_day_nav),
        )
    except Exception as exc:
        log.info(
            "[iopv-probe-same-day-nav-error] dt=%s code=%s type=%s error=%s",
            context.current_dt,
            PROBE_CODE,
            exc.__class__.__name__,
            exc,
        )

    if context.current_dt.hour == 9 and context.current_dt.minute == 35:
        try:
            iopv_field = get_price(
                PROBE_CODE,
                end_date=context.current_dt,
                count=1,
                frequency="1m",
                fields=["iopv"],
                skip_paused=False,
                fq=None,
            )
            log.info(
                "[iopv-probe-field] dt=%s code=%s supported=True data=%s",
                context.current_dt,
                PROBE_CODE,
                _format_frame(iopv_field),
            )
        except Exception as exc:
            log.info(
                "[iopv-probe-field] dt=%s code=%s supported=False type=%s error=%s",
                context.current_dt,
                PROBE_CODE,
                exc.__class__.__name__,
                exc,
            )


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)

    run_daily(probe_iopv_capability, time="09:34")
    run_daily(probe_iopv_capability, time="09:35")
    run_daily(probe_iopv_capability, time="09:36")
    run_daily(probe_iopv_capability, time="15:30")

    log.info(
        "[iopv-probe-init] code=%s target_dates=%s orders_disabled=True",
        PROBE_CODE,
        sorted(str(value) for value in TARGET_DATES),
    )
