# -*- coding: utf-8 -*-
"""Temporary PTrade IOPV capability probe.

This file is isolated from the cross-signal strategy, places no orders, and
must not be used to define a premium threshold. Run it in a separate PTrade
simulation/live strategy for one session, then stop it after collecting only
the ``[ptrade-iopv-...]`` log lines.
"""

import datetime
import math


QDII_CODES = [
    "513100.SS",
    "513500.SS",
    "513880.SS",
    "513050.SS",
]


def _current_dt(context):
    value = getattr(context, "current_dt", None)
    if value is not None:
        return value
    blotter = getattr(context, "blotter", None)
    return getattr(blotter, "current_dt", None)


def _positive_finite(value):
    try:
        number = float(value)
        return number > 0 and math.isfinite(number)
    except (TypeError, ValueError):
        return False


def _snapshot_age_seconds(raw_timestamp, callback_dt):
    if raw_timestamp in (None, "") or callback_dt is None:
        return None
    digits = "".join(ch for ch in str(raw_timestamp) if ch.isdigit())
    if len(digits) < 14:
        return None
    try:
        snapshot_dt = datetime.datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
        return (callback_dt - snapshot_dt).total_seconds()
    except (TypeError, ValueError):
        return None


def _mapping_record(mapping, code):
    if not isinstance(mapping, dict):
        return {}
    value = mapping.get(code, {})
    return value if isinstance(value, dict) else {}


def probe_iopv_capability(context):
    callback_dt = _current_dt(context)
    try:
        snapshots = get_snapshot(QDII_CODES)
    except Exception as exc:
        log.info(
            "[ptrade-iopv-snapshot-error] dt=%s type=%s error=%s"
            % (callback_dt, exc.__class__.__name__, exc)
        )
        snapshots = {}

    try:
        etf_records = get_etf_info(QDII_CODES)
    except Exception as exc:
        log.info(
            "[ptrade-iopv-etf-info-error] dt=%s type=%s error=%s"
            % (callback_dt, exc.__class__.__name__, exc)
        )
        etf_records = {}

    for code in QDII_CODES:
        snapshot = _mapping_record(snapshots, code)
        etf_info = _mapping_record(etf_records, code)
        timestamp = snapshot.get("hsTimeStamp")
        iopv = snapshot.get("iopv")
        log.info(
            "[ptrade-iopv-snapshot] dt=%s code=%s present=%s "
            "last_px=%s iopv=%s iopv_positive=%s hsTimeStamp=%s "
            "age_seconds=%s trade_status=%s business_amount=%s business_count=%s"
            % (
                callback_dt,
                code,
                bool(snapshot),
                snapshot.get("last_px"),
                iopv,
                _positive_finite(iopv),
                timestamp,
                _snapshot_age_seconds(timestamp, callback_dt),
                snapshot.get("trade_status"),
                snapshot.get("business_amount"),
                snapshot.get("business_count"),
            )
        )
        log.info(
            "[ptrade-iopv-etf-info] dt=%s code=%s present=%s publish=%s "
            "nav_pre=%s nav_percu=%s"
            % (
                callback_dt,
                code,
                bool(etf_info),
                etf_info.get("publish"),
                etf_info.get("nav_pre"),
                etf_info.get("nav_percu"),
            )
        )


def initialize(context):
    try:
        set_universe(QDII_CODES)
    except Exception as exc:
        log.info(
            "[ptrade-iopv-init-error] type=%s error=%s"
            % (exc.__class__.__name__, exc)
        )

    run_daily(context, probe_iopv_capability, time="09:34")
    run_daily(context, probe_iopv_capability, time="09:35")
    run_daily(context, probe_iopv_capability, time="09:36")
    log.info(
        "[ptrade-iopv-init] codes=%s callbacks=09:34,09:35,09:36 "
        "orders_disabled=True" % QDII_CODES
    )
