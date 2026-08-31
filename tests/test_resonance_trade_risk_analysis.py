import hashlib
import importlib.util
import json
import pathlib

import pytest


ANALYZER_PATH = (
    pathlib.Path(__file__).parents[1]
    / "resonance_reversal_strategy"
    / "research"
    / "analyze_resonance_trade_risk.py"
)

spec = importlib.util.spec_from_file_location(
    "resonance_trade_risk_analyzer", ANALYZER_PATH,
)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)

FIXTURE_SESSIONS = (
    "2018-12-28", "2019-01-02", "2019-01-03", "2019-01-04",
    "2019-01-07", "2019-01-08", "2019-01-09",
)
COUNTERFACTUAL_SESSIONS = (
    "2018-12-28", "2019-01-02", "2019-01-03", "2019-01-04",
    "2019-01-07", "2019-01-08", "2019-01-09", "2019-01-10",
    "2019-01-11", "2019-01-14", "2019-01-15", "2019-01-16",
    "2019-01-17", "2019-01-18", "2019-01-21", "2019-01-22",
    "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-28",
    "2019-01-29", "2019-01-30", "2019-01-31",
)


def _manifest_bytes(sessions):
    return json.dumps({
        "schema_version": 2,
        "market": "XSHG",
        "calendar_coverage_start": "2018-01-01",
        "calendar_coverage_end": "2021-12-31",
        "evaluation_start": "2019-01-01",
        "evaluation_end": "2021-12-31",
        "source": "JoinQuant get_all_trade_days",
        "sessions": sessions,
    }, sort_keys=True).encode("utf-8")


def _validated_manifest(sessions):
    raw = _manifest_bytes(sessions)
    return analyzer.validate_session_calendar_manifest(
        raw, hashlib.sha256(raw).hexdigest(),
    )


def _line(timestamp, payload):
    return "%s - INFO  - %s" % (
        timestamp,
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )


def _fill_line(timestamp, code, action, price, amount, commission):
    return (
        "%s - INFO  - order StockOrder(security=%s action=%s) "
        "trade price: %s, amount:%s, commission: %s"
    ) % (timestamp, code, action, price, amount, commission)


def _relative_id(direction, signal_date, branch="SOFT_ALL_THREE"):
    sources = {
        "BOLL": "RELATIVE", "KDJ": "RELATIVE", "RSI": "RELATIVE",
    }
    if branch == "HARD_BOLL_SOFT_OSC":
        sources["BOLL"] = "HARD"
    parts = ["RELATIVE", branch, direction, "159928.XSHE"]
    for indicator in ("BOLL", "KDJ", "RSI"):
        parts.append("%s:%s:%s" % (
            indicator, sources[indicator], signal_date,
        ))
    return "RELATIVE:" + hashlib.sha256(
        "|".join(parts).encode("utf-8")
    ).hexdigest()[:20]


def _initialization(timestamp="2019-01-02 00:00:00"):
    return _line(timestamp, {
        "event": "strategy_initialized",
        "build": "20260828.5",
        "parameter_fingerprint": "e1227fbd8b4a884e",
        "pool_fingerprint": "9123995edeb1ed84",
        "event_logic_fingerprint": "1c0b8a22f48c97c3",
        "relative_observation_fingerprint": "f47d32b87be6d926",
        "atr_exit_policy": "OBSERVE_ONLY",
        "relative_buy_policy": "EMPTY_SLOT_BACKFILL",
    })


def _relative_observation(timestamp, observation_id, direction="BUY_TURN"):
    signal_date = {
        "2019-01-02": "2018-12-28",
        "2019-01-03": "2019-01-02",
    }[timestamp[:10]]
    return _line(timestamp, {
        "event": "relative_resonance_observation",
        "relative_observation_id": observation_id,
        "code": "159928.XSHE",
        "direction": direction,
        "branch": "SOFT_ALL_THREE",
        "supporters": ["BOLL", "KDJ", "RSI"],
        "signal_date": signal_date,
        "supporter_event_dates": {
            "BOLL": signal_date,
            "KDJ": signal_date,
            "RSI": signal_date,
        },
        "hard_or_relative_source_by_indicator": {
            "BOLL": "RELATIVE", "KDJ": "RELATIVE", "RSI": "RELATIVE",
        },
        "observation_kind": "RELATIVE_RESONANCE",
        "expires_date": timestamp[:10],
        "event_close": 10.0,
        "build": "20260828.5",
        "parameter_fingerprint": "e1227fbd8b4a884e",
        "pool_fingerprint": "9123995edeb1ed84",
        "event_logic_fingerprint": "1c0b8a22f48c97c3",
        "relative_observation_fingerprint": "f47d32b87be6d926",
    })


def _sorted_relative_buy(timestamp, observation_id):
    return _line(timestamp, {
        "event": "resonance_decision",
        "accepted": True,
        "code": "159928.XSHE",
        "direction": "BUY_TURN",
        "reason": "RELATIVE_BUY_CANDIDATE_SORTED:1",
        "resonance_id": observation_id,
        "signal_date": "2018-12-28",
        "supporters": ["BOLL", "KDJ", "RSI"],
        "support_count": 3,
        "boll_age": 0,
    })


def _signal_snapshot(
        timestamp, code="159928.XSHE", signal_date="2018-12-28",
        close=10.0, atr14=0.5, rsi14=30.0, adx14=25.0,
        boll_width=0.1, boll_mid_slope=0.2, volume_ratio=1.1):
    return _line(timestamp, {
        "event": "signal_snapshot",
        "version": "resonance-v0.1.0",
        "build": "20260828.5",
        "parameter_fingerprint": "e1227fbd8b4a884e",
        "pool_fingerprint": "9123995edeb1ed84",
        "event_logic_fingerprint": "1c0b8a22f48c97c3",
        "relative_observation_fingerprint": "f47d32b87be6d926",
        "code": code,
        "decision_date": timestamp[:10],
        "signal_date": signal_date,
        "valid": True,
        "trade_values": {
            "atr14": atr14,
            "rsi14": rsi14,
        },
        "observation_values": {
            "adx14": adx14,
            "boll_width": boll_width,
            "boll_mid_slope": boll_mid_slope,
            "volume_ratio": volume_ratio,
        },
        "event_detection_trace": {
            "boll": {"current": {"close": close}},
        },
        "kdj_cross": "NONE",
        "active_events": {},
        "invalidated_events": [],
        "relative_active_events": {},
        "relative_invalidated_events": [],
    })


def _portfolio(timestamp, total_value, cash, positions):
    return _line(timestamp, {
        "event": "portfolio_summary",
        "closing_date": timestamp[:10],
        "total_value": total_value,
        "available_cash": cash,
        "positions": positions,
        "highest_close_anchors": {},
    })


def _ordinary_lines():
    buy_id = _relative_id("BUY_TURN", "2018-12-28")
    sell_id = _relative_id("SELL_TURN", "2019-01-02")
    return [
        _initialization(),
        _signal_snapshot("2019-01-02 09:35:00"),
        _relative_observation("2019-01-02 09:35:00", buy_id),
        _sorted_relative_buy("2019-01-02 09:35:00", buy_id),
        _fill_line(
            "2019-01-02 09:35:00", "159928.XSHE", "open", 10.0, 100, 5.0,
        ),
        _line("2019-01-02 09:35:00", {
            "event": "order_transition", "code": "159928.XSHE",
            "side": "BUY", "outcome": "FILLED",
            "before_amount": 0, "after_amount": 100,
        }),
        _portfolio("2019-01-02 15:30:00", 20000.0, 19000.0,
                   {"159928.XSHE": 100}),
        _line("2019-01-03 09:35:00", {
            "event": "atr_check", "code": "159928.XSHE",
            "current_price": 5.5, "execution_policy": "OBSERVE_ONLY",
            "order_submitted": False,
        }),
        _relative_observation(
            "2019-01-03 09:35:00", sell_id, direction="SELL_TURN",
        ),
        _portfolio("2019-01-03 15:30:00", 20100.0, 19000.0,
                   {"159928.XSHE": 200}),
        _line("2019-01-04 09:35:00", {
            "event": "atr_check", "code": "159928.XSHE",
            "current_price": 6.0, "execution_policy": "OBSERVE_ONLY",
            "order_submitted": False,
        }),
        _fill_line(
            "2019-01-04 09:35:00", "159928.XSHE", "close", 6.0, 200, 5.0,
        ),
        _line("2019-01-04 09:35:00", {
            "event": "order_transition", "code": "159928.XSHE",
            "side": "SELL", "outcome": "FILLED",
            "before_amount": 200, "after_amount": 0,
            "exit_reason": "SIGNAL_EXIT",
        }),
        _portfolio("2019-01-04 15:30:00", 20190.0, 20190.0, {}),
        _portfolio("2019-01-07 15:30:00", 20190.0, 20190.0, {}),
        _portfolio("2019-01-08 15:30:00", 20190.0, 20190.0, {}),
        _line("2019-01-09 15:30:00", {
            "event": "observation_outcome",
            "relative_observation_id": sell_id, "resonance_id": sell_id,
            "observation_kind": "RELATIVE_RESONANCE",
            "code": "159928.XSHE", "direction": "SELL_TURN",
            "branch": "SOFT_ALL_THREE", "horizon": 5,
            "event_date": "2019-01-02",
            "supporters": ["BOLL", "KDJ", "RSI"],
            "build": "20260828.5",
            "relative_observation_fingerprint": "f47d32b87be6d926",
            "outcome": {
                "status": "RECORDED", "return": -0.03,
                "direction_adjusted_return": 0.03,
                "closing_price": 9.7, "closing_date": "2019-01-09",
            },
        }),
        _portfolio("2019-01-09 15:30:00", 20190.0, 20190.0, {}),
    ]


def _double_lines():
    lines = _ordinary_lines()
    return [
        line.replace("amount:100, commission: 5.0",
                     "amount:99, commission: 5.0")
            .replace("amount:200, commission: 5.0",
                     "amount:198, commission: 5.0")
            .replace('"after_amount": 100', '"after_amount": 99')
            .replace('"before_amount": 200', '"before_amount": 198')
            .replace('"159928.XSHE": 100', '"159928.XSHE": 99')
            .replace('"159928.XSHE": 200', '"159928.XSHE": 198')
        for line in lines
    ]


def _counterfactual_lines(
        codes, amount=100, anchor=10.0, commission_rate=0.0003):
    lines = [_initialization()]
    buy_commission = max(5.0, 10.0 * amount * commission_rate)
    sell_commission = max(5.0, 8.0 * amount * commission_rate)
    for index, code in enumerate(codes):
        lines.append(_signal_snapshot(
            "2019-01-02 09:35:00", code=code,
        ))
        lines.append(_line("2019-01-02 09:35:00", {
            "event": "resonance_decision",
            "accepted": True,
            "code": code,
            "direction": "BUY_TURN",
            "reason": "BUY_CANDIDATE_SORTED:%s" % (index + 1),
            "resonance_id": "FORMAL:%s" % code,
            "signal_date": "2018-12-28",
            "supporters": ["BOLL", "RSI"],
        }))
        lines.append(_fill_line(
            "2019-01-02 09:35:00", code, "open", 10.0, amount,
            buy_commission,
        ))
        lines.append(_line("2019-01-02 09:35:00", {
            "event": "order_transition", "code": code,
            "side": "BUY", "outcome": "FILLED",
            "before_amount": 0, "after_amount": amount,
        }))
    positions = {code: amount for code in codes}
    lines.append(_portfolio(
        "2019-01-02 15:30:00", 20000.0, 17000.0, positions,
    ))
    for session in COUNTERFACTUAL_SESSIONS[2:-1]:
        current_price = 9.0 if session == "2019-01-30" else 9.5
        for code in codes:
            lines.append(_line(session + " 09:35:00", {
                "event": "atr_check", "code": code,
                "current_price": current_price,
                "highest_close_anchor": anchor,
                "execution_policy": "OBSERVE_ONLY",
                "order_submitted": False,
            }))
        lines.append(_portfolio(
            session + " 15:30:00", 20000.0, 17000.0, positions,
        ))
    for code in codes:
        lines.append(_line("2019-01-31 09:35:00", {
            "event": "atr_check", "code": code,
            "current_price": 8.0,
            "highest_close_anchor": anchor,
            "execution_policy": "OBSERVE_ONLY",
            "order_submitted": False,
        }))
        lines.append(_fill_line(
            "2019-01-31 09:35:00", code, "close", 8.0, amount,
            sell_commission,
        ))
        lines.append(_line("2019-01-31 09:35:00", {
            "event": "order_transition", "code": code,
            "side": "SELL", "outcome": "FILLED",
            "before_amount": amount, "after_amount": 0,
            "exit_reason": "SIGNAL_EXIT",
        }))
    lines.append(_portfolio(
        "2019-01-31 15:30:00", 19000.0, 19000.0, {},
    ))
    return lines


def _write_log(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_split_safe_cash_flow_and_risk_path_keep_relative_entry_identity(tmp_path):
    ordinary_path = _write_log(tmp_path / "ordinary.log", _ordinary_lines())
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    report = analyzer.analyze_paths(
        [ordinary_path], [double_path], manifest,
    )

    assert report["source_files"] == {
        "ordinary": [{
            "path": str(ordinary_path.resolve()),
            "sha256": hashlib.sha256(ordinary_path.read_bytes()).hexdigest(),
        }],
        "double_friction": [{
            "path": str(double_path.resolve()),
            "sha256": hashlib.sha256(double_path.read_bytes()).hexdigest(),
        }],
    }
    assert report["path_reconciliation"] == {
        "fill_count": 2,
        "identity_match": True,
        "amount_difference_count": 2,
    }
    assert report["trade_summary"]["closed_count"] == 1
    trade = report["trades"][0]
    assert trade["entry_source"] == "RELATIVE"
    assert trade["entry_branch"] == "SOFT_ALL_THREE"
    assert trade["pnl"] == pytest.approx(190.0)
    assert trade["return_rate"] == pytest.approx(190.0 / 1005.0)
    assert trade["amount_ratio"] == pytest.approx(2.0)
    assert trade["mfe"] == pytest.approx(190.0 / 1005.0)
    assert trade["mae"] == pytest.approx(0.0)
    assert trade["max_profit_giveback"] == pytest.approx(0.0)
    assert trade["longest_underwater_sessions"] == 0
    assert trade["relative_sell_observation_count"] == 1
    assert report["relative_sell_diagnostics"]["horizon_5_sell_hit_rate"] == 1.0


def test_training_boundary_rejects_portfolio_or_fill_outside_manifest(tmp_path):
    ordinary = _ordinary_lines()
    ordinary[-1] = ordinary[-1].replace("2019-01-09", "2022-01-04")
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="ordinary.*outside training window"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_training_boundary_rejects_nested_outcome_date_after_2021(tmp_path):
    ordinary = [
        line.replace('"closing_date": "2019-01-09"',
                     '"closing_date": "2022-01-04"')
        if '"event": "observation_outcome"' in line else line
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(
            ValueError, match="ordinary observation outcome outside training window"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


@pytest.mark.parametrize("field", [
    "decision_date", "signal_date", "event_date", "expires_date",
])
def test_training_boundary_rejects_future_structured_date_fields(
        tmp_path, field):
    ordinary = _ordinary_lines()
    payload = {
        "event": "resonance_decision",
        "accepted": False,
        "code": "510300.XSHG",
        "direction": "BUY_TURN",
        "reason": "PORTFOLIO_FULL",
        field: "2022-01-04",
    }
    ordinary.insert(1, _line("2019-01-02 09:35:00", payload))
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="ordinary.*outside manifest coverage"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_training_boundary_rejects_future_supporter_date(tmp_path):
    ordinary = [
        line.replace('"RSI": "2019-01-02"', '"RSI": "2022-01-04"')
        if '"event": "relative_resonance_observation"' in line else line
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="ordinary.*outside manifest coverage"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_wrong_build_or_policy_fails_closed(tmp_path):
    ordinary = [
        line.replace('"build": "20260828.5"', '"build": "20260828.4"')
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="ordinary build"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_double_friction_signal_path_drift_is_rejected(tmp_path):
    ordinary_path = _write_log(tmp_path / "ordinary.log", _ordinary_lines())
    double = [
        line.replace("security=159928.XSHE action=close",
                     "security=510300.XSHG action=close")
        for line in _double_lines()
    ]
    double_path = _write_log(tmp_path / "double.log", double)
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="friction.*path.*differ"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_cli_rejects_output_alias_before_overwriting_input(tmp_path):
    ordinary_path = _write_log(tmp_path / "ordinary.log", _ordinary_lines())
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    raw_manifest = _manifest_bytes([
        "2019-01-02", "2019-01-03", "2019-01-04",
    ])
    manifest_path = tmp_path / "calendar.json"
    manifest_path.write_bytes(raw_manifest)
    before = ordinary_path.read_bytes()

    with pytest.raises(ValueError, match="output aliases an input"):
        analyzer.main([
            "--ordinary-log", str(ordinary_path),
            "--double-friction-log", str(double_path),
            "--session-calendar", str(manifest_path),
            "--session-calendar-sha256", hashlib.sha256(raw_manifest).hexdigest(),
            "--output", str(ordinary_path),
        ])

    assert ordinary_path.read_bytes() == before


def test_relative_sell_diagnostics_include_open_positions_at_event_time(tmp_path):
    ordinary = [
        line for line in _ordinary_lines()
        if not (
            "2019-01-04 09:35:00" in line
            and ("action=close" in line or '"side": "SELL"' in line)
        )
    ]
    ordinary = [
        line.replace('"positions": {}',
                     '"positions": {"159928.XSHE": 200}')
        if '"event": "portfolio_summary"' in line else line
        for line in ordinary
    ]
    double = [
        line for line in _double_lines()
        if not (
            "2019-01-04 09:35:00" in line
            and ("action=close" in line or '"side": "SELL"' in line)
        )
    ]
    double = [
        line.replace('"positions": {}',
                     '"positions": {"159928.XSHE": 198}')
        if '"event": "portfolio_summary"' in line else line
        for line in double
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", double)
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    report = analyzer.analyze_paths(
        [ordinary_path], [double_path], manifest,
    )

    assert report["trade_summary"]["closed_count"] == 0
    assert report["trade_summary"]["open_count"] == 1
    assert report["relative_sell_diagnostics"] == {
        "held_observation_count": 1,
        "horizon_5_count": 1,
        "horizon_5_sell_hit_rate": 1.0,
        "horizon_5_mean_forward_return": -0.03,
    }


def test_duplicate_relative_registration_is_rejected(tmp_path):
    ordinary = _ordinary_lines()
    duplicate = _relative_observation(
        "2019-01-02 09:35:00",
        _relative_id("BUY_TURN", "2018-12-28"),
    ).replace('"branch": "SOFT_ALL_THREE"',
              '"branch": "HARD_BOLL_SOFT_OSC"')
    ordinary.insert(3, duplicate)
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="duplicate relative observation id"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_relative_buy_registration_must_match_decision_metadata(tmp_path):
    ordinary = [
        line.replace('"signal_date": "2018-12-28"',
                     '"signal_date": "2019-01-02"')
        if 'RELATIVE_BUY_CANDIDATE_SORTED' in line else line
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="relative buy metadata mismatch"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_filled_buy_requires_accepted_buy_decision(tmp_path):
    ordinary = [
        line.replace('"accepted": true', '"accepted": false')
        if 'RELATIVE_BUY_CANDIDATE_SORTED' in line else line
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="sorted buy decision is invalid"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_fill_order_transition_and_portfolio_ledgers_must_reconcile(tmp_path):
    ordinary = [
        line for line in _ordinary_lines()
        if not ('"event": "order_transition"' in line
                and '"side": "BUY"' in line)
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="ordinary fill and order paths differ"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_portfolio_position_set_must_match_canonical_active_ledger(tmp_path):
    ordinary = [
        line.replace('"positions": {"159928.XSHE": 200}',
                     '"positions": {"510300.XSHG": 200}')
        if '2019-01-03 15:30:00' in line else line
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="ordinary portfolio position set mismatch"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_missing_held_session_atr_mark_is_rejected(tmp_path):
    ordinary = [
        line for line in _ordinary_lines()
        if not ('2019-01-03 09:35:00' in line
                and '"event": "atr_check"' in line)
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="missing atr_check mark"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_exit_session_counts_once_in_underwater_run(tmp_path):
    ordinary = [
        line.replace("trade price: 6.0, amount:200",
                     "trade price: 4.0, amount:200")
            .replace('"current_price": 6.0', '"current_price": 4.0')
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    report = analyzer.analyze_paths(
        [ordinary_path], [double_path], manifest,
    )

    assert report["trades"][0]["longest_underwater_sessions"] == 1


def test_relative_registration_requires_full_fingerprint_contract(tmp_path):
    ordinary = [
        line.replace('"parameter_fingerprint": "e1227fbd8b4a884e"',
                     '"parameter_fingerprint": "wrong"')
        if '"event": "relative_resonance_observation"' in line else line
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="relative observation metadata"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


@pytest.mark.parametrize("replacement", [
    ('"code": "159928.XSHE"', '"code": "510300.XSHG"'),
    ('"closing_price": 9.7', '"closing_price": 9.8'),
])
def test_relative_outcome_identity_and_return_are_reconciled(
        tmp_path, replacement):
    old, new = replacement
    ordinary = [
        line.replace(old, new)
        if '"event": "observation_outcome"' in line else line
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="relative outcome"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_orphan_relative_outcome_is_rejected(tmp_path):
    ordinary = [
        line.replace(
            _relative_id("SELL_TURN", "2019-01-02"),
            "RELATIVE:00000000000000000000",
        ) if '"event": "observation_outcome"' in line else line
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="orphan relative outcome"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_filled_transition_before_amount_must_match_active_ledger(tmp_path):
    ordinary = [
        line.replace("amount:200, commission: 5.0",
                     "amount:198, commission: 5.0")
            .replace('"before_amount": 200', '"before_amount": 198')
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="ordinary order before amount mismatch"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_retry_exit_does_not_require_exit_session_atr_mark(tmp_path):
    ordinary = [
        line for line in _ordinary_lines()
        if not ('2019-01-04 09:35:00' in line
                and '"event": "atr_check"' in line)
    ]
    double = [
        line for line in _double_lines()
        if not ('2019-01-04 09:35:00' in line
                and '"event": "atr_check"' in line)
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", double)
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    report = analyzer.analyze_paths(
        [ordinary_path], [double_path], manifest,
    )

    assert report["trade_summary"]["closed_count"] == 1


def test_relative_registration_requires_previous_manifest_signal(tmp_path):
    old_id = _relative_id("BUY_TURN", "2018-12-28")
    new_id = _relative_id("BUY_TURN", "2019-01-02")
    ordinary = []
    for line in _ordinary_lines():
        if ('"event": "relative_resonance_observation"' in line
                and '"direction": "BUY_TURN"' in line):
            line = line.replace("2018-12-28", "2019-01-02")
        if "RELATIVE_BUY_CANDIDATE_SORTED" in line:
            line = line.replace('"signal_date": "2018-12-28"',
                                '"signal_date": "2019-01-02"')
        ordinary.append(line.replace(old_id, new_id))
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="previous manifest session"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_relative_registration_expiry_is_supporter_derived(tmp_path):
    ordinary = [
        line.replace('"expires_date": "2019-01-02"',
                     '"expires_date": "2019-01-03"')
        if ('"event": "relative_resonance_observation"' in line
            and '"direction": "BUY_TURN"' in line) else line
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="relative observation expiry"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_relative_outcome_closing_session_matches_horizon(tmp_path):
    ordinary = [
        line.replace("2019-01-09 15:30:00", "2019-01-08 15:30:00")
            .replace('"closing_date": "2019-01-09"',
                     '"closing_date": "2019-01-08"')
        if '"event": "observation_outcome"' in line else line
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="relative outcome closing session"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_relative_outcome_log_date_matches_closing_session(tmp_path):
    ordinary = [
        line.replace("2019-01-09 15:30:00", "2019-01-08 15:30:00")
        if '"event": "observation_outcome"' in line else line
        for line in _ordinary_lines()
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match="relative outcome log date"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


@pytest.mark.parametrize("status", ["HORIZON_MISSED", "PRICE_UNAVAILABLE"])
def test_non_recorded_terminal_outcomes_are_valid_but_not_numeric(
        tmp_path, status):
    def terminal_lines(lines):
        result = []
        for line in lines:
            if '"event": "observation_outcome"' not in line:
                result.append(line)
                continue
            timestamp, raw = line.split(" - INFO  - ", 1)
            record = json.loads(raw)
            record["outcome"] = {
                "status": status,
                "closing_date": "2019-01-09",
                "closing_price": None,
                "return": None,
            }
            result.append(_line(timestamp, record))
        return result

    ordinary_path = _write_log(
        tmp_path / "ordinary.log", terminal_lines(_ordinary_lines()),
    )
    double_path = _write_log(
        tmp_path / "double.log", terminal_lines(_double_lines()),
    )
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    report = analyzer.analyze_paths(
        [ordinary_path], [double_path], manifest,
    )

    assert report["relative_sell_diagnostics"]["held_observation_count"] == 1
    assert report["relative_sell_diagnostics"]["horizon_5_count"] == 0


def test_non_recovery_counterfactual_uses_prior_anchor_at_session_20(
        tmp_path):
    code = "159928.XSHE"
    ordinary_path = _write_log(
        tmp_path / "ordinary.log", _counterfactual_lines([code]),
    )
    double_path = _write_log(
        tmp_path / "double.log", _counterfactual_lines([code]),
    )
    manifest = _validated_manifest(COUNTERFACTUAL_SESSIONS)

    report = analyzer.analyze_paths(
        [ordinary_path], [double_path], manifest,
    )

    counterfactual = report["non_recovery_counterfactual"]
    assert counterfactual["rule"] == {
        "completed_holding_sessions": 20,
        "qualification": "PRIOR_HIGHEST_CLOSE_NOT_ABOVE_ENTRY_PRICE",
        "execution": "DECISION_SESSION_0935_ATR_CHECK_PRICE",
        "path_assumption": "ORIGINAL_TRADE_PATH_FIXED",
    }
    ordinary = counterfactual["ordinary"]
    assert ordinary["commission_rate"] == pytest.approx(0.0003)
    assert ordinary["triggered_closed_count"] == 1
    assert ordinary["improved_trade_count"] == 1
    assert ordinary["actual_closed_pnl"] == pytest.approx(-210.0)
    assert ordinary["counterfactual_closed_pnl"] == pytest.approx(-110.0)
    assert ordinary["pnl_delta"] == pytest.approx(100.0)
    assert ordinary["actual_wins"] == 0
    assert ordinary["counterfactual_wins"] == 0
    assert ordinary["actual_worst_trade_pnl"] == pytest.approx(-210.0)
    assert ordinary["counterfactual_worst_trade_pnl"] == pytest.approx(-110.0)
    assert ordinary["gate"]["at_least_three_improved_trades"] is False
    assert ordinary["gate"]["passed"] is False
    assert ordinary["rows"] == [{
        "code": code,
        "entry_date": "2019-01-02",
        "decision_date": "2019-01-30",
        "actual_exit_date": "2019-01-31",
        "entry_source": "FORMAL",
        "entry_branch": None,
        "entry_price": 10.0,
        "prior_highest_close_anchor": 10.0,
        "execution_price": 9.0,
        "execution_amount": 100,
        "execution_commission": 5.0,
        "actual_pnl": pytest.approx(-210.0),
        "counterfactual_pnl": pytest.approx(-110.0),
        "pnl_delta": pytest.approx(100.0),
        "actual_winner": False,
        "counterfactual_winner": False,
    }]


def test_non_recovery_gate_requires_three_distributed_improvements(
        tmp_path):
    codes = ["159928.XSHE", "510300.XSHG", "513050.XSHG"]
    ordinary_path = _write_log(
        tmp_path / "ordinary.log", _counterfactual_lines(codes, amount=100),
    )
    double_path = _write_log(
        tmp_path / "double.log", _counterfactual_lines(codes, amount=90),
    )
    manifest = _validated_manifest(COUNTERFACTUAL_SESSIONS)

    report = analyzer.analyze_paths(
        [ordinary_path], [double_path], manifest,
    )

    counterfactual = report["non_recovery_counterfactual"]
    assert counterfactual["ordinary"]["pnl_delta"] == pytest.approx(300.0)
    assert counterfactual["double_friction"]["pnl_delta"] == pytest.approx(
        270.0,
    )
    assert counterfactual["ordinary"]["gate"]["passed"] is True
    assert counterfactual["double_friction"]["gate"]["passed"] is True
    assert counterfactual["decision"] == {
        "ordinary_passed": True,
        "double_friction_passed": True,
        "proceed_to_strategy_candidate": True,
    }


def test_non_recovery_does_not_trigger_after_prior_close_above_entry(
        tmp_path):
    code = "159928.XSHE"
    ordinary_path = _write_log(
        tmp_path / "ordinary.log",
        _counterfactual_lines([code], anchor=10.01),
    )
    double_path = _write_log(
        tmp_path / "double.log",
        _counterfactual_lines([code], anchor=10.01),
    )
    manifest = _validated_manifest(COUNTERFACTUAL_SESSIONS)

    report = analyzer.analyze_paths(
        [ordinary_path], [double_path], manifest,
    )

    ordinary = report["non_recovery_counterfactual"]["ordinary"]
    assert ordinary["triggered_closed_count"] == 0
    assert ordinary["counterfactual_closed_pnl"] == pytest.approx(-210.0)


def test_non_recovery_uses_distinct_large_order_commission_rates(tmp_path):
    code = "159928.XSHE"
    ordinary_path = _write_log(
        tmp_path / "ordinary.log",
        _counterfactual_lines(
            [code], amount=10000, commission_rate=0.0003,
        ),
    )
    double_path = _write_log(
        tmp_path / "double.log",
        _counterfactual_lines(
            [code], amount=10000, commission_rate=0.0006,
        ),
    )
    manifest = _validated_manifest(COUNTERFACTUAL_SESSIONS)

    report = analyzer.analyze_paths(
        [ordinary_path], [double_path], manifest,
    )["non_recovery_counterfactual"]

    ordinary = report["ordinary"]
    double = report["double_friction"]
    assert ordinary["commission_rate"] == pytest.approx(0.0003)
    assert double["commission_rate"] == pytest.approx(0.0006)
    assert ordinary["rows"][0]["execution_commission"] == pytest.approx(27.0)
    assert double["rows"][0]["execution_commission"] == pytest.approx(54.0)
    assert ordinary["rows"][0]["counterfactual_pnl"] == pytest.approx(
        -10057.0,
    )
    assert double["rows"][0]["counterfactual_pnl"] == pytest.approx(
        -10114.0,
    )


@pytest.mark.parametrize("old,new", [
    ("2019-01-30 09:35:00", "2019-01-30 09:36:00"),
    ('"execution_policy": "OBSERVE_ONLY"',
     '"execution_policy": "EXECUTE"'),
    ('"order_submitted": false', '"order_submitted": true'),
])
def test_non_recovery_rejects_non_0935_observation_evidence(
        tmp_path, old, new):
    code = "159928.XSHE"
    ordinary = [
        line.replace(old, new)
        if ('2019-01-30' in line and '"event": "atr_check"' in line)
        else line
        for line in _counterfactual_lines([code])
    ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(
        tmp_path / "double.log", _counterfactual_lines([code]),
    )
    manifest = _validated_manifest(COUNTERFACTUAL_SESSIONS)

    with pytest.raises(ValueError, match="atr_check identity is invalid"):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def test_entry_quality_links_only_t_minus_one_snapshot_features(tmp_path):
    ordinary_path = _write_log(tmp_path / "ordinary.log", _ordinary_lines())
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    report = analyzer.analyze_paths(
        [ordinary_path], [double_path], manifest,
    )

    attribution = report["entry_quality_attribution"]
    assert attribution["scope"] == {
        "processing_stage": "POST_BACKTEST_READ_ONLY_ATTRIBUTION",
        "path_assumption": "ORIGINAL_TRADE_PATH_FIXED",
        "strategy_behavior_changed": False,
        "open_positions_excluded": True,
    }
    assert attribution["feature_contract"] == {
        "categorical": [
            "entry_source", "entry_branch", "supporters",
            "entry_rank", "entry_market_state", "code",
        ],
        "continuous": [
            "rsi14", "adx14", "atr_to_close", "boll_width",
            "volume_ratio", "normalized_boll_mid_slope",
        ],
        "forbidden_post_entry_predictors": [
            "mfe", "mae", "max_profit_giveback",
            "longest_underwater_sessions",
        ],
    }
    assert attribution["semantic_boundaries"] == {
        "normalized_boll_mid_slope": {
            "positive": "> 0",
            "nonpositive": "<= 0",
        },
        "volume_ratio": {
            "above_one": "> 1.0",
            "at_or_below_one": "<= 1.0",
        },
        "threshold_search_performed": False,
    }
    marginal = attribution["volume_ratio_marginal"]
    assert marginal["scope"]["strategy_behavior_changed"] is False
    assert marginal["scope"]["rule_candidate_created"] is False
    assert marginal["ordinary"]["groups"][
        "VOLUME_ABOVE_ONE"
    ]["count"] == 1
    assert marginal["ordinary"]["comparison_available"] is False
    assert marginal["cross_friction_stability"][
        "all_reported_directions_match"
    ] is None
    ordinary = attribution["ordinary"]
    assert ordinary["closed_count"] == 1
    assert ordinary["wins"] == 1
    assert ordinary["losses"] == 0
    assert ordinary["categorical"]["entry_rank"]["1"]["count"] == 1
    assert ordinary["categorical"]["supporters"][
        "BOLL+KDJ+RSI"
    ]["count"] == 1
    assert ordinary["categorical"]["entry_market_state"][
        "SLOPE_POSITIVE|VOLUME_ABOVE_ONE"
    ]["count"] == 1
    assert ordinary["continuous"]["atr_to_close"]["winner"] == {
        "count": 1,
        "median": pytest.approx(0.05),
        "q1": pytest.approx(0.05),
        "q3": pytest.approx(0.05),
    }
    assert ordinary["continuous"]["normalized_boll_mid_slope"][
        "winner"
    ]["median"] == pytest.approx(0.02)


@pytest.mark.parametrize("mutation,error", [
    ("missing", "filled buy lacks unique signal snapshot"),
    ("duplicate", "duplicate signal snapshot"),
    ("not_previous_session", "signal snapshot is not T-1"),
])
def test_entry_quality_snapshot_evidence_fails_closed(
        tmp_path, mutation, error):
    ordinary = _ordinary_lines()
    if mutation == "missing":
        ordinary = [
            line for line in ordinary
            if '"event": "signal_snapshot"' not in line
        ]
    elif mutation == "duplicate":
        ordinary.insert(2, _signal_snapshot("2019-01-02 09:35:00"))
    else:
        ordinary = [
            line.replace(
                '"signal_date": "2018-12-28"',
                '"signal_date": "2019-01-02"',
            ) if '"event": "signal_snapshot"' in line else line
            for line in ordinary
        ]
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match=error):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


def _entry_quality_rows(group_count=8):
    rows = []
    for index in range(group_count):
        losing = index % 2 == 0
        year = 2019 if index < 4 else 2020
        rows.append({
            "code": "WEAK",
            "entry_date": "%s-01-%02d" % (year, index % 4 + 2),
            "entry_source": "RELATIVE",
            "entry_branch": "WEAK_BRANCH",
            "supporters": "BOLL+KDJ+RSI",
            "entry_rank": 2,
            "pnl": -10.0 if losing else 1.0,
            "return_rate": -0.1 if losing else 0.01,
            "features": {
                "rsi14": 30.0 + index,
                "adx14": 20.0,
                "atr_to_close": 0.05,
                "boll_width": 0.1,
                "volume_ratio": 1.0,
                "normalized_boll_mid_slope": 0.01,
            },
        })
    for index in range(8):
        rows.append({
            "code": "STRONG",
            "entry_date": "%s-02-%02d" % (
                2019 if index < 4 else 2020, index % 4 + 2,
            ),
            "entry_source": "FORMAL",
            "entry_branch": "NONE",
            "supporters": "BOLL+KDJ",
            "entry_rank": 1,
            "pnl": 10.0,
            "return_rate": 0.1,
            "features": {
                "rsi14": 40.0,
                "adx14": 30.0,
                "atr_to_close": 0.04,
                "boll_width": 0.08,
                "volume_ratio": 1.2,
                "normalized_boll_mid_slope": 0.02,
            },
        })
    return rows


def test_entry_quality_gate_marks_only_stable_distributed_weak_cohort():
    ordinary = analyzer._entry_quality_path(_entry_quality_rows())
    double = analyzer._entry_quality_path(_entry_quality_rows())
    decision = analyzer._entry_quality_candidate_decision(ordinary, double)

    weak = ordinary["categorical"]["entry_branch"]["WEAK_BRANCH"]
    assert weak["count"] == 8
    assert weak["losses"] == 4
    assert weak["win_rate"] == pytest.approx(0.5)
    assert weak["median_return"] == pytest.approx(-0.045)
    assert weak["candidate_gate"] == {
        "at_least_eight_trades": True,
        "at_least_four_losses": True,
        "win_rate_lags_overall_by_ten_points": True,
        "median_return_not_positive": True,
        "pnl_or_concentration_condition": True,
        "stable_in_at_least_two_years": True,
        "passed": True,
    }
    assert {
        "field": "entry_branch", "value": "WEAK_BRANCH",
    } in decision["eligible_groups"]
    assert {
        "field": "entry_market_state",
        "value": "SLOPE_POSITIVE|VOLUME_AT_OR_BELOW_ONE",
    } in decision["eligible_groups"]
    assert decision["proceed_to_counterfactual_design"] is True


def test_entry_quality_gate_rejects_seven_trade_boundary():
    ordinary = analyzer._entry_quality_path(_entry_quality_rows(7))
    double = analyzer._entry_quality_path(_entry_quality_rows(7))
    decision = analyzer._entry_quality_candidate_decision(ordinary, double)

    weak = ordinary["categorical"]["entry_branch"]["WEAK_BRANCH"]
    assert weak["candidate_gate"]["at_least_eight_trades"] is False
    assert weak["candidate_gate"]["passed"] is False
    assert {
        "field": "entry_branch", "value": "WEAK_BRANCH",
    } not in decision["eligible_groups"]


def test_entry_quality_zero_pnl_is_breakeven_not_loss():
    rows = _entry_quality_rows()
    for row in rows:
        if row["entry_branch"] == "WEAK_BRANCH":
            row["pnl"] = 1.0
            row["return_rate"] = 0.01
    for row in rows[:5]:
        row["pnl"] = 0.0
        row["return_rate"] = 0.0

    report = analyzer._entry_quality_path(rows)
    weak = report["categorical"]["entry_branch"]["WEAK_BRANCH"]

    assert weak["wins"] == 3
    assert weak["losses"] == 0
    assert weak["breakeven"] == 5
    assert weak["candidate_gate"]["at_least_four_losses"] is False
    assert weak["candidate_gate"]["passed"] is False
    assert report["continuous"]["rsi14"]["loser"]["count"] == 0
    assert report["continuous"]["rsi14"]["breakeven"]["count"] == 5


@pytest.mark.parametrize("mutation", [
    "after_fill_same_second", "afternoon", "noncanonical_reason",
])
def test_entry_quality_sorted_candidate_is_strictly_prefill_0935_evidence(
        tmp_path, mutation):
    ordinary = _ordinary_lines()
    decision_index = next(
        index for index, line in enumerate(ordinary)
        if "RELATIVE_BUY_CANDIDATE_SORTED" in line
    )
    decision = ordinary.pop(decision_index)
    if mutation == "after_fill_same_second":
        fill_index = next(
            index for index, line in enumerate(ordinary)
            if "action=open" in line
        )
        ordinary.insert(fill_index + 1, decision)
        expected = "sorted buy decision must precede fill"
    elif mutation == "afternoon":
        decision = decision.replace(
            "2019-01-02 09:35:00", "2019-01-02 15:30:00", 1,
        )
        portfolio_index = next(
            index for index, line in enumerate(ordinary)
            if "2019-01-02 15:30:00" in line
        )
        ordinary.insert(portfolio_index, decision)
        expected = "sorted buy decision time is invalid"
    else:
        decision = decision.replace(
            "RELATIVE_BUY_CANDIDATE_SORTED:1",
            "RELATIVE_BUY_CANDIDATE_SORTED:garbage:1",
        )
        ordinary.insert(decision_index, decision)
        expected = "sorted buy rank is invalid"
    ordinary_path = _write_log(tmp_path / "ordinary.log", ordinary)
    double_path = _write_log(tmp_path / "double.log", _double_lines())
    manifest = _validated_manifest(FIXTURE_SESSIONS)

    with pytest.raises(ValueError, match=expected):
        analyzer.analyze_paths([ordinary_path], [double_path], manifest)


@pytest.mark.parametrize("slope,volume_ratio,expected", [
    (0.01, 1.01, "SLOPE_POSITIVE|VOLUME_ABOVE_ONE"),
    (0.01, 1.00, "SLOPE_POSITIVE|VOLUME_AT_OR_BELOW_ONE"),
    (0.00, 1.01, "SLOPE_NONPOSITIVE|VOLUME_ABOVE_ONE"),
    (-0.01, 1.00, "SLOPE_NONPOSITIVE|VOLUME_AT_OR_BELOW_ONE"),
])
def test_entry_market_state_uses_fixed_zero_and_one_boundaries(
        slope, volume_ratio, expected):
    assert analyzer._entry_market_state({
        "normalized_boll_mid_slope": slope,
        "volume_ratio": volume_ratio,
    }) == expected


def test_volume_ratio_marginal_reports_fixed_boundary_and_year_deltas():
    rows = _entry_quality_rows()
    for index, row in enumerate(rows):
        row["features"]["normalized_boll_mid_slope"] = (
            -0.01 if index % 2 == 0 else 0.01
        )

    report = analyzer._volume_ratio_marginal_path(rows)

    assert report["boundary"] == {
        "above_one": "> 1.0",
        "at_or_below_one": "<= 1.0",
        "threshold_search_performed": False,
        "delta_zero_absolute_tolerance": 1e-12,
    }
    below = report["groups"]["VOLUME_AT_OR_BELOW_ONE"]
    above = report["groups"]["VOLUME_ABOVE_ONE"]
    assert below["count"] == 8
    assert below["wins"] == 4
    assert below["win_rate"] == pytest.approx(0.5)
    assert above["count"] == 8
    assert above["wins"] == 8
    assert above["win_rate"] == pytest.approx(1.0)
    assert report["comparison_available"] is True
    assert report["overall_delta_at_or_below_minus_above"] == {
        "win_rate": pytest.approx(-0.5),
        "pnl": pytest.approx(-116.0),
        "median_return": pytest.approx(-0.145),
    }
    assert report["by_entry_year_delta_at_or_below_minus_above"] == {
        "2019": {
            "win_rate": pytest.approx(-0.5),
            "pnl": pytest.approx(-58.0),
        },
        "2020": {
            "win_rate": pytest.approx(-0.5),
            "pnl": pytest.approx(-58.0),
        },
    }
    assert report["cross_year_direction_stability"] == {
        "win_rate": True,
        "pnl": True,
    }


def test_volume_ratio_marginal_marks_mixed_year_directions_unstable():
    rows = _entry_quality_rows()
    for row in rows:
        if row["code"] == "STRONG" and row["entry_date"].startswith("2020"):
            row["pnl"] = -10.0
            row["return_rate"] = -0.1

    report = analyzer._volume_ratio_marginal_path(rows)

    assert report["cross_year_direction_stability"] == {
        "win_rate": False,
        "pnl": False,
    }


@pytest.mark.parametrize("value,expected", [
    (0.0, "ZERO"),
    (0.1 + 0.2 - 0.3, "ZERO"),
    (0.5e-12, "ZERO"),
    (-0.5e-12, "ZERO"),
    (2.0e-12, "POSITIVE"),
    (-2.0e-12, "NEGATIVE"),
])
def test_volume_ratio_delta_direction_uses_fixed_numeric_zero_tolerance(
        value, expected):
    assert analyzer._delta_direction(value) == expected


def test_volume_ratio_zero_direction_is_not_stable_evidence():
    by_year = {
        "2019": {"win_rate": 0.0},
        "2020": {"win_rate": 0.1 + 0.2 - 0.3},
    }

    assert analyzer._year_delta_directions_stable(
        by_year, "win_rate",
    ) is False
    assert analyzer._same_available_delta_direction(0.0, 0.0) is False


@pytest.mark.parametrize("rows", [
    [],
    [dict(_entry_quality_rows()[0])],
])
def test_volume_ratio_marginal_marks_missing_group_unavailable(rows):
    report = analyzer._volume_ratio_marginal_path(rows)

    assert report["comparison_available"] is False
    assert report["cross_year_direction_stability"] == {
        "win_rate": None,
        "pnl": None,
    }


def test_volume_ratio_marginal_marks_one_common_year_unavailable():
    rows = _entry_quality_rows()
    rows = [row for row in rows if row["entry_date"].startswith("2019")]

    report = analyzer._volume_ratio_marginal_path(rows)

    assert report["comparison_available"] is True
    assert report["cross_year_direction_stability"] == {
        "win_rate": None,
        "pnl": None,
    }


def test_volume_ratio_marginal_compares_direction_across_friction_paths():
    ordinary_rows = _entry_quality_rows()
    double_rows = _entry_quality_rows()
    for row in double_rows:
        row["pnl"] -= 0.5
        row["return_rate"] -= 0.005

    report = analyzer._volume_ratio_marginal_attribution(
        ordinary_rows, double_rows,
    )

    assert report["scope"] == {
        "processing_stage": "POST_BACKTEST_READ_ONLY_ATTRIBUTION",
        "path_assumption": "ORIGINAL_TRADE_PATH_FIXED",
        "strategy_behavior_changed": False,
        "rule_candidate_created": False,
    }
    assert report["cross_friction_stability"] == {
        "overall_delta_direction_matches": {
            "win_rate": True,
            "pnl": True,
            "median_return": True,
        },
        "by_entry_year_delta_direction_matches": {
            "2019": {"win_rate": True, "pnl": True},
            "2020": {"win_rate": True, "pnl": True},
        },
        "all_reported_directions_match": True,
    }


def test_volume_ratio_cross_friction_marks_single_year_unavailable():
    rows = [
        row for row in _entry_quality_rows()
        if row["entry_date"].startswith("2019")
    ]

    report = analyzer._volume_ratio_marginal_attribution(rows, rows)

    assert report["ordinary"]["cross_year_direction_stability"] == {
        "win_rate": None,
        "pnl": None,
    }
    assert report["double_friction"][
        "cross_year_direction_stability"
    ] == {"win_rate": None, "pnl": None}
    assert report["cross_friction_stability"][
        "all_reported_directions_match"
    ] is None


def test_volume_ratio_cross_friction_requires_two_common_years():
    ordinary_rows = _entry_quality_rows()
    double_rows = _entry_quality_rows()
    for row in double_rows:
        if row["entry_date"].startswith("2019"):
            row["entry_date"] = row["entry_date"].replace(
                "2019", "2021", 1,
            )

    report = analyzer._volume_ratio_marginal_attribution(
        ordinary_rows, double_rows,
    )

    yearly = report["cross_friction_stability"][
        "by_entry_year_delta_direction_matches"
    ]
    assert list(yearly) == ["2020"]
    assert report["cross_friction_stability"][
        "all_reported_directions_match"
    ] is None
