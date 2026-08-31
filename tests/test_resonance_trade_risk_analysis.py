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
    ordinary.insert(2, duplicate)
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
