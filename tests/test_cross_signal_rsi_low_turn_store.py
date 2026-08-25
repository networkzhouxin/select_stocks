from dataclasses import replace
from datetime import date, datetime, time, timedelta
import json
from zoneinfo import ZoneInfo

import pytest

from cross_signal_strategy.research.rsi_low_turn_shadow import RsiTurnInput, detect_rsi_low_turn
from cross_signal_strategy.research.rsi_low_turn_store import ShadowStore, SourceRewriteError


SHANGHAI = ZoneInfo("Asia/Shanghai")
OBSERVED_0826 = datetime(2026, 8, 26, 9, 35, tzinfo=SHANGHAI)
OBSERVED_0827 = datetime(2026, 8, 27, 9, 35, tzinfo=SHANGHAI)
OBSERVED_0828 = datetime(2026, 8, 28, 9, 35, tzinfo=SHANGHAI)


def decision_for(arrival_date: str, signal: bool = True, price: float = 2.035):
    day = date.fromisoformat(arrival_date)
    item = RsiTurnInput(
        code="513100",
        arrival_dt=datetime.combine(day, time(9, 35), SHANGHAI),
        signal_date=day - timedelta(days=1),
        r2=24.0 if signal else 18.0,
        r1=18.0,
        r0=21.0,
        c1=2.00,
        c0=price,
        entry_open=price,
        price_proved=True,
    )
    return detect_rsi_low_turn(item)


def true_decision(day: str):
    return decision_for(day, True)


def false_decision(day: str):
    return decision_for(day, False)


def replace_price(decision, price: float):
    return detect_rsi_low_turn(replace(decision.item, c0=price))


def test_first_true_emits_once_and_consecutive_true_does_not(tmp_path):
    store = ShadowStore(tmp_path)
    first = store.record_evaluation(true_decision("2026-08-26"), OBSERVED_0826)
    second = store.record_evaluation(true_decision("2026-08-27"), OBSERVED_0827)
    assert first.event_created is True
    assert second.event_created is False
    assert second.reason == "same_active_episode"


def test_false_day_resets_episode(tmp_path):
    store = ShadowStore(tmp_path)
    store.record_evaluation(true_decision("2026-08-26"), OBSERVED_0826)
    store.record_evaluation(false_decision("2026-08-27"), OBSERVED_0827)
    assert store.record_evaluation(true_decision("2026-08-28"), OBSERVED_0828).event_created is True


def test_duplicate_is_idempotent_but_conflicting_payload_is_refused(tmp_path):
    store = ShadowStore(tmp_path)
    decision = true_decision("2026-08-26")
    assert store.record_evaluation(decision, OBSERVED_0826).written is True
    assert store.record_evaluation(decision, OBSERVED_0826).written is False
    with pytest.raises(SourceRewriteError, match="conflicting evaluation"):
        store.record_evaluation(replace_price(decision, 9.99), OBSERVED_0826)


def test_changed_source_hash_stops(tmp_path):
    store = ShadowStore(tmp_path)
    store.record_source_hash("daily/513100.csv", "a" * 64, OBSERVED_0826)
    with pytest.raises(SourceRewriteError, match="source hash changed"):
        store.record_source_hash("daily/513100.csv", "b" * 64, OBSERVED_0827)


def test_old_arrival_first_seen_later_is_audit_only(tmp_path):
    store = ShadowStore(tmp_path)
    result = store.record_evaluation(true_decision("2026-08-26"), OBSERVED_0827)
    assert result.event_created is False
    assert result.reason == "late_import"


def test_unproved_signal_starts_episode_without_emitting_event(tmp_path):
    store = ShadowStore(tmp_path)
    unproved = detect_rsi_low_turn(replace(true_decision("2026-08-26").item, price_proved=False))
    first = store.record_evaluation(unproved, OBSERVED_0826)
    second = store.record_evaluation(true_decision("2026-08-27"), OBSERVED_0827)
    assert first.event_created is False
    assert first.reason == "invalid_event"
    assert second.event_created is False
    assert second.reason == "same_active_episode"


def test_labels_are_idempotent_and_conflicts_are_refused(tmp_path):
    store = ShadowStore(tmp_path)
    payload = {"return": 0.1, "close": 2.1}
    first = store.append_label("event-1", 5, payload)
    duplicate = store.append_label("event-1", 5, payload)
    assert first.written is True
    assert duplicate.written is False
    assert store.load_labels() == ({"event_id": "event-1", "horizon": 5, "payload": payload},)
    with pytest.raises(SourceRewriteError, match="conflicting label"):
        store.append_label("event-1", 5, {"return": 0.2, "close": 2.1})


def test_event_snapshot_keeps_complete_immutable_decision_input(tmp_path):
    store = ShadowStore(tmp_path)
    decision = detect_rsi_low_turn(replace(
        true_decision("2026-08-26").item,
        background={"rsi12": 31.5},
        source_hashes=("a" * 64,),
    ))
    store.record_evaluation(decision, OBSERVED_0826)

    event = store.load_events()[0]
    assert event["arrival_date"] == "2026-08-26"
    assert event["entry_open"] == pytest.approx(2.035)
    assert event["decision"] == {
        "event_id": decision.event_id,
        "signal_detected": True,
        "valid_event": True,
        "reasons": [],
    }
    assert event["input"]["price_proved"] is True
    assert event["input"]["background"] == {"rsi12": 31.5}
    assert event["input"]["source_hashes"] == ["a" * 64]


def test_retry_reconciles_event_after_interruption_between_appends(tmp_path, monkeypatch):
    store = ShadowStore(tmp_path)
    decision = true_decision("2026-08-26")
    original_append = store._append_record

    def interrupt_after_evaluation(filename, record):
        original_append(filename, record)
        if filename == "evaluations.jsonl":
            raise RuntimeError("injected interruption")

    monkeypatch.setattr(store, "_append_record", interrupt_after_evaluation)
    with pytest.raises(RuntimeError, match="injected interruption"):
        store.record_evaluation(decision, OBSERVED_0826)
    monkeypatch.setattr(store, "_append_record", original_append)

    recovered = store.record_evaluation(decision, OBSERVED_0826)
    duplicate = store.record_evaluation(decision, OBSERVED_0826)
    assert recovered.written is False
    assert recovered.event_created is True
    assert recovered.reason == "event_recovered"
    assert duplicate.event_created is False
    assert len(store.load_events()) == 1


@pytest.mark.parametrize("filename", ["evaluations.jsonl", "events.jsonl", "labels.jsonl"])
def test_malformed_or_truncated_jsonl_is_a_domain_integrity_error(tmp_path, filename):
    store = ShadowStore(tmp_path)
    (tmp_path / filename).write_text('{"truncated":', encoding="utf-8")
    with pytest.raises(SourceRewriteError, match="invalid JSONL"):
        if filename == "evaluations.jsonl":
            store.record_evaluation(true_decision("2026-08-26"), OBSERVED_0826)
        elif filename == "events.jsonl":
            store.load_events()
        else:
            store.load_labels()


def test_preexisting_later_conflicting_evaluation_duplicate_is_refused(tmp_path):
    store = ShadowStore(tmp_path)
    decision = true_decision("2026-08-26")
    store.record_evaluation(decision, OBSERVED_0826)
    path = tmp_path / "evaluations.jsonl"
    conflicting = json.loads(path.read_text(encoding="utf-8"))
    conflicting["item"]["c0"] = 9.99
    path.write_text(
        path.read_text(encoding="utf-8") + json.dumps(conflicting) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SourceRewriteError, match="conflicting evaluation"):
        store.record_evaluation(decision, OBSERVED_0826)


def test_preexisting_later_conflicting_label_duplicate_is_refused(tmp_path):
    store = ShadowStore(tmp_path)
    payload = {"return": 0.1}
    store.append_label("event-1", 5, payload)
    path = tmp_path / "labels.jsonl"
    conflicting = json.loads(path.read_text(encoding="utf-8"))
    conflicting["payload"]["return"] = 0.2
    path.write_text(
        path.read_text(encoding="utf-8") + json.dumps(conflicting) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SourceRewriteError, match="conflicting label"):
        store.append_label("event-1", 5, payload)
