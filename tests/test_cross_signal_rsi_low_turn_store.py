from dataclasses import replace
from datetime import date, datetime, time, timedelta
import json
import hashlib
import os
from zoneinfo import ZoneInfo

import pytest

from cross_signal_strategy.research.rsi_low_turn_shadow import RsiTurnInput, detect_rsi_low_turn
import cross_signal_strategy.research.rsi_low_turn_store as store_module
from cross_signal_strategy.research.rsi_low_turn_store import (
    ShadowStore,
    SourceRewriteError,
    SourceSnapshotBatch,
)


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
        source_hashes=("a" * 64, "b" * 64, "c" * 64),
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


def test_source_snapshot_history_allows_only_exact_prefix_growth(tmp_path):
    store = ShadowStore(tmp_path)
    assert store.record_source_snapshot("daily/513100.csv", b"first\n", OBSERVED_0826) is True
    assert store.record_source_snapshot("daily/513100.csv", b"first\nsecond\n", OBSERVED_0827) is True
    history = store.load_source_snapshots()

    assert [item["byte_length"] for item in history] == [6, 13]
    assert set(history[0]) == {"relative_path", "sha256", "byte_length", "observed_at"}
    with pytest.raises(SourceRewriteError, match="source hash changed"):
        store.record_source_snapshot("daily/513100.csv", b"first?second\n", OBSERVED_0828)


def _snapshot_record(relative_path, content, observed_at):
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_length": len(content),
        "observed_at": observed_at.isoformat(),
    }


@pytest.mark.parametrize(("record", "message"), [
    (_snapshot_record("daily/513100.csv", b"first", OBSERVED_0827), "source snapshot length"),
    (_snapshot_record("daily/513100.csv", b"other\n", OBSERVED_0827), "source snapshot hash"),
    (_snapshot_record("daily/513100.csv", b"first\nsecond\n", OBSERVED_0826), "source snapshot observed_at"),
])
def test_snapshot_history_rejects_broken_per_path_chain(tmp_path, record, message):
    store = ShadowStore(tmp_path)
    store.record_source_snapshot("daily/513100.csv", b"first\n", OBSERVED_0826)
    path = tmp_path / "source_hashes.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(SourceRewriteError, match=message):
        store.load_source_snapshots()


def test_snapshot_batch_rechecks_chain_before_writing_stale_plan(tmp_path):
    store = ShadowStore(tmp_path)
    store.record_source_snapshot("daily/513100.csv", b"first\n", OBSERVED_0826)
    stale = store.prepare_source_snapshot_batch(
        (("daily/513100.csv", b"first\nsecond\n"),), OBSERVED_0827,
    )
    store.record_source_snapshot("daily/513100.csv", b"first\nsecond\nthird\n", OBSERVED_0828)

    with pytest.raises(SourceRewriteError, match="stale source snapshot batch"):
        store.write_source_snapshot_batch(stale)


def test_snapshot_batch_preflights_every_path_before_appending(tmp_path):
    store = ShadowStore(tmp_path)
    store.record_source_snapshot("daily/513100.csv", b"a\n", OBSERVED_0826)
    store.record_source_snapshot("minute_0935/513100.csv", b"b\n", OBSERVED_0826)
    stale = store.prepare_source_snapshot_batch((
        ("daily/513100.csv", b"a\nplanned\n"),
        ("minute_0935/513100.csv", b"b\nplanned\n"),
    ), OBSERVED_0827)
    store.record_source_snapshot(
        "minute_0935/513100.csv", b"b\nnewer\n", OBSERVED_0828,
    )
    path = tmp_path / "source_hashes.jsonl"
    before = path.read_bytes()

    with pytest.raises(SourceRewriteError, match="stale source snapshot batch"):
        store.write_source_snapshot_batch(stale)

    assert path.read_bytes() == before


def test_snapshot_batch_preserves_same_path_chain_and_retry_idempotency(tmp_path):
    store = ShadowStore(tmp_path)
    first = _snapshot_record("daily/513100.csv", b"header\n", OBSERVED_0826)
    second = _snapshot_record("daily/513100.csv", b"header\nrow\n", OBSERVED_0827)
    other = _snapshot_record("minute_0935/513100.csv", b"minute\n", OBSERVED_0826)
    batch = SourceSnapshotBatch(
        records=(first, second, other),
        bases=(None, first, None),
    )

    assert store.write_source_snapshot_batch(batch) == (True, True, True)
    assert store.write_source_snapshot_batch(batch) == (False, False, False)
    assert [record["relative_path"] for record in store.load_source_snapshots()] == [
        "daily/513100.csv",
        "daily/513100.csv",
        "minute_0935/513100.csv",
    ]


def test_source_snapshot_retry_is_idempotent_after_partial_append(tmp_path):
    store = ShadowStore(tmp_path)
    snapshots = (
        ("manifest.json", b'{"append_only":true}\n'),
        ("daily/513100.csv", b"header\nrow\n"),
    )

    assert store.record_source_snapshot(*snapshots[0], OBSERVED_0826) is True
    assert store.record_source_snapshot(*snapshots[0], OBSERVED_0826) is False
    assert store.record_source_snapshot(*snapshots[1], OBSERVED_0826) is True
    assert len(store.load_source_snapshots()) == 2


def test_marker_write_is_atomic_and_retry_safe_after_replace_failure(tmp_path, monkeypatch):
    store = ShadowStore(tmp_path)
    marker = tmp_path / "observer_state.json"
    real_replace = os.replace

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        store.write_state_marker()
    assert not marker.exists()
    assert not list(tmp_path.glob(".observer-state-*.tmp"))

    monkeypatch.setattr(os, "replace", real_replace)
    store.write_state_marker()
    assert marker.read_text(encoding="utf-8") == '{"observer":"rsi_low_turn_prospective_shadow","schema_version":1}\n'
    store.require_state_marker()


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda item: item.pop("r0"), "stored evaluation item fields"),
    (lambda item: item.update(source_hashes=["a" * 64, "b" * 64]), "stored evaluation source hashes"),
    (lambda item: item.update(source_hashes=["z" * 64] * 3), "stored evaluation source hashes"),
])
def test_evaluation_item_schema_and_hash_provenance_are_strict(tmp_path, mutate, message):
    store = ShadowStore(tmp_path)
    store.record_evaluation(true_decision("2026-08-26"), OBSERVED_0826)
    path = tmp_path / "evaluations.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    mutate(record["item"])
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(SourceRewriteError, match=message):
        store.load_evaluations()


def test_unproved_item_allows_missing_entry_open_when_other_fields_are_valid(tmp_path):
    decision = detect_rsi_low_turn(replace(
        true_decision("2026-08-26").item,
        entry_open=None, price_proved=False, price_reason="price_unproved",
    ))
    store = ShadowStore(tmp_path)
    store.record_evaluation(decision, OBSERVED_0826)

    assert len(store.load_evaluations()) == 1


def test_source_snapshot_batch_is_compact_and_reuses_one_history_load(tmp_path, monkeypatch):
    store = ShadowStore(tmp_path)
    snapshots = (
        ("manifest.json", b'{"append_only":true}\n'),
        ("daily/513100.csv", b"header\nrow\n"),
    )
    calls = 0
    original = store.load_source_snapshots

    def counted_history():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(store, "load_source_snapshots", counted_history)
    plan = store.prepare_source_snapshot_batch(snapshots, OBSERVED_0826)
    assert calls == 1
    store.write_source_snapshot_batch(plan)
    assert calls == 2
    assert all("content_b64" not in record for record in original())


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
        source_hashes=("a" * 64, "b" * 64, "c" * 64),
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
    assert event["input"]["source_hashes"] == ["a" * 64, "b" * 64, "c" * 64]


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


@pytest.mark.parametrize(
    "filename",
    ["evaluations.jsonl", "events.jsonl", "labels.jsonl", "source_hashes.jsonl"],
)
def test_malformed_or_truncated_jsonl_is_a_domain_integrity_error(tmp_path, filename):
    store = ShadowStore(tmp_path)
    (tmp_path / filename).write_text('{"truncated":', encoding="utf-8")
    with pytest.raises(SourceRewriteError, match="invalid JSONL"):
        if filename == "evaluations.jsonl":
            store.record_evaluation(true_decision("2026-08-26"), OBSERVED_0826)
        elif filename == "events.jsonl":
            store.load_events()
        elif filename == "labels.jsonl":
            store.load_labels()
        else:
            store.record_source_snapshot("daily/513100.csv", b"fixture", OBSERVED_0826)


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

    with pytest.raises(SourceRewriteError, match="duplicate evaluation"):
        store.record_evaluation(decision, OBSERVED_0826)


def test_load_evaluations_rejects_identical_duplicate_event_id(tmp_path):
    store = ShadowStore(tmp_path)
    store.record_evaluation(true_decision("2026-08-26"), OBSERVED_0826)
    path = tmp_path / "evaluations.jsonl"
    original = path.read_text(encoding="utf-8")
    path.write_text(original + original, encoding="utf-8")

    with pytest.raises(SourceRewriteError, match="duplicate evaluation"):
        store.load_evaluations()


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


def test_load_events_rejects_later_conflicting_duplicate_event_id(tmp_path):
    store = ShadowStore(tmp_path)
    store.record_evaluation(true_decision("2026-08-26"), OBSERVED_0826)
    path = tmp_path / "events.jsonl"
    conflicting = json.loads(path.read_text(encoding="utf-8"))
    conflicting["entry_open"] = 9.99
    path.write_text(
        path.read_text(encoding="utf-8") + json.dumps(conflicting) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SourceRewriteError, match="conflicting event"):
        store.load_events()


def test_load_events_rejects_identical_duplicate_event_ids(tmp_path):
    store = ShadowStore(tmp_path)
    store.record_evaluation(true_decision("2026-08-26"), OBSERVED_0826)
    path = tmp_path / "events.jsonl"
    original = path.read_text(encoding="utf-8")
    path.write_text(original + original, encoding="utf-8")

    with pytest.raises(SourceRewriteError, match="conflicting event"):
        store.load_events()
