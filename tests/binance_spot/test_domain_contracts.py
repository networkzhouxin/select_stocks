from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import unittest

from binance_spot_strategy.domain import (
    Bar,
    BarInterval,
    DecisionKey,
    DecisionPlan,
    EquitySnapshot,
    ExecutionGroup,
    Fill,
    OrderIntent,
    OrderState,
    PaperObservationEpochId,
    Position,
    ScopeType,
    Side,
    SignalIntent,
    StageActivationId,
    Symbol,
    TrainingStageId,
    canonical_utc_timestamp,
)
from binance_spot_strategy.protocols import Q18, quantize_q18


OPEN_TIME = datetime(2026, 8, 1, tzinfo=timezone.utc)
CLOSE_TIME = datetime(2026, 8, 1, 4, tzinfo=timezone.utc)
CANDIDATE_FINGERPRINT = "a" * 64
RUN_FINGERPRINT = "b" * 64


def q18(value: str) -> Q18:
    return quantize_q18(Decimal(value))


def training_key(
    *,
    bar_close_time: datetime = CLOSE_TIME,
) -> DecisionKey:
    return DecisionKey(
        "run-1",
        ScopeType.TRAINING_STAGE,
        TrainingStageId("train-1"),
        CANDIDATE_FINGERPRINT,
        RUN_FINGERPRINT,
        bar_close_time,
    )


def signal(
    side: Side,
    *,
    symbol: Symbol = Symbol.BTCUSDT,
    bar_close_time: datetime = CLOSE_TIME,
    risk_atr: Q18 | None = None,
) -> SignalIntent:
    return SignalIntent(
        f"signal-{side.value}",
        symbol,
        side,
        bar_close_time,
        "ranked-signal",
        risk_atr,
    )


def order(
    intent_id: str,
    side: Side,
    state: OrderState = OrderState.PENDING,
    *,
    group_id: str = "group-1",
    symbol: Symbol = Symbol.BTCUSDT,
    depends_on: str | None = None,
) -> OrderIntent:
    return OrderIntent(
        intent_id,
        group_id,
        symbol,
        side,
        state,
        q18("1"),
        depends_on,
    )


class EnumAndIdentifierContractTests(unittest.TestCase):
    def test_enums_have_the_exact_wire_values(self) -> None:
        self.assertEqual(Symbol.BTCUSDT.value, "BTCUSDT")
        self.assertEqual(Symbol.ETHUSDT.value, "ETHUSDT")
        self.assertEqual(BarInterval.FOUR_HOURS.value, "4h")
        self.assertEqual(Side.BUY.value, "buy")
        self.assertEqual(Side.SELL.value, "sell")
        self.assertEqual(OrderState.PENDING.value, "pending")
        self.assertEqual(OrderState.BLOCKED.value, "blocked")
        self.assertEqual(ScopeType.TRAINING_STAGE.value, "training_stage")
        self.assertEqual(ScopeType.RESERVED_STAGE.value, "reserved_stage")
        self.assertEqual(ScopeType.PAPER_EPOCH.value, "paper_epoch")

    def test_typed_scope_ids_accept_only_the_identifier_grammar(self) -> None:
        for identifier_type in (
            TrainingStageId,
            StageActivationId,
            PaperObservationEpochId,
        ):
            with self.subTest(identifier_type=identifier_type):
                self.assertEqual(identifier_type("A0._:-").value, "A0._:-")
                self.assertEqual(identifier_type("a" * 128).value, "a" * 128)
                for invalid in ("", "-starts-wrong", "has space", "a" * 129, 1):
                    with self.subTest(invalid=invalid):
                        with self.assertRaises((TypeError, ValueError)):
                            identifier_type(invalid)


class TimestampAndBarContractTests(unittest.TestCase):
    def test_canonical_utc_timestamp_has_second_or_millisecond_precision(self) -> None:
        self.assertEqual(
            canonical_utc_timestamp(
                datetime(2026, 8, 1, tzinfo=timezone.utc)
            ),
            "2026-08-01T00:00:00Z",
        )
        self.assertEqual(
            canonical_utc_timestamp(
                datetime(
                    2026,
                    8,
                    1,
                    3,
                    59,
                    59,
                    999000,
                    tzinfo=timezone.utc,
                )
            ),
            "2026-08-01T03:59:59.999Z",
        )

    def test_canonical_utc_timestamp_rejects_noncanonical_time(self) -> None:
        for invalid in (
            datetime(2026, 8, 1),
            datetime(2026, 8, 1, tzinfo=timezone(timedelta(hours=1))),
            datetime(2026, 8, 1, 0, 0, 0, 1, tzinfo=timezone.utc),
            "2026-08-01T00:00:00Z",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    canonical_utc_timestamp(invalid)

    def test_bar_exposes_the_exact_identity_and_is_immutable(self) -> None:
        bar = Bar(
            Symbol.BTCUSDT,
            BarInterval.FOUR_HOURS,
            OPEN_TIME,
            CLOSE_TIME,
            Decimal("100"),
            Decimal("120"),
            Decimal("90"),
            Decimal("110"),
            Decimal("5"),
        )

        self.assertEqual(
            bar.identity,
            (Symbol.BTCUSDT, BarInterval.FOUR_HOURS, OPEN_TIME),
        )
        with self.assertRaises(FrozenInstanceError):
            bar.close = Decimal("111")

    def test_bar_rejects_invalid_time_type_and_numeric_values(self) -> None:
        valid = {
            "symbol": Symbol.BTCUSDT,
            "interval": BarInterval.FOUR_HOURS,
            "open_time": OPEN_TIME,
            "close_time": CLOSE_TIME,
            "open": Decimal("100"),
            "high": Decimal("120"),
            "low": Decimal("90"),
            "close": Decimal("110"),
            "volume": Decimal("5"),
        }
        invalid_changes = (
            {"symbol": "BTCUSDT"},
            {"interval": "4h"},
            {"open_time": OPEN_TIME.replace(tzinfo=None)},
            {"close_time": CLOSE_TIME.replace(microsecond=1)},
            {"close_time": OPEN_TIME},
            {"open": 100},
            {"open": Decimal("0")},
            {"high": Decimal("NaN")},
            {"volume": -1},
            {"volume": Decimal("-1")},
            {"high": Decimal("99")},
            {"low": Decimal("101")},
            {"close": Decimal("121")},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    Bar(**(valid | changes))


class DecisionKeyContractTests(unittest.TestCase):
    def test_scope_specific_keys_are_hashable_immutable_and_pairwise_unequal(self) -> None:
        keys = (
            DecisionKey(
                "run-1",
                ScopeType.TRAINING_STAGE,
                TrainingStageId("same"),
                CANDIDATE_FINGERPRINT,
                RUN_FINGERPRINT,
                CLOSE_TIME,
            ),
            DecisionKey(
                "run-1",
                ScopeType.RESERVED_STAGE,
                StageActivationId("same"),
                CANDIDATE_FINGERPRINT,
                RUN_FINGERPRINT,
                CLOSE_TIME,
            ),
            DecisionKey(
                "run-1",
                ScopeType.PAPER_EPOCH,
                PaperObservationEpochId("same"),
                CANDIDATE_FINGERPRINT,
                RUN_FINGERPRINT,
                CLOSE_TIME,
            ),
        )

        self.assertEqual(len(set(keys)), 3)
        with self.assertRaises(FrozenInstanceError):
            keys[0].run_id = "other-run"

    def test_decision_key_rejects_every_scope_id_type_mismatch(self) -> None:
        valid_type_by_scope = {
            ScopeType.TRAINING_STAGE: TrainingStageId,
            ScopeType.RESERVED_STAGE: StageActivationId,
            ScopeType.PAPER_EPOCH: PaperObservationEpochId,
        }
        for scope_type, expected_id_type in valid_type_by_scope.items():
            for actual_id_type in valid_type_by_scope.values():
                if actual_id_type is expected_id_type:
                    continue
                with self.subTest(
                    scope_type=scope_type,
                    actual_id_type=actual_id_type,
                ):
                    with self.assertRaises(TypeError):
                        DecisionKey(
                            "run-1",
                            scope_type,
                            actual_id_type("same"),
                            CANDIDATE_FINGERPRINT,
                            RUN_FINGERPRINT,
                            CLOSE_TIME,
                        )

    def test_decision_key_rejects_noncanonical_fingerprints(self) -> None:
        for field_name in (
            "candidate_strategy_fingerprint",
            "run_fingerprint",
        ):
            for invalid in ("A" * 64, "a" * 63, "g" * 64, 1):
                values = {
                    "run_id": "run-1",
                    "scope_type": ScopeType.TRAINING_STAGE,
                    "scope_id": TrainingStageId("train-1"),
                    "candidate_strategy_fingerprint": CANDIDATE_FINGERPRINT,
                    "run_fingerprint": RUN_FINGERPRINT,
                    "bar_close_time": CLOSE_TIME,
                }
                values[field_name] = invalid
                with self.subTest(field_name=field_name, invalid=invalid):
                    with self.assertRaises((TypeError, ValueError)):
                        DecisionKey(**values)

    def test_decision_key_rejects_invalid_run_scope_or_close_time(self) -> None:
        valid = {
            "run_id": "run-1",
            "scope_type": ScopeType.TRAINING_STAGE,
            "scope_id": TrainingStageId("train-1"),
            "candidate_strategy_fingerprint": CANDIDATE_FINGERPRINT,
            "run_fingerprint": RUN_FINGERPRINT,
            "bar_close_time": CLOSE_TIME,
        }
        for changes in (
            {"run_id": "bad run"},
            {"scope_type": "training_stage"},
            {"bar_close_time": CLOSE_TIME.replace(tzinfo=None)},
            {"bar_close_time": CLOSE_TIME.replace(microsecond=1)},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    DecisionKey(**(valid | changes))


class DecisionPlanContractTests(unittest.TestCase):
    def test_decision_plan_accepts_exactly_the_four_supported_shapes(self) -> None:
        key = training_key()
        sell = signal(Side.SELL)
        buy = signal(
            Side.BUY,
            symbol=Symbol.ETHUSDT,
            risk_atr=q18("2"),
        )

        plans = (
            DecisionPlan(key, "group-hold"),
            DecisionPlan(key, "group-sell", sell=sell),
            DecisionPlan(key, "group-buy", buy=buy),
            DecisionPlan(key, "group-replace", sell=sell, buy=buy),
        )

        self.assertEqual(len(plans), 4)
        self.assertIsNone(plans[0].sell)
        self.assertIsNone(plans[0].buy)

    def test_signal_intent_enforces_side_specific_atr_rules(self) -> None:
        with self.assertRaises(ValueError):
            signal(Side.BUY)
        with self.assertRaises(ValueError):
            signal(Side.BUY, risk_atr=q18("0"))
        with self.assertRaises(ValueError):
            signal(Side.BUY, risk_atr=q18("-1"))
        with self.assertRaises(ValueError):
            signal(Side.SELL, risk_atr=q18("1"))

    def test_decision_plan_rejects_wrong_sides_same_symbol_and_close_mismatch(self) -> None:
        key = training_key()
        valid_sell = signal(Side.SELL)
        valid_buy = signal(Side.BUY, risk_atr=q18("1"))
        other_buy = signal(
            Side.BUY,
            symbol=Symbol.ETHUSDT,
            risk_atr=q18("1"),
        )
        wrong_close = CLOSE_TIME + timedelta(hours=4)
        invalid_plans = (
            {"sell": other_buy},
            {"buy": valid_sell},
            {"sell": valid_sell, "buy": valid_buy},
            {"sell": signal(Side.SELL, bar_close_time=wrong_close)},
            {
                "buy": signal(
                    Side.BUY,
                    symbol=Symbol.ETHUSDT,
                    bar_close_time=wrong_close,
                    risk_atr=q18("1"),
                )
            },
        )
        for plan_parts in invalid_plans:
            with self.subTest(plan_parts=plan_parts):
                with self.assertRaises((TypeError, ValueError)):
                    DecisionPlan(key, "group-1", **plan_parts)

    def test_signal_and_plan_reject_invalid_types_and_identifiers(self) -> None:
        with self.assertRaises(ValueError):
            SignalIntent(
                "bad id",
                Symbol.BTCUSDT,
                Side.SELL,
                CLOSE_TIME,
                "reason",
            )
        with self.assertRaises(TypeError):
            SignalIntent(
                "signal-1",
                Symbol.BTCUSDT,
                "sell",
                CLOSE_TIME,
                "reason",
            )
        with self.assertRaises(TypeError):
            SignalIntent(
                "signal-1",
                Symbol.BTCUSDT,
                Side.BUY,
                CLOSE_TIME,
                "reason",
                Decimal("1"),
            )
        with self.assertRaises(ValueError):
            DecisionPlan(training_key(), "bad group")


class ExecutionGroupContractTests(unittest.TestCase):
    def test_execution_group_accepts_pending_standalone_buy_or_sell(self) -> None:
        for side in (Side.BUY, Side.SELL):
            with self.subTest(side=side):
                standalone = order(f"intent-{side.value}", side)
                group = ExecutionGroup("group-1", (standalone,))
                self.assertEqual(group.order_intents, (standalone,))

    def test_execution_group_accepts_ordered_sell_parent_and_blocked_buy_child(self) -> None:
        parent = order("intent-sell", Side.SELL)
        child = order(
            "intent-buy",
            Side.BUY,
            OrderState.BLOCKED,
            symbol=Symbol.ETHUSDT,
            depends_on=parent.intent_id,
        )

        group = ExecutionGroup("group-1", (parent, child))

        self.assertEqual(group.order_intents, (parent, child))

    def test_execution_group_rejects_every_unsupported_shape(self) -> None:
        sell = order("intent-sell", Side.SELL)
        buy = order("intent-buy", Side.BUY, symbol=Symbol.ETHUSDT)
        child = order(
            "intent-child",
            Side.BUY,
            OrderState.BLOCKED,
            symbol=Symbol.ETHUSDT,
            depends_on=sell.intent_id,
        )
        invalid_groups = (
            (sell, buy),
            (
                order(
                    "intent-blocked-sell",
                    Side.SELL,
                    OrderState.BLOCKED,
                    depends_on="intent-buy",
                ),
            ),
            (child, sell),
            (
                sell,
                order(
                    "intent-wrong-dependency",
                    Side.BUY,
                    OrderState.BLOCKED,
                    symbol=Symbol.ETHUSDT,
                    depends_on="other-parent",
                ),
            ),
            (order("intent-wrong-group", Side.BUY, group_id="group-2"),),
        )
        for order_intents in invalid_groups:
            with self.subTest(order_intents=order_intents):
                with self.assertRaises(ValueError):
                    ExecutionGroup("group-1", order_intents)

    def test_order_intent_rejects_nonpositive_quantity_or_invalid_dependency(self) -> None:
        for invalid_quantity in (q18("0"), q18("-1"), Decimal("1")):
            with self.subTest(invalid_quantity=invalid_quantity):
                with self.assertRaises((TypeError, ValueError)):
                    OrderIntent(
                        "intent-1",
                        "group-1",
                        Symbol.BTCUSDT,
                        Side.BUY,
                        OrderState.PENDING,
                        invalid_quantity,
                    )
        with self.assertRaises(ValueError):
            order("intent-1", Side.BUY, depends_on="bad dependency")


class PortfolioRecordContractTests(unittest.TestCase):
    def test_fill_position_and_equity_snapshot_accept_their_valid_ranges(self) -> None:
        fill = Fill(
            "fill-1",
            "intent-1",
            Symbol.BTCUSDT,
            Side.BUY,
            q18("1"),
            q18("100"),
            q18("0"),
            CLOSE_TIME,
        )
        position = Position(Symbol.BTCUSDT, q18("1"), q18("100"))
        snapshot = EquitySnapshot(
            CLOSE_TIME,
            q18("0"),
            q18("100"),
            q18("100"),
        )

        self.assertEqual(fill.commission, q18("0"))
        self.assertEqual(position.average_cost, q18("100"))
        self.assertEqual(snapshot.cash, q18("0"))

    def test_fill_rejects_nonpositive_trade_values_and_negative_commission(self) -> None:
        valid = {
            "fill_id": "fill-1",
            "intent_id": "intent-1",
            "symbol": Symbol.BTCUSDT,
            "side": Side.BUY,
            "quantity": q18("1"),
            "price": q18("100"),
            "commission": q18("0"),
            "event_time": CLOSE_TIME,
        }
        for changes in (
            {"quantity": q18("0")},
            {"price": q18("0")},
            {"commission": q18("-1")},
            {"quantity": Decimal("1")},
            {"event_time": CLOSE_TIME.replace(tzinfo=None)},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    Fill(**(valid | changes))

    def test_position_requires_positive_quantity_and_average_cost(self) -> None:
        for quantity, average_cost in (
            (q18("0"), q18("1")),
            (q18("-1"), q18("1")),
            (q18("1"), q18("0")),
            (Decimal("1"), q18("1")),
        ):
            with self.subTest(quantity=quantity, average_cost=average_cost):
                with self.assertRaises((TypeError, ValueError)):
                    Position(Symbol.BTCUSDT, quantity, average_cost)

    def test_equity_snapshot_requires_nonnegative_q18_components(self) -> None:
        for field_name in ("cash", "position_value", "total_equity"):
            for invalid in (q18("-1"), Decimal("0")):
                values = {
                    "event_time": CLOSE_TIME,
                    "cash": q18("0"),
                    "position_value": q18("0"),
                    "total_equity": q18("0"),
                }
                values[field_name] = invalid
                with self.subTest(field_name=field_name, invalid=invalid):
                    with self.assertRaises((TypeError, ValueError)):
                        EquitySnapshot(**values)

    def test_every_domain_value_object_is_frozen(self) -> None:
        sell = signal(Side.SELL)
        standalone = order("intent-sell", Side.SELL)
        objects_and_mutations = (
            (TrainingStageId("train-1"), "value", "train-2"),
            (StageActivationId("activation-1"), "value", "activation-2"),
            (PaperObservationEpochId("epoch-1"), "value", "epoch-2"),
            (training_key(), "run_id", "run-2"),
            (sell, "reason_code", "other-reason"),
            (DecisionPlan(training_key(), "group-1"), "buy", sell),
            (standalone, "state", OrderState.BLOCKED),
            (
                ExecutionGroup("group-1", (standalone,)),
                "order_intents",
                (),
            ),
            (
                Fill(
                    "fill-1",
                    "intent-1",
                    Symbol.BTCUSDT,
                    Side.SELL,
                    q18("1"),
                    q18("100"),
                    q18("0"),
                    CLOSE_TIME,
                ),
                "price",
                q18("99"),
            ),
            (
                Position(Symbol.BTCUSDT, q18("1"), q18("100")),
                "quantity",
                q18("2"),
            ),
            (
                EquitySnapshot(
                    CLOSE_TIME,
                    q18("0"),
                    q18("100"),
                    q18("100"),
                ),
                "cash",
                q18("1"),
            ),
        )
        for instance, attribute, replacement in objects_and_mutations:
            with self.subTest(instance=instance, attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, attribute, replacement)


if __name__ == "__main__":
    unittest.main()
