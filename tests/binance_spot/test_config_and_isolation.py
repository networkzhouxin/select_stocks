from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal
import sys
import unittest


_MODULES_BEFORE_IMPORT = set(sys.modules)
import binance_spot_strategy
from binance_spot_strategy.config import frozen


_INTRODUCED_MODULES = set(sys.modules).difference(_MODULES_BEFORE_IMPORT)
UTC = timezone.utc


class ConfigAndIsolationTests(unittest.TestCase):
    def test_package_import_does_not_introduce_cross_signal_strategy_modules(self) -> None:
        offending_modules = tuple(
            module_name
            for module_name in _INTRODUCED_MODULES
            if module_name == "cross_signal_strategy"
            or module_name.startswith("cross_signal_strategy.")
        )

        self.assertEqual(offending_modules, ())

    def test_frozen_common_constants_match_the_approved_design(self) -> None:
        self.assertEqual(frozen.SYMBOLS, ("BTCUSDT", "ETHUSDT"))
        self.assertEqual(frozen.BAR_INTERVAL, "4h")
        self.assertEqual(frozen.FORMAL_STARTING_BALANCE, Decimal("500.00"))
        self.assertEqual(frozen.SLIPPAGE_RATE, Decimal("0.0005"))
        self.assertEqual(frozen.COMMISSION_RATE, Decimal("0.001"))
        self.assertEqual(frozen.RISK_BUDGET, Decimal("0.01"))
        self.assertEqual(frozen.ALLOCATION_CAP, Decimal("0.30"))
        self.assertEqual(frozen.ATR_MULTIPLIER, Decimal("2.5"))
        self.assertEqual(frozen.STOP_FLOOR, Decimal("0.05"))
        self.assertEqual(frozen.STOP_CAP, Decimal("0.15"))
        self.assertEqual(
            frozen.DESIGN_REVISION_SHA256,
            "18adf5fd56177354e5e4194a416ae1c9de5a4371ae6c6872783cb08d7a3026a0",
        )

    def test_stage_windows_are_ordered_half_open_intervals_with_540_bar_warmup(self) -> None:
        self.assertEqual(
            frozen.STAGE_WINDOWS,
            (
                frozen.StageWindow(
                    frozen.StageName.TRAINING,
                    datetime(2018, 1, 1, tzinfo=UTC),
                    datetime(2022, 1, 1, tzinfo=UTC),
                ),
                frozen.StageWindow(
                    frozen.StageName.VALIDATION,
                    datetime(2022, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),
                ),
                frozen.StageWindow(
                    frozen.StageName.HOLDOUT,
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2026, 8, 1, tzinfo=UTC),
                ),
            ),
        )
        self.assertTrue(all(window.warmup_bars == 540 for window in frozen.STAGE_WINDOWS))

    def test_training_stage_admits_final_2021_bar_but_excludes_2022_open(self) -> None:
        training = frozen.STAGE_WINDOWS[0]

        self.assertTrue(
            training.admits(
                datetime(2021, 12, 31, 20, tzinfo=UTC),
                datetime(2021, 12, 31, 23, 59, 59, 999000, tzinfo=UTC),
            )
        )
        self.assertFalse(
            training.admits(
                datetime(2022, 1, 1, tzinfo=UTC),
                datetime(2022, 1, 1, 3, 59, 59, 999000, tzinfo=UTC),
            )
        )

    def test_stage_window_rejects_naive_utc_reversed_boundaries_and_wrong_warmup(self) -> None:
        with self.assertRaises(ValueError):
            frozen.StageWindow(
                frozen.StageName.TRAINING,
                datetime(2018, 1, 1),
                datetime(2022, 1, 1, tzinfo=UTC),
            )
        with self.assertRaises(ValueError):
            frozen.StageWindow(
                frozen.StageName.TRAINING,
                datetime(2022, 1, 1, tzinfo=UTC),
                datetime(2018, 1, 1, tzinfo=UTC),
            )
        with self.assertRaises(ValueError):
            frozen.StageWindow(
                frozen.StageName.TRAINING,
                datetime(2018, 1, 1, tzinfo=UTC),
                datetime(2022, 1, 1, tzinfo=UTC),
                warmup_bars=539,
            )

    def test_stage_window_rejects_naive_bar_times(self) -> None:
        training = frozen.STAGE_WINDOWS[0]

        with self.assertRaises(ValueError):
            training.admits(
                datetime(2021, 12, 31, 20),
                datetime(2021, 12, 31, 23, 59, 59, 999000, tzinfo=UTC),
            )

    def test_stage_windows_are_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            frozen.STAGE_WINDOWS[0].start = datetime(2019, 1, 1, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
