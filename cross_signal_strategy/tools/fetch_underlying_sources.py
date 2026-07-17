# -*- coding: utf-8 -*-
"""Download the locked 2018-2021 underlying-index histories to raw staging."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from cross_signal_strategy.research.underlying_source_acquisition import (
    run_source_acquisition,
)


DEFAULT_STAGING_ROOT = Path(
    r"G:\financial\history_data\cross_signal_underlying_staging_2018_2021"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "采集四个预登记底层指数的 2018-2021 原始历史值。"
            "该命令不会发布正式 available_at 数据。"
        )
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=DEFAULT_STAGING_ROOT,
        help="原始暂存目录，不得指向正式只读根目录",
    )
    args = parser.parse_args()
    acquired_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest = run_source_acquisition(
        staging_root=args.staging_root,
        acquired_at=acquired_at,
    )
    print("原始来源采集完成：%s" % manifest)
    print("正式发布状态：阻断（SPX/H30533 的历史发布时间证据尚未闭环）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
