# -*- coding: utf-8 -*-
"""归档未来 PTrade 实盘日志，不读取行情，也不评价策略表现。"""

import argparse
from pathlib import Path
import sys

from cross_signal_strategy.research.prospective_log_archive import (
    LogProtocolError,
    archive_log_bundle,
)
from cross_signal_strategy.tools.verify_release import verify_release


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="校验并归档 cross-signal 未来 PTrade 实盘日志"
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="仓库根目录",
    )
    parser.add_argument(
        "--archive-root",
        required=True,
        help="独立归档目录；不能位于任何源日志路径内部",
    )
    parser.add_argument("logs", nargs="+", help="PTrade 导出的原始日志文件")
    args = parser.parse_args(argv)

    release = verify_release(args.repo_root, run_tests=False)
    if release["status"] != "通过":
        print("归档拒绝：正式发布核验未通过", file=sys.stderr)
        return 2

    try:
        manifest = archive_log_bundle(
            [Path(value) for value in args.logs],
            Path(args.archive_root),
            expected_version=release["strategy_version"],
            expected_build=release["deployment_build"],
            expected_fingerprint=release["business_fingerprint"],
        )
    except (LogProtocolError, OSError, ValueError) as exc:
        print("归档拒绝：%s" % exc, file=sys.stderr)
        return 2

    print(
        "归档完成：文件数=%d 构建=%s 业务配置=%s"
        % (
            len(manifest["files"]),
            manifest["release"]["deployment_build"],
            manifest["release"]["business_fingerprint"],
        )
    )
    print("清单：%s" % (Path(args.archive_root).resolve() / "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

