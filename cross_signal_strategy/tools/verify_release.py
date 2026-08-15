# -*- coding: utf-8 -*-
"""只读检查 cross-v0.3.3 正式发布文件，不调用行情或交易 API。"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


FORMAL_FILES = (
    "local_training_run.py",
    "smart_trade_joinquant_cross_signal_etf.py",
    "smart_trade_ptrade_cross_signal_etf.py",
)

PURE_BUSINESS_FUNCTIONS = {
    "_as_float_array",
    "_date_key",
    "_latest_cross_age_by_diff_recent",
    "_numeric_score",
    "_valid_pair",
    "build_signal_snapshot",
    "business_config_fingerprint",
    "buy_position_scale",
    "calc_atr",
    "calc_bollinger",
    "calc_buy_target_value",
    "calc_dmi_adx",
    "calc_kdj",
    "calc_macd",
    "calc_rsi",
    "calc_stop_price",
    "calc_stress_adjusted_buy_target_value",
    "can_sell_by_signal",
    "crossed_above_by_diff_recent",
    "crossed_above_recent",
    "crossed_below_by_diff_recent",
    "crossed_below_recent",
    "filter_buy_candidates",
    "format_cross_flags",
    "format_indicator_params",
    "format_indicator_values",
    "format_self_check",
    "get_a_share_etf_codes",
    "get_default_params",
    "has_new_buy_position",
    "has_signal_sell_confirmation",
    "is_blocked_entry_combo",
    "is_protected_by_strong_adx_uptrend",
    "is_strong_adx_uptrend",
    "latest_cross_direction_by_diff_recent",
    "portfolio_atr_stress_buy_scale",
    "rsi_group_direction",
    "score_buy_snapshot",
    "score_sell_snapshot",
    "score_skip_reason",
    "should_force_sell",
    "sort_candidates",
    "summarize_cross_signal_candidates",
    "summarize_loose_reversal_candidates",
    "trading_days_between",
}

EXECUTION_CONTRACT_TESTS = {
    "test_cross_signal_strategy.py": {
        "test_same_day_sell_exclusion_backfills_next_ranked_candidate",
        "test_failed_buy_order_consumes_intended_slot_and_does_not_backfill",
        "test_only_confirmed_pause_releases_top_candidate_slot",
    },
    "test_cross_signal_ptrade_strategy.py": {
        "test_buy_execution_waits_for_submitted_sells_to_finish",
        "test_full_sell_callback_immediately_resumes_buy_with_stale_portfolio",
        "test_live_sell_blocks_noncontinuous_status_and_keeps_bounded_retry",
    },
}


def _check(key, label, passed, detail):
    return {
        "key": key,
        "label": label,
        "status": "通过" if passed else "失败",
        "detail": str(detail),
    }


def _constant(tree, name):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    return None


def _literal_return(tree, function_name):
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != function_name:
            continue
        for child in node.body:
            if isinstance(child, ast.Return):
                return ast.literal_eval(child.value)
    return None


def _functions(tree):
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _business_fingerprint(version, params, pool):
    param_text = "|".join(
        "%s=%r" % (key, params[key]) for key in sorted(params)
    )
    pool_text = ",".join(str(code).split(".")[0] for code in pool)
    payload = "%s|%s|%s" % (version, param_text, pool_text)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _forbidden_os_references(tree):
    references = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            references.extend(
                node.lineno
                for alias in node.names
                if alias.name.split(".")[0] == "os"
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "os":
                references.append(node.lineno)
        elif isinstance(node, ast.Name) and node.id == "os":
            references.append(node.lineno)
    return sorted(set(references))


def _early_failure(checks):
    return {
        "status": "失败",
        "strategy_version": None,
        "deployment_build": None,
        "business_fingerprint": None,
        "checks": checks,
    }


def _pytest_result_detail(output, returncode):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    summary = lines[-1] if lines else "无测试输出"
    if returncode == 0:
        return summary
    failed_nodes = []
    for line in lines:
        if not line.startswith("FAILED "):
            continue
        node_id = line[len("FAILED "):].split(" ", 1)[0]
        if node_id and node_id not in failed_nodes:
            failed_nodes.append(node_id)
    if not failed_nodes:
        return summary
    suffix = ""
    if len(failed_nodes) > 5:
        suffix = " 等%d项" % len(failed_nodes)
    return "失败用例=%s%s | %s" % (
        ",".join(failed_nodes[:5]),
        suffix,
        summary,
    )


def verify_release(repo_root, run_tests=False):
    root = Path(repo_root).resolve()
    strategy_root = root / "cross_signal_strategy"
    paths = {name: strategy_root / name for name in FORMAL_FILES}
    checks = []

    missing = [name for name, path in paths.items() if not path.is_file()]
    checks.append(_check(
        "formal_files",
        "三个正式入口存在",
        not missing,
        "完整" if not missing else "缺少=" + ",".join(missing),
    ))
    if missing:
        return _early_failure(checks)

    sources = {
        name: path.read_text(encoding="utf-8")
        for name, path in paths.items()
    }
    trees = {}
    syntax_errors = []
    for name, source in sources.items():
        try:
            trees[name] = ast.parse(source, filename=str(paths[name]))
        except SyntaxError as exc:
            syntax_errors.append("%s:%s" % (name, exc))
    checks.append(_check(
        "syntax",
        "正式入口语法",
        not syntax_errors,
        "全部可解析" if not syntax_errors else "; ".join(syntax_errors),
    ))
    if syntax_errors:
        return _early_failure(checks)

    jq_name = "smart_trade_joinquant_cross_signal_etf.py"
    pt_name = "smart_trade_ptrade_cross_signal_etf.py"
    jq_tree, pt_tree = trees[jq_name], trees[pt_name]
    jq_version = _constant(jq_tree, "STRATEGY_VERSION")
    pt_version = _constant(pt_tree, "STRATEGY_VERSION")
    jq_build = _constant(jq_tree, "DEPLOYMENT_BUILD_ID")
    pt_build = _constant(pt_tree, "DEPLOYMENT_BUILD_ID")
    checks.append(_check(
        "release_identity",
        "版本与构建编号一致",
        bool(
            jq_version
            and jq_version == pt_version
            and jq_build
            and jq_build == pt_build
        ),
        "版本=%s 构建=%s" % (jq_version, jq_build),
    ))

    jq_params = _literal_return(jq_tree, "get_default_params")
    pt_params = _literal_return(pt_tree, "get_default_params")
    jq_pool = _literal_return(jq_tree, "get_default_etf_pool")
    pt_pool = _literal_return(pt_tree, "get_default_etf_pool")
    normalized_jq_pool = [str(code).split(".")[0] for code in (jq_pool or [])]
    normalized_pt_pool = [str(code).split(".")[0] for code in (pt_pool or [])]
    config_matches = (
        isinstance(jq_params, dict)
        and jq_params == pt_params
        and normalized_jq_pool == normalized_pt_pool
    )
    fingerprint = (
        _business_fingerprint(jq_version, jq_params, jq_pool)
        if config_matches and jq_version
        else None
    )
    checks.append(_check(
        "business_config",
        "参数与ETF池一致",
        config_matches,
        "业务配置指纹=%s" % fingerprint,
    ))

    jq_functions = _functions(jq_tree)
    pt_functions = _functions(pt_tree)
    missing_functions = sorted(
        (PURE_BUSINESS_FUNCTIONS - jq_functions.keys())
        | (PURE_BUSINESS_FUNCTIONS - pt_functions.keys())
    )
    mismatches = []
    if not missing_functions:
        for name in sorted(PURE_BUSINESS_FUNCTIONS):
            if ast.dump(
                pt_functions[name], include_attributes=False
            ) != ast.dump(jq_functions[name], include_attributes=False):
                mismatches.append(name)
    parity_ok = not missing_functions and not mismatches
    parity_detail = "核心纯函数全部一致"
    if missing_functions:
        parity_detail = "缺少函数=" + ",".join(missing_functions)
    elif mismatches:
        parity_detail = "不一致函数=" + ",".join(mismatches)
    checks.append(_check(
        "business_parity",
        "聚宽与PTrade核心纯函数一致",
        parity_ok,
        parity_detail,
    ))

    missing_contracts = []
    contract_count = 0
    for test_file, required_names in EXECUTION_CONTRACT_TESTS.items():
        test_path = root / "tests" / test_file
        if not test_path.is_file():
            missing_contracts.append("%s:<missing-file>" % test_file)
            continue
        try:
            test_tree = ast.parse(
                test_path.read_text(encoding="utf-8"),
                filename=str(test_path),
            )
            available_names = set(_functions(test_tree))
        except (OSError, SyntaxError) as exc:
            missing_contracts.append("%s:<unreadable:%s>" % (test_file, exc))
            continue
        contract_count += len(required_names)
        for name in sorted(required_names - available_names):
            missing_contracts.append("%s:%s" % (test_file, name))
    checks.append(_check(
        "execution_contract_tests",
        "跨平台高风险执行合同",
        not missing_contracts,
        (
            "已登记%d条必需合同" % contract_count
            if not missing_contracts
            else "缺少=" + ",".join(missing_contracts)
        ),
    ))

    stale = []
    for name in (jq_name, pt_name):
        if (
            "[cross-v0.1]" in sources[name]
            or "Strategy v0.1 for JoinQuant" in sources[name]
        ):
            stale.append(name)
    checks.append(_check(
        "release_labels",
        "正式源码无过期版本标签",
        not stale,
        "无" if not stale else "过期标签=" + ",".join(stale),
    ))

    os_lines = _forbidden_os_references(pt_tree)
    checks.append(_check(
        "ptrade_forbidden_os",
        "PTrade未使用禁用os模块",
        not os_lines,
        "无" if not os_lines else "行=" + ",".join(map(str, os_lines)),
    ))

    schema_version = _constant(pt_tree, "LIVE_STATE_SCHEMA_VERSION")
    checks.append(_check(
        "state_schema",
        "PTrade状态结构版本有效",
        isinstance(schema_version, int) and schema_version > 0,
        "状态结构=%s" % schema_version,
    ))

    if run_tests:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests",
            "-k",
            "cross_signal",
        ]
        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = (completed.stdout + completed.stderr).strip()
        detail = _pytest_result_detail(output, completed.returncode)
        if completed.returncode != 0 and (
            "PermissionError" in output or "WinError 5" in output
        ):
            detail = "测试环境不可用: " + detail
        checks.append(_check(
            "tests",
            "cross-signal完整自动化测试",
            completed.returncode == 0,
            detail,
        ))

    return {
        "status": (
            "通过"
            if all(item["status"] == "通过" for item in checks)
            else "失败"
        ),
        "strategy_version": jq_version,
        "deployment_build": jq_build,
        "business_fingerprint": fingerprint,
        "checks": checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="检查 cross-signal 正式发布文件")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="仓库根目录",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="同时运行完整测试",
    )
    parser.add_argument("--json", action="store_true", help="输出JSON")
    args = parser.parse_args(argv)
    report = verify_release(args.repo_root, run_tests=args.run_tests)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("发布检查: %s" % report["status"])
        print("策略版本: %s" % report["strategy_version"])
        print("构建编号: %s" % report["deployment_build"])
        print("业务配置指纹: %s" % report["business_fingerprint"])
        for item in report["checks"]:
            print("- [%s] %s: %s" % (
                item["status"], item["label"], item["detail"]
            ))
    return 0 if report["status"] == "通过" else 1


if __name__ == "__main__":
    raise SystemExit(main())
