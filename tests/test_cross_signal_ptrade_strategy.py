# -*- coding: utf-8 -*-
"""Parity and live-safety tests for the PTrade cross-signal strategy."""

from datetime import date, datetime, timedelta
import ast
import importlib.util
from pathlib import Path
import pickle
import re
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.log = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    return module


jq = load_module(
    "cross_signal_joinquant_for_ptrade_parity",
    ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf.py",
)
pt = load_module(
    "cross_signal_ptrade",
    ROOT / "cross_signal_strategy" / "smart_trade_ptrade_cross_signal_etf.py",
)


def make_g(**overrides):
    values = {
        "params": pt.get_default_params(),
        "etf_pool": pt.get_default_etf_pool(),
        "highest_since_buy": {},
        "entry_atr": {},
        "buy_date": {},
        "last_scores": {},
        "sold_today": {},
        "sell_retry_reasons": {},
        "paused_pool_codes": set(),
        "unverified_positions": set(),
        "execution_date": None,
        "deferred_scores": [],
        "deferred_signal_date": None,
        "live_state_schema_version": None,
        "live_state_business_fingerprint": None,
        "live_state_generation": None,
        "live_state_broker_positions": None,
        "state_instance_id": "test-instance-default",
        "__last_snapshot": {},
        "__pending_orders": {},
        "__pending_sells": {},
        "__deferred_buy_after_sell": False,
        "__order_state_unknown": False,
        "__is_live": True,
        "__mode_verified": True,
        "__data": None,
        "__state_path": None,
        "__state_restore_source": None,
        "__state_restore_generation": None,
        "__persisted_g_status": None,
        "__persisted_g_reason": None,
        "__persisted_g_generation": None,
        "__position_recovery_source": {},
        "__startup_recovery_done": False,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_buy_score(code="513100.SS"):
    return {
        "code": code,
        "buy_allowed": True,
        "buy_score": 70,
        "sell_score": 0,
        "reversal_score": 35,
        "location_score": 17,
        "trend_score": 20,
        "volume_score": 0,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": True,
        "close_far_above_ma20": False,
        "rsi6": 50,
        "atr": 0.05,
    }


def make_sell_score(code="513100.SS"):
    score = make_buy_score(code)
    score.update({
        "buy_allowed": False,
        "buy_score": 20,
        "sell_score": 35,
        "close_below_ma20": True,
        "close_below_boll_mid": True,
        "close_below_falling_ma10": False,
        "downside_continuation": False,
        "far_above_ma20_and_rsi6_down": False,
        "adx": 10.0,
        "plus_di": 10.0,
        "minus_di": 20.0,
        "ma20_slope_non_negative": False,
    })
    return score


def test_ptrade_business_configuration_matches_frozen_joinquant_mainline():
    assert pt.STRATEGY_VERSION == jq.STRATEGY_VERSION == "cross-v0.3.2"
    assert pt.DEPLOYMENT_BUILD_ID == jq.DEPLOYMENT_BUILD_ID == "20260723.1"
    assert pt.LIVE_STATE_SCHEMA_VERSION == 3
    assert pt.get_default_params() == jq.get_default_params()
    assert pt.get_default_etf_pool() == [
        "159915.SZ",
        "512100.SS",
        "159928.SZ",
        "513100.SS",
        "513500.SS",
        "513880.SS",
        "513050.SS",
        "518880.SS",
        "159985.SZ",
    ]
    assert pt.business_config_fingerprint() == jq.business_config_fingerprint()


def test_live_audit_log_uses_dedicated_directory_and_mirrors_full_messages(
        tmp_path, monkeypatch):
    platform_messages = []
    raw_log = types.SimpleNamespace(
        info=lambda message, *args: platform_messages.append(
            ("info", message % args if args else str(message))),
        warning=lambda message, *args: platform_messages.append(
            ("warning", message % args if args else str(message))),
        error=lambda message, *args: platform_messages.append(
            ("error", message % args if args else str(message))),
    )
    created = []

    def create_audit_dir(relative_path):
        created.append(relative_path)
        (tmp_path / relative_path).mkdir(parents=True, exist_ok=True)
        return True

    monkeypatch.setattr(pt, "log", raw_log)
    monkeypatch.setattr(
        pt, "get_research_path", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(pt, "create_dir", create_audit_dir, raising=False)
    monkeypatch.setattr(
        pt, "_audit_now", lambda: datetime(2026, 7, 22, 9, 35, 1),
        raising=False)

    assert pt._install_live_audit_log(enabled=True) is True
    detail = (
        "[候选排名] 513500.SS 买入评分=40 卖出评分=55 "
        "RSI[6/12/24]=50.7/50.9/52.4"
    )
    pt.log.info(detail)
    pt.log.warning("[状态恢复] %s", "使用状态台账")
    pt.log.error("[委托恢复] 测试错误")

    audit_path = (
        tmp_path / pt.AUDIT_LOG_DIR / pt.AUDIT_LOG_FILENAME)
    text = audit_path.read_text(encoding="utf-8")
    assert created == [pt.AUDIT_LOG_DIR]
    assert text.splitlines() == [
        "2026-07-22 09:35:01 - INFO - " + detail,
        "2026-07-22 09:35:01 - WARNING - [状态恢复] 使用状态台账",
        "2026-07-22 09:35:01 - ERROR - [委托恢复] 测试错误",
    ]
    assert platform_messages == [
        ("info", detail),
        ("warning", "[状态恢复] 使用状态台账"),
        ("error", "[委托恢复] 测试错误"),
    ]


def test_audit_log_rolls_at_complete_utf8_lines_and_stays_bounded(tmp_path):
    path = tmp_path / "cross_signal_v032_audit.log"
    old_lines = [
        ("旧明细%02d-" % index) + ("甲" * 20)
        for index in range(12)
    ]
    path.write_text("\n".join(old_lines) + "\n", encoding="utf-8")
    newest = "2026-07-22 09:35:00 - INFO - [候选排名] 最新完整明细\n"

    assert pt._append_audit_log_text(
        path, newest, max_bytes=420, compact_target_bytes=260) is True

    raw = path.read_bytes()
    text = raw.decode("utf-8")
    assert len(raw) <= 420
    assert text.endswith(newest)
    assert "旧明细00" not in text
    assert "最新完整明细" in text
    assert all(line.startswith(("旧明细", "2026-07-22")) for line in text.splitlines())


def test_audit_log_compaction_failure_preserves_original_file(
        tmp_path, monkeypatch):
    path = tmp_path / "cross_signal_v032_audit.log"
    original = (("原始完整日志行\n" * 30).encode("utf-8"))
    path.write_bytes(original)

    def fail_replace(self, target):
        raise OSError("simulated replacement interruption")

    monkeypatch.setattr(Path, "replace", fail_replace)

    assert pt._append_audit_log_text(
        path,
        "2026-07-22 09:35:00 - INFO - 新日志\n",
        max_bytes=300,
        compact_target_bytes=180,
    ) is False
    assert path.read_bytes() == original


def test_formal_ptrade_source_has_no_stale_release_labels_and_logs_fingerprint():
    source = (
        ROOT
        / "cross_signal_strategy"
        / "smart_trade_ptrade_cross_signal_etf.py"
    ).read_text(encoding="utf-8")

    assert "[cross-v0.1]" not in source
    assert "[发布指纹]" in source
    assert "LIVE_STATE_SCHEMA_VERSION" in source


def test_ptrade_strategy_does_not_use_platform_forbidden_os_module():
    path = (
        ROOT
        / "cross_signal_strategy"
        / "smart_trade_ptrade_cross_signal_etf.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = set()
    os_reference_lines = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])
        elif isinstance(node, ast.Name) and node.id == "os":
            os_reference_lines.append(node.lineno)

    assert "os" not in imported_modules
    assert os_reference_lines == []


def test_ptrade_source_documents_platform_boundaries_in_chinese():
    path = (
        ROOT
        / "cross_signal_strategy"
        / "smart_trade_ptrade_cross_signal_etf.py"
    )
    source = path.read_text(encoding="utf-8")
    required_sections = (
        "# 一、冻结的业务配置与持久化边界",
        "# 二、PTrade 生命周期与任务调度",
        "# 三、技术指标与交叉信号",
        "# 四、T-1 日线数据与交易日证明",
        "# 五、账户、行情与委托执行",
        "# 六、09:35 主流程与 10:35 补偿",
        "# 七、实盘状态恢复与成交回报",
    )
    legacy_english_comments = (
        "Cross-Signal ETF Strategy v0.3.2 for Guojin PTrade",
        "Business rules are frozen",
        "Reassert code-owned strategy configuration",
        "PTrade backtest entry",
        "Normalize PTrade callbacks",
        "Load pre-adjusted daily bars",
        "Submit buys only against broker-confirmed holdings and cash",
        "Fetch broker delivery records",
        "Verify state, then adopt account positions",
    )

    assert all(section in source for section in required_sections)
    assert not any(comment in source for comment in legacy_english_comments)


def test_ptrade_direct_log_templates_are_chinese():
    path = (
        ROOT
        / "cross_signal_strategy"
        / "smart_trade_ptrade_cross_signal_etf.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned_words = {
        "initialized", "failed", "unavailable", "empty", "invalid",
        "disabled", "blocked", "unverified", "unknown", "unusable",
        "aborted", "skipped", "ignored", "reason", "amount", "limit",
        "price", "order", "paused", "deferred", "waiting", "positions",
        "candidates", "threshold", "date", "rebalance", "scores",
        "summary", "samples", "count", "none", "full", "pool", "loose",
        "close", "total", "cash", "holdings", "cost", "high", "pnl",
        "stop", "resumed", "evaluated", "tracked", "delivery", "records",
        "range", "current", "strategy", "trades", "query", "candidate",
        "source", "generation", "status", "baseline", "partial", "fill",
        "retained", "preserved", "unmatched", "buy", "sell", "qty",
        "cumulative", "valid", "payload", "event", "age_seconds",
    }

    def static_template(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            return static_template(node.left)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = static_template(node.left)
            right = static_template(node.right)
            return left + right if left is not None and right is not None else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return static_template(node.left) or static_template(node.right)
        return None

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "log":
            continue
        if node.func.attr not in {"info", "warning", "error"} or not node.args:
            continue
        template = static_template(node.args[0])
        if template is None or set(template) <= {"="}:
            continue
        words = {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z_-]*", template)}
        if not re.search(r"[\u4e00-\u9fff]", template) or words & banned_words:
            violations.append((node.lineno, template))

    assert violations == []


def test_ptrade_dynamic_log_formatters_translate_business_text():
    self_check = pt._format_self_check_for_log()
    assert "位置差值上穿已启用" in self_check
    assert "自检=通过" in self_check
    assert "enabled" not in self_check

    values = pt._format_indicator_values_for_log(make_buy_score())
    assert "前值" in values
    assert "prev" not in values

    flags = pt._format_cross_flags_for_log({
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_down": False,
    })
    assert "RSI12上穿=是" in flags
    assert "RSI24下穿=否" in flags
    assert "_UP" not in flags
    assert "_DOWN" not in flags

    assert pt._format_reason_for_log("short_data:10<110") == "数据长度不足:10<110"
    assert pt._format_reason_for_log("sell_score 35") == "卖出分 35"
    assert pt._format_reason_for_log("atr_stop 1.000<=1.100") == "ATR止损 1.000<=1.100"


def test_ptrade_pure_business_functions_are_ast_identical_to_joinquant():
    function_names = {
        "_as_float_array",
        "_date_key",
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
        "rsi_group_direction",
        "score_buy_snapshot",
        "score_sell_snapshot",
        "score_skip_reason",
        "should_force_sell",
        "sort_candidates",
        "summarize_cross_signal_candidates",
        "summarize_loose_reversal_candidates",
    }

    def functions(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    jq_functions = functions(
        ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf.py"
    )
    pt_functions = functions(
        ROOT / "cross_signal_strategy" / "smart_trade_ptrade_cross_signal_etf.py"
    )
    assert function_names <= jq_functions.keys()
    assert function_names <= pt_functions.keys()
    for name in sorted(function_names):
        assert ast.dump(
            pt_functions[name], include_attributes=False
        ) == ast.dump(
            jq_functions[name], include_attributes=False
        ), name


def test_before_trading_start_relocks_frozen_business_config_after_restore(monkeypatch):
    today = date(2026, 7, 13)
    stale_params = pt.get_default_params()
    stale_params["buy_threshold"] = 999
    pt.g = make_g(
        params=stale_params,
        etf_pool=["510300.SS"],
        execution_date=today,
        __is_live=False,
    )
    monkeypatch.setattr(pt, "_restore_live_state", lambda: True, raising=False)

    context = types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 8, 30))
    pt.before_trading_start(context, data=None)

    assert pt.g.params == pt.get_default_params()
    assert pt.g.etf_pool == pt.get_default_etf_pool()


def test_explicit_live_state_round_trip_excludes_business_configuration(tmp_path):
    state_path = tmp_path / "cross-signal-state.pkl"
    today = date(2026, 7, 13)
    signal_date = date(2026, 7, 10)
    pt.g = make_g(
        highest_since_buy={"513100.SS": 2.5},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": signal_date},
        last_scores={"513100.SS": make_buy_score()},
        sold_today={"159915.SZ": True},
        sell_retry_reasons={"513500.SS": "ATR止损重试"},
        paused_pool_codes={"513880.SS"},
        unverified_positions={"159985.SZ"},
        execution_date=today,
        deferred_scores=[make_buy_score()],
        deferred_signal_date=signal_date,
    )

    assert pt._persist_live_state(path=state_path) is True

    pt.g.highest_since_buy = {}
    pt.g.entry_atr = {}
    pt.g.buy_date = {}
    pt.g.last_scores = {}
    pt.g.sold_today = {}
    pt.g.sell_retry_reasons = {}
    pt.g.paused_pool_codes = set()
    pt.g.unverified_positions = set()
    pt.g.execution_date = None
    pt.g.deferred_scores = []
    pt.g.deferred_signal_date = None
    pt.g.params = {"buy_threshold": 777}
    pt.g.etf_pool = ["510300.SS"]

    assert pt._restore_live_state(path=state_path) is True
    assert pt.g.highest_since_buy == {"513100.SS": 2.5}
    assert pt.g.entry_atr == {"513100.SS": 0.05}
    assert pt.g.buy_date == {"513100.SS": signal_date}
    assert pt.g.sold_today == {"159915.SZ": True}
    assert pt.g.sell_retry_reasons == {"513500.SS": "ATR止损重试"}
    assert pt.g.paused_pool_codes == {"513880.SS"}
    assert pt.g.unverified_positions == {"159985.SZ"}
    assert pt.g.execution_date == today
    assert pt.g.deferred_signal_date == signal_date
    assert pt.g.params == {"buy_threshold": 777}
    assert pt.g.etf_pool == ["510300.SS"]


def test_state_journal_rejects_state_when_broker_position_changed(tmp_path):
    state_path = tmp_path / "cross-signal-state.pkl"
    original_position = types.SimpleNamespace(
        amount=500, cost_basis=1.0, last_sale_price=1.1)
    original_context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(
            positions={"513100.SS": original_position}))
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": date(2026, 7, 10)},
    )

    assert pt._persist_live_state(original_context, path=state_path) is True

    changed_position = types.SimpleNamespace(
        amount=600, cost_basis=1.0, last_sale_price=1.1)
    changed_context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(
            positions={"513100.SS": changed_position}))
    pt.g = make_g()

    assert pt._restore_live_state(changed_context, path=state_path) is False
    assert pt.g.highest_since_buy == {}
    assert pt.g.entry_atr == {}
    assert pt.g.buy_date == {}


def test_state_journal_restores_state_when_broker_position_is_unchanged(tmp_path):
    state_path = tmp_path / "cross-signal-state.pkl"
    position = types.SimpleNamespace(
        amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(
            positions={"513100.SS": position}))
    buy_date = date(2026, 7, 10)
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": buy_date},
    )

    assert pt._persist_live_state(context, path=state_path) is True

    pt.g = make_g()
    assert pt._restore_live_state(context, path=state_path) is True
    assert pt.g.highest_since_buy == {"513100.SS": 1.2}
    assert pt.g.entry_atr == {"513100.SS": 0.05}
    assert pt.g.buy_date == {"513100.SS": buy_date}


def test_persisted_g_state_is_accepted_when_broker_position_is_unchanged():
    code = "513100.SS"
    buy_date = date(2026, 7, 10)
    position = types.SimpleNamespace(
        amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 8, 30)),
        portfolio=types.SimpleNamespace(positions={code: position}),
    )
    pt.g = make_g(
        highest_since_buy={code: 1.2},
        entry_atr={code: 0.05},
        buy_date={code: buy_date},
        live_state_schema_version=pt.LIVE_STATE_SCHEMA_VERSION,
        live_state_business_fingerprint=pt.business_config_fingerprint(),
        live_state_generation=7,
        live_state_broker_positions={
            code: {"amount": 500.0, "cost": 1.0},
        },
    )

    generation, state = pt._load_persisted_g_state(context)

    assert generation == 7
    assert state["buy_date"] == {code: buy_date}
    assert state["entry_atr"] == {code: 0.05}
    assert state["highest_since_buy"] == {code: 1.2}
    assert pt.g.__persisted_g_status == "accepted"
    assert pt.g.__persisted_g_reason == "validated"
    assert pt.g.__persisted_g_generation == 7


def test_missing_persisted_g_state_records_explicit_not_provided_diagnostic():
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 8, 30)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    pt.g = make_g()

    assert pt._load_persisted_g_state(context) is None
    assert pt.g.__persisted_g_status == "not-provided"
    assert pt.g.__persisted_g_reason == "metadata-missing"
    assert pt.g.__persisted_g_generation is None


@pytest.mark.parametrize(
    ("current_amount", "current_cost"),
    [(600, 1.0), (500, 1.01)],
)
def test_persisted_g_state_is_rejected_when_broker_position_changed(
    current_amount, current_cost
):
    code = "513100.SS"
    position = types.SimpleNamespace(
        amount=current_amount, cost_basis=current_cost, last_sale_price=1.1)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 8, 30)),
        portfolio=types.SimpleNamespace(positions={code: position}),
    )
    pt.g = make_g(
        highest_since_buy={code: 1.2},
        entry_atr={code: 0.05},
        buy_date={code: date(2026, 7, 10)},
        live_state_schema_version=pt.LIVE_STATE_SCHEMA_VERSION,
        live_state_business_fingerprint=pt.business_config_fingerprint(),
        live_state_generation=7,
        live_state_broker_positions={
            code: {"amount": 500.0, "cost": 1.0},
        },
    )

    assert pt._load_persisted_g_state(context) is None
    assert pt.g.__persisted_g_status == "rejected"
    assert pt.g.__persisted_g_reason == "broker-position-snapshot-mismatch"
    assert pt.g.__persisted_g_generation == 7


def test_persist_live_state_records_broker_bound_g_metadata(tmp_path):
    code = "513100.SS"
    state_path = tmp_path / "cross-signal-state.journal"
    position = types.SimpleNamespace(
        amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={code: position}))
    pt.g = make_g(
        highest_since_buy={code: 1.2},
        entry_atr={code: 0.05},
        buy_date={code: date(2026, 7, 10)},
    )

    assert pt._persist_live_state(context, path=state_path) is True

    assert pt.g.live_state_schema_version == pt.LIVE_STATE_SCHEMA_VERSION
    assert (
        pt.g.live_state_business_fingerprint ==
        pt.business_config_fingerprint()
    )
    assert pt.g.live_state_generation == 1
    assert pt.g.live_state_broker_positions == {
        code: {"amount": 500.0, "cost": 1.0},
    }


def test_automatic_live_state_path_is_isolated_by_account_and_trade(monkeypatch, tmp_path):
    user_name_calls = []

    def get_user_name(real_trade):
        user_name_calls.append(real_trade)
        return "account-a"

    monkeypatch.setattr(pt, "get_research_path", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(pt, "get_user_name", get_user_name, raising=False)
    monkeypatch.setattr(pt, "get_trade_name", lambda: "simulation", raising=False)

    simulation_path = pt._live_state_path()
    monkeypatch.setattr(
        pt, "get_user_name", lambda real_trade: "account-b", raising=False
    )
    other_account_path = pt._live_state_path()
    monkeypatch.setattr(pt, "get_user_name", get_user_name, raising=False)
    monkeypatch.setattr(pt, "get_trade_name", lambda: "live", raising=False)
    live_path = pt._live_state_path()

    assert len({simulation_path, other_account_path, live_path}) == 3
    assert Path(simulation_path).name.startswith(
        "cross_signal_v032_live_state_v3_")
    assert Path(simulation_path).suffix == ".journal"
    assert {
        state_parent(simulation_path),
        state_parent(other_account_path),
        state_parent(live_path),
    } == {str(tmp_path)}
    assert "account-a" not in simulation_path
    assert user_name_calls == [False, False]


def test_automatic_live_state_path_survives_manual_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(pt, "get_research_path", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(
        pt, "get_user_name", lambda real_trade: "account-a", raising=False
    )
    monkeypatch.setattr(pt, "get_trade_name", lambda: "cross-signal", raising=False)

    pt.g = make_g(state_instance_id="strategy-instance-a")
    first_path = pt._live_state_path()
    pt.g = make_g(state_instance_id="strategy-instance-b")
    second_path = pt._live_state_path()

    assert first_path == second_path


def test_automatic_live_state_path_fails_closed_without_instance_identity(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(pt, "get_research_path", lambda: str(tmp_path), raising=False)
    monkeypatch.delattr(pt, "get_user_name", raising=False)
    monkeypatch.delattr(pt, "get_trade_name", raising=False)

    assert pt._live_state_path() is None


@pytest.mark.parametrize(
    ("user_name", "trade_name"),
    [(None, "live"), ("account-a", None), ("", "live"), ("account-a", "")],
)
def test_automatic_live_state_path_requires_complete_instance_identity(
    monkeypatch, tmp_path, user_name, trade_name
):
    monkeypatch.setattr(pt, "get_research_path", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(
        pt, "get_user_name", lambda real_trade: user_name, raising=False
    )
    monkeypatch.setattr(pt, "get_trade_name", lambda: trade_name, raising=False)

    assert pt._live_state_path() is None


def state_parent(path):
    return str(Path(path).parent)


def test_missing_live_state_is_a_clean_first_start(monkeypatch, tmp_path):
    errors = []
    monkeypatch.setattr(
        pt,
        "log",
        types.SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: errors.append(args),
        ),
    )
    pt.g = make_g()

    assert pt._restore_live_state(path=tmp_path / "not-created.pkl") is False
    assert errors == []


def test_malformed_live_state_is_rejected_without_partial_restore(tmp_path):
    state_path = tmp_path / "malformed-state.pkl"
    payload = {
        "strategy_version": pt.STRATEGY_VERSION,
        "state": {
            "highest_since_buy": {"513100.SS": 9.9},
            "entry_atr": ["not", "a", "mapping"],
        },
    }
    state_path.write_bytes(pickle.dumps(payload))
    pt.g = make_g(
        highest_since_buy={"513100.SS": 2.5},
        entry_atr={"513100.SS": 0.05},
    )

    assert pt._restore_live_state(path=state_path) is False
    assert pt.g.highest_since_buy == {"513100.SS": 2.5}
    assert pt.g.entry_atr == {"513100.SS": 0.05}


def test_live_state_io_uses_initialize_cached_path(monkeypatch, tmp_path):
    state_path = tmp_path / "cached-cross-signal-state.pkl"
    pt.g = make_g(__state_path=str(state_path))
    monkeypatch.setattr(
        pt,
        "_live_state_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("platform path APIs must not run after initialize")
        ),
    )

    assert pt._persist_live_state() is True
    pt.g.highest_since_buy = {"513100.SS": 9.9}
    assert pt._restore_live_state() is True
    assert pt.g.highest_since_buy == {}


def test_before_trading_prefers_valid_persisted_g_over_broker_history(
    monkeypatch, tmp_path
):
    code = "513100.SS"
    today = date(2026, 7, 13)
    buy_date = date(2026, 7, 10)
    position = types.SimpleNamespace(
        amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 8, 30)),
        portfolio=types.SimpleNamespace(positions={code: position}),
    )
    pt.g = make_g(
        highest_since_buy={code: 1.2},
        entry_atr={code: 0.05},
        buy_date={code: buy_date},
        execution_date=today,
        __state_path=str(tmp_path / "missing-state.journal"),
        live_state_schema_version=pt.LIVE_STATE_SCHEMA_VERSION,
        live_state_business_fingerprint=pt.business_config_fingerprint(),
        live_state_generation=7,
        live_state_broker_positions={
            code: {"amount": 500.0, "cost": 1.0},
        },
    )
    calls = []
    monkeypatch.setattr(pt, "_lock_frozen_business_config", lambda: None)
    monkeypatch.setattr(
        pt, "_reconcile_open_orders", lambda context: calls.append("orders") or True
    )
    monkeypatch.setattr(
        pt,
        "_recover_live_state_with_available_sources",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("valid persisted g must not query broker history")
        ),
    )
    original_recover = pt.recover_live_state

    def validate(context, *args, **kwargs):
        calls.append("validate")
        return original_recover(context, *args, **kwargs)

    monkeypatch.setattr(pt, "recover_live_state", validate)
    monkeypatch.setattr(pt, "_log_live_recovery_summary", lambda context: None)
    monkeypatch.setattr(pt, "_persist_live_state", lambda context: True)

    pt.before_trading_start(context, data=None)

    assert calls == ["orders", "validate"]
    assert pt.g.buy_date == {code: buy_date}
    assert pt.g.entry_atr == {code: 0.05}
    assert pt.g.highest_since_buy == {code: 1.2}
    assert pt.g.unverified_positions == set()
    assert pt.g.__state_restore_source == "ptrade-g"
    assert pt.g.__position_recovery_source == {code: "ptrade-g"}


def test_before_trading_uses_newer_complete_matching_journal_without_broker_history(
    monkeypatch, tmp_path
):
    code = "513100.SS"
    today = date(2026, 7, 13)
    position = types.SimpleNamespace(
        amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 8, 30)),
        portfolio=types.SimpleNamespace(positions={code: position}),
    )
    pt.g = make_g(
        highest_since_buy={code: 1.2},
        entry_atr={code: 0.05},
        buy_date={code: date(2026, 7, 10)},
        execution_date=today,
        __state_path=str(tmp_path / "state.journal"),
        live_state_schema_version=pt.LIVE_STATE_SCHEMA_VERSION,
        live_state_business_fingerprint=pt.business_config_fingerprint(),
        live_state_generation=7,
        live_state_broker_positions={
            code: {"amount": 500.0, "cost": 1.0},
        },
    )
    journal_state = {
        field: getattr(pt.g, field) for field in pt.LIVE_STATE_FIELDS
    }
    journal_state["highest_since_buy"] = {code: 1.4}
    journal_state["entry_atr"] = {code: 0.06}
    calls = []

    def load_journal(context):
        pt.g.__state_restore_source = "journal"
        pt.g.__state_restore_generation = 8
        return journal_state

    monkeypatch.setattr(pt, "_load_live_state", load_journal)
    monkeypatch.setattr(pt, "_lock_frozen_business_config", lambda: None)
    monkeypatch.setattr(pt, "_reconcile_open_orders", lambda context: True)
    monkeypatch.setattr(
        pt,
        "_recover_live_state_with_available_sources",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete broker-bound journal must not query history")
        ),
    )
    monkeypatch.setattr(pt, "_log_live_recovery_summary", lambda context: None)
    monkeypatch.setattr(pt, "_persist_live_state", lambda context: True)

    pt.before_trading_start(context, data=None)

    assert calls == []
    assert pt.g.highest_since_buy == {code: 1.4}
    assert pt.g.entry_atr == {code: 0.06}
    assert pt.g.__state_restore_source == "journal"
    assert pt.g.__state_restore_generation == 8
    assert pt.g.__persisted_g_status == "superseded"
    assert pt.g.__persisted_g_reason == "newer-journal"
    assert pt.g.__persisted_g_generation == 7
    assert pt.g.__position_recovery_source == {code: "journal"}


def test_before_trading_uses_complete_matching_journal_when_framework_g_missing(
    monkeypatch, tmp_path
):
    code = "513100.SS"
    today = date(2026, 7, 13)
    buy_date = date(2026, 7, 10)
    position = types.SimpleNamespace(
        amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 8, 30)),
        portfolio=types.SimpleNamespace(positions={code: position}),
    )
    pt.g = make_g(
        execution_date=today,
        __state_path=str(tmp_path / "state.journal"),
    )
    journal_state = {
        field: getattr(pt.g, field) for field in pt.LIVE_STATE_FIELDS
    }
    journal_state["highest_since_buy"] = {code: 1.4}
    journal_state["entry_atr"] = {code: 0.06}
    journal_state["buy_date"] = {code: buy_date}

    def load_journal(context):
        pt.g.__state_restore_source = "journal"
        pt.g.__state_restore_generation = 8
        return journal_state

    monkeypatch.setattr(pt, "_load_live_state", load_journal)
    monkeypatch.setattr(pt, "_lock_frozen_business_config", lambda: None)
    monkeypatch.setattr(pt, "_reconcile_open_orders", lambda context: True)
    monkeypatch.setattr(
        pt,
        "_recover_live_state_with_available_sources",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete broker-bound journal must not query history")
        ),
    )
    monkeypatch.setattr(pt, "_log_live_recovery_summary", lambda context: None)
    monkeypatch.setattr(pt, "_persist_live_state", lambda context: True)

    pt.before_trading_start(context, data=None)

    assert pt.g.buy_date == {code: buy_date}
    assert pt.g.entry_atr == {code: 0.06}
    assert pt.g.highest_since_buy == {code: 1.4}
    assert pt.g.unverified_positions == set()
    assert pt.g.__position_recovery_source == {code: "journal"}


def test_before_trading_keeps_broker_recovery_for_incomplete_matching_journal(
    monkeypatch, tmp_path
):
    code = "513100.SS"
    today = date(2026, 7, 13)
    position = types.SimpleNamespace(
        amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 8, 30)),
        portfolio=types.SimpleNamespace(positions={code: position}),
    )
    pt.g = make_g(
        execution_date=today,
        __state_path=str(tmp_path / "state.journal"),
    )
    journal_state = {
        field: getattr(pt.g, field) for field in pt.LIVE_STATE_FIELDS
    }
    journal_state["highest_since_buy"] = {code: 1.4}
    journal_state["entry_atr"] = {code: 0.06}
    journal_state["buy_date"] = {}
    calls = []

    def load_journal(context):
        pt.g.__state_restore_source = "journal"
        pt.g.__state_restore_generation = 8
        return journal_state

    def broker_recovery(context, allow_deliver):
        calls.append(("broker", allow_deliver))
        pt.g.buy_date = {code: date(2026, 7, 10)}
        pt.g.entry_atr = {code: 0.05}
        pt.g.highest_since_buy = {code: 1.3}
        pt.g.unverified_positions = set()
        pt.g.__position_recovery_source = {
            code: "account-takeover:get-deliver"
        }

    monkeypatch.setattr(pt, "_load_live_state", load_journal)
    monkeypatch.setattr(pt, "_lock_frozen_business_config", lambda: None)
    monkeypatch.setattr(pt, "_reconcile_open_orders", lambda context: True)
    monkeypatch.setattr(
        pt, "_recover_live_state_with_available_sources", broker_recovery)
    monkeypatch.setattr(pt, "_log_live_recovery_summary", lambda context: None)
    monkeypatch.setattr(pt, "_persist_live_state", lambda context: True)

    pt.before_trading_start(context, data=None)

    assert calls == [("broker", True)]
    assert pt.g.buy_date == {code: date(2026, 7, 10)}
    assert pt.g.__position_recovery_source == {
        code: "account-takeover:get-deliver"
    }


def test_before_trading_recovers_broker_before_journal_risk_fallback(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "cross-signal-state.pkl"
    today = date(2026, 7, 13)
    pt.g = make_g(__state_path=None, execution_date=today)
    calls = []

    monkeypatch.setattr(
        pt,
        "_live_state_path",
        lambda: calls.append("path") or str(state_path),
    )

    journal_state = {"journal": "state"}

    def load(context):
        assert pt.g.__state_path == str(state_path)
        calls.append("load-journal")
        return journal_state

    monkeypatch.setattr(pt, "_load_live_state", load, raising=False)
    monkeypatch.setattr(
        pt,
        "_restore_live_state_continuity",
        lambda state: calls.append(("continuity", state)),
        raising=False,
    )
    monkeypatch.setattr(pt, "_lock_frozen_business_config", lambda: calls.append("lock"))
    monkeypatch.setattr(
        pt,
        "_clear_live_risk_state_for_broker_recovery",
        lambda: calls.append("clear-risk"),
        raising=False,
    )
    monkeypatch.setattr(
        pt, "_reconcile_open_orders", lambda context: calls.append("orders") or True
    )
    monkeypatch.setattr(
        pt,
        "_recover_live_state_with_available_sources",
        lambda context, allow_deliver: calls.append(("recover", allow_deliver)),
    )
    monkeypatch.setattr(
        pt,
        "_restore_live_state_risk_fallback",
        lambda context, state: calls.append(("fallback", state)),
        raising=False,
    )
    monkeypatch.setattr(
        pt, "recover_live_state", lambda context: calls.append("validate")
    )
    monkeypatch.setattr(pt, "_log_live_recovery_summary", lambda context: None)
    monkeypatch.setattr(
        pt,
        "_persist_live_state",
        lambda context: calls.append("persist") or True,
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 8, 30)),
        portfolio=types.SimpleNamespace(positions={}),
    )

    pt.before_trading_start(context, data=None)

    assert calls == [
        "path",
        "load-journal",
        ("continuity", journal_state),
        "lock",
        "clear-risk",
        "orders",
        ("recover", True),
        ("fallback", journal_state),
        "validate",
        "persist",
    ]


def test_journal_risk_fallback_does_not_overwrite_broker_recovery():
    code = "513100.SS"
    position = types.SimpleNamespace(
        amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={code: position}))
    broker_buy_date = date(2026, 7, 10)
    pt.g = make_g(
        highest_since_buy={code: 1.3},
        entry_atr={code: 0.06},
        buy_date={code: broker_buy_date},
        unverified_positions=set(),
        __position_recovery_source={code: "account-takeover:get-deliver"},
    )
    journal_state = {
        "highest_since_buy": {code: 1.2},
        "entry_atr": {code: 0.05},
        "buy_date": {code: date(2026, 7, 9)},
    }

    restored = pt._restore_live_state_risk_fallback(context, journal_state)

    assert restored == set()
    assert pt.g.highest_since_buy == {code: 1.3}
    assert pt.g.entry_atr == {code: 0.06}
    assert pt.g.buy_date == {code: broker_buy_date}
    assert pt.g.__position_recovery_source == {
        code: "account-takeover:get-deliver"
    }


def test_journal_risk_fallback_fills_only_unproved_old_position():
    code = "513100.SS"
    position = types.SimpleNamespace(
        amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={code: position}))
    journal_buy_date = date(2020, 1, 2)
    pt.g = make_g(unverified_positions={code})
    journal_state = {
        "highest_since_buy": {code: 1.2},
        "entry_atr": {code: 0.05},
        "buy_date": {code: journal_buy_date},
    }

    restored = pt._restore_live_state_risk_fallback(context, journal_state)

    assert restored == {code}
    assert pt.g.highest_since_buy == {code: 1.2}
    assert pt.g.entry_atr == {code: 0.05}
    assert pt.g.buy_date == {code: journal_buy_date}
    assert pt.g.__position_recovery_source == {code: "journal"}


def test_before_trading_uses_matching_journal_when_delivery_cannot_prove_old_buy(
    monkeypatch, tmp_path
):
    code = "513100.SS"
    state_path = tmp_path / "cross-signal-state.pkl"
    position = types.SimpleNamespace(
        amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 20, 8, 30)),
        portfolio=types.SimpleNamespace(positions={code: position}),
    )
    old_buy_date = date(2020, 1, 2)
    pt.g = make_g(
        highest_since_buy={code: 1.2},
        entry_atr={code: 0.05},
        buy_date={code: old_buy_date},
    )
    assert pt._persist_live_state(context, path=state_path) is True

    pt.g = make_g(__state_path=str(state_path))
    monkeypatch.setattr(pt, "_lock_frozen_business_config", lambda: None)
    monkeypatch.setattr(pt, "_reconcile_open_orders", lambda context: True)

    def broker_recovery(context, allow_deliver):
        assert allow_deliver is True
        pt.g.unverified_positions = {code}
        pt.g.__position_recovery_source = {code: "unverified"}

    monkeypatch.setattr(
        pt, "_recover_live_state_with_available_sources", broker_recovery
    )
    monkeypatch.setattr(pt, "_log_live_recovery_summary", lambda context: None)

    pt.before_trading_start(context, data=None)

    assert pt.g.buy_date == {code: old_buy_date}
    assert pt.g.entry_atr == {code: 0.05}
    assert pt.g.highest_since_buy == {code: 1.2}
    assert pt.g.unverified_positions == set()
    assert pt.g.__position_recovery_source == {code: "journal"}
    assert pt.g.__startup_recovery_done is True


def test_live_state_uses_one_bounded_journal(tmp_path):
    state_path = tmp_path / "cross-signal-state.pkl"
    pt.g = make_g(highest_since_buy={"513100.SS": 2.5})

    assert pt._persist_live_state(path=state_path) is True
    pt.g.highest_since_buy = {"513100.SS": 3.0}
    assert pt._persist_live_state(path=state_path) is True

    assert state_path.exists()
    assert not Path(str(state_path) + ".a").exists()
    assert not Path(str(state_path) + ".b").exists()
    with state_path.open("rb") as handle:
        first = pickle.load(handle)
        second = pickle.load(handle)
        with pytest.raises(EOFError):
            pickle.load(handle)

    assert first["schema_version"] == pt.LIVE_STATE_SCHEMA_VERSION
    assert first["business_config_fingerprint"] == pt.business_config_fingerprint()
    assert first["generation"] == 1
    assert first["payload"][:2] == b"\x80\x04"
    assert second["generation"] == 2
    assert second["payload"][:2] == b"\x80\x04"


def test_live_state_journal_keeps_only_two_latest_valid_records(tmp_path):
    state_path = tmp_path / "cross-signal-state.journal"
    pt.g = make_g(highest_since_buy={"513100.SS": 2.5})

    assert pt._persist_live_state(path=state_path) is True
    pt.g.highest_since_buy = {"513100.SS": 3.0}
    assert pt._persist_live_state(path=state_path) is True
    pt.g.highest_since_buy = {"513100.SS": 3.5}
    assert pt._persist_live_state(path=state_path) is True

    records = pt._read_live_state_journal(state_path)
    assert [generation for generation, _, _ in records] == [2, 3]
    assert records[0][1]["highest_since_buy"] == {"513100.SS": 3.0}
    assert records[1][1]["highest_since_buy"] == {"513100.SS": 3.5}


def test_live_state_compaction_failure_keeps_original_journal_recoverable(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "cross-signal-state.journal"
    pt.g = make_g(highest_since_buy={"513100.SS": 2.5})

    assert pt._persist_live_state(path=state_path) is True
    pt.g.highest_since_buy = {"513100.SS": 3.0}
    assert pt._persist_live_state(path=state_path) is True

    def fail_replace(self, target):
        raise OSError("simulated interruption before replace")

    monkeypatch.setattr(Path, "replace", fail_replace)
    pt.g.highest_since_buy = {"513100.SS": 3.5}
    assert pt._persist_live_state(path=state_path) is True

    records = pt._read_live_state_journal(state_path)
    assert [generation for generation, _, _ in records] == [1, 2, 3]

    pt.g.highest_since_buy = {"513100.SS": 9.9}
    assert pt._restore_live_state(path=state_path) is True
    assert pt.g.highest_since_buy == {"513100.SS": 3.5}
    assert pt.g.__state_restore_generation == 3


def test_live_state_does_not_append_an_identical_snapshot(tmp_path):
    state_path = tmp_path / "cross-signal-state.journal"
    pt.g = make_g(highest_since_buy={"513100.SS": 2.5})

    assert pt._persist_live_state(path=state_path) is True
    first_size = state_path.stat().st_size
    assert pt.g.live_state_generation == 1

    assert pt._persist_live_state(path=state_path) is True

    assert state_path.stat().st_size == first_size
    assert pt.g.live_state_generation == 1
    with state_path.open("rb") as handle:
        assert pickle.load(handle)["generation"] == 1
        with pytest.raises(EOFError):
            pickle.load(handle)


def test_live_state_reuses_verified_journal_tail_for_changed_snapshot(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "cross-signal-state.journal"
    pt.g = make_g(highest_since_buy={"513100.SS": 2.5})
    original_scan = pt._scan_live_state_journal
    scans = []

    def tracking_scan(path):
        scans.append(str(path))
        return original_scan(path)

    monkeypatch.setattr(pt, "_scan_live_state_journal", tracking_scan)
    assert pt._persist_live_state(path=state_path) is True

    pt.g.highest_since_buy = {"513100.SS": 3.0}
    assert pt._persist_live_state(path=state_path) is True

    assert scans == [str(state_path)]
    assert pt.g.live_state_generation == 2
    with state_path.open("rb") as handle:
        assert pickle.load(handle)["generation"] == 1
        assert pickle.load(handle)["generation"] == 2
        with pytest.raises(EOFError):
            pickle.load(handle)


def test_live_state_restore_uses_last_complete_record_when_tail_is_truncated(tmp_path):
    state_path = tmp_path / "cross-signal-state.pkl"
    pt.g = make_g(highest_since_buy={"513100.SS": 2.5})
    assert pt._persist_live_state(path=state_path) is True
    pt.g.highest_since_buy = {"513100.SS": 3.0}
    assert pt._persist_live_state(path=state_path) is True

    with state_path.open("ab") as handle:
        handle.write(b"truncated")
    pt.g.highest_since_buy = {"513100.SS": 9.9}

    assert pt._restore_live_state(path=state_path) is True
    assert pt.g.highest_since_buy == {"513100.SS": 3.0}
    assert pt.g.__state_restore_source == "journal"
    assert pt.g.__state_restore_generation == 2


def test_live_state_persist_repairs_truncated_tail_before_appending(tmp_path):
    state_path = tmp_path / "cross-signal-state.pkl"
    pt.g = make_g(highest_since_buy={"513100.SS": 2.5})
    assert pt._persist_live_state(path=state_path) is True
    with state_path.open("ab") as handle:
        handle.write(b"truncated")

    pt.g.highest_since_buy = {"513100.SS": 3.0}
    assert pt._persist_live_state(path=state_path) is True

    pt.g.highest_since_buy = {"513100.SS": 9.9}
    assert pt._restore_live_state(path=state_path) is True
    assert pt.g.highest_since_buy == {"513100.SS": 3.0}
    assert pt.g.__state_restore_generation == 2


def test_live_state_envelope_rejects_checksum_mismatch():
    state = {
        field: getattr(make_g(), field)
        for field in pt.LIVE_STATE_FIELDS
    }
    envelope = pt._encode_live_state_envelope(state, generation=1)
    envelope["checksum"] = "0" * 64

    with pytest.raises(ValueError, match="checksum"):
        pt._decode_live_state_envelope(envelope)


def test_live_state_envelope_rejects_business_fingerprint_mismatch():
    state = {
        field: getattr(make_g(), field)
        for field in pt.LIVE_STATE_FIELDS
    }
    envelope = pt._encode_live_state_envelope(state, generation=1)
    envelope["business_config_fingerprint"] = "incompatible-business-config"

    with pytest.raises(ValueError, match="business fingerprint"):
        pt._decode_live_state_envelope(envelope)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("state schema mismatch", "状态结构版本不匹配"),
        ("business fingerprint mismatch", "业务配置指纹不匹配"),
        ("state checksum mismatch", "状态校验和不匹配"),
        ("missing state fields: buy_date", "缺少状态字段: buy_date"),
        ("invalid broker position snapshot", "券商持仓快照结构无效"),
        ("invalid broker position facts: 513100.SS", "券商持仓事实无效: 513100.SS"),
        ("unprovable broker position: BAD", "券商持仓无法证明: BAD"),
    ],
)
def test_state_journal_internal_errors_are_formatted_in_chinese(raw, expected):
    assert pt._format_state_error_for_log(ValueError(raw)) == expected


def test_live_state_schema_accepts_compatible_producer_strategy_version(tmp_path):
    state_path = tmp_path / "cross-signal-state.pkl"
    state = {
        field: getattr(make_g(highest_since_buy={"513100.SS": 2.5}), field)
        for field in pt.LIVE_STATE_FIELDS
    }
    envelope = pt._encode_live_state_envelope(state, generation=7)
    envelope["producer_strategy_version"] = "cross-v0.3.1"
    envelope["checksum"] = pt._live_state_checksum(
        envelope["schema_version"],
        envelope["generation"],
        envelope["producer_strategy_version"],
        envelope["payload"],
    )
    state_path.write_bytes(pickle.dumps(envelope, protocol=4))
    pt.g = make_g(highest_since_buy={"513100.SS": 9.9})

    assert pt._restore_live_state(path=state_path) is True
    assert pt.g.highest_since_buy == {"513100.SS": 2.5}
    assert pt.g.__state_restore_source == "journal"
    assert pt.g.__state_restore_generation == 7


def test_live_state_unknown_schema_is_rejected_without_partial_restore(tmp_path):
    state_path = tmp_path / "cross-signal-state.pkl"
    state = {
        field: getattr(make_g(highest_since_buy={"513100.SS": 7.7}), field)
        for field in pt.LIVE_STATE_FIELDS
    }
    envelope = pt._encode_live_state_envelope(state, generation=1)
    envelope["schema_version"] = pt.LIVE_STATE_SCHEMA_VERSION + 1
    envelope["checksum"] = pt._live_state_checksum(
        envelope["schema_version"],
        envelope["generation"],
        envelope["producer_strategy_version"],
        envelope["payload"],
    )
    state_path.write_bytes(pickle.dumps(envelope, protocol=4))
    pt.g = make_g(
        highest_since_buy={"513100.SS": 2.5},
        entry_atr={"513100.SS": 0.05},
    )

    assert pt._restore_live_state(path=state_path) is False
    assert pt.g.highest_since_buy == {"513100.SS": 2.5}
    assert pt.g.entry_atr == {"513100.SS": 0.05}
    assert pt.g.__state_restore_source is None


def test_live_state_unframed_legacy_file_is_not_trusted(tmp_path):
    state_path = tmp_path / "cross-signal-state.pkl"
    legacy_state = {
        field: getattr(make_g(highest_since_buy={"513100.SS": 2.5}), field)
        for field in pt.LIVE_STATE_FIELDS
    }
    state_path.write_bytes(pickle.dumps({
        "strategy_version": pt.STRATEGY_VERSION,
        "state": legacy_state,
    }, protocol=4))
    pt.g = make_g(highest_since_buy={"513100.SS": 9.9})

    assert pt._restore_live_state(path=state_path) is False
    assert pt.g.highest_since_buy == {"513100.SS": 9.9}
    assert pt.g.__state_restore_source is None
    assert not Path(str(state_path) + ".a").exists()
    assert not Path(str(state_path) + ".b").exists()


def test_live_schedule_and_official_after_trading_callback_checkpoint_state(monkeypatch):
    calls = []
    monkeypatch.setattr(pt, "do_trading", lambda context: calls.append("trade"))
    monkeypatch.setattr(pt, "halt_recover", lambda context: calls.append("recover"))
    monkeypatch.setattr(pt, "after_close", lambda context: calls.append("close"))
    monkeypatch.setattr(
        pt,
        "_recover_live_state_with_available_sources",
        lambda context, allow_deliver: calls.append(("state-recovery", allow_deliver)),
    )
    monkeypatch.setattr(
        pt, "_persist_live_state", lambda context: calls.append("persist") or True,
        raising=False,
    )
    monkeypatch.setattr(
        pt, "get_open_orders", lambda: calls.append("open-orders") or [],
        raising=False,
    )
    pt.g = make_g(sold_today={"513100.SS": True})
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={})
    )

    pt._do_trading_wrapper(context)
    pt._halt_recover_wrapper(context)
    pt.after_trading_end(context, data=None)

    assert calls == [
        "trade", "persist",
        "recover", "persist",
        ("state-recovery", True), "open-orders", "close", "persist",
    ]
    assert pt.g.sold_today == {}


def test_after_trading_end_logs_unfinished_orders_without_mutating_guards(monkeypatch):
    warnings = []
    pending_orders = {"513100.SS": {"order_id": "buy-1"}}
    pending_sells = {"513500.SS": {"order_id": "sell-1"}}
    pt.g = make_g(
        __pending_orders=pending_orders.copy(),
        __pending_sells=pending_sells.copy(),
    )
    monkeypatch.setattr(
        pt,
        "get_open_orders",
        lambda: [
            types.SimpleNamespace(
                id="broker-order-1",
                symbol="513100.SS",
                status="2",
                amount=1000,
                filled=200,
            )
        ],
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "_recover_live_state_with_available_sources",
        lambda context, allow_deliver: None,
    )
    monkeypatch.setattr(pt, "after_close", lambda context: None)
    monkeypatch.setattr(pt, "_persist_live_state", lambda context: True)
    monkeypatch.setattr(pt.log, "warning", lambda message: warnings.append(message))
    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(positions={}))

    pt.after_trading_end(context, data=None)

    assert any(
        "[盘后委托核对]" in message and "broker-order-1" in message
        for message in warnings
    )
    assert pt.g.__pending_orders == pending_orders
    assert pt.g.__pending_sells == pending_sells


def test_order_and_trade_callbacks_checkpoint_state(monkeypatch):
    persisted = []
    monkeypatch.setattr(
        pt, "_persist_live_state", lambda context: persisted.append(context) or True,
        raising=False,
    )
    pt.g = make_g()
    context = types.SimpleNamespace()

    pt.on_order_response(context, [])
    pt.on_trade_response(context, [])

    assert persisted == [context, context]


def test_callbacks_reject_non_dict_records_without_crashing(monkeypatch):
    class CallbackLike:
        def get(self, key, default=None):
            return default

    warnings = []
    persisted = []
    pt.g = make_g()
    monkeypatch.setattr(pt.log, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(
        pt,
        "_persist_live_state",
        lambda context: persisted.append(context) or True,
        raising=False,
    )
    context = types.SimpleNamespace()

    pt.on_order_response(context, [CallbackLike()])
    pt.on_trade_response(context, [CallbackLike()])

    assert persisted == [context, context]
    assert sum("回报格式异常" in message for message in warnings) == 2


def test_ptrade_scoring_and_stop_math_match_joinquant_mainline():
    snapshot = {
        "rsi6": 61.0,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": True,
        "macd_cross_up": False,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": True,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": True,
        "close_far_above_ma20": False,
        "ma5_gt_ma10": True,
        "ma10_gt_ma20": False,
        "ma20_slope_non_negative": True,
        "close_gt_ma60": True,
        "downside_continuation": False,
        "volume_above_vol20_and_up": False,
        "vol5_gt_vol20": True,
    }

    assert pt.score_buy_snapshot(snapshot) == jq.score_buy_snapshot(snapshot)
    assert pt.calc_stop_price(10.0, 0.2, 8.0) == jq.calc_stop_price(10.0, 0.2, 8.0)


def test_joinquant_and_ptrade_choose_same_buy_on_same_synthetic_day(monkeypatch):
    today = date(2026, 7, 14)
    prev_date = date(2026, 7, 13)
    jq_code = "513100.XSHG"
    pt_code = "513100.SS"
    jq_score = make_buy_score(jq_code)
    pt_score = make_buy_score(pt_code)
    for score in (jq_score, pt_score):
        score.update({
            "close": 2.0,
            "loose_reversal_count": 0,
            "rsi_turn_up": False,
            "rsi6_delta": 0.0,
            "macd_turn_up": False,
            "dif_delta": 0.0,
            "kdj_turn_up": False,
            "k_delta": 0.0,
            "j_delta": 0.0,
        })

    jq_orders = []
    jq.g = types.SimpleNamespace(
        params=jq.get_default_params(),
        etf_pool=[jq_code],
        highest_since_buy={},
        entry_atr={},
        buy_date={},
        last_scores={},
    )
    jq_context = types.SimpleNamespace(
        current_dt=datetime(2026, 7, 14, 9, 35),
        portfolio=types.SimpleNamespace(
            positions={}, total_value=20000.0, available_cash=20000.0),
    )
    monkeypatch.setattr(jq, "get_prev_trade_date", lambda context: prev_date)
    monkeypatch.setattr(
        jq,
        "get_current_data",
        lambda: {jq_code: types.SimpleNamespace(paused=False, last_price=2.0)},
        raising=False,
    )
    monkeypatch.setattr(
        jq,
        "calc_cross_signal_score",
        lambda code, end_date, return_reason=False: (dict(jq_score), None),
    )
    monkeypatch.setattr(
        jq, "get_trade_days", lambda **kwargs: [prev_date, today], raising=False)
    monkeypatch.setattr(
        jq,
        "order_target_value",
        lambda code, value: jq_orders.append(("buy", code.split(".")[0])),
        raising=False,
    )

    pt_orders = []
    pt.g = make_g(etf_pool=[pt_code])
    pt_context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 14, 9, 35)),
        portfolio=types.SimpleNamespace(
            positions={}, portfolio_value=20000.0, cash=20000.0),
    )
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: prev_date)
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_current_price", lambda code: 2.0)
    monkeypatch.setattr(
        pt,
        "calc_cross_signal_score",
        lambda code, end_date, return_reason=False: (dict(pt_score), None),
    )
    monkeypatch.setattr(
        pt, "get_trade_days", lambda **kwargs: [prev_date, today], raising=False)
    monkeypatch.setattr(pt, "log_iopv_buy_observation", lambda *args: None)
    monkeypatch.setattr(
        pt,
        "order",
        lambda code, shares, limit_price=None: (
            pt_orders.append(("buy", code.split(".")[0])) or "buy-order-1"
        ),
        raising=False,
    )

    jq.do_trading(jq_context)
    pt.do_trading(pt_context)

    assert jq_orders == pt_orders == [("buy", "513100")]


def test_ptrade_normalizes_callback_codes_to_universe_format():
    assert pt.normalize_code("513100") == "513100.SS"
    assert pt.normalize_code("159915") == "159915.SZ"
    assert pt.normalize_code("513100.XSHG") == "513100.SS"
    assert pt.normalize_code("159915.XSHE") == "159915.SZ"


def test_live_price_fails_closed_when_snapshot_is_unavailable(monkeypatch):
    pt.g = make_g()
    history_called = []
    monkeypatch.setattr(pt, "get_snapshot", lambda code: {}, raising=False)
    monkeypatch.setattr(
        pt, "get_history", lambda *args, **kwargs: history_called.append(True), raising=False
    )

    assert pt.get_current_price("513100.SS") is None
    assert history_called == []


def test_live_price_rejects_snapshot_from_previous_session(monkeypatch):
    pt.g = make_g()
    stale_time = datetime.now() - timedelta(days=1)
    monkeypatch.setattr(
        pt,
        "get_snapshot",
        lambda code: {
            code: {
                "last_px": 2.0,
                "hsTimeStamp": stale_time.strftime("%Y%m%d%H%M%S"),
            }
        },
        raising=False,
    )

    assert pt.get_current_price("513100.SS") is None
    assert pt.g.__last_snapshot == {}


def test_live_price_accepts_snapshot_from_current_session(monkeypatch):
    pt.g = make_g()
    current_time = datetime.now()
    monkeypatch.setattr(
        pt,
        "get_snapshot",
        lambda code: {
            code: {
                "last_px": 2.0,
                "hsTimeStamp": current_time.strftime("%Y%m%d%H%M%S"),
            }
        },
        raising=False,
    )

    assert pt.get_current_price("513100.SS") == pytest.approx(2.0)
    assert pt.g.__last_snapshot["513100.SS"]["last_px"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "timestamp",
    [
        "20260720",
        "20260720092500",
        "20260720093600",
    ],
)
def test_live_price_rejects_malformed_stale_or_future_snapshot(
    monkeypatch, timestamp
):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 7, 20, 9, 35, 0)

    pt.g = make_g()
    monkeypatch.setattr(pt, "datetime", FixedDateTime)
    monkeypatch.setattr(
        pt,
        "get_snapshot",
        lambda code: {
            code: {
                "last_px": 2.0,
                "hsTimeStamp": timestamp,
            }
        },
        raising=False,
    )

    assert pt.get_current_price("513100.SS") is None
    assert pt.g.__last_snapshot == {}


def test_live_price_accepts_snapshot_inside_five_minute_safety_window(monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 7, 20, 9, 35, 0)

    pt.g = make_g()
    monkeypatch.setattr(pt, "datetime", FixedDateTime)
    monkeypatch.setattr(
        pt,
        "get_snapshot",
        lambda code: {
            code: {
                "last_px": 2.0,
                "hsTimeStamp": "20260720093100",
            }
        },
        raising=False,
    )

    assert pt.get_current_price("513100.SS") == pytest.approx(2.0)


def test_iopv_observation_uses_the_same_snapshot_as_the_live_buy_price():
    observation = pt.build_iopv_observation(
        "513100.SS",
        {
            "last_px": 2.0,
            "iopv": "1.95",
            "hsTimeStamp": "20260713093500123",
        },
        execution_price=2.0,
        observed_at=datetime(2026, 7, 13, 9, 35, 1),
    )

    assert observation["code"] == "513100.SS"
    assert observation["valid"] is True
    assert observation["market_price"] == pytest.approx(2.0)
    assert observation["iopv"] == pytest.approx(1.95)
    assert observation["premium"] == pytest.approx(2.0 / 1.95 - 1.0)
    assert observation["snapshot_age_seconds"] == pytest.approx(1.0)


def test_iopv_observation_is_limited_to_qdii_and_tolerates_missing_values():
    assert pt.build_iopv_observation(
        "159915.SZ", {"last_px": 2.0, "iopv": 1.95}, 2.0
    ) is None

    observation = pt.build_iopv_observation(
        "513500.SS", {"last_px": 2.0, "iopv": 0}, 2.0
    )
    assert observation["valid"] is False
    assert observation["iopv"] is None
    assert observation["premium"] is None


def test_prev_trade_date_does_not_guess_weekdays_when_apis_fail(monkeypatch):
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 35))
    )
    monkeypatch.setattr(
        pt, "get_trading_day", lambda day: (_ for _ in ()).throw(RuntimeError()),
        raising=False,
    )
    monkeypatch.setattr(
        pt, "get_trade_days", lambda **kwargs: (_ for _ in ()).throw(RuntimeError()),
        raising=False,
    )
    monkeypatch.setattr(
        pt, "get_all_trades_days", lambda **kwargs: (_ for _ in ()).throw(RuntimeError()),
        raising=False,
    )

    assert pt.get_prev_trade_date(context) is None


def test_prev_trade_date_prefers_documented_get_trading_day(monkeypatch):
    calls = []

    def get_trading_day(day):
        calls.append(day)
        return date(2026, 7, 13) if day == 0 else date(2026, 7, 10)

    monkeypatch.setattr(pt, "get_trading_day", get_trading_day, raising=False)
    monkeypatch.setattr(
        pt,
        "get_trade_days",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("fallback calendar should not run")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "get_all_trades_days",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("fallback calendar should not run")
        ),
        raising=False,
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 2, 46))
    )

    assert pt.get_prev_trade_date(context) == date(2026, 7, 10)
    assert calls == [0, -1]


def test_prev_trade_date_logs_unusable_calendar_payloads(monkeypatch):
    messages = []
    monkeypatch.setattr(
        pt,
        "log",
        types.SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda message, *args: messages.append(
                message % args if args else message
            ),
            error=lambda message, *args: messages.append(
                message % args if args else message
            ),
        ),
    )
    monkeypatch.setattr(
        pt, "get_trading_day", lambda day: "not-a-trading-date", raising=False
    )
    monkeypatch.setattr(pt, "get_trade_days", lambda **kwargs: [], raising=False)
    monkeypatch.setattr(pt, "get_all_trades_days", lambda **kwargs: [], raising=False)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 2, 46))
    )

    assert pt.get_prev_trade_date(context) is None
    assert any(
        "get_trading_day返回值不可用" in message and "类型=str" in message
        for message in messages
    )
    assert any(
        "get_trade_days返回值不可用" in message and "类型=list" in message
        for message in messages
    )


def test_ptrade_calendar_queries_use_documented_string_dates(monkeypatch):
    captured_end_dates = []
    pt.g = make_g()

    def fake_get_trade_days(**kwargs):
        captured_end_dates.append(kwargs.get("end_date"))
        return [date(2026, 7, 10), date(2026, 7, 13)]

    monkeypatch.setattr(pt, "get_trade_days", fake_get_trade_days, raising=False)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 35))
    )

    assert pt.get_prev_trade_date(context) == date(2026, 7, 10)
    assert pt._get_signal_hold_days(date(2026, 7, 13)) == [
        date(2026, 7, 10),
        date(2026, 7, 13),
    ]
    assert pt._previous_trade_date_before(date(2026, 7, 13)) == date(2026, 7, 10)
    assert captured_end_dates == ["20260713", "20260713", "20260713"]


@pytest.mark.parametrize("raw_date", ["2026-07-15", "20260715"])
def test_as_date_accepts_ptrade_numpy_string_scalars(raw_date):
    assert pt._as_date(pt.np.str_(raw_date)) == date(2026, 7, 15)


def test_previous_trade_date_accepts_ptrade_unicode_ndarray(monkeypatch):
    calls = []

    def get_trade_days(**kwargs):
        calls.append(("get_trade_days", kwargs))
        return pt.np.array(["2026-07-15", "2026-07-16"], dtype="<U10")

    monkeypatch.setattr(pt, "get_trade_days", get_trade_days, raising=False)
    monkeypatch.setattr(
        pt,
        "get_all_trades_days",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("first documented calendar result should be sufficient")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "get_trading_day_by_date",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("non-binding diagnostic probe should not run")
        ),
        raising=False,
    )

    assert pt._previous_trade_date_before(date(2026, 7, 16)) == date(2026, 7, 15)
    assert calls == [("get_trade_days", {"end_date": "20260716", "count": 2})]


def test_signal_sell_requires_verified_trading_calendar_for_five_day_hold():
    buy_date = date(2026, 7, 10)
    today = date(2026, 7, 15)

    assert not pt.can_sell_with_verified_calendar(
        buy_date, today, min_hold_days=5, trade_days=None
    )
    assert pt.can_sell_with_verified_calendar(
        buy_date,
        date(2026, 7, 17),
        min_hold_days=5,
        trade_days=[
            date(2026, 7, 10),
            date(2026, 7, 13),
            date(2026, 7, 14),
            date(2026, 7, 15),
            date(2026, 7, 16),
            date(2026, 7, 17),
        ],
    )


def test_signal_sell_accepts_ptrade_unicode_ndarray_calendar():
    trade_days = pt.np.array(
        [
            "2026-07-10",
            "2026-07-13",
            "2026-07-14",
            "2026-07-15",
            "2026-07-16",
            "2026-07-17",
        ],
        dtype="<U10",
    )

    assert pt.can_sell_with_verified_calendar(
        date(2026, 7, 10),
        date(2026, 7, 17),
        min_hold_days=5,
        trade_days=trade_days,
    )


def test_daily_signal_loader_uses_pre_adjusted_data_ending_at_t_minus_one(monkeypatch):
    captured = {}

    def fake_get_price(code, **kwargs):
        captured.update({"code": code, **kwargs})
        return pt.pd.DataFrame({
            "open": [1.0, 1.1],
            "close": [1.1, 1.2],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "volume": [100, 200],
        })

    monkeypatch.setattr(pt, "get_price", fake_get_price, raising=False)
    frame = pt.get_price_data("513100.SS", date(2021, 12, 30), 120)

    assert len(frame) == 2
    assert captured["end_date"] == "2021-12-30"
    assert captured["frequency"] == "1d"
    assert captured["fq"] == "pre"


def test_signal_loader_supports_python311_long_get_history_fallback(monkeypatch):
    code = "510300.SS"
    index = pt.pd.to_datetime(["2026-07-09", "2026-07-10"])
    values = {
        "open": [4.0, 4.1],
        "high": [4.2, 4.3],
        "low": [3.9, 4.0],
        "close": [4.1, 4.2],
        "volume": [1000.0, 1200.0],
    }

    def fail_get_price(*args, **kwargs):
        raise RuntimeError("force documented get_history fallback")

    def fake_get_history(count, frequency, field, security_list, fq, include, **kwargs):
        return pt.pd.DataFrame(
            {"code": [code, code], field: values[field]},
            index=index,
        )

    monkeypatch.setattr(pt, "get_price", fail_get_price, raising=False)
    monkeypatch.setattr(pt, "get_history", fake_get_history, raising=False)

    frame = pt.get_price_data(code, date(2026, 7, 10), 2)

    assert frame is not None
    assert list(frame.columns) == ["open", "close", "high", "low", "volume"]
    assert frame.index.tolist() == index.tolist()
    assert frame["close"].tolist() == [4.1, 4.2]


def test_signal_loader_supports_legacy_wide_get_history_fallback(monkeypatch):
    code = "510300.SS"
    index = pt.pd.to_datetime(["2026-07-09", "2026-07-10"])
    values = {
        "open": [4.0, 4.1],
        "high": [4.2, 4.3],
        "low": [3.9, 4.0],
        "close": [4.1, 4.2],
        "volume": [1000.0, 1200.0],
    }

    monkeypatch.setattr(
        pt,
        "get_price",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fallback")),
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "get_history",
        lambda count, frequency, field, security_list, fq, include, **kwargs:
            pt.pd.DataFrame({code: values[field]}, index=index),
        raising=False,
    )

    frame = pt.get_price_data(code, date(2026, 7, 10), 2)

    assert frame is not None
    assert frame["close"].tolist() == [4.1, 4.2]


def test_signal_loader_rejects_history_without_provable_date_index(monkeypatch):
    code = "510300.SS"
    values = {
        "open": [4.0, 4.1],
        "high": [4.2, 4.3],
        "low": [3.9, 4.0],
        "close": [4.1, 4.2],
        "volume": [1000.0, 1200.0],
    }

    monkeypatch.setattr(
        pt,
        "get_price",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fallback")),
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "get_history",
        lambda count, frequency, field, security_list, fq, include, **kwargs:
            pt.pd.DataFrame(
                {"code": [code, code], field: values[field]}
            ),
        raising=False,
    )

    assert pt.get_price_data(code, date(2026, 7, 10), 2) is None


def test_backtest_current_price_supports_python311_long_history_fallback(monkeypatch):
    code = "510300.SS"
    pt.g = make_g(__is_live=False, __data=None)

    def fake_get_history(count, frequency, field, security_list, fq, include, **kwargs):
        return pt.pd.DataFrame(
            {"code": [code], "close": [4.23]},
            index=pt.pd.to_datetime(["2026-07-10"]),
        )

    monkeypatch.setattr(pt, "get_history", fake_get_history, raising=False)

    assert pt.get_current_price(code) == pytest.approx(4.23)


def test_sell_submission_keeps_state_until_full_fill(monkeypatch):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(positions={"513100.SS": position}))
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": date(2026, 6, 1)},
        last_scores={"513100.SS": {"buy_score": 60}},
    )
    orders = []
    messages = []
    monkeypatch.setattr(pt.log, "info", lambda message: messages.append(message))
    monkeypatch.setattr(pt, "get_current_price", lambda code: 1.1)
    monkeypatch.setattr(pt, "get_sell_limit_price", lambda code, price: 1.0)
    monkeypatch.setattr(
        pt,
        "order_target",
        lambda code, amount, limit_price=None: (
            orders.append((code, amount, limit_price)) or "sell-order-1"
        ),
        raising=False,
    )

    assert pt.execute_sell("513100.SS", context, "test")
    assert orders == [("513100.SS", 0, 1.0)]
    assert "513100.SS" in pt.g.highest_since_buy
    assert pt.g.__pending_sells["513100.SS"]["requested_qty"] == 500
    assert pt.g.__pending_sells["513100.SS"]["order_id"] == "sell-order-1"
    assert any(
        message.startswith("[卖出委托]") and "sell-order-1" in message
        for message in messages
    )


def test_sell_submission_failure_does_not_create_guard(monkeypatch):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g(highest_since_buy={"513100.SS": 1.2})
    monkeypatch.setattr(pt, "get_current_price", lambda code: 1.1)
    monkeypatch.setattr(pt, "get_sell_limit_price", lambda code, price: 1.0)
    monkeypatch.setattr(pt, "order_target", lambda *args, **kwargs: None, raising=False)

    assert not pt.execute_sell("513100.SS", context, "test")
    assert pt.g.__pending_sells == {}
    assert pt.g.sold_today == {}


def test_partial_buy_callbacks_accumulate_before_pending_is_cleared():
    pt.g = make_g(
        __pending_orders={
            "513100.SS": {
                "requested_qty": 500,
                "filled_qty": 0,
                "filled_value": 0.0,
                "atr": 0.05,
                "buy_date": date(2026, 7, 10),
                "order_id": "buy-order-1",
            }
        }
    )
    context = types.SimpleNamespace()

    pt.on_trade_response(context, {
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 200,
        "business_price": 1.10,
        "order_id": "buy-order-1",
    })
    assert pt.g.__pending_orders["513100.SS"]["filled_qty"] == 200
    assert pt.g.highest_since_buy["513100.SS"] == pytest.approx(1.10)

    pt.on_trade_response(context, {
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 300,
        "business_price": 1.20,
        "order_id": "buy-order-1",
    })
    assert "513100.SS" not in pt.g.__pending_orders
    assert pt.g.highest_since_buy["513100.SS"] == pytest.approx(1.16)


def test_duplicate_buy_trade_callback_is_counted_once():
    pt.g = make_g(
        __pending_orders={
            "513100.SS": {
                "requested_qty": 500,
                "filled_qty": 0,
                "filled_value": 0.0,
                "atr": 0.05,
                "buy_date": date(2026, 7, 10),
                "order_id": "buy-order-1",
            }
        }
    )
    trade = {
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 200,
        "business_price": 1.10,
        "business_id": "buy-fill-1",
        "order_id": "buy-order-1",
    }

    pt.on_trade_response(types.SimpleNamespace(), trade)
    pt.on_trade_response(types.SimpleNamespace(), dict(trade))

    pending = pt.g.__pending_orders["513100.SS"]
    assert pending["filled_qty"] == 200
    assert pending["filled_value"] == pytest.approx(220.0)


def test_duplicate_sell_trade_callback_is_counted_once():
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": date(2026, 6, 1)},
        sold_today={"513100.SS": True},
        __pending_sells={
            "513100.SS": {
                "requested_qty": 500,
                "filled_qty": 0,
                "order_id": "sell-order-1",
            }
        },
    )
    trade = {
        "stock_code": "513100",
        "entrust_bs": "2",
        "business_amount": 200,
        "business_price": 1.15,
        "business_id": "sell-fill-1",
        "order_id": "sell-order-1",
    }

    pt.on_trade_response(types.SimpleNamespace(), trade)
    pt.on_trade_response(types.SimpleNamespace(), dict(trade))

    assert pt.g.__pending_sells["513100.SS"]["filled_qty"] == 200
    assert "513100.SS" in pt.g.highest_since_buy


def test_cancel_trade_push_is_not_counted_as_a_fill():
    pt.g = make_g(
        __pending_orders={
            "513100.SS": {
                "requested_qty": 500,
                "filled_qty": 0,
                "filled_value": 0.0,
                "atr": 0.05,
                "buy_date": date(2026, 7, 10),
                "order_id": "buy-order-1",
            }
        }
    )

    pt.on_trade_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 500,
        "business_price": 1.10,
        "order_id": "buy-order-1",
        "real_type": "2",
    })

    assert pt.g.__pending_orders["513100.SS"]["filled_qty"] == 0
    assert "513100.SS" not in pt.g.highest_since_buy


def test_delayed_callback_for_old_order_does_not_touch_current_guard():
    pt.g = make_g(
        __pending_sells={
            "513100.SS": {
                "requested_qty": 500,
                "filled_qty": 0,
                "order_id": "sell-order-new",
            }
        }
    )

    pt.on_trade_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "2",
        "business_amount": 500,
        "business_price": 1.15,
        "order_id": "sell-order-old",
        "real_type": "0",
    })

    assert pt.g.__pending_sells["513100.SS"]["filled_qty"] == 0


def test_full_sell_callback_clears_strategy_state():
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": date(2026, 6, 1)},
        last_scores={"513100.SS": {"buy_score": 60}},
        sold_today={"513100.SS": True},
        __pending_sells={
            "513100.SS": {
                "requested_qty": 500, "filled_qty": 0, "order_id": "sell-order-1"
            }
        },
    )

    pt.on_trade_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "2",
        "business_amount": 500,
        "business_price": 1.15,
        "order_id": "sell-order-1",
    })

    assert "513100.SS" not in pt.g.__pending_sells
    assert "513100.SS" not in pt.g.highest_since_buy
    assert "513100.SS" not in pt.g.entry_atr
    assert "513100.SS" not in pt.g.buy_date


def test_0936_trade_query_recovers_missing_sell_callback_and_resumes_buy(
        monkeypatch):
    today = date(2026, 7, 23)
    score = make_buy_score("518880.SS")
    context = types.SimpleNamespace(
        current_dt=datetime(2026, 7, 23, 9, 36),
        portfolio=types.SimpleNamespace(
            positions={},
            cash=5000.0,
            portfolio_value=20000.0,
        ),
    )
    pt.g = make_g(
        highest_since_buy={"513050.SS": 1.145},
        entry_atr={"513050.SS": 0.036429},
        buy_date={"513050.SS": date(2026, 7, 14)},
        last_scores={"513050.SS": make_sell_score("513050.SS")},
        sold_today={"513050.SS": True},
        execution_date=today,
        deferred_signal_date=date(2026, 7, 22),
        deferred_scores=[score],
        __pending_sells={
            "513050.SS": {
                "requested_qty": 2100,
                "filled_qty": 0.0,
                "reason": "sell_score 34",
                "order_id": "sell-order-1",
            }
        },
        __deferred_buy_after_sell=True,
    )
    monkeypatch.setattr(
        pt,
        "_fetch_current_strategy_trades",
        lambda: [{
            "stock_code": "513050.XSHG",
            "entrust_bs": "2",
            "business_amount": 2100.0,
            "business_price": 1.082,
            "business_id": "sell-fill-1",
            "order_id": "sell-order-1",
            "_recovery_source": "get-trades",
        }],
    )
    buy_calls = []
    monkeypatch.setattr(
        pt,
        "execute_buy_candidates",
        lambda received_context, scores, received_today: (
            buy_calls.append((received_context, scores, received_today)) or 1
        ),
    )

    pt.reconcile_recent_fills_and_resume_buys(context)

    assert pt.g.__pending_sells == {}
    assert "513050.SS" not in pt.g.highest_since_buy
    assert "513050.SS" not in pt.g.entry_atr
    assert "513050.SS" not in pt.g.buy_date
    assert pt.g.__deferred_buy_after_sell is False
    assert buy_calls == [(context, [score], today)]


def test_live_pause_check_fails_closed_when_status_is_unknown(monkeypatch):
    pt.g = make_g()
    monkeypatch.setattr(pt, "get_stock_status", lambda *args, **kwargs: {}, raising=False)

    assert pt.is_paused("513100.SS") is True


def test_live_pause_check_fails_closed_when_status_value_is_none(monkeypatch):
    pt.g = make_g()
    monkeypatch.setattr(
        pt,
        "get_stock_status",
        lambda *args, **kwargs: {"513100.SS": None},
        raising=False,
    )

    assert pt.is_paused("513100.SS") is True


def test_live_pause_check_refreshes_status_instead_of_trusting_stale_snapshot(monkeypatch):
    pt.g = make_g(__last_snapshot={"513100.SS": {"trade_status": "HALT"}})
    monkeypatch.setattr(
        pt,
        "get_stock_status",
        lambda *args, **kwargs: {"513100.SS": False},
        raising=False,
    )

    assert pt.is_paused("513100.SS") is False


def test_partial_sell_callbacks_keep_state_until_cumulative_full_fill():
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": date(2026, 6, 1)},
        last_scores={"513100.SS": {"buy_score": 60}},
        sold_today={"513100.SS": True},
        __pending_sells={
            "513100.SS": {
                "requested_qty": 500, "filled_qty": 0, "order_id": "sell-order-1"
            }
        },
    )

    pt.on_trade_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "2",
        "business_amount": 200,
        "business_price": 1.15,
        "order_id": "sell-order-1",
    })
    assert pt.g.__pending_sells["513100.SS"]["filled_qty"] == 200
    assert "513100.SS" in pt.g.highest_since_buy

    pt.on_trade_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "2",
        "business_amount": 300,
        "business_price": 1.14,
        "order_id": "sell-order-1",
    })
    assert "513100.SS" not in pt.g.__pending_sells
    assert "513100.SS" not in pt.g.highest_since_buy


def test_rejected_sell_releases_retry_guard_without_clearing_position_state():
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": date(2026, 6, 1)},
        sold_today={"513100.SS": True},
        __pending_sells={
            "513100.SS": {
                "requested_qty": 500,
                "filled_qty": 0,
                "order_id": "sell-order-1",
                "reason": "atr_stop 0.900<=0.950",
            }
        },
    )

    pt.on_order_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "2",
        "status": "9",
        "business_amount": 0,
        "error_info": "rejected",
        "order_id": "sell-order-1",
    })

    assert "513100.SS" not in pt.g.__pending_sells
    assert "513100.SS" not in pt.g.sold_today
    assert "513100.SS" in pt.g.highest_since_buy
    assert pt.g.sell_retry_reasons == {
        "513100.SS": "atr_stop 0.900<=0.950",
    }


def test_partial_cancelled_sell_keeps_risk_state_for_remaining_position():
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": date(2026, 6, 1)},
        sold_today={"513100.SS": True},
        __pending_sells={
            "513100.SS": {
                "requested_qty": 500,
                "filled_qty": 200,
                "order_id": "sell-order-1",
                "reason": "atr_stop 0.900<=0.950",
            }
        },
    )

    pt.on_order_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "2",
        "status": "5",
        "business_amount": 200,
        "error_info": "partial cancel",
        "order_id": "sell-order-1",
    })

    assert "513100.SS" not in pt.g.__pending_sells
    assert "513100.SS" not in pt.g.sold_today
    assert "513100.SS" in pt.g.highest_since_buy
    assert "513100.SS" in pt.g.entry_atr
    assert pt.g.sell_retry_reasons == {
        "513100.SS": "atr_stop 0.900<=0.950",
    }


def test_before_trading_clears_expired_day_order_guards(monkeypatch):
    pt.g = make_g(
        __pending_orders={"513100.SS": {"requested_qty": 100}},
        __pending_sells={"159915.SZ": {"requested_qty": 100}},
        sold_today={"159915.SZ": True},
        sell_retry_reasons={"159915.SZ": "atr_stop 0.900<=0.950"},
    )
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)
    monkeypatch.setattr(pt, "get_open_orders", lambda: [], raising=False)

    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 0)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    pt.before_trading_start(context, data={})

    assert pt.g.__pending_orders == {}
    assert pt.g.__pending_sells == {}
    assert pt.g.sold_today == {}
    assert pt.g.sell_retry_reasons == {}


def test_before_trading_rebuilds_guards_from_broker_open_orders(monkeypatch):
    pt.g = make_g(last_scores={"513100.SS": {"atr": 0.05}})
    open_orders = [
        types.SimpleNamespace(
            id="buy-order-open", symbol="513100.XSHG", amount=500, filled=200
        ),
        types.SimpleNamespace(
            id="sell-order-open", symbol="159915.XSHE", amount=-300, filled=100
        ),
    ]
    monkeypatch.setattr(pt, "get_open_orders", lambda: open_orders, raising=False)
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 40)),
        portfolio=types.SimpleNamespace(positions={}),
    )

    pt.before_trading_start(context, data={})

    assert pt.g.__pending_orders["513100.SS"]["requested_qty"] == 500
    assert pt.g.__pending_orders["513100.SS"]["filled_qty"] == 200
    assert pt.g.__pending_orders["513100.SS"]["order_id"] == "buy-order-open"
    assert pt.g.__pending_sells["159915.SZ"]["requested_qty"] == 300
    assert pt.g.__pending_sells["159915.SZ"]["filled_qty"] == 100
    assert pt.g.__pending_sells["159915.SZ"]["order_id"] == "sell-order-open"
    assert pt.g.sold_today["159915.SZ"] is True
    assert pt.g.__order_state_unknown is False


def test_before_trading_fails_closed_on_duplicate_symbol_open_orders(monkeypatch):
    pt.g = make_g()
    open_orders = [
        types.SimpleNamespace(
            id="buy-order-1", symbol="513100.XSHG", amount=500, filled=0
        ),
        types.SimpleNamespace(
            id="buy-order-2", symbol="513100.XSHG", amount=500, filled=0
        ),
    ]
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 40)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    monkeypatch.setattr(pt, "get_open_orders", lambda: open_orders, raising=False)
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)

    pt.before_trading_start(context, data={})

    assert pt.g.__order_state_unknown is True
    assert pt.g.__pending_orders == {}
    assert pt.g.__pending_sells == {}


@pytest.mark.parametrize(
    ("amount", "filled"),
    [(float("nan"), 0), (500, float("nan")), (500, float("inf")), (500, 600)],
)
def test_before_trading_fails_closed_on_invalid_open_order_quantities(
    monkeypatch, amount, filled
):
    pt.g = make_g()
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 40)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    monkeypatch.setattr(
        pt,
        "get_open_orders",
        lambda: [
            types.SimpleNamespace(
                id="buy-order-1",
                symbol="513100.XSHG",
                amount=amount,
                filled=filled,
            )
        ],
        raising=False,
    )
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)

    pt.before_trading_start(context, data={})

    assert pt.g.__order_state_unknown is True
    assert pt.g.__pending_orders == {}


def test_recovered_partial_buy_without_synced_cost_remains_unverified(monkeypatch):
    pt.g = make_g(last_scores={"513100.SS": {"atr": 0.05}})
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 40)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    monkeypatch.setattr(
        pt,
        "get_open_orders",
        lambda: [
            types.SimpleNamespace(
                id="buy-order-1",
                symbol="513100.XSHG",
                amount=500,
                filled=200,
            )
        ],
        raising=False,
    )

    pt.before_trading_start(context, data={})
    pt.on_trade_response(context, {
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 300,
        "business_price": 1.20,
        "order_id": "buy-order-1",
        "real_type": "0",
    })

    assert "513100.SS" not in pt.g.__pending_orders
    assert "513100.SS" in pt.g.unverified_positions
    assert "513100.SS" not in pt.g.highest_since_buy


def test_nonfinite_buy_fill_price_keeps_position_unverified():
    pt.g = make_g(
        __pending_orders={
            "513100.SS": {
                "requested_qty": 500,
                "filled_qty": 0,
                "filled_value": 0.0,
                "fill_value_complete": True,
                "atr": 0.05,
                "buy_date": date(2026, 7, 10),
                "order_id": "buy-order-1",
            }
        }
    )

    pt.on_trade_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 500,
        "business_price": float("nan"),
        "order_id": "buy-order-1",
        "real_type": "0",
    })

    assert "513100.SS" not in pt.g.__pending_orders
    assert "513100.SS" in pt.g.unverified_positions
    assert "513100.SS" not in pt.g.highest_since_buy


def test_before_trading_marks_order_state_unknown_when_reconciliation_fails(monkeypatch):
    pt.g = make_g()
    monkeypatch.setattr(
        pt,
        "get_open_orders",
        lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
        raising=False,
    )
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)

    pt.before_trading_start(types.SimpleNamespace(), data={})

    assert pt.g.__order_state_unknown is True


def test_before_trading_fails_closed_when_open_order_response_is_none(monkeypatch):
    pt.g = make_g()
    monkeypatch.setattr(pt, "get_open_orders", lambda: None, raising=False)
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)

    pt.before_trading_start(types.SimpleNamespace(), data={})

    assert pt.g.__order_state_unknown is True


def test_trading_aborts_when_broker_order_state_is_unknown(monkeypatch):
    pt.g = make_g(__order_state_unknown=True)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 35))
    )
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: date(2026, 7, 10))
    monkeypatch.setattr(
        pt,
        "check_atr_stops",
        lambda context: (_ for _ in ()).throw(AssertionError("must not evaluate")),
    )

    pt.do_trading(context)


def test_0935_defers_paused_pool_codes_but_processes_open_codes(monkeypatch):
    paused = "513100.SS"
    open_code = "159985.SZ"
    today = date(2026, 7, 13)
    prev_date = date(2026, 7, 10)
    pt.g = make_g(etf_pool=[paused, open_code])
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 35)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    stop_checks = []
    scored = []
    buys = []
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: prev_date)
    monkeypatch.setattr(pt, "is_paused", lambda code: code == paused)
    monkeypatch.setattr(
        pt,
        "check_atr_stops",
        lambda context: stop_checks.append(True) or [],
    )
    monkeypatch.setattr(
        pt,
        "calc_cross_signal_score",
        lambda code, end_date, return_reason=False: (
            scored.append((code, end_date)) or {
                **make_buy_score(code), "close": 1.0}, None),
    )
    monkeypatch.setattr(
        pt,
        "get_trade_days",
        lambda end_date, count: [prev_date, today],
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "execute_buy_candidates",
        lambda context, scores, execution_date: (
            buys.append([score["code"] for score in scores]) or 0),
    )

    pt.do_trading(context)

    assert stop_checks == [True]
    assert pt.g.paused_pool_codes == {paused}
    assert scored == [(open_code, prev_date)]
    assert buys == [[open_code]]


def test_buy_execution_waits_for_submitted_sells_to_finish(monkeypatch):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(
            positions={"159915.SZ": position}, portfolio_value=20000, cash=500
        )
    )
    pt.g = make_g(
        __pending_sells={"159915.SZ": {"requested_qty": 500, "filled_qty": 0}}
    )
    orders = []
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_current_price", lambda code: 2.0)
    monkeypatch.setattr(
        pt,
        "order",
        lambda *args, **kwargs: orders.append((args, kwargs)) or "buy-order-1",
        raising=False,
    )

    assert pt.execute_buy_candidates(
        context, [make_buy_score()], date(2026, 7, 13)
    ) == 0
    assert orders == []


def test_buy_execution_uses_confirmed_cash_and_creates_fill_guard(monkeypatch):
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={}, portfolio_value=20000, cash=20000)
    )
    pt.g = make_g()
    orders = []
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_current_price", lambda code: 2.0)
    monkeypatch.setattr(
        pt,
        "order",
        lambda *args, **kwargs: orders.append((args, kwargs)) or "buy-order-1",
        raising=False,
    )

    assert pt.execute_buy_candidates(
        context, [make_buy_score()], date(2026, 7, 13)
    ) == 1
    assert orders == [(('513100.SS', 3100), {'limit_price': 2.0})]
    assert pt.g.__pending_orders["513100.SS"]["requested_qty"] == 3100
    assert pt.g.__pending_orders["513100.SS"]["order_id"] == "buy-order-1"


def test_unverified_held_position_blocks_every_new_buy(monkeypatch):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(
            positions={"159915.SZ": position}, portfolio_value=20000, cash=15000
        )
    )
    pt.g = make_g(unverified_positions={"159915.SZ"})
    orders = []
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_current_price", lambda code: 2.0)
    monkeypatch.setattr(
        pt,
        "order",
        lambda *args, **kwargs: orders.append((args, kwargs)) or "buy-order-1",
        raising=False,
    )

    assert pt.execute_buy_candidates(
        context, [make_buy_score("513100.SS")], date(2026, 7, 13)
    ) == 0
    assert orders == []


def test_unverified_holding_does_not_block_verified_holding_signal_exit(monkeypatch):
    today = date(2026, 7, 13)
    verified_buy_date = date(2026, 6, 1)
    positions = {
        "513100.SS": types.SimpleNamespace(
            amount=500, cost_basis=1.0, last_sale_price=0.9),
        "159915.SZ": types.SimpleNamespace(
            amount=500, cost_basis=1.0, last_sale_price=0.9),
    }
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions=positions)
    )
    pt.g = make_g(
        buy_date={"159915.SZ": verified_buy_date},
        entry_atr={"159915.SZ": 0.05},
        highest_since_buy={"159915.SZ": 1.1},
        unverified_positions={"513100.SS"},
    )
    sold = []
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(
        pt,
        "execute_sell",
        lambda code, context, reason: sold.append((code, reason)) or True,
    )

    assert pt._evaluate_signal_sell(
        context,
        "159915.SZ",
        make_sell_score("159915.SZ"),
        today,
        [
            verified_buy_date,
            date(2026, 6, 2),
            date(2026, 6, 3),
            date(2026, 6, 4),
            date(2026, 6, 5),
            today,
        ],
    ) is True
    assert sold == [("159915.SZ", "sell_score 35")]


def test_observation_only_sell_risk_log_does_not_claim_stop_tightening(monkeypatch):
    code = "513100.SS"
    buy_date = date(2026, 6, 1)
    today = date(2026, 7, 13)
    position = types.SimpleNamespace(
        amount=500, cost_basis=1.0, last_sale_price=1.0)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={code: position}))
    pt.g = make_g(
        buy_date={code: buy_date},
        entry_atr={code: 0.05},
        highest_since_buy={code: 1.1},
    )
    score = make_sell_score(code)
    score.update({
        "sell_score": 18,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
    })
    messages = []
    monkeypatch.setattr(pt, "is_paused", lambda candidate: False)
    monkeypatch.setattr(
        pt.log, "info", lambda message: messages.append(str(message)))
    monkeypatch.setattr(
        pt,
        "execute_sell",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("observation-only score must not sell")
        ),
    )

    assert pt._evaluate_signal_sell(
        context,
        code,
        score,
        today,
        [
            buy_date,
            date(2026, 6, 2),
            date(2026, 6, 3),
            date(2026, 6, 4),
            date(2026, 6, 5),
            today,
        ],
    ) is False

    assert any("[卖出风险观察]" in message for message in messages)
    assert all("[风险收紧]" not in message for message in messages)


@pytest.mark.parametrize(
    ("iopv", "expected_valid"),
    [("1.95", "有效=True"), (0, "有效=False")],
)
def test_qdii_buy_logs_iopv_but_never_changes_order_path(
    monkeypatch, iopv, expected_valid
):
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 35, 1)),
        portfolio=types.SimpleNamespace(positions={}, portfolio_value=20000, cash=20000),
    )
    pt.g = make_g(
        __last_snapshot={
            "513100.SS": {
                "last_px": 2.0,
                "iopv": iopv,
                "hsTimeStamp": "20260713093500123",
            }
        }
    )
    events = []

    def log_info(message, *args):
        rendered = message % args if args else message
        events.append(("log", rendered))

    monkeypatch.setattr(
        pt,
        "log",
        types.SimpleNamespace(info=log_info, warning=log_info, error=log_info),
    )
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_current_price", lambda code: 2.0)
    monkeypatch.setattr(
        pt,
        "order",
        lambda *args, **kwargs: events.append(("order", (args, kwargs))) or "buy-order-1",
        raising=False,
    )

    assert pt.execute_buy_candidates(
        context, [make_buy_score()], date(2026, 7, 13)
    ) == 1
    assert ("order", (("513100.SS", 3100), {"limit_price": 2.0})) in events
    assert any(
        kind == "log" and value.startswith("[买入委托]") and "buy-order-1" in value
        for kind, value in events
    )
    observations = [
        (index, value)
        for index, (kind, value) in enumerate(events)
        if kind == "log" and value.startswith("[IOPV观察]")
    ]
    assert len(observations) == 1
    assert expected_valid in observations[0][1]
    order_index = next(index for index, item in enumerate(events) if item[0] == "order")
    assert observations[0][0] < order_index


def test_non_qdii_buy_does_not_emit_iopv_observation(monkeypatch):
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 35)),
        portfolio=types.SimpleNamespace(positions={}, portfolio_value=20000, cash=20000),
    )
    pt.g = make_g()
    messages = []
    monkeypatch.setattr(
        pt,
        "log",
        types.SimpleNamespace(
            info=lambda message, *args: messages.append(message),
            warning=lambda *args: None,
            error=lambda *args: None,
        ),
    )
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_current_price", lambda code: 2.0)
    monkeypatch.setattr(pt, "order", lambda *args, **kwargs: "buy-order-1", raising=False)

    assert pt.execute_buy_candidates(
        context, [make_buy_score("159915.SZ")], date(2026, 7, 13)
    ) == 1
    assert not any(message.startswith("[IOPV观察]") for message in messages)


def test_release_docs_keep_iopv_observation_non_binding():
    deployment = (
        ROOT / "cross_signal_strategy" / "docs" / "ptrade_deployment.md"
    ).read_text(encoding="utf-8")
    decisions = (
        ROOT / "cross_signal_strategy" / "docs" / "decisions.md"
    ).read_text(encoding="utf-8")
    readme = (ROOT / "cross_signal_strategy" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "[IOPV观察]" in deployment
    assert "must never block or resize an order" in deployment
    assert "Observe PTrade IOPV Without Changing Frozen Orders" in decisions
    assert "observation-only IOPV" in readme


def test_buy_submission_failure_does_not_create_guard(monkeypatch):
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={}, portfolio_value=20000, cash=20000)
    )
    pt.g = make_g()
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_current_price", lambda code: 2.0)
    monkeypatch.setattr(pt, "order", lambda *args, **kwargs: None, raising=False)

    assert pt.execute_buy_candidates(
        context, [make_buy_score()], date(2026, 7, 13)
    ) == 0
    assert pt.g.__pending_orders == {}


def test_halt_recovery_merges_resumed_scores_without_second_portfolio_pass(monkeypatch):
    pt.g = make_g(
        paused_pool_codes={"513100.SS"},
        execution_date=date(2026, 7, 13),
        deferred_signal_date=date(2026, 7, 10),
        deferred_scores=[make_buy_score("159985.SZ")],
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    scored = []
    executed = []
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: date(2026, 7, 10))
    monkeypatch.setattr(pt, "get_open_orders", lambda: [], raising=False)
    monkeypatch.setattr(
        pt,
        "calc_cross_signal_score",
        lambda code, end_date, return_reason=False: (
            scored.append((code, end_date)) or make_buy_score(code),
            None,
        ),
    )
    monkeypatch.setattr(
        pt,
        "execute_buy_candidates",
        lambda context, scores, today: executed.append([s["code"] for s in scores]) or 0,
    )
    monkeypatch.setattr(
        pt,
        "do_trading",
        lambda context: (_ for _ in ()).throw(AssertionError("no second full pass")),
    )

    pt.halt_recover(context)

    assert scored == [("513100.SS", date(2026, 7, 10))]
    assert executed == [["159985.SZ", "513100.SS"]]
    assert pt.g.paused_pool_codes == set()


def test_halt_recovery_runs_atr_stop_for_resumed_holding(monkeypatch):
    code = "513100.SS"
    today = date(2026, 7, 13)
    prev_date = date(2026, 7, 10)
    pt.g = make_g(
        paused_pool_codes={code},
        execution_date=today,
        deferred_signal_date=prev_date,
        highest_since_buy={code: 10.0},
        entry_atr={code: 0.5},
        buy_date={code: date(2026, 6, 30)},
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35)),
        portfolio=types.SimpleNamespace(
            positions={
                code: types.SimpleNamespace(
                    amount=100,
                    cost_basis=10.0,
                    last_sale_price=8.5,
                )
            }
        ),
    )
    sold = []
    monkeypatch.setattr(pt, "is_paused", lambda candidate: False)
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: prev_date)
    monkeypatch.setattr(pt, "_reconcile_open_orders", lambda context: True)
    monkeypatch.setattr(
        pt, "_recover_live_state_with_available_sources",
        lambda context, allow_deliver: None,
    )
    monkeypatch.setattr(pt, "get_current_price", lambda candidate: 8.5)
    monkeypatch.setattr(
        pt,
        "calc_cross_signal_score",
        lambda candidate, end_date, return_reason=False: (
            make_buy_score(candidate), None),
    )
    monkeypatch.setattr(
        pt,
        "execute_sell",
        lambda candidate, context, reason: sold.append((candidate, reason)) or True,
    )
    monkeypatch.setattr(pt, "execute_buy_candidates", lambda *args, **kwargs: 0)

    pt.halt_recover(context)

    assert len(sold) == 1
    assert sold[0][0] == code
    assert sold[0][1].startswith("atr_stop ")


def test_halt_recovery_retries_rejected_atr_sell_for_nonpaused_holding(monkeypatch):
    code = "513100.SS"
    today = date(2026, 7, 13)
    prev_date = date(2026, 7, 10)
    pt.g = make_g(
        sell_retry_reasons={code: "atr_stop 8.500<=8.750"},
        execution_date=today,
        deferred_signal_date=prev_date,
        highest_since_buy={code: 10.0},
        entry_atr={code: 0.5},
        buy_date={code: date(2026, 6, 30)},
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35)),
        portfolio=types.SimpleNamespace(
            positions={
                code: types.SimpleNamespace(
                    amount=100,
                    cost_basis=10.0,
                    last_sale_price=8.5,
                )
            }
        ),
    )
    sold = []
    monkeypatch.setattr(pt, "is_paused", lambda candidate: False)
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: prev_date)
    monkeypatch.setattr(pt, "_reconcile_open_orders", lambda context: True)
    monkeypatch.setattr(
        pt, "_recover_live_state_with_available_sources",
        lambda context, allow_deliver: None,
    )
    monkeypatch.setattr(pt, "get_current_price", lambda candidate: 8.5)
    monkeypatch.setattr(
        pt,
        "execute_sell",
        lambda candidate, context, reason: sold.append((candidate, reason)) or True,
    )
    monkeypatch.setattr(pt, "execute_buy_candidates", lambda *args, **kwargs: 0)

    pt.halt_recover(context)

    assert sold == [(code, "atr_stop 8.500<=8.750")]
    assert pt.g.sell_retry_reasons == {}


def test_halt_recovery_drops_rejected_atr_retry_when_risk_has_cleared(monkeypatch):
    code = "513100.SS"
    today = date(2026, 7, 13)
    prev_date = date(2026, 7, 10)
    pt.g = make_g(
        sell_retry_reasons={code: "atr_stop 8.500<=8.750"},
        execution_date=today,
        deferred_signal_date=prev_date,
        highest_since_buy={code: 10.0},
        entry_atr={code: 0.5},
        buy_date={code: date(2026, 6, 30)},
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35)),
        portfolio=types.SimpleNamespace(
            positions={
                code: types.SimpleNamespace(
                    amount=100,
                    cost_basis=10.0,
                    last_sale_price=9.5,
                )
            }
        ),
    )
    monkeypatch.setattr(pt, "is_paused", lambda candidate: False)
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: prev_date)
    monkeypatch.setattr(pt, "_reconcile_open_orders", lambda context: True)
    monkeypatch.setattr(
        pt, "_recover_live_state_with_available_sources",
        lambda context, allow_deliver: None,
    )
    monkeypatch.setattr(pt, "get_current_price", lambda candidate: 9.5)
    monkeypatch.setattr(
        pt,
        "execute_sell",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cleared ATR risk must not be sold")
        ),
    )
    monkeypatch.setattr(pt, "execute_buy_candidates", lambda *args, **kwargs: 0)

    pt.halt_recover(context)

    assert pt.g.sell_retry_reasons == {}


def test_halt_recovery_runs_signal_sell_only_for_resumed_holding(monkeypatch):
    resumed = "513100.SS"
    already_open = "159985.SZ"
    today = date(2026, 7, 13)
    prev_date = date(2026, 7, 10)
    pt.g = make_g(
        paused_pool_codes={resumed},
        execution_date=today,
        deferred_signal_date=prev_date,
        deferred_scores=[make_sell_score(already_open)],
        highest_since_buy={resumed: 10.0, already_open: 10.0},
        entry_atr={resumed: 0.2, already_open: 0.2},
        buy_date={
            resumed: date(2026, 6, 30),
            already_open: date(2026, 6, 30),
        },
    )
    positions = {
        resumed: types.SimpleNamespace(amount=100, cost_basis=10.0, last_sale_price=10.0),
        already_open: types.SimpleNamespace(amount=100, cost_basis=10.0, last_sale_price=10.0),
    }
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35)),
        portfolio=types.SimpleNamespace(positions=positions),
    )
    sold = []
    monkeypatch.setattr(pt, "is_paused", lambda candidate: False)
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: prev_date)
    monkeypatch.setattr(pt, "_reconcile_open_orders", lambda context: True)
    monkeypatch.setattr(
        pt, "_recover_live_state_with_available_sources",
        lambda context, allow_deliver: None,
    )
    monkeypatch.setattr(pt, "get_current_price", lambda candidate: 10.0)
    monkeypatch.setattr(
        pt,
        "calc_cross_signal_score",
        lambda candidate, end_date, return_reason=False: (
            make_sell_score(candidate), None),
    )
    monkeypatch.setattr(
        pt,
        "get_trade_days",
        lambda end_date, count: [
            date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8),
            date(2026, 7, 9), date(2026, 7, 10), today,
        ],
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "execute_sell",
        lambda candidate, context, reason: sold.append((candidate, reason)) or True,
    )
    monkeypatch.setattr(pt, "execute_buy_candidates", lambda *args, **kwargs: 0)

    pt.halt_recover(context)

    assert sold == [(resumed, "sell_score 35")]


def test_halt_recovery_retries_rejected_signal_sell_for_nonpaused_holding(monkeypatch):
    code = "513100.SS"
    today = date(2026, 7, 13)
    prev_date = date(2026, 7, 10)
    score = make_sell_score(code)
    pt.g = make_g(
        sell_retry_reasons={code: "sell_score 35"},
        execution_date=today,
        deferred_signal_date=prev_date,
        deferred_scores=[score],
        highest_since_buy={code: 10.0},
        entry_atr={code: 0.2},
        buy_date={code: date(2026, 6, 30)},
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35)),
        portfolio=types.SimpleNamespace(
            positions={
                code: types.SimpleNamespace(
                    amount=100,
                    cost_basis=10.0,
                    last_sale_price=10.0,
                )
            }
        ),
    )
    sold = []
    monkeypatch.setattr(pt, "is_paused", lambda candidate: False)
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: prev_date)
    monkeypatch.setattr(pt, "_reconcile_open_orders", lambda context: True)
    monkeypatch.setattr(
        pt, "_recover_live_state_with_available_sources",
        lambda context, allow_deliver: None,
    )
    monkeypatch.setattr(pt, "get_current_price", lambda candidate: 10.0)
    monkeypatch.setattr(
        pt,
        "get_trade_days",
        lambda end_date, count: [
            date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8),
            date(2026, 7, 9), date(2026, 7, 10), today,
        ],
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "execute_sell",
        lambda candidate, context, reason: sold.append((candidate, reason)) or True,
    )
    monkeypatch.setattr(pt, "execute_buy_candidates", lambda *args, **kwargs: 0)

    pt.halt_recover(context)

    assert sold == [(code, "sell_score 35")]
    assert pt.g.sell_retry_reasons == {}


def test_halt_recovery_keeps_minimum_hold_protection_for_resumed_holding(monkeypatch):
    code = "513100.SS"
    today = date(2026, 7, 13)
    prev_date = date(2026, 7, 10)
    pt.g = make_g(
        paused_pool_codes={code},
        execution_date=today,
        deferred_signal_date=prev_date,
        highest_since_buy={code: 10.0},
        entry_atr={code: 0.2},
        buy_date={code: date(2026, 7, 10)},
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35)),
        portfolio=types.SimpleNamespace(
            positions={
                code: types.SimpleNamespace(
                    amount=100,
                    cost_basis=10.0,
                    last_sale_price=10.0,
                )
            }
        ),
    )
    sold = []
    monkeypatch.setattr(pt, "is_paused", lambda candidate: False)
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: prev_date)
    monkeypatch.setattr(pt, "_reconcile_open_orders", lambda context: True)
    monkeypatch.setattr(
        pt, "_recover_live_state_with_available_sources",
        lambda context, allow_deliver: None,
    )
    monkeypatch.setattr(pt, "get_current_price", lambda candidate: 10.0)
    monkeypatch.setattr(
        pt,
        "calc_cross_signal_score",
        lambda candidate, end_date, return_reason=False: (
            make_sell_score(candidate), None),
    )
    monkeypatch.setattr(
        pt,
        "get_trade_days",
        lambda end_date, count: [date(2026, 7, 10), today],
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "execute_sell",
        lambda candidate, context, reason: sold.append((candidate, reason)) or True,
    )
    monkeypatch.setattr(pt, "execute_buy_candidates", lambda *args, **kwargs: 0)

    pt.halt_recover(context)

    assert sold == []


def test_new_trading_day_clears_date_scoped_deferred_state(monkeypatch):
    pt.g = make_g(
        execution_date=date(2026, 7, 10),
        deferred_signal_date=date(2026, 7, 9),
        deferred_scores=[make_buy_score()],
        paused_pool_codes={"513100.SS"},
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 0)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    monkeypatch.setattr(pt, "get_open_orders", lambda: [], raising=False)
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)

    pt.before_trading_start(context, data={})

    assert pt.g.execution_date == date(2026, 7, 13)
    assert pt.g.deferred_signal_date is None
    assert pt.g.deferred_scores == []
    assert pt.g.paused_pool_codes == set()


def test_intraday_restart_preserves_same_day_deferred_and_halt_state(monkeypatch):
    deferred = [make_buy_score()]
    pt.g = make_g(
        execution_date=date(2026, 7, 13),
        deferred_signal_date=date(2026, 7, 10),
        deferred_scores=deferred,
        paused_pool_codes={"513100.SS"},
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 40)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    monkeypatch.setattr(pt, "get_open_orders", lambda: [], raising=False)
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)

    pt.before_trading_start(context, data={})

    assert pt.g.deferred_scores == deferred
    assert pt.g.deferred_signal_date == date(2026, 7, 10)
    assert pt.g.paused_pool_codes == {"513100.SS"}


def test_halt_recovery_rejects_stale_deferred_scores(monkeypatch):
    pt.g = make_g(
        execution_date=date(2026, 7, 10),
        deferred_signal_date=date(2026, 7, 9),
        deferred_scores=[make_buy_score()],
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35))
    )
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: date(2026, 7, 10))
    monkeypatch.setattr(
        pt,
        "execute_buy_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("stale scores must not execute")
        ),
    )

    pt.halt_recover(context)


def test_halt_recovery_reconciles_open_orders_before_deferred_buy(monkeypatch):
    pt.g = make_g(
        execution_date=date(2026, 7, 13),
        deferred_signal_date=date(2026, 7, 10),
        deferred_scores=[make_buy_score()],
        __pending_sells={
            "159915.SZ": {
                "requested_qty": 500,
                "filled_qty": 0,
                "order_id": "sell-order-1",
            }
        },
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    calls = []
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: date(2026, 7, 10))

    def reconcile(context):
        calls.append("reconcile")
        pt.g.__pending_sells = {}
        return True

    monkeypatch.setattr(pt, "_reconcile_open_orders", reconcile)
    monkeypatch.setattr(pt, "recover_live_state", lambda context: calls.append("recover"))
    monkeypatch.setattr(
        pt,
        "execute_buy_candidates",
        lambda context, scores, today: calls.append("buy") or 0,
    )

    pt.halt_recover(context)

    assert calls == ["reconcile", "recover", "buy"]


def test_live_recovery_does_not_invent_missing_entry_risk_state(monkeypatch):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 0)),
        portfolio=types.SimpleNamespace(positions={"513100.SS": position}),
    )
    pt.g = make_g()
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: date(2026, 7, 10))
    monkeypatch.setattr(pt, "get_current_price", lambda code: 1.1)
    monkeypatch.setattr(
        pt,
        "get_price_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not synthesize entry ATR")
        ),
    )

    pt.recover_live_state(context)

    assert pt.g.unverified_positions == {"513100.SS"}
    assert "513100.SS" not in pt.g.buy_date
    assert "513100.SS" not in pt.g.highest_since_buy
    assert "513100.SS" not in pt.g.entry_atr


def test_delivery_reconstruction_finds_current_open_position_episode():
    records = [
        {
            "stock_code": "513100",
            "entrust_bs": "1",
            "business_amount": 1000,
            "init_date": 20260105,
            "business_time": 93501,
        },
        {
            "stock_code": "513100",
            "entrust_bs": "2",
            "business_amount": 1000,
            "init_date": 20260202,
            "business_time": 93502,
        },
        {
            "stock_code": "513100",
            "business_name": "\u8bc1\u5238\u4e70\u5165",
            "occur_amount": 500,
            "init_date": 20260303,
            "business_time": 93503,
        },
        {
            "stock_code": "513100",
            "business_name": "\u8bc1\u5238\u5356\u51fa",
            "occur_amount": 100,
            "init_date": 20260310,
            "business_time": 93504,
        },
    ]

    recovered = pt._reconstruct_open_position(records, "513100.SS", 400)

    assert recovered["buy_date"] == date(2026, 3, 3)
    assert recovered["amount"] == pytest.approx(400)


def test_delivery_reconstruction_rejects_quantity_mismatch():
    records = [{
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 500,
        "init_date": 20260303,
        "business_time": 93503,
    }]

    assert pt._reconstruct_open_position(records, "513100.SS", 400) is None


def test_live_recovery_logs_sanitized_delivery_replay_on_quantity_mismatch(
    monkeypatch,
):
    position = types.SimpleNamespace(amount=400, cost_basis=1.18, last_sale_price=1.28)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g()
    records = [{
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 500,
        "business_price": 1.18,
        "init_date": 20260303,
        "business_time": 93503,
        "fund_account": "secret-account",
        "stock_account": "secret-stock-account",
    }]
    messages = []
    monkeypatch.setattr(
        pt,
        "log",
        types.SimpleNamespace(
            info=lambda message, *args: messages.append(message % args if args else message),
            warning=lambda message, *args: messages.append(message % args if args else message),
            error=lambda message, *args: messages.append(message % args if args else message),
        ),
    )

    pt.recover_live_state(
        context,
        deliver_records=records,
        prev_date=date(2026, 7, 10),
    )

    summary = next(
        message for message in messages
        if "阶段=交割单重放" in message
    )
    assert "代码=513100.SS" in summary
    assert "券商持仓=400" in summary
    assert "总记录数=1" in summary
    assert "标的记录数=1" in summary
    assert "有效记录数=1" in summary
    assert "买入笔数=1" in summary
    assert "卖出笔数=0" in summary
    assert "净数量=500" in summary
    assert "日期范围=2026-03-03~2026-03-03" in summary
    assert "可用代码=513100.SS" in summary
    rendered = "\n".join(messages)
    assert "secret-account" not in rendered
    assert "secret-stock-account" not in rendered


def test_live_recovery_probes_historical_calendar_without_adopting_probe(
    monkeypatch,
):
    position = types.SimpleNamespace(amount=400, cost_basis=1.18, last_sale_price=1.28)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g()
    records = [{
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 400,
        "business_price": 1.18,
        "init_date": 20260303,
    }]
    messages = []
    probe_calls = []
    score_calls = []
    monkeypatch.setattr(
        pt,
        "log",
        types.SimpleNamespace(
            info=lambda message, *args: messages.append(message % args if args else message),
            warning=lambda message, *args: messages.append(message % args if args else message),
            error=lambda message, *args: messages.append(message % args if args else message),
        ),
    )
    monkeypatch.setattr(pt, "get_trade_days", lambda **kwargs: [], raising=False)
    monkeypatch.setattr(pt, "get_all_trades_days", lambda **kwargs: [], raising=False)
    monkeypatch.setattr(
        pt,
        "get_trading_day_by_date",
        lambda query_date, day=0: probe_calls.append((query_date, day)) or "20260302",
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "calc_cross_signal_score",
        lambda *args, **kwargs: score_calls.append((args, kwargs)) or make_buy_score(),
    )

    pt.recover_live_state(
        context,
        deliver_records=records,
        prev_date=date(2026, 7, 10),
    )

    assert probe_calls == [("20260303", -1)]
    assert score_calls == []
    assert pt.g.unverified_positions == {"513100.SS"}
    assert any(
        "接口=get_trade_days" in message
        and "返回值不可用" in message
        and "类型=list" in message
        for message in messages
    )
    failure = next(
        message for message in messages
        if "阶段=历史交易日历" in message
    )
    assert "买入日期=2026-03-03" in failure
    assert "日期探针=2026-03-02" in failure
    assert "不参与交易判断=是" in failure


def test_live_recovery_logs_entry_atr_stage_when_score_is_unavailable(monkeypatch):
    position = types.SimpleNamespace(amount=400, cost_basis=1.18, last_sale_price=1.28)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g()
    records = [{
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 400,
        "business_price": 1.18,
        "init_date": 20260303,
    }]
    messages = []
    monkeypatch.setattr(
        pt,
        "log",
        types.SimpleNamespace(
            info=lambda message, *args: messages.append(message % args if args else message),
            warning=lambda message, *args: messages.append(message % args if args else message),
            error=lambda message, *args: messages.append(message % args if args else message),
        ),
    )
    monkeypatch.setattr(
        pt,
        "_previous_trade_date_before",
        lambda value: date(2026, 3, 2),
        raising=False,
    )
    monkeypatch.setattr(pt, "calc_cross_signal_score", lambda *args: None)

    pt.recover_live_state(
        context,
        deliver_records=records,
        prev_date=date(2026, 7, 10),
    )

    failure = next(
        message for message in messages
        if "阶段=入场ATR" in message
    )
    assert "代码=513100.SS" in failure
    assert "原因=评分不可用" in failure
    assert "信号日期=2026-03-02" in failure
    assert pt.g.unverified_positions == {"513100.SS"}


def test_live_recovery_adopts_account_position_from_broker_facts(monkeypatch):
    position = types.SimpleNamespace(amount=400, cost_basis=1.18, last_sale_price=1.28)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g()
    records = [{
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 400,
        "init_date": 20260303,
        "business_time": 93503,
        "business_price": 1.18,
        "_recovery_source": "get-deliver",
    }]
    score = make_buy_score()
    info_messages = []
    warning_messages = []
    monkeypatch.setattr(
        pt,
        "log",
        types.SimpleNamespace(
            info=lambda message, *args: info_messages.append(
                message % args if args else message),
            warning=lambda message, *args: warning_messages.append(
                message % args if args else message),
            error=lambda *args: None,
        ),
    )
    monkeypatch.setattr(
        pt,
        "_previous_trade_date_before",
        lambda value: date(2026, 3, 2),
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "calc_cross_signal_score",
        lambda code, end_date: score,
    )
    monkeypatch.setattr(
        pt,
        "_get_recovery_close_data",
        lambda code, start_date, end_date: pt.pd.DataFrame(
            {"close": [1.20, 1.35, 1.28], "volume": [100, 200, 150]},
            index=pt.pd.to_datetime(["2026-03-03", "2026-03-20", "2026-07-10"]),
        ),
        raising=False,
    )

    pt.recover_live_state(
        context,
        deliver_records=records,
        prev_date=date(2026, 7, 10),
    )

    assert pt.g.buy_date["513100.SS"] == date(2026, 3, 3)
    assert pt.g.entry_atr["513100.SS"] == pytest.approx(0.05)
    assert pt.g.highest_since_buy["513100.SS"] == pytest.approx(1.35)
    assert pt.g.unverified_positions == set()
    assert pt.g.__position_recovery_source == {
        "513100.SS": "account-takeover:get-deliver"
    }
    assert any("已依据券商事实接管" in message for message in info_messages)
    assert not any("已依据券商事实接管" in message for message in warning_messages)


def test_live_recovery_adopts_existing_account_position_without_cross_signal_entry(
    monkeypatch,
):
    position = types.SimpleNamespace(amount=400, cost_basis=1.18, last_sale_price=1.28)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g()
    records = [{
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 400,
        "init_date": 20260303,
        "business_price": 1.18,
        "_recovery_source": "get-deliver",
    }]
    ineligible = make_buy_score()
    ineligible["buy_score"] = 10
    monkeypatch.setattr(
        pt,
        "_previous_trade_date_before",
        lambda value: date(2026, 3, 2),
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "calc_cross_signal_score",
        lambda code, end_date: ineligible,
    )
    monkeypatch.setattr(
        pt,
        "_get_recovery_close_data",
        lambda code, start_date, end_date: pt.pd.DataFrame(
            {"close": [1.20, 1.35, 1.28], "volume": [100, 200, 150]},
            index=pt.pd.to_datetime(["2026-03-03", "2026-03-20", "2026-07-10"]),
        ),
        raising=False,
    )

    pt.recover_live_state(
        context,
        deliver_records=records,
        prev_date=date(2026, 7, 10),
    )

    assert pt.g.buy_date["513100.SS"] == date(2026, 3, 3)
    assert pt.g.entry_atr["513100.SS"] == pytest.approx(0.05)
    assert pt.g.highest_since_buy["513100.SS"] == pytest.approx(1.35)
    assert pt.g.unverified_positions == set()
    assert pt.g.__position_recovery_source == {
        "513100.SS": "account-takeover:get-deliver"
    }


def test_live_recovery_prunes_state_for_positions_no_longer_held():
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={})
    )
    pt.g = make_g(
        buy_date={"513100.SS": date(2026, 3, 3)},
        entry_atr={"513100.SS": 0.05},
        highest_since_buy={"513100.SS": 1.35},
        unverified_positions={"513100.SS"},
    )

    pt.recover_live_state(context)

    assert pt.g.buy_date == {}
    assert pt.g.entry_atr == {}
    assert pt.g.highest_since_buy == {}
    assert pt.g.unverified_positions == set()


def test_live_recovery_requires_valid_broker_cost_for_automatic_exits():
    position = types.SimpleNamespace(amount=400, cost_basis=0, last_sale_price=1.28)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g(
        buy_date={"513100.SS": date(2026, 3, 3)},
        entry_atr={"513100.SS": 0.05},
        highest_since_buy={"513100.SS": 1.35},
    )

    pt.recover_live_state(context)

    assert pt.g.unverified_positions == {"513100.SS"}


def test_live_recovery_rebuilds_same_day_buy_from_strategy_trades(monkeypatch):
    today = date(2026, 7, 13)
    position = types.SimpleNamespace(amount=400, cost_basis=1.181, last_sale_price=1.22)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g()
    monkeypatch.setattr(
        pt,
        "get_trades",
        lambda: {
            "order-1": [[
                "trade-1",
                "entrust-1",
                "513100.XSHG",
                "\u4e70",
                400.0,
                1.18,
                472.0,
                "2026-07-13 09:35:01",
            ]]
        },
        raising=False,
    )
    score = make_buy_score()
    monkeypatch.setattr(
        pt,
        "_previous_trade_date_before",
        lambda value: date(2026, 7, 10),
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "calc_cross_signal_score",
        lambda code, end_date: score,
    )
    monkeypatch.setattr(
        pt,
        "_get_recovery_close_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("same-day entry has no completed daily close yet")
        ),
        raising=False,
    )

    current_records = pt._fetch_current_strategy_trades()
    pt.recover_live_state(
        context,
        current_trade_records=current_records,
        prev_date=date(2026, 7, 10),
    )

    assert current_records[0]["stock_code"] == "513100.XSHG"
    assert current_records[0]["entrust_bs"] == "1"
    assert current_records[0]["business_id"] == "trade-1"
    assert current_records[0]["_recovery_source"] == "get-trades"
    assert pt.g.buy_date["513100.SS"] == today
    assert pt.g.entry_atr["513100.SS"] == pytest.approx(0.05)
    assert pt.g.highest_since_buy["513100.SS"] == pytest.approx(1.18)
    assert pt.g.unverified_positions == set()
    assert pt.g.__position_recovery_source == {"513100.SS": "get-trades"}


def test_live_recovery_summary_logs_state_journal_and_each_holding(monkeypatch):
    position = types.SimpleNamespace(amount=400, cost_basis=1.18, last_sale_price=1.28)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g(
        buy_date={"513100.SS": date(2026, 3, 3)},
        entry_atr={"513100.SS": 0.05},
        highest_since_buy={"513100.SS": 1.35},
        __state_restore_source="journal",
        __state_restore_generation=8,
        __persisted_g_status="not-provided",
        __persisted_g_reason="metadata-missing",
        __position_recovery_source={"513100.SS": "journal"},
    )
    messages = []
    monkeypatch.setattr(
        pt,
        "log",
        types.SimpleNamespace(
            info=lambda message, *args: messages.append(message % args if args else message),
            warning=lambda *args: None,
            error=lambda *args: None,
        ),
    )

    pt._log_live_recovery_summary(context)

    assert any(
        "[PTrade框架g] 状态=未提供 代次=不适用 原因=未发现持久状态元数据"
        in message
        for message in messages
    )
    assert any(
        "[连续状态恢复] 来源=状态台账 代次=8" in message
        for message in messages
    )
    assert any(
        "[持仓风险恢复] 来源=状态台账" in message
        for message in messages
    )
    holding = next(message for message in messages if "代码=513100.SS" in message)
    assert "数量=400" in holding
    assert "成本=1.180000" in holding
    assert "买入日期=2026-03-03" in holding
    assert "ATR=0.050000" in holding
    assert "持仓最高收盘价=1.350000" in holding
    assert "状态=已验证" in holding
    assert "来源=状态台账" in holding


def test_live_recovery_summary_reports_delivery_takeover_source(monkeypatch):
    position = types.SimpleNamespace(amount=400, cost_basis=1.18, last_sale_price=1.28)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g(
        buy_date={"513100.SS": date(2026, 3, 3)},
        entry_atr={"513100.SS": 0.05},
        highest_since_buy={"513100.SS": 1.35},
        __state_restore_source="journal",
        __state_restore_generation=3,
        __persisted_g_status="not-provided",
        __persisted_g_reason="metadata-missing",
        __position_recovery_source={
            "513100.SS": "account-takeover:get-deliver",
        },
    )
    messages = []
    monkeypatch.setattr(
        pt,
        "log",
        types.SimpleNamespace(
            info=lambda message, *args: messages.append(message % args if args else message),
            warning=lambda *args: None,
            error=lambda *args: None,
        ),
    )

    pt._log_live_recovery_summary(context)

    assert any(
        "[连续状态恢复] 来源=状态台账 代次=3" in message
        for message in messages
    )
    assert any(
        "[持仓风险恢复] 来源=账户接管:交割单" in message
        for message in messages
    )


def test_live_recovery_summary_reports_mixed_position_sources(monkeypatch):
    positions = {
        "513100.SS": types.SimpleNamespace(
            amount=400, cost_basis=1.18, last_sale_price=1.28),
        "518880.SS": types.SimpleNamespace(
            amount=200, cost_basis=4.68, last_sale_price=4.72),
    }
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions=positions)
    )
    pt.g = make_g(
        buy_date={
            "513100.SS": date(2026, 3, 3),
            "518880.SS": date(2026, 3, 4),
        },
        entry_atr={"513100.SS": 0.05, "518880.SS": 0.08},
        highest_since_buy={"513100.SS": 1.35, "518880.SS": 4.90},
        __state_restore_source="journal",
        __state_restore_generation=9,
        __persisted_g_status="accepted",
        __persisted_g_reason="validated",
        __persisted_g_generation=7,
        __position_recovery_source={
            "513100.SS": "ptrade-g",
            "518880.SS": "account-takeover:get-deliver",
        },
    )
    messages = []
    monkeypatch.setattr(
        pt,
        "log",
        types.SimpleNamespace(
            info=lambda message, *args: messages.append(message % args if args else message),
            warning=lambda *args: None,
            error=lambda *args: None,
        ),
    )

    pt._log_live_recovery_summary(context)

    assert any(
        "[PTrade框架g] 状态=已接受 代次=7 原因=校验通过" in message
        for message in messages
    )
    assert any(
        "[连续状态恢复] 来源=状态台账 代次=9" in message
        for message in messages
    )
    assert any(
        "[持仓风险恢复] 来源=混合恢复" in message
        for message in messages
    )


@pytest.mark.parametrize(
    ("allow_deliver", "expected_calls"),
    [(True, ["current", "deliver", "recover"]), (False, ["current", "recover"])],
)
def test_recovery_source_query_respects_ptrade_call_phase(
        monkeypatch, allow_deliver, expected_calls):
    position = types.SimpleNamespace(amount=400, cost_basis=1.18, last_sale_price=1.22)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35)),
        portfolio=types.SimpleNamespace(positions={"513100.SS": position}),
    )
    pt.g = make_g()
    calls = []
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: date(2026, 7, 10))
    monkeypatch.setattr(
        pt,
        "_fetch_current_strategy_trades",
        lambda: calls.append("current") or [{"source": "current"}],
    )
    monkeypatch.setattr(
        pt,
        "_fetch_deliver_records",
        lambda prev_date: calls.append("deliver") or [{"source": "deliver"}],
    )

    def recover(context, deliver_records=None, current_trade_records=None, prev_date=None):
        calls.append("recover")
        assert current_trade_records == [{"source": "current"}]
        assert deliver_records == ([{"source": "deliver"}] if allow_deliver else None)
        assert prev_date == date(2026, 7, 10)

    monkeypatch.setattr(pt, "recover_live_state", recover)

    pt._recover_live_state_with_available_sources(context, allow_deliver)

    assert calls == expected_calls


def test_unverified_position_is_excluded_from_atr_stop_execution(monkeypatch):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=0.8)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 35)),
        portfolio=types.SimpleNamespace(positions={"513100.SS": position}),
    )
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": None},
        unverified_positions={"513100.SS"},
    )
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_current_price", lambda code: 0.8)

    assert pt.check_atr_stops(context) == []


@pytest.mark.parametrize(
    ("atr", "highest"),
    [("not-a-number", 1.2), (0.05, float("inf")), (float("nan"), 1.2)],
)
def test_live_recovery_rejects_malformed_or_nonfinite_risk_state(atr, highest):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g(
        buy_date={"513100.SS": date(2026, 6, 1)},
        highest_since_buy={"513100.SS": highest},
        entry_atr={"513100.SS": atr},
    )

    pt.recover_live_state(context)

    assert pt.g.unverified_positions == {"513100.SS"}


def test_initialize_live_schedules_only_cross_signal_tasks(monkeypatch):
    scheduled = []
    platform_parameters = []
    commission_calls = []
    slippage_calls = []
    state_path_calls = []
    pt.g = types.SimpleNamespace()
    monkeypatch.setattr(pt, "set_benchmark", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        pt,
        "set_commission",
        lambda *args, **kwargs: commission_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "set_slippage",
        lambda *args, **kwargs: slippage_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(pt, "set_universe", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        pt,
        "set_parameters",
        lambda **kwargs: platform_parameters.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(pt, "is_trade", lambda: True, raising=False)
    monkeypatch.setattr(
        pt,
        "_live_state_path",
        lambda: state_path_calls.append(True) or "unexpected",
    )
    monkeypatch.setattr(
        pt,
        "run_daily",
        lambda context, func, time: scheduled.append((func.__name__, time)),
        raising=False,
    )

    pt.initialize(types.SimpleNamespace())

    assert scheduled == [
        ("_do_trading_wrapper", "09:35"),
        ("_recent_fill_reconcile_wrapper", "09:36"),
        ("_halt_recover_wrapper", "10:35"),
    ]
    assert pt.g.params == jq.get_default_params()
    assert pt.g.sell_retry_reasons == {}
    assert not hasattr(pt.g, "base_weights")
    assert platform_parameters == [{
        "receive_cancel_response": "1",
        "not_restart_trade": "0",
        "server_restart_not_do_before": "0",
    }]
    assert commission_calls == []
    assert slippage_calls == []
    assert state_path_calls == []
    assert pt.g.__state_path is None
    assert pt.g.__mode_verified is True
    assert not hasattr(pt.g, "state_instance_id")


def test_initialize_backtest_uses_only_backtest_cost_configuration(monkeypatch):
    scheduled = []
    platform_parameters = []
    commission_calls = []
    slippage_calls = []
    pt.g = types.SimpleNamespace()
    monkeypatch.setattr(pt, "set_benchmark", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        pt,
        "set_commission",
        lambda *args, **kwargs: commission_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "set_slippage",
        lambda *args, **kwargs: slippage_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(pt, "set_universe", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        pt,
        "set_parameters",
        lambda **kwargs: platform_parameters.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(pt, "is_trade", lambda: False, raising=False)
    monkeypatch.setattr(
        pt,
        "run_daily",
        lambda context, func, time: scheduled.append((func.__name__, time)),
        raising=False,
    )

    pt.initialize(types.SimpleNamespace())

    assert scheduled == []
    assert platform_parameters == []
    assert len(commission_calls) == 1
    assert len(slippage_calls) == 1
    assert pt.g.__mode_verified is True


def test_initialize_mode_detection_failure_blocks_all_trading(monkeypatch):
    scheduled = []
    platform_parameters = []
    commission_calls = []
    slippage_calls = []
    trading_calls = []
    pt.g = types.SimpleNamespace()
    monkeypatch.setattr(pt, "set_benchmark", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        pt,
        "is_trade",
        lambda: (_ for _ in ()).throw(RuntimeError("mode unavailable")),
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "set_parameters",
        lambda **kwargs: platform_parameters.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "set_commission",
        lambda *args, **kwargs: commission_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        pt,
        "set_slippage",
        lambda *args, **kwargs: slippage_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(pt, "set_universe", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        pt,
        "run_daily",
        lambda context, func, time: scheduled.append((func.__name__, time)),
        raising=False,
    )
    monkeypatch.setattr(pt, "do_trading", lambda context: trading_calls.append(True))

    context = types.SimpleNamespace()
    pt.initialize(context)
    pt.handle_data(context, data=object())

    assert pt.g.__mode_verified is False
    assert scheduled == []
    assert platform_parameters == []
    assert commission_calls == []
    assert slippage_calls == []
    assert trading_calls == []


def test_ptrade_deployment_notes_pin_frozen_version_and_live_schedule():
    notes = (
        ROOT / "cross_signal_strategy" / "docs" / "ptrade_deployment.md"
    ).read_text(encoding="utf-8")

    assert "cross-v0.3.2" in notes
    assert "09:35" in notes
    assert "09:36" in notes
    assert "10:35" in notes
    assert "15:30" in notes
    assert "JoinQuant" in notes
    assert "PTrade" in notes
    assert "configuration lock" in notes
    assert "bounded state journal" in notes
    assert "three tasks" in notes
    assert "`get_trades()`" in notes
    assert "after_trading_end" in notes
    assert "path is resolved and cached at the start of" in notes
    assert "`before_trading_start`" in notes
    assert "resumed holdings repeat the 09:35 ATR-stop and signal-sell checks" in notes
    assert "does not rerun already processed ETFs" in notes
    assert "[发布指纹]" in notes
    assert "20260723.1" in notes
    assert "1506a0e834fe" in notes


def test_ptrade_deployment_notes_define_bounded_full_audit_log():
    notes = (
        ROOT / "cross_signal_strategy" / "docs" / "ptrade_deployment.md"
    ).read_text(encoding="utf-8")

    assert "cross_signal_logs" in notes
    assert "cross_signal_v032_audit.log" in notes
    assert "完整镜像" in notes
    assert "20 MB" in notes
    assert "16 MB" in notes
    assert "最旧的完整日志行" in notes
    assert "PTrade 平台自身" in notes


def test_release_docs_describe_resilient_ptrade_state_recovery():
    deployment = (
        ROOT / "cross_signal_strategy" / "docs" / "ptrade_deployment.md"
    ).read_text(encoding="utf-8")
    decisions = (
        ROOT / "cross_signal_strategy" / "docs" / "decisions.md"
    ).read_text(encoding="utf-8")

    assert "single bounded journal" in deployment
    assert "state-schema version" in deployment
    assert "broker position snapshot" in deployment
    assert "validated PTrade-persisted `g` state is attempted first" in deployment
    assert "newer matching journal" in deployment
    assert "truncated tail" in deployment
    assert "all new buys are blocked" in deployment
    assert "[PTrade框架g]" in deployment
    assert "[连续状态恢复]" in deployment
    assert "[持仓风险恢复]" in deployment
    assert "账户接管:交割单" in deployment
    assert "one account runs one active" in deployment
    assert "same calendar date as the running process" in deployment
    assert "reads `get_open_orders()` without cancelling or resubmitting" in deployment
    assert "business-configuration fingerprint" in deployment
    assert "Malformed callback records" in deployment
    assert "Harden PTrade Checkpoints And Recovery Gating" in decisions
    assert "Harden Cross-Signal Live Engineering Without Changing Business Rules" in decisions
    assert "Adopt Existing PTrade Account Positions On Strategy Handover" in decisions
    assert "Replace A/B Checkpoints With Broker-First State Journal" in decisions
    assert "Prefer Broker-Validated PTrade G State On Restart" in decisions
