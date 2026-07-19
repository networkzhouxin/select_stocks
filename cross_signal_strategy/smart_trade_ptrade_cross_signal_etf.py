# -*- coding: utf-8 -*-
"""国金证券 PTrade 上穿下穿 ETF 策略 v0.3.2 正式版。

业务规则冻结并对齐聚宽 v0.3.2 主线；两版只允许在平台接口、实盘委托、
重启恢复和停复牌处理上存在差异。PTrade 回测只用于验证代码可运行，
策略收益仍以聚宽回测为准。
"""

import numpy as np
import pandas as pd
import builtins as _builtins
import hashlib
import pickle
from datetime import datetime


# 一、冻结的业务配置与持久化边界
# 参数和 ETF 池由代码固定，防止 PTrade 在重启恢复 g 时覆盖正式版业务配置。
# 单一追加式状态台账保存风险与当日连续性状态；行情快照、在途委托等临时状态使用双下划线变量。

STRATEGY_VERSION = "cross-v0.3.2"
DEPLOYMENT_BUILD_ID = "20260720.4"
LIVE_STATE_SCHEMA_VERSION = 3
LIVE_STATE_PICKLE_PROTOCOL = 4
IOPV_OBSERVE_CODES = frozenset((
    "513100.SS",
    "513500.SS",
    "513880.SS",
    "513050.SS",
))
LIVE_STATE_FILENAME = "cross_signal_v032_live_state_v3_%s.journal"
DELIVER_RECOVERY_START_DATE = "20100101"
LIVE_STATE_FIELDS = (
    "highest_since_buy",
    "entry_atr",
    "buy_date",
    "last_scores",
    "sold_today",
    "sell_retry_reasons",
    "paused_pool_codes",
    "unverified_positions",
    "execution_date",
    "deferred_scores",
    "deferred_signal_date",
)


try:
    log
except NameError:
    class _LocalLog(object):
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    log = _LocalLog()


def get_default_params():
    return {
        "lookback": 120,
        "rebalance_weekdays": [0, 1, 2, 3, 4],
        "max_hold": 3,
        "base_ratio": 0.95,
        "min_signal_hold_days": 5,
        "buy_threshold": 60,
        "strong_buy_threshold": 70,
        "sell_threshold": 30,
        "risk_tighten_threshold": 18,
        "cross_window": 3,
        "rsi_fast": 6,
        "rsi_mid": 12,
        "rsi_slow": 24,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "kdj_n": 9,
        "kdj_m1": 3,
        "kdj_m2": 3,
        "boll_period": 20,
        "boll_std": 2.0,
        "atr_period": 14,
        "adx_period": 14,
        "adx_trend_threshold": 25,
        "trailing_atr_mult": 2.5,
        "stop_floor": 0.05,
        "stop_cap": 0.15,
        "overheat_rsi": 85,
        "a_share_zero_volume_buy_scale": 0.50,
    }


def get_default_etf_pool():
    return [
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


def business_config_fingerprint(params=None, etf_pool=None):
    p = params or get_default_params()
    pool = etf_pool or get_default_etf_pool()
    param_text = "|".join(
        "%s=%r" % (key, p[key]) for key in sorted(p)
    )
    pool_text = ",".join(str(code).split(".")[0] for code in pool)
    payload = "%s|%s|%s" % (STRATEGY_VERSION, param_text, pool_text)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _lock_frozen_business_config():
    """PTrade 恢复 g 后，重新写入由代码冻结的策略参数和 ETF 池。"""
    g.params = get_default_params()
    g.etf_pool = get_default_etf_pool()
    try:
        set_universe(g.etf_pool)
    except Exception as exc:
        log.warning("[配置锁定] 设置标的池失败: %s" % exc)


def _live_state_path(path=None):
    if path is not None:
        return str(path)
    try:
        root = get_research_path()
    except Exception as exc:
        log.warning("[状态] 无法获取研究路径: %s" % exc)
        return None
    if not root:
        log.warning("[状态] 研究路径为空")
        return None
    identity = []
    identity_getters = (
        ("get_user_name", (False,)),
        ("get_trade_name", ()),
    )
    for getter_name, getter_args in identity_getters:
        getter = globals().get(getter_name)
        if getter is None:
            log.error("[状态] 接口%s不可用，检查点已停用" % getter_name)
            return None
        try:
            value = getter(*getter_args)
        except Exception as exc:
            log.error("[状态] 接口%s调用失败，检查点已停用: %s" % (
                getter_name, exc))
            return None
        if value in (None, ""):
            log.error("[状态] 接口%s返回空值，检查点已停用" % getter_name)
            return None
        identity.append(str(value))
    identity_text = "|".join(identity)
    identity_hash = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()[:12]
    root_text = str(root).rstrip("/\\")
    return root_text + "/" + (LIVE_STATE_FILENAME % identity_hash)


def _cached_live_state_path(path=None):
    if path is not None:
        return str(path)
    return getattr(g, "__state_path", None)


def _live_state_checksum(
        schema_version, generation, producer_version, payload,
        business_fingerprint=None):
    fingerprint = business_fingerprint or business_config_fingerprint()
    header = "%s|%s|%s|%s|" % (
        schema_version, generation, producer_version, fingerprint)
    return hashlib.sha256(header.encode("utf-8") + payload).hexdigest()


def _validated_broker_position_snapshot(snapshot):
    if snapshot is None:
        return None
    if not isinstance(snapshot, dict):
        raise ValueError("invalid broker position snapshot")
    validated = {}
    for raw_code, raw_position in snapshot.items():
        code = normalize_code(raw_code)
        if not code or not isinstance(raw_position, dict):
            raise ValueError("invalid broker position entry")
        amount = _safe_float(raw_position.get("amount"), np.nan)
        cost = _safe_float(raw_position.get("cost"), np.nan)
        if not _is_positive_finite(amount) or not _is_positive_finite(cost):
            raise ValueError("invalid broker position facts: %s" % code)
        validated[code] = {
            "amount": float(amount),
            "cost": float(cost),
        }
    return validated


def _broker_position_snapshot(context):
    if context is None:
        return None
    snapshot = {}
    for raw_code, position in _positions(context).items():
        amount = _pos_amount(position)
        if amount <= 0:
            continue
        code = normalize_code(raw_code)
        cost = _pos_cost(position)
        if not code or not _is_positive_finite(amount) or not _is_positive_finite(cost):
            raise ValueError("unprovable broker position: %s" % raw_code)
        if code in snapshot:
            raise ValueError("duplicate broker position: %s" % code)
        snapshot[code] = {"amount": amount, "cost": cost}
    return _validated_broker_position_snapshot(snapshot)


def _broker_position_snapshots_match(recorded, current):
    recorded = _validated_broker_position_snapshot(recorded)
    current = _validated_broker_position_snapshot(current)
    if recorded is None or current is None:
        return recorded is None and current is None
    if set(recorded) != set(current):
        return False
    for code in recorded:
        old = recorded[code]
        new = current[code]
        if old["amount"] != new["amount"]:
            return False
        tolerance = max(1e-8, abs(old["cost"]) * 1e-8)
        if abs(old["cost"] - new["cost"]) > tolerance:
            return False
    return True


def _load_persisted_g_state(context):
    """读取并验证 PTrade 框架自动恢复的普通 g 状态。"""
    schema_version = getattr(g, "live_state_schema_version", None)
    fingerprint = getattr(g, "live_state_business_fingerprint", None)
    generation = getattr(g, "live_state_generation", None)
    recorded_positions = getattr(g, "live_state_broker_positions", None)
    metadata = (schema_version, fingerprint, generation, recorded_positions)
    if all(value is None for value in metadata):
        _set_persisted_g_diagnostic(
            "not-provided", "metadata-missing", None)
        return None
    try:
        if schema_version != LIVE_STATE_SCHEMA_VERSION:
            raise ValueError("state-schema-mismatch")
        if fingerprint != business_config_fingerprint():
            raise ValueError("business-fingerprint-mismatch")
        if not isinstance(generation, int) or generation <= 0:
            raise ValueError("invalid-state-generation")
        state = _validated_live_state({
            field: getattr(g, field, None)
            for field in LIVE_STATE_FIELDS
        })
        current_positions = _broker_position_snapshot(context)
        if not _broker_position_snapshots_match(
                recorded_positions, current_positions):
            raise ValueError("broker-position-snapshot-mismatch")

        buy_dates = _normalized_state_mapping(state["buy_date"])
        entry_atr = _normalized_state_mapping(state["entry_atr"])
        highest = _normalized_state_mapping(state["highest_since_buy"])
        unverified = set(
            normalize_code(code) for code in state["unverified_positions"])
        today = _as_date(get_context_datetime(context))
        for code in current_positions:
            buy_date = _as_date(buy_dates.get(code))
            if (
                code in unverified or
                buy_date is None or
                (today is not None and buy_date > today) or
                not _is_positive_finite(entry_atr.get(code)) or
                not _is_positive_finite(highest.get(code))
            ):
                raise ValueError("incomplete-position-risk-state:%s" % code)
        _set_persisted_g_diagnostic("accepted", "validated", generation)
        return generation, state
    except Exception as exc:
        reason = str(exc)
        _set_persisted_g_diagnostic("rejected", reason, generation)
        log.warning(
            "[PTrade框架g] 校验失败，已拒绝恢复: %s" %
            _format_persisted_g_reason_for_log(reason)
        )
        return None


def _set_persisted_g_diagnostic(status, reason, generation):
    """记录本次启动对普通 g 的判断；双下划线字段不交给平台持久化。"""
    g.__persisted_g_status = status
    g.__persisted_g_reason = reason
    g.__persisted_g_generation = generation


def _format_persisted_g_status_for_log(status):
    return {
        "not-provided": "未提供",
        "accepted": "已接受",
        "rejected": "已拒绝",
        "superseded": "已接受但未采用",
    }.get(str(status or ""), "未检查")


def _format_persisted_g_reason_for_log(reason):
    text = str(reason or "")
    if text.startswith("incomplete-position-risk-state:"):
        return "持仓风险状态不完整:%s" % text.split(":", 1)[1]
    return {
        "metadata-missing": "未发现持久状态元数据",
        "validated": "校验通过",
        "state-schema-mismatch": "状态结构版本不匹配",
        "business-fingerprint-mismatch": "业务配置指纹不匹配",
        "invalid-state-generation": "状态代次无效",
        "broker-position-snapshot-mismatch": "券商持仓快照不匹配",
        "newer-journal": "状态台账代次更新",
    }.get(text, text or "无")


def _record_persisted_g_state(state, generation, broker_positions):
    g.live_state_schema_version = LIVE_STATE_SCHEMA_VERSION
    g.live_state_business_fingerprint = business_config_fingerprint()
    g.live_state_generation = int(generation)
    g.live_state_broker_positions = _validated_broker_position_snapshot(
        broker_positions)


def _encode_live_state_envelope(state, generation, broker_positions=None):
    generation = int(generation)
    if generation <= 0:
        raise ValueError("invalid state generation")
    validated = _validated_live_state(state)
    payload = pickle.dumps({
        "state": validated,
        "broker_positions": _validated_broker_position_snapshot(
            broker_positions),
    }, protocol=LIVE_STATE_PICKLE_PROTOCOL)
    envelope = {
        "schema_version": LIVE_STATE_SCHEMA_VERSION,
        "generation": generation,
        "producer_strategy_version": STRATEGY_VERSION,
        "business_config_fingerprint": business_config_fingerprint(),
        "payload": payload,
    }
    envelope["checksum"] = _live_state_checksum(
        envelope["schema_version"],
        generation,
        envelope["producer_strategy_version"],
        payload,
        envelope["business_config_fingerprint"],
    )
    return envelope


def _decode_live_state_envelope(envelope):
    if not isinstance(envelope, dict):
        raise ValueError("invalid state envelope")
    schema_version = envelope.get("schema_version")
    if schema_version != LIVE_STATE_SCHEMA_VERSION:
        raise ValueError("state schema mismatch")
    generation = envelope.get("generation")
    if not isinstance(generation, int) or generation <= 0:
        raise ValueError("invalid state generation")
    producer_version = envelope.get("producer_strategy_version")
    if not isinstance(producer_version, str) or not producer_version:
        raise ValueError("invalid producer strategy version")
    fingerprint = envelope.get("business_config_fingerprint")
    if fingerprint != business_config_fingerprint():
        raise ValueError("business fingerprint mismatch")
    payload = envelope.get("payload")
    if not isinstance(payload, bytes):
        raise ValueError("invalid state payload bytes")
    expected = _live_state_checksum(
        schema_version, generation, producer_version, payload, fingerprint)
    if envelope.get("checksum") != expected:
        raise ValueError("state checksum mismatch")
    body = pickle.loads(payload)
    if not isinstance(body, dict):
        raise ValueError("invalid state payload")
    return (
        generation,
        _validated_live_state(body.get("state")),
        _validated_broker_position_snapshot(body.get("broker_positions")),
    )


def _scan_live_state_journal(state_path):
    records = []
    last_complete_offset = 0
    tail_damaged = False
    try:
        with open(state_path, "rb") as handle:
            handle.seek(0, 2)
            file_size = handle.tell()
            handle.seek(0)
            while True:
                record_start = handle.tell()
                try:
                    envelope = pickle.load(handle)
                except EOFError:
                    if record_start < file_size:
                        tail_damaged = True
                        log.warning(
                            "[状态台账] 尾部记录不完整，已使用此前有效记录")
                    break
                except Exception as exc:
                    tail_damaged = True
                    log.warning("[状态台账] 尾部记录不完整，已使用此前有效记录: %s" % exc)
                    break
                last_complete_offset = handle.tell()
                try:
                    generation, state, broker_positions = (
                        _decode_live_state_envelope(envelope))
                except Exception as exc:
                    log.error("[状态台账] 记录无效: %s" % exc)
                    continue
                records.append((generation, state, broker_positions))
    except FileNotFoundError:
        return [], 0, False
    except Exception as exc:
        log.error("[状态台账] 读取失败: %s" % exc)
        return [], 0, False
    return records, last_complete_offset, tail_damaged


def _read_live_state_journal(state_path):
    records, _, _ = _scan_live_state_journal(state_path)
    return records


def _persist_live_state(context=None, path=None):
    state_path = _cached_live_state_path(path)
    state = {
        field: getattr(g, field, None)
        for field in LIVE_STATE_FIELDS
    }
    try:
        validated_state = _validated_live_state(state)
        broker_positions = _broker_position_snapshot(context)
        records = []
        last_complete_offset = 0
        tail_damaged = False
        if state_path is not None:
            records, last_complete_offset, tail_damaged = (
                _scan_live_state_journal(state_path))
        persisted_generation = getattr(g, "live_state_generation", 0)
        if not isinstance(persisted_generation, int) or persisted_generation < 0:
            persisted_generation = 0
        generation = max(
            [persisted_generation] + [item[0] for item in records]
        ) + 1
        envelope = _encode_live_state_envelope(
            validated_state, generation, broker_positions)
        _record_persisted_g_state(
            validated_state, generation, broker_positions)
        if state_path is None:
            return True
        if tail_damaged:
            with open(state_path, "r+b") as handle:
                handle.truncate(last_complete_offset)
            log.warning("[状态台账] 已移除不完整尾部，继续追加新记录")
        with open(state_path, "ab") as handle:
            pickle.dump(
                envelope, handle, protocol=LIVE_STATE_PICKLE_PROTOCOL)
        return True
    except Exception as exc:
        log.error("[状态] 保存失败: %s" % exc)
        return False


def _validated_live_state(state):
    if not isinstance(state, dict):
        raise ValueError("invalid state body")
    missing = [field for field in LIVE_STATE_FIELDS if field not in state]
    if missing:
        raise ValueError("missing state fields: %s" % ",".join(missing))
    mapping_fields = (
        "highest_since_buy",
        "entry_atr",
        "buy_date",
        "last_scores",
        "sold_today",
        "sell_retry_reasons",
    )
    for field in mapping_fields:
        if not isinstance(state[field], dict):
            raise ValueError("invalid mapping field: %s" % field)
    for field in ("paused_pool_codes", "unverified_positions"):
        if not isinstance(state[field], set):
            raise ValueError("invalid set field: %s" % field)
    if not isinstance(state["deferred_scores"], list):
        raise ValueError("invalid deferred scores")

    validated = dict(state)
    for field in ("execution_date", "deferred_signal_date"):
        value = state[field]
        normalized = _as_date(value) if value is not None else None
        if value is not None and normalized is None:
            raise ValueError("invalid date field: %s" % field)
        validated[field] = normalized
    return validated


def _load_live_state(context=None, path=None):
    state_path = _cached_live_state_path(path)
    if state_path is None:
        return None
    g.__state_restore_source = None
    g.__state_restore_generation = None
    records = _read_live_state_journal(state_path)
    if not records:
        return None
    generation, state, recorded_positions = max(
        records, key=lambda item: item[0])
    current_positions = _broker_position_snapshot(context)
    if not _broker_position_snapshots_match(
            recorded_positions, current_positions):
        log.warning("[状态台账] 当前券商持仓与记录不一致，已拒绝恢复")
        return None
    g.__state_restore_source = "journal"
    g.__state_restore_generation = generation
    return state


def _restore_live_state(context=None, path=None):
    state = _load_live_state(context=context, path=path)
    if state is None:
        return False
    for field in LIVE_STATE_FIELDS:
        setattr(g, field, state[field])
    return True


def _restore_live_state_continuity(state):
    state = _validated_live_state(state)
    for field in (
        "last_scores",
        "sold_today",
        "sell_retry_reasons",
        "paused_pool_codes",
        "execution_date",
        "deferred_scores",
        "deferred_signal_date",
    ):
        setattr(g, field, state[field])


def _restore_persisted_g_risk_state(context, state):
    state = _validated_live_state(state)
    g.highest_since_buy = dict(state["highest_since_buy"])
    g.entry_atr = dict(state["entry_atr"])
    g.buy_date = dict(state["buy_date"])
    g.unverified_positions = set(state["unverified_positions"])
    g.__position_recovery_source = {
        code: "ptrade-g" for code in current_hold_codes(context)
    }


def _clear_live_risk_state_for_broker_recovery():
    g.highest_since_buy = {}
    g.entry_atr = {}
    g.buy_date = {}
    g.unverified_positions = set()
    g.__position_recovery_source = {}


def _normalized_state_mapping(mapping):
    if not isinstance(mapping, dict):
        return {}
    return {
        normalize_code(code): value
        for code, value in mapping.items()
        if normalize_code(code)
    }


def _restore_live_state_risk_fallback(context, state):
    if not isinstance(state, dict):
        return set()
    journal_highest = _normalized_state_mapping(
        state.get("highest_since_buy"))
    journal_atr = _normalized_state_mapping(state.get("entry_atr"))
    journal_buy_date = _normalized_state_mapping(state.get("buy_date"))
    restored = set()
    source_map = getattr(g, "__position_recovery_source", None)
    if not isinstance(source_map, dict):
        source_map = {}
        g.__position_recovery_source = source_map

    for code in current_hold_codes(context):
        current_complete = (
            _as_date(g.buy_date.get(code)) is not None and
            _is_positive_finite(g.entry_atr.get(code)) and
            _is_positive_finite(g.highest_since_buy.get(code))
        )
        if current_complete:
            continue
        buy_date = _as_date(journal_buy_date.get(code))
        atr = journal_atr.get(code)
        highest = journal_highest.get(code)
        if (
            buy_date is None or
            not _is_positive_finite(atr) or
            not _is_positive_finite(highest)
        ):
            continue
        g.buy_date[code] = buy_date
        g.entry_atr[code] = float(atr)
        g.highest_since_buy[code] = float(highest)
        g.unverified_positions.discard(code)
        source_map[code] = "journal"
        restored.add(code)
    return restored


def get_a_share_etf_codes():
    return set([
        "510300",
        "159915",
        "512100",
        "159928",
        "510880",
    ])


def buy_position_scale(score, params=None):
    p = params or get_default_params()
    code = str(score.get("code", "")).split(".")[0]
    if code in get_a_share_etf_codes() and score.get("volume_score", 0) <= 0:
        scale = float(p.get("a_share_zero_volume_buy_scale", 1.0))
        return max(0.0, min(1.0, scale))
    return 1.0


def calc_buy_target_value(total_value, score, params=None):
    p = params or get_default_params()
    base_target = float(total_value) * float(p["base_ratio"]) / int(p["max_hold"])
    return base_target * buy_position_scale(score, p)


def format_indicator_params(params):
    return (
        "RSI(%d,%d,%d) MACD(%d,%d,%d) KDJ(%d,%d,%d) BOLL(%d,%.1f) "
        "ATR(%d) ADX(%d) MA(5,10,20,60)" % (
            params["rsi_fast"], params["rsi_mid"], params["rsi_slow"],
            params["macd_fast"], params["macd_slow"], params["macd_signal"],
            params["kdj_n"], params["kdj_m1"], params["kdj_m2"],
            params["boll_period"], params["boll_std"],
            params["atr_period"], params["adx_period"],
        )
    )


def format_self_check():
    fast = pd.Series([40.0, 41.0, 42.0, 45.9, 48.1])
    slow = pd.Series([42.0, 42.0, 42.0, 50.5, 46.2])
    diff_cross_ok = crossed_above_recent(fast, slow, window=3)
    score = score_buy_snapshot({
        "rsi6": 48.1,
        "rsi6_cross_rsi12_up": False,
        "rsi6_cross_rsi24_up": diff_cross_ok,
        "macd_cross_up": False,
        "kdj_k_cross_up": False,
        "kdj_j_cross_up": False,
    })
    return (
        "[%s] positional-diff-cross enabled | "
        "diff_cross_self_check=%s expected=True | self_rev=%.0f" % (
            STRATEGY_VERSION, diff_cross_ok, score["reversal_score"])
    )


def format_indicator_values(item):
    rsi_diff_12 = item.get("rsi6", np.nan) - item.get("rsi12", np.nan)
    rsi_diff_24 = item.get("rsi6", np.nan) - item.get("rsi24", np.nan)
    rsi_diff_12_prev = item.get("rsi6_prev", np.nan) - item.get("rsi12_prev", np.nan)
    rsi_diff_24_prev = item.get("rsi6_prev", np.nan) - item.get("rsi24_prev", np.nan)
    macd_diff = item.get("dif", np.nan) - item.get("dea", np.nan)
    macd_diff_prev = item.get("dif_prev", np.nan) - item.get("dea_prev", np.nan)
    kdj_diff_k = item.get("k", np.nan) - item.get("d", np.nan)
    kdj_diff_j = item.get("j", np.nan) - item.get("d", np.nan)
    kdj_diff_k_prev = item.get("k_prev", np.nan) - item.get("d_prev", np.nan)
    kdj_diff_j_prev = item.get("j_prev", np.nan) - item.get("d_prev", np.nan)
    return (
        "RSI[6/12/24]=%.1f/%.1f/%.1f "
        "MACD[DIF/DEA/HIST]=%.4f/%.4f/%.4f "
        "KDJ[K/D/J]=%.1f/%.1f/%.1f "
        "BOLL[U/M/L]=%.3f/%.3f/%.3f "
        "MA[5/10/20/60]=%.3f/%.3f/%.3f/%.3f "
        "VOL[5/20]=%.0f/%.0f "
        "ATR14=%.4f "
        "DMI[+DI/-DI/ADX]=%.1f/%.1f/%.1f "
        "RSI_DIFF[6-12/6-24]=%.1f/%.1f(prev %.1f/%.1f) "
        "MACD_DIFF[DIF-DEA]=%.4f(prev %.4f) "
        "KDJ_DIFF[K-D/J-D]=%.1f/%.1f(prev %.1f/%.1f)" % (
            item.get("rsi6", np.nan), item.get("rsi12", np.nan), item.get("rsi24", np.nan),
            item.get("dif", np.nan), item.get("dea", np.nan), item.get("macd_hist", np.nan),
            item.get("k", np.nan), item.get("d", np.nan), item.get("j", np.nan),
            item.get("boll_upper", np.nan), item.get("boll_mid", np.nan), item.get("boll_lower", np.nan),
            item.get("ma5", np.nan), item.get("ma10", np.nan),
            item.get("ma20", np.nan), item.get("ma60", np.nan),
            item.get("vol5", np.nan), item.get("vol20", np.nan),
            item.get("atr", np.nan),
            item.get("plus_di", np.nan), item.get("minus_di", np.nan), item.get("adx", np.nan),
            rsi_diff_12, rsi_diff_24, rsi_diff_12_prev, rsi_diff_24_prev,
            macd_diff, macd_diff_prev,
            kdj_diff_k, kdj_diff_j, kdj_diff_k_prev, kdj_diff_j_prev,
        )
    )


def format_cross_flags(item):
    return (
        "RSI12_UP=%s RSI24_UP=%s MACD_UP=%s KDJ_K_UP=%s KDJ_J_UP=%s "
        "RSI12_DOWN=%s RSI24_DOWN=%s MACD_DOWN=%s KDJ_K_DOWN=%s KDJ_J_DOWN=%s" % (
            item.get("rsi6_cross_rsi12_up"),
            item.get("rsi6_cross_rsi24_up"),
            item.get("macd_cross_up"),
            item.get("kdj_k_cross_up"),
            item.get("kdj_j_cross_up"),
            item.get("rsi6_cross_rsi12_down"),
            item.get("rsi6_cross_rsi24_down"),
            item.get("macd_cross_down"),
            item.get("kdj_k_cross_down"),
            item.get("kdj_j_cross_down"),
        )
    )


def _format_self_check_for_log():
    text = format_self_check()
    replacements = (
        ("positional-diff-cross enabled", "位置差值上穿已启用"),
        ("diff_cross_self_check=", "自检="),
        ("expected=", "预期="),
        ("self_rev=", "自检反转分="),
        ("True", "通过"),
        ("False", "未通过"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def _format_indicator_values_for_log(item):
    return format_indicator_values(item).replace("(prev ", "(前值 ")


def _format_cross_flags_for_log(item):
    text = format_cross_flags(item)
    replacements = (
        ("RSI12_UP", "RSI12上穿"),
        ("RSI24_UP", "RSI24上穿"),
        ("MACD_UP", "MACD上穿"),
        ("KDJ_K_UP", "KDJ_K上穿"),
        ("KDJ_J_UP", "KDJ_J上穿"),
        ("RSI12_DOWN", "RSI12下穿"),
        ("RSI24_DOWN", "RSI24下穿"),
        ("MACD_DOWN", "MACD下穿"),
        ("KDJ_K_DOWN", "KDJ_K下穿"),
        ("KDJ_J_DOWN", "KDJ_J下穿"),
        ("True", "是"),
        ("False", "否"),
        ("None", "未知"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def _format_reason_for_log(reason):
    text = str(reason or "")
    exact = {
        "no_data": "无数据",
        "zero_recent_volume": "近期成交量为零",
        "paused": "停牌",
        "unknown": "未知原因",
        "recovered_open_order": "恢复的未完成委托",
    }
    if text in exact:
        return exact[text]
    prefixes = (
        ("short_data:", "数据长度不足:"),
        ("invalid_close:", "收盘价无效:"),
        ("nan_fields:", "指标缺失:"),
        ("exception:", "异常:"),
        ("sell_score ", "卖出分 "),
        ("atr_stop ", "ATR止损 "),
    )
    for source, target in prefixes:
        if text.startswith(source):
            return target + text[len(source):]
    return text or "无"


def _format_recovery_source_for_log(source):
    text = str(source or "")
    if text.startswith("account-takeover:"):
        suffix = text.split(":", 1)[1]
        suffix = {"get-deliver": "交割单"}.get(suffix, suffix)
        return "账户接管:" + suffix
    if text.startswith("checkpoint-"):
        return "检查点-" + text.split("-", 1)[1]
    return {
        "get-trades": "当前策略成交",
        "get-deliver": "交割单",
        "journal": "状态台账",
        "mixed": "混合恢复",
        "no-position": "无持仓",
        "unverified": "未验证",
        "ptrade-g": "PTrade持久状态",
    }.get(text, text or "无")


# 二、PTrade 生命周期与任务调度
# 实盘只注册 09:35 主流程和 10:35 停复牌/废单补偿，共两个定时任务。
# 回测由 handle_data 驱动且固定在收盘执行，因此只能用于冒烟检查，不能评价收益。

def initialize(context):
    set_benchmark("000300.SS")
    try:
        is_live = bool(is_trade())
        mode_verified = True
    except Exception as exc:
        is_live = False
        mode_verified = False
        log.error("[初始化] 交易模式检测失败，交易已停用: %s" % exc)
    if mode_verified and is_live:
        try:
            set_parameters(
                receive_cancel_response="1",
                not_restart_trade="0",
                server_restart_not_do_before="0",
            )
        except Exception as exc:
            log.warning("[初始化] 平台参数设置失败: %s" % exc)
    elif mode_verified:
        try:
            set_commission(commission_ratio=0.0003, min_commission=5.0, type="ETF")
            set_slippage(slippage=0.001)
        except Exception as exc:
            log.warning("[初始化] 佣金或滑点设置失败: %s" % exc)

    g.params = get_default_params()
    g.etf_pool = get_default_etf_pool()
    g.highest_since_buy = {}
    g.entry_atr = {}
    g.buy_date = {}
    g.last_scores = {}
    g.sold_today = {}
    g.sell_retry_reasons = {}
    g.paused_pool_codes = set()
    g.unverified_positions = set()
    g.execution_date = None
    g.deferred_scores = []
    g.deferred_signal_date = None
    # 普通 g 字段由 PTrade 框架自动持久化；盘前必须与券商持仓重新核验后才能使用。
    g.live_state_schema_version = None
    g.live_state_business_fingerprint = None
    g.live_state_generation = None
    g.live_state_broker_positions = None
    g.__last_snapshot = {}
    g.__pending_orders = {}
    g.__pending_sells = {}
    g.__order_state_unknown = False
    g.__data = None
    g.__is_live = is_live
    g.__mode_verified = mode_verified
    # initialize 阶段不允许调用 get_trade_name；实例隔离的检查点路径延后到盘前阶段解析。
    g.__state_path = None
    g.__state_restore_source = None
    g.__state_restore_generation = None
    g.__persisted_g_status = None
    g.__persisted_g_reason = None
    g.__persisted_g_generation = None
    g.__position_recovery_source = {}
    g.__startup_recovery_done = False

    try:
        set_universe(g.etf_pool)
    except Exception as exc:
        log.warning("[初始化] 设置标的池失败: %s" % exc)

    if g.__is_live:
        run_daily(context, _do_trading_wrapper, time="09:35")
        run_daily(context, _halt_recover_wrapper, time="10:35")

    log.info("[%s] 初始化完成: 最大持仓=%d 基础仓位比例=%.2f 普通信号最短持有=%d" % (
        STRATEGY_VERSION,
        g.params["max_hold"],
        g.params["base_ratio"],
        g.params["min_signal_hold_days"]))
    log.info("[发布指纹] 构建=%s 业务配置=%s 状态结构=%d" % (
        DEPLOYMENT_BUILD_ID,
        business_config_fingerprint(g.params, g.etf_pool),
        LIVE_STATE_SCHEMA_VERSION))
    log.info(_format_self_check_for_log())
    log.info("[指标参数] %s" % format_indicator_params(g.params))


def handle_data(context, data):
    """PTrade 回测入口；日线回测会被平台固定在收盘时点执行。"""
    g.__data = data
    if not getattr(g, "__mode_verified", False):
        log.error("[数据处理] 交易模式未验证，交易已阻止")
        return
    if g.__is_live:
        return
    do_trading(context)


def before_trading_start(context, data):
    g.__data = data
    g.__last_snapshot = {}
    startup_recovery = bool(
        g.__is_live and not getattr(g, "__startup_recovery_done", False))
    persisted_g_state = None
    use_persisted_g = False
    journal_state = None
    if g.__is_live:
        if _cached_live_state_path() is None:
            g.__state_path = _live_state_path()
        if startup_recovery:
            persisted_g_candidate = _load_persisted_g_state(context)
            journal_state = _load_live_state(context)
            if persisted_g_candidate is not None:
                persisted_g_generation, persisted_g_state = persisted_g_candidate
                journal_generation = getattr(
                    g, "__state_restore_generation", None)
                journal_is_newer = (
                    journal_state is not None and
                    isinstance(journal_generation, int) and
                    journal_generation > persisted_g_generation
                )
                if not journal_is_newer:
                    use_persisted_g = True
                    g.__state_restore_source = "ptrade-g"
                    g.__state_restore_generation = persisted_g_generation
                else:
                    _set_persisted_g_diagnostic(
                        "superseded", "newer-journal",
                        persisted_g_generation)
            continuity_state = (
                persisted_g_state if use_persisted_g else journal_state)
            if continuity_state is not None:
                _restore_live_state_continuity(continuity_state)
    _lock_frozen_business_config()
    today = _as_date(get_context_datetime(context))
    if today is None:
        g.__order_state_unknown = True
        log.error("[每日重置] 无法确定当前交易日，交易已阻止")
        return
    if g.execution_date != today:
        g.execution_date = today
        g.sold_today = {}
        g.sell_retry_reasons = {}
        g.paused_pool_codes = set()
        g.deferred_scores = []
        g.deferred_signal_date = None
    if g.__is_live:
        if startup_recovery:
            if use_persisted_g:
                _restore_persisted_g_risk_state(context, persisted_g_state)
            else:
                _clear_live_risk_state_for_broker_recovery()
        _reconcile_open_orders(context)
        if startup_recovery and use_persisted_g:
            recover_live_state(context)
        else:
            _recover_live_state_with_available_sources(context, allow_deliver=True)
            if startup_recovery and journal_state is not None:
                _restore_live_state_risk_fallback(context, journal_state)
                recover_live_state(context)
        g.__startup_recovery_done = True
        _log_live_recovery_summary(context)
    else:
        g.__pending_orders = {}
        g.__pending_sells = {}
        g.__order_state_unknown = False
    if g.__is_live:
        _persist_live_state(context)


def after_trading_end(context, data):
    g.__data = data
    if g.__is_live:
        _recover_live_state_with_available_sources(context, allow_deliver=True)
        _audit_after_close_open_orders()
    after_close(context)
    g.sold_today = {}
    if g.__is_live:
        _persist_live_state(context)


def _do_trading_wrapper(context):
    do_trading(context)
    _persist_live_state(context)


def _halt_recover_wrapper(context):
    halt_recover(context)
    _persist_live_state(context)


# 三、技术指标与交叉信号
# 本段纯计算逻辑与聚宽正式版保持一致；交叉按数组位置比较最近有效差值，避免索引错位。
# 所有买卖评分只消费已经截断到 T-1 的日线快照，不读取 T 日完整行情。

def calc_rsi(close, period):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    rsi[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return rsi


def calc_macd(close, fast, slow, signal):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, 2 * (dif - dea)


def calc_kdj(high, low, close, n, m1, m2):
    lowest = low.rolling(n).min()
    highest = high.rolling(n).max()
    rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def calc_bollinger(close, period, std_mult):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + std_mult * std, mid, mid - std_mult * std


def calc_atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_dmi_adx(high, low, close, period):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    atr = calc_atr(high, low, close, period)
    plus_di = 100 * plus_dm.rolling(period).sum() / atr.rolling(period).sum()
    minus_di = 100 * minus_dm.rolling(period).sum() / atr.rolling(period).sum()
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return plus_di, minus_di, adx


def _valid_pair(a_prev, a_cur, b_prev, b_cur):
    return not _builtins.any(pd.isna(v) for v in [a_prev, a_cur, b_prev, b_cur])


def _as_float_array(values):
    if hasattr(values, "values"):
        values = values.values
    return np.asarray(values, dtype=float)


def latest_cross_direction_by_diff_recent(fast, slow, window=3):
    fast_values = _as_float_array(fast)
    slow_values = _as_float_array(slow)
    if len(fast_values) < window + 1 or len(slow_values) < window + 1:
        return None
    diff = fast_values - slow_values
    latest_direction = None
    for offset in range(window, 0, -1):
        prev_idx = -offset - 1
        cur_idx = -offset
        prev_diff, cur_diff = diff[prev_idx], diff[cur_idx]
        if not _builtins.any(pd.isna(v) for v in [prev_diff, cur_diff]) and prev_diff <= 0 and cur_diff > 0:
            latest_direction = "above"
        elif not _builtins.any(pd.isna(v) for v in [prev_diff, cur_diff]) and prev_diff >= 0 and cur_diff < 0:
            latest_direction = "below"
    return latest_direction


def crossed_above_by_diff_recent(fast, slow, window=3):
    return latest_cross_direction_by_diff_recent(fast, slow, window) == "above"


def crossed_below_by_diff_recent(fast, slow, window=3):
    return latest_cross_direction_by_diff_recent(fast, slow, window) == "below"


def crossed_above_recent(fast, slow, window=3):
    return crossed_above_by_diff_recent(fast, slow, window)


def crossed_below_recent(fast, slow, window=3):
    return crossed_below_by_diff_recent(fast, slow, window)


def rsi_group_direction(snapshot):
    rsi_up = (
        snapshot.get("rsi6_cross_rsi12_up") or
        snapshot.get("rsi6_cross_rsi24_up")
    )
    rsi_down = (
        snapshot.get("rsi6_cross_rsi12_down") or
        snapshot.get("rsi6_cross_rsi24_down")
    )
    if rsi_up and not rsi_down:
        return "up"
    if rsi_down and not rsi_up:
        return "down"
    return None


def score_buy_snapshot(snapshot, params=None):
    p = params or get_default_params()
    reversal = 0
    if rsi_group_direction(snapshot) == "up":
        reversal += 12 if snapshot.get("rsi6_cross_rsi12_up") else 0
        reversal += 12 if snapshot.get("rsi6_cross_rsi24_up") else 0
    reversal += 10 if snapshot.get("macd_cross_up") else 0
    reversal += 6 if snapshot.get("kdj_k_cross_up") else 0
    reversal += 5 if snapshot.get("kdj_j_cross_up") else 0

    location = 0
    location += 10 if snapshot.get("close_between_boll_lower_mid") else 0
    location += 8 if snapshot.get("close_cross_boll_mid_up") else 0
    location += 7 if snapshot.get("close_near_ma20") else 0
    location -= 10 if snapshot.get("close_far_above_ma20") else 0

    trend = 0
    trend += 6 if snapshot.get("ma5_gt_ma10") else 0
    trend += 6 if snapshot.get("ma10_gt_ma20") else 0
    trend += 5 if snapshot.get("ma20_slope_non_negative") else 0
    trend += 3 if snapshot.get("close_gt_ma60") else 0
    trend -= 15 if snapshot.get("downside_continuation") else 0

    volume = 0
    volume += 6 if snapshot.get("volume_above_vol20_and_up") else 0
    volume += 4 if snapshot.get("vol5_gt_vol20") else 0

    total = max(0, reversal + location + trend + volume)
    rsi6 = snapshot.get("rsi6")
    buy_allowed = rsi6 is None or pd.isna(rsi6) or rsi6 < p["overheat_rsi"]
    return {
        "buy_score": total,
        "reversal_score": reversal,
        "location_score": location,
        "trend_score": trend,
        "volume_score": volume,
        "buy_allowed": buy_allowed,
    }


def score_sell_snapshot(snapshot):
    reversal = 0
    if rsi_group_direction(snapshot) == "down":
        reversal += 12 if snapshot.get("rsi6_cross_rsi12_down") else 0
        reversal += 12 if snapshot.get("rsi6_cross_rsi24_down") else 0
    reversal += 10 if snapshot.get("macd_cross_down") else 0
    reversal += 6 if snapshot.get("kdj_k_cross_down") else 0
    reversal += 5 if snapshot.get("kdj_j_cross_down") else 0

    risk = 0
    risk += 8 if snapshot.get("far_above_ma20_and_rsi6_down") else 0
    risk += 10 if snapshot.get("close_below_falling_ma10") else 0
    risk += 6 if snapshot.get("fell_back_inside_boll") else 0

    return {
        "sell_score": max(0, reversal + risk),
        "sell_reversal_score": reversal,
        "sell_risk_score": risk,
    }


def should_force_sell(sell_score_result, atr_stop_triggered=False, params=None):
    p = params or get_default_params()
    if atr_stop_triggered:
        return True
    return (
        sell_score_result.get("sell_score", 0) >= p["sell_threshold"] and
        has_signal_sell_confirmation(sell_score_result) and
        not is_protected_by_strong_adx_uptrend(sell_score_result, p)
    )


def is_protected_by_strong_adx_uptrend(snapshot, params=None):
    p = params or get_default_params()
    severe_break = (
        snapshot.get("close_below_ma20") or
        snapshot.get("close_below_falling_ma10") or
        snapshot.get("downside_continuation")
    )
    if severe_break:
        return False
    return is_strong_adx_uptrend(snapshot, p)


def is_strong_adx_uptrend(snapshot, params=None):
    p = params or get_default_params()
    adx = snapshot.get("adx")
    plus_di = snapshot.get("plus_di")
    minus_di = snapshot.get("minus_di")
    if _builtins.any(pd.isna(v) for v in [adx, plus_di, minus_di]):
        return False
    return (
        adx >= p["adx_trend_threshold"] and
        plus_di > minus_di and
        snapshot.get("ma20_slope_non_negative")
    )


def has_signal_sell_confirmation(snapshot):
    return (
        snapshot.get("close_below_ma20") or
        snapshot.get("close_below_boll_mid") or
        snapshot.get("close_below_falling_ma10") or
        snapshot.get("downside_continuation") or
        snapshot.get("far_above_ma20_and_rsi6_down")
    )


def _date_key(value):
    if isinstance(value, np.str_):
        value = str(value)
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def can_sell_by_signal(buy_date, today, min_hold_days=1, trade_days=None):
    if buy_date is None:
        return True
    if int(min_hold_days) <= 1:
        return _date_key(buy_date) != _date_key(today)
    buy_key = _date_key(buy_date)
    today_key = _date_key(today)
    if trade_days is not None:
        keys = [_date_key(day) for day in trade_days]
        if buy_key in keys and today_key in keys:
            return keys.index(today_key) - keys.index(buy_key) >= int(min_hold_days)
    return (pd.Timestamp(today_key) - pd.Timestamp(buy_key)).days >= int(min_hold_days)


def can_sell_with_verified_calendar(buy_date, today, min_hold_days=1, trade_days=None):
    if int(min_hold_days) > 1 and trade_days is None:
        return False
    return can_sell_by_signal(buy_date, today, min_hold_days, trade_days)


def sort_candidates(candidates):
    return sorted(candidates, key=lambda x: (
        -x.get("buy_score", 0),
        -x.get("reversal_score", 0),
        x.get("code", "")
    ))


def has_new_buy_position(snapshot, params=None):
    if snapshot.get("close_far_above_ma20"):
        return False
    return (
        snapshot.get("close_between_boll_lower_mid") or
        snapshot.get("close_cross_boll_mid_up") or
        snapshot.get("close_near_ma20")
    )


def filter_buy_candidates(scores, held_codes, params=None):
    p = params or get_default_params()
    held = set(held_codes)
    return [
        s for s in scores
        if s.get("buy_allowed")
        and s.get("buy_score", 0) >= p["buy_threshold"]
        and s.get("sell_score", 0) < p["sell_threshold"]
        and has_new_buy_position(s, p)
        and not is_blocked_entry_combo(s)
        and s.get("code") not in held
    ]


def is_blocked_entry_combo(score):
    rsi_up = score.get("rsi6_cross_rsi12_up") or score.get("rsi6_cross_rsi24_up")
    kdj_up = score.get("kdj_k_cross_up") or score.get("kdj_j_cross_up")
    return (
        bool(rsi_up)
        and bool(score.get("macd_cross_up"))
        and not bool(kdj_up)
        and _numeric_score(score.get("volume_score")) > 0
        and 0 < _numeric_score(score.get("trend_score")) < 20
    )


def _numeric_score(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_cross_signal_candidates(scores, limit=5):
    items = [s for s in scores if s.get("reversal_score", 0) > 0]
    items = sorted(items, key=lambda x: (
        -x.get("reversal_score", 0),
        -x.get("buy_score", 0),
        x.get("code", "")
    ))
    return {
        "count": len(items),
        "items": items[:limit],
    }


def summarize_loose_reversal_candidates(scores, limit=5):
    items = []
    for score in scores:
        rsi6_delta = score.get("rsi6", np.nan) - score.get("rsi6_prev", np.nan)
        dif_delta = score.get("dif", np.nan) - score.get("dif_prev", np.nan)
        k_delta = score.get("k", np.nan) - score.get("k_prev", np.nan)
        j_delta = score.get("j", np.nan) - score.get("j_prev", np.nan)
        rsi_turn_up = rsi6_delta > 0
        macd_turn_up = dif_delta > 0
        kdj_turn_up = k_delta > 0 or j_delta > 0
        loose_count = int(rsi_turn_up) + int(macd_turn_up) + int(kdj_turn_up)
        if loose_count <= 0:
            continue
        item = dict(score)
        item.update({
            "loose_reversal_count": loose_count,
            "rsi_turn_up": rsi_turn_up,
            "macd_turn_up": macd_turn_up,
            "kdj_turn_up": kdj_turn_up,
            "rsi6_delta": round(float(rsi6_delta), 4),
            "dif_delta": round(float(dif_delta), 4),
            "k_delta": round(float(k_delta), 4),
            "j_delta": round(float(j_delta), 4),
        })
        items.append(item)

    items = sorted(items, key=lambda x: (
        -x.get("loose_reversal_count", 0),
        -x.get("buy_score", 0),
        x.get("code", "")
    ))
    return {
        "count": len(items),
        "items": items[:limit],
    }


def score_skip_reason(df, snapshot, required_fields, min_len):
    if df is None:
        return "no_data"
    if len(df) < min_len:
        return "short_data:%d<%d" % (len(df), min_len)
    if "close" in df and df["close"].iloc[-1] <= 0:
        return "invalid_close:%.4f" % df["close"].iloc[-1]
    if "volume" in df and df["volume"].iloc[-5:].sum() == 0:
        return "zero_recent_volume"
    if snapshot is None:
        return None
    nan_fields = [k for k in required_fields if pd.isna(snapshot.get(k))]
    if nan_fields:
        return "nan_fields:%s" % ",".join(nan_fields)
    return None


# 四、T-1 日线数据与交易日证明
# 交易日必须由平台日历证明；无法证明 T-1 时直接停止交易，不用自然日猜测。
# PTrade 停牌日线会以昨值填充且成交量为零，数据层统一剔除这些行以对齐聚宽 skip_paused。

def normalize_code(code):
    """把 PTrade 回调或聚宽格式代码统一转换为 PTrade 标的代码。"""
    text = str(code or "").strip().upper()
    if not text:
        return ""
    base = text.split(".")[0]
    if text.endswith((".XSHG", ".SH", ".SS")):
        return base + ".SS"
    if text.endswith((".XSHE", ".SZ")):
        return base + ".SZ"
    return base + (".SS" if base.startswith(("5", "6", "9")) else ".SZ")


def get_context_datetime(context):
    blotter = getattr(context, "blotter", None)
    value = getattr(blotter, "current_dt", None)
    if value is None:
        value = getattr(context, "current_dt", None)
    return value


def _as_date(value):
    if value is None:
        return None
    if isinstance(value, np.str_):
        value = str(value)
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "date") and not isinstance(value, str):
        try:
            return value.date()
        except Exception:
            pass
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _api_date_text(value):
    trade_date = _as_date(value)
    return trade_date.strftime("%Y%m%d") if trade_date is not None else None


def _previous_day_from_result(result, today):
    if result is None:
        return None
    if isinstance(result, tuple):
        values = []
        for item in result:
            if isinstance(item, (list, tuple, np.ndarray, pd.Index)):
                values.extend(list(item))
            else:
                values.append(item)
    elif isinstance(result, (list, tuple, np.ndarray, pd.Index, pd.Series)):
        values = list(result)
    else:
        values = [result]
    dates = [_as_date(item) for item in values]
    dates = sorted(set(item for item in dates if item is not None and item < today))
    return dates[-1] if dates else None


def _calendar_payload_summary(result):
    type_name = type(result).__name__
    shape = getattr(result, "shape", None)
    try:
        value_text = repr(result)
    except Exception:
        value_text = "<unrepresentable>"
    if len(value_text) > 240:
        value_text = value_text[:237] + "..."
    return "类型=%s 形状=%s 值=%s" % (type_name, shape, value_text)


def get_prev_trade_date(context):
    now = get_context_datetime(context)
    today = _as_date(now)
    if today is None:
        log.error("[交易日] 上下文当前时间不可用，交易已中止")
        return None
    trading_day_getter = globals().get("get_trading_day")
    if trading_day_getter is not None:
        try:
            current_raw = trading_day_getter(0)
            current_day = _as_date(current_raw)
            if current_day is not None and current_day < today:
                return current_day
            if current_day == today:
                previous_raw = trading_day_getter(-1)
                previous_day = _as_date(previous_raw)
                if previous_day is not None and previous_day < today:
                    return previous_day
                log.warning(
                    "[交易日] get_trading_day(-1)返回值不可用: %s" %
                    _calendar_payload_summary(previous_raw)
                )
            else:
                log.warning(
                    "[交易日] get_trading_day返回值不可用: %s" %
                    _calendar_payload_summary(current_raw)
                )
        except Exception as exc:
            log.warning("[交易日] get_trading_day调用失败: %s" % exc)
    try:
        result = get_trade_days(end_date=_api_date_text(today), count=2)
        prev = _previous_day_from_result(result, today)
        if prev is not None:
            return prev
        log.warning("[交易日] get_trade_days返回值不可用: %s" % (
            _calendar_payload_summary(result)))
    except Exception as exc:
        log.warning("[交易日] get_trade_days调用失败: %s" % exc)
    try:
        result = get_all_trades_days(date=today.strftime("%Y%m%d"))
        prev = _previous_day_from_result(result, today)
        if prev is not None:
            return prev
        log.warning("[交易日] get_all_trades_days返回值不可用: %s" % (
            _calendar_payload_summary(result)))
    except Exception as exc:
        log.warning("[交易日] get_all_trades_days调用失败: %s" % exc)
    log.error("[交易日] 无法证实T-1交易日，交易已中止")
    return None


def get_price_data(code, end_date, count):
    """读取截至已证明 T-1 的前复权日线，并拒绝越过信号日期的数据。"""
    end_date_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)
    fields = ["open", "close", "high", "low", "volume"]
    try:
        frame = get_price(
            code,
            end_date=end_date_str,
            count=count,
            frequency="1d",
            fields=fields,
            fq="pre",
        )
        if frame is not None and len(frame) > 0:
            frame = pd.DataFrame(frame).copy()
            if "code" in frame.columns:
                frame = frame[frame["code"].map(normalize_code) == normalize_code(code)]
            if all(field in frame.columns for field in fields):
                return frame[frame["volume"] > 0][fields]
    except Exception as exc:
        log.warning("[日线数据] get_price调用失败 %s: %s" % (code, exc))

    try:
        series = {}
        for field in fields:
            raw = get_history(
                count,
                "1d",
                field,
                [code],
                fq="pre",
                include=False,
            )
            values = _extract_history_field_series(raw, code, field)
            if values is None:
                raise ValueError("unsupported get_history result")
            series[field] = values
        frame = pd.DataFrame(series)
        frame = _history_frame_through_end_date(frame, end_date_str)
        return frame[frame["volume"] > 0]
    except Exception as exc:
        log.error("[日线数据] 数据不可用 %s: %s" % (code, exc))
        return None


def _extract_history_field_series(raw, code, field):
    """兼容官方新版及旧版 PTrade 历史行情返回结构。"""
    normalized_code = normalize_code(code)
    if isinstance(raw, pd.Series):
        return raw.copy()
    if isinstance(raw, pd.DataFrame):
        if field in raw.columns:
            selected = raw
            if "code" in selected.columns:
                selected = selected[
                    selected["code"].map(normalize_code) == normalized_code
                ]
            return selected[field].copy()
        for column in raw.columns:
            if isinstance(column, tuple):
                parts = [str(part) for part in column]
                if field in parts and any(
                    normalize_code(part) == normalized_code for part in parts
                ):
                    return raw[column].copy()
            elif normalize_code(column) == normalized_code:
                return raw[column].copy()
        return None
    if isinstance(raw, dict):
        for key, values in raw.items():
            if normalize_code(key) != normalized_code:
                continue
            if isinstance(values, pd.DataFrame):
                if field not in values.columns:
                    return None
                return values[field].copy()
            if isinstance(values, pd.Series):
                return values.copy()
            return pd.Series(values)
    return None


def _history_frame_through_end_date(frame, end_date):
    index = frame.index
    if isinstance(index, pd.DatetimeIndex):
        timestamps = index
    else:
        values = list(index)
        if any(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            raise ValueError("history index is not date-like")
        if any(_as_date(value) is None for value in values):
            raise ValueError("history index contains an unprovable date")
        timestamps = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if timestamps.isna().any():
        raise ValueError("history index contains an invalid date")
    if timestamps.tz is not None:
        timestamps = timestamps.tz_localize(None)
    end_timestamp = pd.Timestamp(end_date)
    result = frame.copy()
    result.index = timestamps
    return result[result.index <= end_timestamp]


def build_signal_snapshot(df, params):
    C, H, L, V = df["close"], df["high"], df["low"], df["volume"]
    rsi6 = calc_rsi(C, params["rsi_fast"])
    rsi12 = calc_rsi(C, params["rsi_mid"])
    rsi24 = calc_rsi(C, params["rsi_slow"])
    dif, dea, macd_hist = calc_macd(C, params["macd_fast"], params["macd_slow"], params["macd_signal"])
    k, d, j = calc_kdj(H, L, C, params["kdj_n"], params["kdj_m1"], params["kdj_m2"])
    boll_upper, boll_mid, boll_lower = calc_bollinger(C, params["boll_period"], params["boll_std"])
    atr = calc_atr(H, L, C, params["atr_period"])
    plus_di, minus_di, adx = calc_dmi_adx(H, L, C, params["adx_period"])

    ma5 = C.rolling(5).mean()
    ma10 = C.rolling(10).mean()
    ma20 = C.rolling(20).mean()
    ma60 = C.rolling(60).mean()
    vol5 = V.rolling(5).mean()
    vol20 = V.rolling(20).mean()

    latest = C.iloc[-1]
    prev = C.iloc[-2]
    ma20_slope = ma20.iloc[-1] - ma20.iloc[-6] if len(ma20) >= 6 else np.nan
    rsi6_down = rsi6.iloc[-1] < rsi6.iloc[-2] if len(rsi6) >= 2 else False

    snapshot = {
        "close": latest,
        "rsi6": rsi6.iloc[-1],
        "rsi6_prev": rsi6.iloc[-2],
        "rsi12": rsi12.iloc[-1],
        "rsi12_prev": rsi12.iloc[-2],
        "rsi24": rsi24.iloc[-1],
        "rsi24_prev": rsi24.iloc[-2],
        "dif": dif.iloc[-1],
        "dif_prev": dif.iloc[-2],
        "dea": dea.iloc[-1],
        "dea_prev": dea.iloc[-2],
        "macd_hist": macd_hist.iloc[-1],
        "macd_hist_prev": macd_hist.iloc[-2],
        "k": k.iloc[-1],
        "k_prev": k.iloc[-2],
        "d": d.iloc[-1],
        "d_prev": d.iloc[-2],
        "j": j.iloc[-1],
        "j_prev": j.iloc[-2],
        "ma20": ma20.iloc[-1],
        "ma5": ma5.iloc[-1],
        "ma10": ma10.iloc[-1],
        "ma60": ma60.iloc[-1],
        "vol5": vol5.iloc[-1],
        "vol20": vol20.iloc[-1],
        "boll_upper": boll_upper.iloc[-1],
        "boll_mid": boll_mid.iloc[-1],
        "boll_lower": boll_lower.iloc[-1],
        "atr": atr.iloc[-1],
        "plus_di": plus_di.iloc[-1],
        "minus_di": minus_di.iloc[-1],
        "adx": adx.iloc[-1],
        "rsi6_cross_rsi12_up": crossed_above_recent(rsi6, rsi12, params["cross_window"]),
        "rsi6_cross_rsi24_up": crossed_above_recent(rsi6, rsi24, params["cross_window"]),
        "rsi6_cross_rsi12_down": crossed_below_recent(rsi6, rsi12, params["cross_window"]),
        "rsi6_cross_rsi24_down": crossed_below_recent(rsi6, rsi24, params["cross_window"]),
        "macd_cross_up": crossed_above_recent(dif, dea, params["cross_window"]),
        "macd_cross_down": crossed_below_recent(dif, dea, params["cross_window"]),
        "kdj_k_cross_up": crossed_above_recent(k, d, params["cross_window"]),
        "kdj_j_cross_up": crossed_above_recent(j, d, params["cross_window"]),
        "kdj_k_cross_down": crossed_below_recent(k, d, params["cross_window"]),
        "kdj_j_cross_down": crossed_below_recent(j, d, params["cross_window"]),
        "close_between_boll_lower_mid": boll_lower.iloc[-1] <= latest <= boll_mid.iloc[-1],
        "close_cross_boll_mid_up": crossed_above_recent(C, boll_mid, params["cross_window"]),
        "close_near_ma20": abs(latest / ma20.iloc[-1] - 1) <= 0.05 if ma20.iloc[-1] > 0 else False,
        "close_far_above_ma20": latest / ma20.iloc[-1] - 1 > 0.12 if ma20.iloc[-1] > 0 else False,
        "close_below_ma20": latest < ma20.iloc[-1],
        "close_below_boll_mid": latest < boll_mid.iloc[-1],
        "ma5_gt_ma10": ma5.iloc[-1] > ma10.iloc[-1],
        "ma10_gt_ma20": ma10.iloc[-1] > ma20.iloc[-1],
        "ma20_slope_non_negative": ma20_slope >= 0,
        "close_gt_ma60": latest > ma60.iloc[-1],
        "downside_continuation": latest < ma60.iloc[-1] and ma20_slope < 0,
        "volume_above_vol20_and_up": V.iloc[-1] > vol20.iloc[-1] and latest > prev,
        "vol5_gt_vol20": vol5.iloc[-1] > vol20.iloc[-1],
        "far_above_ma20_and_rsi6_down": latest / ma20.iloc[-1] - 1 > 0.10 and rsi6_down if ma20.iloc[-1] > 0 else False,
        "close_below_falling_ma10": latest < ma10.iloc[-1] and ma10.iloc[-1] < ma10.iloc[-2],
        "fell_back_inside_boll": C.iloc[-2] > boll_upper.iloc[-2] and latest <= boll_upper.iloc[-1],
    }
    return snapshot


def calc_cross_signal_score(code, end_date, return_reason=False):
    p = g.params
    min_len = p["lookback"] - 10
    required = ["rsi6", "rsi12", "rsi24", "dif", "dea", "k", "d", "j", "ma20", "atr", "adx"]
    try:
        df = get_price_data(code, end_date, p["lookback"])
        reason = score_skip_reason(df, None, required, min_len)
        if reason is not None:
            return (None, reason) if return_reason else None

        snapshot = build_signal_snapshot(df, p)
    except Exception as exc:
        reason = "exception:%s" % exc.__class__.__name__
        return (None, reason) if return_reason else None

    reason = score_skip_reason(df, snapshot, required, min_len)
    if reason is not None:
        return (None, reason) if return_reason else None

    buy_score = score_buy_snapshot(snapshot, p)
    sell_score = score_sell_snapshot(snapshot)
    result = {}
    result.update(snapshot)
    result.update(buy_score)
    result.update(sell_score)
    result["code"] = code
    return (result, None) if return_reason else result


def calc_stop_price(highest, atr_val, cost, params=None):
    p = params or get_default_params()
    if highest <= 0 or atr_val <= 0:
        return cost * (1 - p["stop_cap"])
    pct_stop = p["trailing_atr_mult"] * atr_val / highest
    pct_stop = max(p["stop_floor"], min(p["stop_cap"], pct_stop))
    return highest * (1 - pct_stop)


# 五、账户、行情与委托执行
# 实盘以券商持仓、可用资金和行情快照为事实来源；未知状态一律按不可交易处理。
# 卖出使用 pending/sold_today 双重防重，避免 order_target 在持仓同步延迟期间重复下单。

def current_hold_codes(context):
    return [
        normalize_code(code) for code, pos in _positions(context).items()
        if _pos_amount(pos) > 0
    ]


def has_position(context, code):
    code = normalize_code(code)
    for held_code, pos in _positions(context).items():
        if normalize_code(held_code) == code and _pos_amount(pos) > 0:
            return True
    return False


def _total_value(context):
    return float(context.portfolio.portfolio_value)


def _available_cash(context):
    return float(context.portfolio.cash)


def _positions(context):
    return context.portfolio.positions


def _pos_amount(pos):
    return float(getattr(pos, "amount", 0) or 0)


def _pos_cost(pos):
    return float(getattr(pos, "cost_basis", 0) or 0)


def _pos_price(pos):
    return float(getattr(pos, "last_sale_price", 0) or 0)


def _get_position(context, code):
    code = normalize_code(code)
    for held_code, pos in _positions(context).items():
        if normalize_code(held_code) == code:
            return pos
    return None


def _order_field(order_obj, name, default=None):
    if isinstance(order_obj, dict):
        return order_obj.get(name, default)
    return getattr(order_obj, name, default)


def _reconcile_open_orders(context):
    g.__pending_orders = {}
    g.__pending_sells = {}
    g.__order_state_unknown = False
    try:
        open_orders = get_open_orders()
    except Exception as exc:
        g.__order_state_unknown = True
        log.error("[委托恢复] get_open_orders调用失败，交易已阻止: %s" % exc)
        return False

    if not isinstance(open_orders, (list, tuple)):
        g.__order_state_unknown = True
        log.error("[委托恢复] get_open_orders返回值无效，交易已阻止")
        return False

    today = _as_date(get_context_datetime(context))
    pending_buys = {}
    pending_sells = {}
    sold_guards = {}
    for order_obj in open_orders:
        code = normalize_code(_order_field(order_obj, "symbol", ""))
        order_id = str(_order_field(order_obj, "id", "") or "")
        amount = _safe_float(_order_field(order_obj, "amount", 0))
        filled = abs(_safe_float(_order_field(order_obj, "filled", 0)))
        requested = abs(amount)
        quantities_valid = (
            np.isfinite(amount) and
            np.isfinite(filled) and
            requested > 0 and
            0 <= filled <= requested
        )
        if not code or not order_id or not quantities_valid:
            g.__order_state_unknown = True
            log.error("[委托恢复] 未完成委托格式异常，交易已阻止")
            return False
        if code in pending_buys or code in pending_sells:
            g.__order_state_unknown = True
            log.error("[委托恢复] %s存在多笔未完成委托，交易已阻止" % code)
            return False
        if amount > 0:
            score = g.last_scores.get(code, {})
            pos = _get_position(context, code)
            filled_price = _pos_cost(pos) if pos is not None else 0.0
            fill_value_complete = filled == 0 or _is_positive_finite(filled_price)
            pending_buys[code] = {
                "requested_qty": requested,
                "filled_qty": filled,
                "filled_value": filled * filled_price if fill_value_complete else 0.0,
                "fill_value_complete": fill_value_complete,
                "atr": g.entry_atr.get(code, score.get("atr")),
                "buy_date": g.buy_date.get(code, today),
                "order_id": order_id,
                "recovered_guard": True,
            }
        else:
            pending_sells[code] = {
                "requested_qty": requested,
                "filled_qty": filled,
                "reason": "recovered_open_order",
                "order_id": order_id,
                "recovered_guard": True,
            }
            sold_guards[code] = True
    g.__pending_orders = pending_buys
    g.__pending_sells = pending_sells
    g.sold_today.update(sold_guards)
    if g.__pending_orders or g.__pending_sells:
        log.warning("[委托恢复] 未完成买单=%d 未完成卖单=%d" % (
            len(g.__pending_orders), len(g.__pending_sells)))
    return True


def _audit_after_close_open_orders():
    """盘后只读核对未完成委托，不撤单、不重建盘中防重守卫。"""
    try:
        open_orders = get_open_orders()
    except Exception as exc:
        log.error("[盘后委托核对] get_open_orders调用失败: %s" % exc)
        return False
    if not isinstance(open_orders, list):
        log.error("[盘后委托核对] 返回类型异常: %s" % type(open_orders).__name__)
        return False
    if not open_orders:
        log.info("[盘后委托核对] 未发现未完成委托")
        return True
    for order_obj in open_orders:
        order_id = str(_order_field(order_obj, "id", "") or "")
        raw_code = (
            _order_field(order_obj, "symbol", "")
            or _order_field(order_obj, "stock_code", "")
        )
        code = normalize_code(raw_code) if raw_code else "未知"
        status = str(_order_field(order_obj, "status", "") or "")
        amount = _safe_float(_order_field(order_obj, "amount", 0))
        filled = _safe_float(_order_field(order_obj, "filled", 0))
        log.warning(
            "[盘后委托核对] 仍有未完成委托 代码=%s 委托编号=%s "
            "状态=%s 委托数量=%.0f 已成交=%.0f" % (
                code, order_id or "未知", status or "未知", amount, filled)
        )
    return False


def _clear_position_state(code):
    code = normalize_code(code)
    g.highest_since_buy.pop(code, None)
    g.entry_atr.pop(code, None)
    g.buy_date.pop(code, None)
    g.last_scores.pop(code, None)
    g.unverified_positions.discard(code)
    source_map = getattr(g, "__position_recovery_source", {})
    if isinstance(source_map, dict):
        source_map.pop(code, None)


def _snapshot_record(raw, code):
    if not isinstance(raw, dict):
        return None
    if "last_px" in raw or "trade_status" in raw:
        return raw
    for key in (code, code.split(".")[0]):
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    for key, value in raw.items():
        if normalize_code(key) == code and isinstance(value, dict):
            return value
    return None


def _positive_float_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def _snapshot_age_seconds(raw_timestamp, observed_at):
    if raw_timestamp in (None, "") or not isinstance(observed_at, datetime):
        return None
    digits = "".join(ch for ch in str(raw_timestamp) if ch.isdigit())
    if len(digits) < 14:
        return None
    try:
        snapshot_dt = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
        return (observed_at - snapshot_dt).total_seconds()
    except (TypeError, ValueError):
        return None


def _snapshot_session_date(raw_timestamp):
    if raw_timestamp in (None, ""):
        return None
    digits = "".join(ch for ch in str(raw_timestamp) if ch.isdigit())
    if len(digits) < 8:
        return None
    try:
        return datetime.strptime(digits[:8], "%Y%m%d").date()
    except (TypeError, ValueError):
        return None


def build_iopv_observation(
    code,
    snapshot,
    execution_price,
    observed_at=None,
):
    code = normalize_code(code)
    if code not in IOPV_OBSERVE_CODES:
        return None
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    snapshot_price = _positive_float_or_none(snapshot.get("last_px"))
    fallback_price = _positive_float_or_none(execution_price)
    market_price = snapshot_price if snapshot_price is not None else fallback_price
    iopv = _positive_float_or_none(snapshot.get("iopv"))
    premium = (
        market_price / iopv - 1.0
        if market_price is not None and iopv is not None
        else None
    )
    timestamp = snapshot.get("hsTimeStamp")
    return {
        "code": code,
        "valid": bool(market_price is not None and iopv is not None),
        "market_price": market_price,
        "iopv": iopv,
        "premium": premium,
        "snapshot_timestamp": timestamp,
        "snapshot_age_seconds": _snapshot_age_seconds(timestamp, observed_at),
    }


def log_iopv_buy_observation(context, code, execution_price):
    if not getattr(g, "__is_live", False):
        return
    try:
        normalized = normalize_code(code)
        snapshot = getattr(g, "__last_snapshot", {}).get(normalized, {})
        observation = build_iopv_observation(
            normalized,
            snapshot,
            execution_price,
            observed_at=get_context_datetime(context),
        )
        if observation is None:
            return
        premium_pct = (
            observation["premium"] * 100.0
            if observation["premium"] is not None
            else None
        )
        log.info(
            "[IOPV观察] 事件=买入 时间=%s 代码=%s 有效=%s 市价=%s "
            "IOPV=%s 溢价率百分比=%s 行情时间戳=%s 行情延迟秒数=%s"
            % (
                get_context_datetime(context),
                observation["code"],
                observation["valid"],
                observation["market_price"],
                observation["iopv"],
                premium_pct,
                observation["snapshot_timestamp"],
                observation["snapshot_age_seconds"],
            )
        )
    except Exception as exc:
        try:
            log.warning("[IOPV观察] 代码=%s 数据不可用: %s" % (code, exc))
        except Exception:
            pass


def get_current_price(code):
    code = normalize_code(code)
    if getattr(g, "__is_live", False):
        try:
            snapshot = _snapshot_record(get_snapshot(code), code)
            snapshot_date = _snapshot_session_date(
                snapshot.get("hsTimeStamp") if snapshot else None)
            current_date = datetime.now().date()
            if snapshot_date != current_date:
                log.warning(
                    "[行情快照] %s时间戳不是当前交易日，已拒绝使用: %s" % (
                        code,
                        snapshot.get("hsTimeStamp") if snapshot else None,
                    )
                )
                return None
            price = float(snapshot.get("last_px", 0)) if snapshot else 0.0
            if price > 0:
                g.__last_snapshot[code] = snapshot
                return price
        except Exception as exc:
            log.warning("[行情快照] %s价格不可用: %s" % (code, exc))
        return None

    data = getattr(g, "__data", None)
    if data is not None:
        try:
            price = float(data[code].price)
            if price > 0:
                return price
        except Exception:
            pass
    try:
        raw = get_history(1, "1d", "close", [code], fq="pre", include=True)
        values = _extract_history_field_series(raw, code, "close")
        if values is None or len(values) == 0:
            return None
        price = float(values.iloc[-1])
        return price if price > 0 else None
    except Exception:
        return None


def is_paused(code):
    code = normalize_code(code)
    if getattr(g, "__is_live", False):
        try:
            result = get_stock_status([code], "HALT")
            if isinstance(result, dict):
                for key, value in result.items():
                    if normalize_code(key) == code:
                        if isinstance(value, (bool, np.bool_)):
                            return bool(value)
                        log.warning("[交易状态] %s停牌值无法识别=%r" % (code, value))
                        return True
        except Exception as exc:
            log.warning("[交易状态] %s停牌查询失败: %s" % (code, exc))
        snapshot = getattr(g, "__last_snapshot", {}).get(code)
        if snapshot:
            status = str(snapshot.get("trade_status", "")).upper()
            if status in ("HALT", "SUSP", "STOPT", "DELISTED"):
                return True
            if status in ("TRADE", "OCALL", "BREAK", "ENDTR", "POSTR"):
                return False
        return True

    data = getattr(g, "__data", None)
    if data is not None:
        try:
            return int(data[code].is_open) == 0
        except Exception:
            pass
    try:
        result = get_stock_status([code], "HALT")
        if isinstance(result, dict):
            for key, value in result.items():
                if normalize_code(key) == code:
                    return bool(value)
    except Exception:
        pass
    return False


def _find_paused_pool_codes(pool, pause_check):
    return set(code for code in pool if pause_check(code))


def get_sell_limit_price(code, current):
    code = normalize_code(code)
    price = round(float(current), 3)
    snapshot = getattr(g, "__last_snapshot", {}).get(code, {})
    if getattr(g, "__is_live", False):
        try:
            down_price = float(snapshot.get("down_px", 0))
            if down_price > 0:
                return round(down_price, 3)
        except (TypeError, ValueError):
            pass
    return price


def execute_sell(code, context, reason):
    code = normalize_code(code)
    if g.sold_today.get(code) or code in getattr(g, "__pending_sells", {}):
        return False
    pos = _get_position(context, code)
    amount = int(_pos_amount(pos)) if pos is not None else 0
    if amount <= 0:
        return False
    price = get_current_price(code)
    if price is None or price <= 0:
        log.warning("[卖出] %s价格不可用，已跳过委托" % code)
        return False
    limit_price = get_sell_limit_price(code, price)
    log.info("[卖出] %s 原因=%s 数量=%s 限价=%.3f" % (
        code, _format_reason_for_log(reason), amount, limit_price))
    try:
        order_id = order_target(code, 0, limit_price=limit_price)
    except Exception as exc:
        log.error("[卖出] %s委托提交失败: %s" % (code, exc))
        return False
    if order_id is None:
        log.error("[卖出] %s委托提交后未返回委托编号" % code)
        return False
    log.info("[卖出委托] %s 委托编号=%s" % (code, order_id))
    if getattr(g, "__is_live", False):
        g.sell_retry_reasons.pop(code, None)
        g.sold_today[code] = True
        g.__pending_sells[code] = {
            "requested_qty": amount,
            "filled_qty": 0.0,
            "reason": reason,
            "order_id": str(order_id),
        }
    else:
        _clear_position_state(code)
    return True


def check_atr_stops(context, codes=None):
    triggered = []
    today = _as_date(get_context_datetime(context))
    allowed = None if codes is None else set(normalize_code(code) for code in codes)
    for code in current_hold_codes(context):
        if allowed is not None and code not in allowed:
            continue
        if code in g.unverified_positions:
            continue
        if g.buy_date.get(code) == today:
            continue
        if g.sold_today.get(code) or code in getattr(g, "__pending_sells", {}):
            continue
        if is_paused(code):
            continue
        pos = _get_position(context, code)
        price = get_current_price(code)
        if price is None or price <= 0:
            continue
        if code not in g.highest_since_buy or code not in g.entry_atr:
            continue
        stop_price = calc_stop_price(
            g.highest_since_buy[code], g.entry_atr[code], _pos_cost(pos), g.params)
        if price <= stop_price:
            triggered.append((code, stop_price, price))
    return triggered


def _get_signal_hold_days(today, params=None):
    p = params or g.params
    try:
        return get_trade_days(
            end_date=_api_date_text(today),
            count=max(2, int(p.get("min_signal_hold_days", 1)) + 1),
        )
    except Exception as exc:
        log.warning("[最短持有] 交易日查询失败，信号卖出已阻止: %s" % exc)
        return None


def _evaluate_signal_sell(context, code, score, today, signal_hold_days):
    code = normalize_code(code)
    p = g.params
    if g.sold_today.get(code) or code in getattr(g, "__pending_sells", {}):
        return False
    if code in g.unverified_positions:
        log.error("[持仓] %s风险状态未验证，自动信号卖出已阻止" % code)
        return False
    if is_paused(code):
        log.info("[持仓] %s处于停牌，已跳过信号卖出" % code)
        return False
    if not can_sell_with_verified_calendar(
        g.buy_date.get(code),
        today,
        min_hold_days=p.get("min_signal_hold_days", 1),
        trade_days=signal_hold_days,
    ):
        log.info("[持仓] %s尚未满足最短持有期，已跳过信号卖出" % code)
        return False
    if score["buy_score"] >= p["strong_buy_threshold"] and score["sell_score"] < p["sell_threshold"]:
        log.info("[持仓] %s买入评分较强 买入评分=%.0f 卖出评分=%.0f" % (
            code, score["buy_score"], score["sell_score"]))
        return False
    if should_force_sell(score, False, p):
        return execute_sell(code, context, "sell_score %.0f" % score["sell_score"])
    if score["sell_score"] >= p["risk_tighten_threshold"]:
        log.info("[风险收紧] %s卖出评分=%.0f" % (code, score["sell_score"]))
    return False


def execute_buy_candidates(context, all_scores, today):
    """只依据券商已确认的持仓和可用资金提交买单。"""
    if getattr(g, "__order_state_unknown", False):
        log.error("[买入] 券商委托状态无法确认，延后买入已阻止")
        return 0
    if getattr(g, "__pending_sells", {}):
        log.info("[买入延后] 正在等待%d笔卖出委托完成" % len(g.__pending_sells))
        return 0

    held = set(current_hold_codes(context))
    unverified_held = held & set(getattr(g, "unverified_positions", set()))
    if unverified_held:
        log.error("[买入] 存在未验证持仓=%s，全部新买入已阻止" % (
            ",".join(sorted(unverified_held))))
        return 0
    pending_buys = set(getattr(g, "__pending_orders", {}).keys())
    slots = g.params["max_hold"] - len(held | pending_buys)
    if slots <= 0:
        return 0
    candidates = filter_buy_candidates(all_scores, held | pending_buys, g.params)
    if not candidates:
        log.info("[%s] 没有达到阈值的买入候选" % STRATEGY_VERSION)
        return 0

    available = _available_cash(context)
    bought = 0
    for score in candidates:
        if bought >= slots:
            break
        code = score["code"]
        if code in getattr(g, "__pending_orders", {}):
            continue
        if is_paused(code):
            continue
        price = get_current_price(code)
        if price is None or price <= 0:
            log.warning("[买入跳过] %s当前价格不可用" % code)
            continue
        target_value = min(calc_buy_target_value(_total_value(context), score, g.params), available)
        shares = int(target_value / price / 100) * 100
        if shares < 100:
            log.info("[买入跳过] %s可用资金不足 当前可用=%.0f" % (code, available))
            continue
        log_iopv_buy_observation(context, code, price)
        log.info(
            "[买入] %s 买入评分=%.0f 反转评分=%.0f 位置评分=%.0f "
            "趋势评分=%.0f 量能评分=%.0f 目标金额=%.0f 股数=%d" % (
                code, score["buy_score"], score["reversal_score"],
                score["location_score"], score["trend_score"],
                score["volume_score"], target_value, shares))
        try:
            order_id = order(code, shares, limit_price=round(price, 3))
        except Exception as exc:
            log.error("[买入] %s委托提交失败: %s" % (code, exc))
            continue
        if order_id is None:
            log.error("[买入] %s委托提交后未返回委托编号" % code)
            continue
        log.info("[买入委托] %s 委托编号=%s" % (code, order_id))
        if getattr(g, "__is_live", False):
            g.__pending_orders[code] = {
                "requested_qty": shares,
                "filled_qty": 0.0,
                "filled_value": 0.0,
                "fill_value_complete": True,
                "atr": score["atr"],
                "buy_date": today,
                "order_id": str(order_id),
            }
        else:
            g.buy_date[code] = today
            g.highest_since_buy[code] = price
            g.entry_atr[code] = score["atr"]
        available -= shares * price
        bought += 1
    return bought


# 六、09:35 主流程与 10:35 补偿
# 09:35 对可交易标的执行完整流程；早盘停牌标的在 10:35 复牌后补做同一套风险与信号判断。
# 卖单拒绝或部分撤单只获得一次 10:35 有界重评机会，条件消失时不再机械追单。

def do_trading(context):
    p = g.params
    if getattr(g, "__is_live", False) and getattr(g, "__order_state_unknown", False):
        log.error("[交易] 券商委托状态无法确认，本次不提交委托")
        return
    today = _as_date(get_context_datetime(context))
    prev_date = get_prev_trade_date(context)
    if today is None or prev_date is None:
        log.error("[交易] 日期边界不可用，本次不提交委托")
        return
    g.execution_date = today
    g.deferred_signal_date = prev_date
    g.deferred_scores = []
    is_rebalance = today.weekday() in p["rebalance_weekdays"]
    g.paused_pool_codes = _find_paused_pool_codes(g.etf_pool, is_paused)

    log.info("[%s] 执行日期=%s 信号日期=%s 是否调仓=%s" % (
        STRATEGY_VERSION, today, prev_date, "是" if is_rebalance else "否"))

    stop_hits = check_atr_stops(context)
    for code, stop_price, price in stop_hits:
        execute_sell(code, context, "atr_stop %.3f<=%.3f" % (price, stop_price))

    if not is_rebalance:
        if not stop_hits:
            log.info("[%s] 非调仓日，止损检查完成且未触发" % STRATEGY_VERSION)
        return

    all_scores = []
    skip_reasons = {}
    for code in g.etf_pool:
        if code in g.paused_pool_codes:
            skip_reasons[code] = "paused"
            continue
        score, reason = calc_cross_signal_score(code, prev_date, return_reason=True)
        if score is not None:
            all_scores.append(score)
        else:
            skip_reasons[code] = reason or "unknown"

    if not all_scores:
        reason_counts = {}
        for reason in skip_reasons.values():
            label = _format_reason_for_log(reason)
            reason_counts[label] = reason_counts.get(label, 0) + 1
        summary = " | ".join("%s=%d" % (k, v) for k, v in sorted(reason_counts.items()))
        samples = " | ".join(
            "%s:%s" % (c, _format_reason_for_log(r))
            for c, r in sorted(skip_reasons.items())[:6]
        )
        log.info("[%s] 没有有效评分" % STRATEGY_VERSION)
        log.info("[评分跳过汇总] %s" % summary)
        log.info("[评分跳过样例] %s" % samples)
        return

    if skip_reasons:
        reason_counts = {}
        for reason in skip_reasons.values():
            label = _format_reason_for_log(reason)
            reason_counts[label] = reason_counts.get(label, 0) + 1
        summary = " | ".join("%s=%d" % (k, v) for k, v in sorted(reason_counts.items()))
        log.info("[评分跳过汇总] %s" % summary)

    all_scores = sort_candidates(all_scores)
    score_map = {s["code"]: s for s in all_scores}
    g.last_scores = score_map

    log.info("[候选排名]")
    for item in all_scores[:5]:
        log.info(
            "  %s 买入评分=%.0f 反转评分=%.0f 位置评分=%.0f 趋势评分=%.0f "
            "量能评分=%.0f 卖出评分=%.0f 收盘价=%.3f %s" % (
                item["code"], item["buy_score"], item["reversal_score"],
                item["location_score"], item["trend_score"], item["volume_score"],
                item["sell_score"], item["close"], _format_indicator_values_for_log(item)))

    cross_summary = summarize_cross_signal_candidates(all_scores)
    if cross_summary["count"] == 0:
        log.info("[上穿信号] 全部标的均未出现")
    else:
        log.info("[上穿信号] 数量=%d" % cross_summary["count"])
        for item in cross_summary["items"]:
            log.info(
                "  %s 反转评分=%.0f 买入评分=%.0f 卖出评分=%.0f %s %s" % (
                    item["code"], item["reversal_score"], item["buy_score"], item["sell_score"],
                    _format_cross_flags_for_log(item),
                    _format_indicator_values_for_log(item)))

    loose_summary = summarize_loose_reversal_candidates(all_scores)
    if loose_summary["count"] == 0:
        log.info("[宽松反转] 全部标的均未出现")
    else:
        log.info("[宽松反转] 数量=%d" % loose_summary["count"])
        for item in loose_summary["items"]:
            log.info(
                "  %s 宽松反转数=%d 买入评分=%.0f 反转评分=%.0f "
                "RSI转强=%s RSI变化=%.2f MACD转强=%s DIF变化=%.4f "
                "KDJ转强=%s K值变化=%.2f J值变化=%.2f %s" % (
                    item["code"], item["loose_reversal_count"],
                    item["buy_score"], item["reversal_score"],
                    item["rsi_turn_up"], item["rsi6_delta"],
                    item["macd_turn_up"], item["dif_delta"],
                    item["kdj_turn_up"], item["k_delta"], item["j_delta"],
                    _format_indicator_values_for_log(item)))

    held = current_hold_codes(context)
    signal_hold_days = _get_signal_hold_days(today, p)
    for code in list(held):
        if code not in score_map:
            continue
        _evaluate_signal_sell(
            context, code, score_map[code], today, signal_hold_days)

    g.deferred_scores = list(all_scores)
    execute_buy_candidates(context, all_scores, today)


def after_close(context):
    total = _total_value(context)
    cash = _available_cash(context)
    holds = current_hold_codes(context)
    log.info("=" * 60)
    log.info("[%s 收盘] 总资产=%.2f 可用资金=%.2f 持仓数=%d/%d" % (
        STRATEGY_VERSION, total, cash, len(holds), g.params["max_hold"]))
    for code in holds:
        if code in g.unverified_positions:
            log.error("  %s风险状态未验证，收盘价和ATR状态未更新" % code)
            continue
        price = get_current_price(code)
        pos = _get_position(context, code)
        if price is None or price <= 0:
            continue
        prev_high = g.highest_since_buy.get(code, price)
        g.highest_since_buy[code] = max(prev_high, price)
        if code not in g.entry_atr:
            score = g.last_scores.get(code)
            if score is not None and score.get("atr") and not pd.isna(score.get("atr")):
                g.entry_atr[code] = score["atr"]
        atr_val = g.entry_atr.get(code, np.nan)
        cost = _pos_cost(pos)
        stop_price = calc_stop_price(g.highest_since_buy[code], atr_val, cost, g.params) \
            if not pd.isna(atr_val) else np.nan
        pnl = (price - cost) / cost if cost > 0 else 0
        score = g.last_scores.get(code, {})
        log.info("  %s 成本价=%.3f 当前价=%.3f 持仓最高收盘价=%.3f "
                 "收益率=%.1f%% 买入评分=%.0f 卖出评分=%.0f 止损价=%.3f" % (
            code, cost, price, g.highest_since_buy[code], pnl * 100,
            score.get("buy_score", 0), score.get("sell_score", 0), stop_price))
    log.info("=" * 60)


def halt_recover(context):
    if getattr(g, "__order_state_unknown", False):
        log.error("[复牌补偿] 券商委托状态无法确认，本次不提交委托")
        return
    today = _as_date(get_context_datetime(context))
    prev_date = get_prev_trade_date(context) if today is not None else None
    if (
        today is None or prev_date is None or
        g.execution_date != today or
        g.deferred_signal_date != prev_date
    ):
        log.error("[复牌补偿] 延后评分日期不匹配，本次不提交委托")
        return
    if not _reconcile_open_orders(context):
        return
    _recover_live_state_with_available_sources(context, allow_deliver=False)
    previous = set(getattr(g, "paused_pool_codes", set()))
    retry_reasons = dict(getattr(g, "sell_retry_reasons", {}))
    g.sell_retry_reasons = {}
    held_now = set(current_hold_codes(context))
    retry_codes = set(retry_reasons) & held_now
    atr_retry_codes = set(
        code for code in retry_codes
        if str(retry_reasons.get(code, "")).startswith("atr_stop ")
    )
    signal_retry_codes = set(
        code for code in retry_codes
        if str(retry_reasons.get(code, "")).startswith("sell_score ")
    )
    still_paused = set(code for code in previous if is_paused(code))
    recovered = sorted(previous - still_paused)
    g.paused_pool_codes = still_paused
    atr_stopped = set()
    atr_review_codes = set(recovered) | atr_retry_codes
    for code, stop_price, price in check_atr_stops(context, atr_review_codes):
        if execute_sell(code, context, "atr_stop %.3f<=%.3f" % (price, stop_price)):
            atr_stopped.add(code)
    scores = list(getattr(g, "deferred_scores", []))
    score_review_codes = set(recovered) | signal_retry_codes
    if score_review_codes:
        by_code = {item["code"]: item for item in scores}
        for code in sorted(score_review_codes):
            if code in by_code:
                continue
            score, reason = calc_cross_signal_score(code, prev_date, return_reason=True)
            if score is not None:
                by_code[code] = score
                g.last_scores[code] = score
            else:
                log.warning("[复牌补偿] %s评分不可用: %s" % (
                    code, _format_reason_for_log(reason)))
        scores = sort_candidates(list(by_code.values()))
        g.deferred_scores = scores
        signal_review_holds = held_now & score_review_codes
        signal_hold_days = _get_signal_hold_days(today, g.params) if signal_review_holds else None
        for code in sorted(signal_review_holds - atr_stopped):
            score = by_code.get(code)
            if score is not None:
                _evaluate_signal_sell(
                    context, code, score, today, signal_hold_days)
        for code in sorted(retry_codes - atr_stopped):
            if code not in getattr(g, "__pending_sells", {}):
                log.info("[卖出重试] %s风险条件已解除，本次不再卖出" % code)
        if retry_codes:
            log.info("[卖出重试] 已重新评估=%s" % ",".join(sorted(retry_codes)))
    elif retry_codes:
        for code in sorted(retry_codes - atr_stopped):
            if code not in getattr(g, "__pending_sells", {}):
                log.info("[卖出重试] %s风险条件已解除，本次不再卖出" % code)
        log.info("[卖出重试] 已重新评估=%s" % ",".join(sorted(retry_codes)))
    if recovered:
        log.info("[复牌补偿] 已复牌=%s，已执行延后卖出与买入评估" % ",".join(recovered))
    elif previous:
        log.info("[复牌补偿] 受跟踪的ETF均未复牌")
    if scores:
        execute_buy_candidates(context, scores, today)


# 七、实盘状态恢复与成交回报
# 启动时先用当日成交、历史交割单和券商持仓重建；仅对无法覆盖的老持仓使用状态台账回退。
# 未验证持仓禁止自动卖出和新增买入，避免凭空构造买入日、ATR 或最高收盘价。

def _has_incomplete_position_state(context):
    for code in current_hold_codes(context):
        if (
            _as_date(g.buy_date.get(code)) is None or
            not _is_positive_finite(g.entry_atr.get(code)) or
            not _is_positive_finite(g.highest_since_buy.get(code))
        ):
            return True
    return False


def _recover_live_state_with_available_sources(context, allow_deliver):
    if not _has_incomplete_position_state(context):
        recover_live_state(context)
        return
    prev_date = get_prev_trade_date(context)
    current_records = _fetch_current_strategy_trades()
    deliver_records = (
        _fetch_deliver_records(prev_date)
        if allow_deliver and prev_date is not None else None
    )
    recover_live_state(
        context,
        deliver_records=deliver_records,
        current_trade_records=current_records,
        prev_date=prev_date,
    )


def _fetch_deliver_records(prev_date):
    """只在官方允许的生命周期回调中读取券商历史交割单。"""
    end_date = _as_date(prev_date)
    if end_date is None:
        return []
    getter = globals().get("get_deliver")
    if getter is None:
        log.error("[状态恢复] get_deliver接口不可用")
        return []
    end_text = end_date.strftime("%Y%m%d")
    try:
        records = getter(DELIVER_RECOVERY_START_DATE, end_text)
    except Exception as exc:
        log.error("[状态恢复] get_deliver调用失败: %s" % exc)
        return []
    if not isinstance(records, (list, tuple)):
        log.error("[状态恢复] get_deliver返回值无效")
        return []
    log.info("[状态恢复] 交割记录数=%d 查询范围=%s~%s" % (
        len(records), DELIVER_RECOVERY_START_DATE, end_text))
    tagged = []
    for record in records:
        if isinstance(record, dict):
            record = dict(record)
            record["_recovery_source"] = "get-deliver"
        tagged.append(record)
    return tagged


def _fetch_current_strategy_trades():
    """把 PTrade 当前策略的当日成交统一转换成交割单式记录。"""
    getter = globals().get("get_trades")
    if getter is None:
        log.error("[状态恢复] get_trades接口不可用")
        return []
    try:
        payload = getter()
    except Exception as exc:
        log.error("[状态恢复] get_trades调用失败: %s" % exc)
        return []
    if not isinstance(payload, dict):
        log.error("[状态恢复] get_trades返回值无效")
        return []

    records = []
    for order_id, fills in payload.items():
        if not isinstance(fills, (list, tuple)):
            continue
        for fill in fills:
            if not isinstance(fill, (list, tuple)) or len(fill) < 8:
                continue
            side = str(fill[3] or "").strip()
            if "\u4e70" in side:
                entrust_bs = "1"
            elif "\u5356" in side:
                entrust_bs = "2"
            else:
                continue
            try:
                trade_time = pd.Timestamp(fill[7])
            except Exception:
                continue
            records.append({
                "stock_code": fill[2],
                "entrust_bs": entrust_bs,
                "business_amount": fill[4],
                "business_price": fill[5],
                "init_date": trade_time.strftime("%Y%m%d"),
                "business_time": trade_time.strftime("%H%M%S"),
                "order_id": str(order_id),
                "_recovery_source": "get-trades",
            })
    log.info("[状态恢复] 当前策略成交记录数=%d" % len(records))
    return records


def _delivery_trade_date(record):
    for field in ("init_date", "entrust_date", "date_back", "business_date"):
        raw = record.get(field)
        digits = "".join(ch for ch in str(raw or "") if ch.isdigit())
        if len(digits) < 8:
            continue
        try:
            return datetime.strptime(digits[:8], "%Y%m%d").date()
        except Exception:
            continue
    return None


def _delivery_direction(record):
    side = str(record.get("entrust_bs", "") or "").strip()
    business_name = str(record.get("business_name", "") or "")
    if side == "1" or "\u4e70\u5165" in business_name:
        return 1
    if side == "2" or "\u5356\u51fa" in business_name:
        return -1
    return 0


def _delivery_quantity(record):
    for field in ("business_amount", "occur_amount"):
        value = _safe_float(record.get(field), np.nan)
        if np.isfinite(value) and abs(value) > 0:
            return abs(value)
    return 0.0


def _delivery_sort_key(record):
    trade_date = _delivery_trade_date(record)
    if trade_date is None:
        return (datetime.max.date(), 0, 0)
    trade_time = int(abs(_safe_float(
        record.get("business_time", record.get("report_time", 0)), 0
    )))
    serial = int(abs(_safe_float(
        record.get("serial_no", record.get("business_no", 0)), 0
    )))
    return (trade_date, trade_time, serial)


def _diagnostic_number(value):
    number = _safe_float(value, np.nan)
    if not np.isfinite(number):
        return "非数值"
    rounded = round(number)
    if abs(number - rounded) <= 1e-9:
        return str(int(rounded))
    return ("%.6f" % number).rstrip("0").rstrip(".")


def _limited_diagnostic_values(values, limit=30):
    ordered = sorted(set(str(value) for value in values if value not in (None, "")))
    if len(ordered) <= limit:
        return ",".join(ordered) if ordered else "无"
    return "%s,+%d" % (",".join(ordered[:limit]), len(ordered) - limit)


def _diagnose_delivery_replay(records, code, broker_amount):
    """生成不含账户标识和原始交割单的安全诊断摘要。"""
    target = normalize_code(code)
    all_records = list(records or [])
    available_codes = []
    field_keys = []
    code_rows = []
    valid_rows = []
    side_values = []
    for record in all_records:
        if not isinstance(record, dict):
            continue
        if not field_keys:
            field_keys = sorted(str(key) for key in record.keys())
        raw_code = record.get("stock_code")
        normalized = normalize_code(raw_code)
        if normalized:
            available_codes.append(normalized)
        if normalized != target:
            continue
        code_rows.append(record)
        side_values.append("%s/%s" % (
            str(record.get("entrust_bs", "") or "").strip() or "空",
            str(record.get("business_name", "") or "").strip() or "空",
        ))
        direction = _delivery_direction(record)
        quantity = _delivery_quantity(record)
        trade_date = _delivery_trade_date(record)
        if direction != 0 and quantity > 0 and trade_date is not None:
            valid_rows.append(record)

    amount = 0.0
    min_running = 0.0
    buys = 0
    sells = 0
    dates = []
    samples = []
    ordered_valid = sorted(valid_rows, key=_delivery_sort_key)
    for record in ordered_valid:
        direction = _delivery_direction(record)
        quantity = _delivery_quantity(record)
        trade_date = _delivery_trade_date(record)
        amount += direction * quantity
        min_running = min(min_running, amount)
        buys += int(direction > 0)
        sells += int(direction < 0)
        dates.append(trade_date)
    sample_rows = ordered_valid
    if len(sample_rows) > 6:
        sample_rows = sample_rows[:3] + sample_rows[-3:]
    for record in sample_rows:
        direction = _delivery_direction(record)
        trade_date = _delivery_trade_date(record)
        quantity = _delivery_quantity(record)
        price = _safe_float(record.get("business_price"), np.nan)
        samples.append("%s:%s:%s@%s" % (
            trade_date.isoformat() if trade_date is not None else "未知日期",
            "买" if direction > 0 else "卖" if direction < 0 else "未知",
            _diagnostic_number(quantity),
            _diagnostic_number(price),
        ))

    date_range = "无"
    if dates:
        date_range = "%s~%s" % (min(dates).isoformat(), max(dates).isoformat())
    return (
        "券商持仓=%s 总记录数=%d 标的记录数=%d 有效记录数=%d "
        "买入笔数=%d 卖出笔数=%d 净数量=%s 最低累计数量=%s 日期范围=%s "
        "可用代码=%s 方向值=%s 字段名=%s 样例=%s" % (
            _diagnostic_number(broker_amount),
            len(all_records),
            len(code_rows),
            len(valid_rows),
            buys,
            sells,
            _diagnostic_number(amount),
            _diagnostic_number(min_running),
            date_range,
            _limited_diagnostic_values(available_codes),
            _limited_diagnostic_values(side_values, limit=12),
            _limited_diagnostic_values(field_keys, limit=40),
            ";".join(samples) if samples else "无",
        )
    )


def _log_recovery_failure(code, stage, reason, details=None):
    stage_text = {
        "pool": "标的池",
        "broker-position": "券商持仓",
        "delivery-replay": "交割单重放",
        "historical-calendar": "历史交易日历",
        "entry-atr": "入场ATR",
        "delivery-entry-price": "交割单入场价",
        "current-calendar": "当前交易日历",
        "trailing-high": "持仓最高价",
        "same-day-entry": "当日买入",
    }.get(stage, str(stage))
    reason_text = {
        "outside-frozen-pool": "不在锁定标的池内",
        "invalid-amount-or-cost": "数量或成本无效",
        "unreconciled": "无法与券商持仓核对一致",
        "previous-trade-date-unresolved": "无法确定前一交易日",
        "score-unavailable": "评分不可用",
        "atr-invalid": "ATR无效",
        "weighted-fill-price-unavailable": "成交加权价不可用",
        "current-prev-date-unavailable": "当前前一交易日不可用",
        "close-history-unavailable": "收盘价历史不可用",
        "no-positive-closes": "没有有效正收盘价",
        "signal-date-mismatch": "信号日期不匹配",
    }.get(reason, str(reason))
    message = "[恢复诊断] 代码=%s 阶段=%s 原因=%s" % (
        normalize_code(code), stage_text, reason_text)
    if details:
        message += " " + str(details)
    log.error(message)
    return False


def _reconstruct_open_position(records, code, broker_amount):
    """重放当前持仓区间，并要求重放数量与券商持仓严格一致。"""
    target = normalize_code(code)
    expected = _safe_float(broker_amount, np.nan)
    if not _is_positive_finite(expected):
        return None
    matched = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        if normalize_code(record.get("stock_code")) != target:
            continue
        direction = _delivery_direction(record)
        quantity = _delivery_quantity(record)
        trade_date = _delivery_trade_date(record)
        if direction == 0 or quantity <= 0 or trade_date is None:
            continue
        matched.append(record)
    if not matched:
        return None

    amount = 0.0
    buy_date = None
    entry_quantity = 0.0
    entry_value = 0.0
    entry_source = None
    tolerance = max(1e-6, expected * 1e-8)
    for record in sorted(matched, key=_delivery_sort_key):
        direction = _delivery_direction(record)
        quantity = _delivery_quantity(record)
        if direction > 0:
            if amount <= tolerance:
                buy_date = _delivery_trade_date(record)
                entry_quantity = 0.0
                entry_value = 0.0
                entry_source = record.get("_recovery_source")
            price = _safe_float(record.get("business_price"), np.nan)
            if _is_positive_finite(price):
                entry_quantity += quantity
                entry_value += quantity * price
            amount += quantity
        else:
            amount -= quantity
            if amount < -tolerance:
                return None
            if abs(amount) <= tolerance:
                amount = 0.0
                buy_date = None
                entry_quantity = 0.0
                entry_value = 0.0
                entry_source = None

    if buy_date is None or abs(amount - expected) > tolerance:
        return None
    entry_price = entry_value / entry_quantity if entry_quantity > 0 else None
    return {
        "buy_date": buy_date,
        "amount": amount,
        "entry_price": entry_price,
        "recovery_source": entry_source,
    }


def _previous_trade_date_before(value):
    trade_date = _as_date(value)
    if trade_date is None:
        return None
    try:
        result = get_trade_days(end_date=_api_date_text(trade_date), count=2)
        previous = _previous_day_from_result(result, trade_date)
        if previous is not None:
            return previous
        log.warning(
            "[恢复交易日历] 查询日期=%s 接口=get_trade_days 返回值不可用 %s" % (
                trade_date.isoformat(), _calendar_payload_summary(result)))
    except Exception as exc:
        log.warning("[状态恢复] get_trade_days调用失败: %s" % exc)
    try:
        result = get_all_trades_days(date=trade_date.strftime("%Y%m%d"))
        previous = _previous_day_from_result(result, trade_date)
        if previous is not None:
            return previous
        log.warning(
            "[恢复交易日历] 查询日期=%s 接口=get_all_trades_days 返回值不可用 %s" % (
                trade_date.isoformat(), _calendar_payload_summary(result)))
    except Exception as exc:
        log.warning("[状态恢复] get_all_trades_days调用失败: %s" % exc)
    return None


def _probe_previous_trade_date_by_date(value):
    """仅诊断历史交易日接口，不把其返回值直接用于恢复决策。"""
    trade_date = _as_date(value)
    getter = globals().get("get_trading_day_by_date")
    if trade_date is None or getter is None:
        return None
    try:
        raw = getter(trade_date.strftime("%Y%m%d"), -1)
    except Exception as exc:
        log.warning(
            "[恢复交易日历探针] 查询日期=%s 接口=get_trading_day_by_date "
            "调用失败=%s 不参与交易判断=是" % (trade_date.isoformat(), exc))
        return None
    candidate = _as_date(raw)
    valid = candidate is not None and candidate < trade_date
    log.info(
        "[恢复交易日历探针] 查询日期=%s 接口=get_trading_day_by_date "
        "候选日期=%s 有效=%s 不参与交易判断=是 返回值=%s" % (
            trade_date.isoformat(),
            candidate.isoformat() if candidate is not None else "无",
            "是" if valid else "否",
            _calendar_payload_summary(raw),
        )
    )
    return candidate if valid else None


def _get_recovery_close_data(code, start_date, end_date):
    """仅为灾难恢复读取可比的前复权收盘价。"""
    start = _as_date(start_date)
    end = _as_date(end_date)
    if start is None or end is None or start > end:
        return None
    try:
        frame = get_price(
            code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            frequency="1d",
            fields=["close", "volume"],
            fq="pre",
        )
        frame = pd.DataFrame(frame).copy()
    except Exception as exc:
        log.error("[状态恢复] %s收盘价历史不可用: %s" % (code, exc))
        return None
    if "code" in frame.columns:
        frame = frame[frame["code"].map(normalize_code) == normalize_code(code)]
    if "close" not in frame.columns or "volume" not in frame.columns:
        return None
    frame = frame[frame["volume"] > 0]
    try:
        index_dates = pd.to_datetime(frame.index).date
        frame = frame[(index_dates >= start) & (index_dates <= end)]
    except Exception:
        return None
    return frame


def _recover_position_from_broker(code, pos, records, prev_date):
    if code not in set(g.etf_pool):
        return _log_recovery_failure(code, "pool", "outside-frozen-pool")
    amount = _pos_amount(pos)
    cost = _pos_cost(pos)
    if not _is_positive_finite(amount) or not _is_positive_finite(cost):
        return _log_recovery_failure(
            code,
            "broker-position",
            "invalid-amount-or-cost",
            "数量=%s 成本=%s" % (
                _diagnostic_number(amount), _diagnostic_number(cost)),
        )
    open_position = _reconstruct_open_position(records, code, amount)
    if open_position is None:
        return _log_recovery_failure(
            code,
            "delivery-replay",
            "unreconciled",
            _diagnose_delivery_replay(records, code, amount),
        )
    buy_date = open_position["buy_date"]
    signal_date = _previous_trade_date_before(buy_date)
    if signal_date is None:
        probe_date = _probe_previous_trade_date_by_date(buy_date)
        return _log_recovery_failure(
            code,
            "historical-calendar",
            "previous-trade-date-unresolved",
            "买入日期=%s 日期探针=%s 不参与交易判断=是" % (
                buy_date.isoformat(),
                probe_date.isoformat() if probe_date is not None else "无",
            ),
        )
    score = calc_cross_signal_score(code, signal_date)
    if not isinstance(score, dict):
        return _log_recovery_failure(
            code,
            "entry-atr",
            "score-unavailable",
            "信号日期=%s" % signal_date.isoformat(),
        )
    atr = score.get("atr")
    if not _is_positive_finite(atr):
        return _log_recovery_failure(
            code,
            "entry-atr",
            "atr-invalid",
            "信号日期=%s ATR=%s" % (
                signal_date.isoformat(), _diagnostic_number(atr)),
        )
    entry_price = open_position.get("entry_price")
    if not _is_positive_finite(entry_price):
        return _log_recovery_failure(
            code,
            "delivery-entry-price",
            "weighted-fill-price-unavailable",
            "买入日期=%s" % buy_date.isoformat(),
        )
    prev_date = _as_date(prev_date)
    if prev_date is None:
        return _log_recovery_failure(
            code, "current-calendar", "current-prev-date-unavailable")
    if buy_date <= prev_date:
        closes = _get_recovery_close_data(code, buy_date, prev_date)
        if closes is None or len(closes) == 0:
            return _log_recovery_failure(
                code,
                "trailing-high",
                "close-history-unavailable",
                "日期范围=%s~%s" % (buy_date.isoformat(), prev_date.isoformat()),
            )
        valid_closes = pd.to_numeric(closes["close"], errors="coerce")
        valid_closes = valid_closes[np.isfinite(valid_closes) & (valid_closes > 0)]
        if len(valid_closes) == 0:
            return _log_recovery_failure(
                code,
                "trailing-high",
                "no-positive-closes",
                "日期范围=%s~%s" % (buy_date.isoformat(), prev_date.isoformat()),
            )
        highest = max(float(entry_price), float(valid_closes.max()))
    else:
        if signal_date != prev_date:
            return _log_recovery_failure(
                code,
                "same-day-entry",
                "signal-date-mismatch",
                "买入日期=%s 信号日期=%s 当前前一交易日=%s" % (
                    buy_date.isoformat(),
                    signal_date.isoformat(),
                    prev_date.isoformat(),
                ),
            )
        highest = float(entry_price)

    g.buy_date[code] = buy_date
    g.entry_atr[code] = float(atr)
    g.highest_since_buy[code] = highest
    source_map = getattr(g, "__position_recovery_source", None)
    if not isinstance(source_map, dict):
        source_map = {}
        g.__position_recovery_source = source_map
    recovery_source = open_position.get("recovery_source") or "get-deliver"
    if recovery_source == "get-deliver":
        recovery_source = "account-takeover:get-deliver"
    source_map[code] = recovery_source
    log.warning(
        "[状态恢复] %s已依据券商事实接管: 买入日期=%s "
        "信号日期=%s ATR=%.6f 持仓最高收盘价=%.6f 成本=%.6f" % (
            code, buy_date, signal_date, atr, highest, cost)
    )
    return True


def _prune_closed_position_state(held):
    for field in ("highest_since_buy", "entry_atr", "buy_date"):
        mapping = getattr(g, field, {})
        for code in list(mapping.keys()):
            if normalize_code(code) not in held:
                mapping.pop(code, None)
    g.unverified_positions.intersection_update(held)
    source_map = getattr(g, "__position_recovery_source", {})
    if isinstance(source_map, dict):
        for code in list(source_map.keys()):
            if normalize_code(code) not in held:
                source_map.pop(code, None)


def recover_live_state(
        context, deliver_records=None, current_trade_records=None, prev_date=None):
    """先验证已有状态，再用可证明的券商事实接管账户持仓。"""
    held = set(current_hold_codes(context))
    _prune_closed_position_state(held)
    source_map = getattr(g, "__position_recovery_source", None)
    if not isinstance(source_map, dict):
        source_map = {}
        g.__position_recovery_source = source_map
    recovery_records = None
    if deliver_records is not None or current_trade_records is not None:
        recovery_records = []
        for records, source in (
            (deliver_records, "get-deliver"),
            (current_trade_records, "get-trades"),
        ):
            for record in records or []:
                if isinstance(record, dict):
                    record = dict(record)
                    record.setdefault("_recovery_source", source)
                recovery_records.append(record)
    for code in held:
        pos = _get_position(context, code)
        buy_date = _as_date(g.buy_date.get(code))
        complete = (
            pos is not None and
            _is_positive_finite(_pos_cost(pos)) and
            buy_date is not None and
            _is_positive_finite(g.highest_since_buy.get(code)) and
            _is_positive_finite(g.entry_atr.get(code))
        )
        if not complete and recovery_records is not None and prev_date is not None:
            complete = _recover_position_from_broker(
                code, pos, recovery_records, prev_date
            )
        if complete:
            g.unverified_positions.discard(code)
            if code not in source_map:
                source_map[code] = (
                    getattr(g, "__state_restore_source", None) or "ptrade-g")
        else:
            g.unverified_positions.add(code)
            source_map[code] = "unverified"
            log.error(
                "[状态恢复] %s历史买入日期、ATR、最高价或券商成本无法证实，"
                "自动卖出已阻止" % code
            )


def _recovery_summary_source(context):
    """按逐仓实际来源生成汇总标签，避免把交割单接管误报为持久状态。"""
    source_map = getattr(g, "__position_recovery_source", {})
    if not isinstance(source_map, dict):
        source_map = {}
    fallback = getattr(g, "__state_restore_source", None)
    sources = []
    held = sorted(current_hold_codes(context))
    for code in held:
        source = source_map.get(code)
        if not source and code in set(getattr(g, "unverified_positions", set())):
            source = "unverified"
        if not source:
            source = fallback
        if source:
            sources.append(str(source))
    if not held:
        return fallback or "no-position"
    unique_sources = set(sources)
    if len(unique_sources) == 1 and len(sources) == len(held):
        return sources[0]
    return "mixed"


def _log_live_recovery_summary(context):
    persisted_g_status = getattr(g, "__persisted_g_status", None)
    persisted_g_reason = getattr(g, "__persisted_g_reason", None)
    persisted_g_generation = getattr(g, "__persisted_g_generation", None)
    persisted_g_generation_text = (
        persisted_g_generation
        if isinstance(persisted_g_generation, int) and
        persisted_g_generation > 0
        else "不适用"
    )
    log.info("[PTrade框架g] 状态=%s 代次=%s 原因=%s" % (
        _format_persisted_g_status_for_log(persisted_g_status),
        persisted_g_generation_text,
        _format_persisted_g_reason_for_log(persisted_g_reason),
    ))

    continuity_source = getattr(g, "__state_restore_source", None)
    generation = getattr(g, "__state_restore_generation", None)
    generation_text = (
        generation
        if isinstance(generation, int) and generation > 0
        else "不适用"
    )
    log.info("[连续状态恢复] 来源=%s 代次=%s" % (
        _format_recovery_source_for_log(continuity_source), generation_text))

    source = _recovery_summary_source(context)
    log.info("[持仓风险恢复] 来源=%s" % (
        _format_recovery_source_for_log(source)))

    source_map = getattr(g, "__position_recovery_source", {})
    if not isinstance(source_map, dict):
        source_map = {}
    unverified = set(getattr(g, "unverified_positions", set()))
    for code in sorted(current_hold_codes(context)):
        pos = _get_position(context, code)
        amount = _pos_amount(pos) if pos is not None else 0.0
        cost = _pos_cost(pos) if pos is not None else np.nan
        buy_date = _as_date(getattr(g, "buy_date", {}).get(code))
        atr = _safe_float(getattr(g, "entry_atr", {}).get(code), np.nan)
        highest = _safe_float(
            getattr(g, "highest_since_buy", {}).get(code), np.nan)
        verified = (
            code not in unverified and
            pos is not None and
            _is_positive_finite(cost) and
            buy_date is not None and
            _is_positive_finite(atr) and
            _is_positive_finite(highest)
        )
        status = "已验证" if verified else "未验证"
        position_source = source_map.get(code)
        if not position_source:
            position_source = source if verified else "unverified"
        log.info(
            "[持仓风险恢复] 代码=%s 数量=%.0f 成本=%.6f 买入日期=%s "
            "ATR=%.6f 持仓最高收盘价=%.6f 状态=%s 来源=%s" % (
                code,
                amount,
                cost,
                buy_date.isoformat() if buy_date is not None else "无",
                atr,
                highest,
                status,
                _format_recovery_source_for_log(position_source),
            )
        )


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_positive_finite(value):
    number = _safe_float(value, np.nan)
    return bool(np.isfinite(number) and number > 0)


def _complete_buy(code, pending):
    quantity = pending.get("filled_qty", 0.0)
    if quantity <= 0:
        return
    _apply_buy_fill_state(code, pending)
    g.__pending_orders.pop(code, None)


def _apply_buy_fill_state(code, pending):
    quantity = pending.get("filled_qty", 0.0)
    value_complete = bool(pending.get("fill_value_complete", True))
    average = pending.get("filled_value", 0.0) / quantity if quantity > 0 else np.nan
    buy_date = _as_date(pending.get("buy_date"))
    atr = pending.get("atr")
    verified = (
        quantity > 0 and
        value_complete and
        _is_positive_finite(average) and
        buy_date is not None and
        _is_positive_finite(atr)
    )
    if buy_date is not None:
        g.buy_date[code] = buy_date
    if _is_positive_finite(atr):
        g.entry_atr[code] = atr
    if verified:
        g.highest_since_buy[code] = average
        g.unverified_positions.discard(code)
        source_map = getattr(g, "__position_recovery_source", None)
        if not isinstance(source_map, dict):
            source_map = {}
            g.__position_recovery_source = source_map
        source_map[code] = "get-trades"
    else:
        g.unverified_positions.add(code)
        source_map = getattr(g, "__position_recovery_source", None)
        if not isinstance(source_map, dict):
            source_map = {}
            g.__position_recovery_source = source_map
        source_map[code] = "unverified"
        log.error("[成交回报] %s入场成交基线未验证，自动卖出已阻止" % code)


def _pending_completion_qty(pending):
    return float(pending.get("terminal_qty", pending.get("requested_qty", 0.0)))


def _response_matches_pending(response, pending):
    response_id = str(response.get("order_id", "") or "")
    pending_id = str(pending.get("order_id", "") or "")
    return bool(response_id and pending_id and response_id == pending_id)


def _finish_terminal_sell(code, pending):
    if pending.get("filled_qty", 0.0) < _pending_completion_qty(pending):
        return False
    g.__pending_sells.pop(code, None)
    if pending.get("filled_qty", 0.0) >= pending.get("requested_qty", 0.0):
        _clear_position_state(code)
    else:
        g.sold_today.pop(code, None)
        g.sell_retry_reasons[code] = pending.get("reason", "sell_retry")
        log.warning("[卖出余量] %s部分成交后仍有持仓，风险状态已保留" % code)
    return True


def on_order_response(context, order_list):
    if not getattr(g, "__is_live", False):
        return
    orders = order_list if isinstance(order_list, list) else [order_list]
    for response in orders:
        if not isinstance(response, dict):
            log.warning("[委托回报格式异常] 已忽略类型=%s" % type(response).__name__)
            continue
        code = normalize_code(response.get("stock_code"))
        status = str(response.get("status", ""))
        if not code or status not in ("5", "6", "9"):
            continue
        filled = _safe_float(response.get("business_amount", 0))
        error = response.get("error_info", "")
        buy_pending = getattr(g, "__pending_orders", {}).get(code)
        sell_pending = getattr(g, "__pending_sells", {}).get(code)

        if buy_pending is not None:
            if not _response_matches_pending(response, buy_pending):
                log.warning("[委托回报] 已忽略无法匹配的买入委托 %s" % code)
                continue
            if filled > 0 and status in ("5", "6"):
                buy_pending["terminal_qty"] = filled
                if buy_pending.get("filled_qty", 0.0) >= filled:
                    _complete_buy(code, buy_pending)
                log.warning("[买入部分成交或撤单] %s 已成交=%.0f 原因=%s" % (
                    code, filled, error))
            else:
                g.__pending_orders.pop(code, None)
                log.warning("[买入拒绝或撤单] %s 状态=%s 原因=%s" % (
                    code, status, error))
            continue

        if sell_pending is not None:
            if not _response_matches_pending(response, sell_pending):
                log.warning("[委托回报] 已忽略无法匹配的卖出委托 %s" % code)
                continue
            if filled > 0 and status in ("5", "6"):
                sell_pending["terminal_qty"] = filled
                _finish_terminal_sell(code, sell_pending)
                log.warning("[卖出部分成交或撤单] %s 已成交=%.0f 原因=%s" % (
                    code, filled, error))
            else:
                g.__pending_sells.pop(code, None)
                g.sold_today.pop(code, None)
                g.sell_retry_reasons[code] = sell_pending.get("reason", "sell_retry")
                log.error("[卖出拒绝或撤单] %s 状态=%s 原因=%s" % (
                    code, status, error))
    _persist_live_state(context)


def on_trade_response(context, trade_list):
    if not getattr(g, "__is_live", False):
        return
    trades = trade_list if isinstance(trade_list, list) else [trade_list]
    for trade in trades:
        if not isinstance(trade, dict):
            log.warning("[成交回报格式异常] 已忽略类型=%s" % type(trade).__name__)
            continue
        if str(trade.get("real_type", "")) == "2":
            log.info("[成交回报] 已忽略撤单推送")
            continue
        code = normalize_code(trade.get("stock_code"))
        direction = str(trade.get("entrust_bs", ""))
        quantity = _safe_float(trade.get("business_amount", 0))
        price = _safe_float(trade.get("business_price", 0))
        if not code or quantity <= 0:
            continue

        if direction == "1":
            pending = getattr(g, "__pending_orders", {}).get(code)
            if pending is None:
                log.warning("[成交回报] 无法匹配的买入成交 %s 数量=%.0f" % (code, quantity))
                continue
            if not _response_matches_pending(trade, pending):
                log.warning("[成交回报] 已忽略旧的或无法匹配的买入委托 %s" % code)
                continue
            pending["filled_qty"] = pending.get("filled_qty", 0.0) + quantity
            if _is_positive_finite(price):
                pending["filled_value"] = pending.get("filled_value", 0.0) + quantity * price
            else:
                pending["fill_value_complete"] = False
            _apply_buy_fill_state(code, pending)
            if pending["filled_qty"] >= _pending_completion_qty(pending):
                g.__pending_orders.pop(code, None)
            log.info("[成交回报] 买入 %s 数量=%.0f 价格=%.3f 累计成交=%.0f" % (
                code, quantity, price, pending["filled_qty"]))

        elif direction == "2":
            pending = getattr(g, "__pending_sells", {}).get(code)
            if pending is None:
                log.warning("[成交回报] 无法匹配的卖出成交 %s 数量=%.0f" % (code, quantity))
                continue
            if not _response_matches_pending(trade, pending):
                log.warning("[成交回报] 已忽略旧的或无法匹配的卖出委托 %s" % code)
                continue
            pending["filled_qty"] = pending.get("filled_qty", 0.0) + quantity
            _finish_terminal_sell(code, pending)
            log.info("[成交回报] 卖出 %s 数量=%.0f 价格=%.3f 累计成交=%.0f" % (
                code, quantity, price, pending["filled_qty"]))
    _persist_live_state(context)
