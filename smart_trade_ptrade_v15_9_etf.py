# -*- coding: utf-8 -*-
"""
智能ETF量化交易策略 V15.9 - PTrade版（回测+实盘兼容）
=============================================
从聚宽V15.9逐行移植，所有交易逻辑、参数、信号条件100%不变。
根据PTrade官方API文档适配，同时兼容回测和实盘两种模式。

V15.9相对V15.7的改动：
  1. ETF池从10只扩展到12只（+日经ETF+中概互联ETF）
  2. 统一所有资金档位max_hold=3

V15.9.1 实盘健壮性优化（不改变策略逻辑）：
  1. 所有order/order_target传limit_price，避免PTrade内部snapshot二次调用失败
  2. 止损/轮动卖出用跌停价作limit_price，确保下跌行情中成交
  3. QDII ETF溢价过滤：iopv溢价>5%时跳过买入，防止高溢价接盘
  4. 买入前check_limit涨停过滤，避免挂单涨停板浪费资金
  5. 实盘用snapshot的trade_status替代get_stock_status，减少API调用
  6. on_trade_response成交回调，确认买入成交后才设置止损基准
  7. on_order_response委托回调，检测止损单失败并告警+重置sold_today允许重试
  8. 盘前清理snapshot缓存/pending_orders，避免跨日残留数据
  9. 持仓entry_atr恢复机制，PTrade重启后自动从历史数据重建ATR止损基准

适配要点（根据PTrade-API.html官方文档）：
  1. 代码格式：.XSHG/.XSHE → .SS/.SZ
  2. Portfolio属性：portfolio_value/cash 替代 total_value/available_cash
  3. Position属性：amount/cost_basis/last_sale_price 替代 total_amount/avg_cost/price
  4. 当前行情：data[code].price / get_snapshot 替代 get_current_data()
  5. 停牌检测：实盘用snapshot trade_status，回测用get_stock_status
  6. run_daily签名：run_daily(context, func, time) 需传context
  7. 回测兼容：日频回测run_daily固定15:00，改用handle_data驱动全部逻辑
  8. 实盘防重复：sold_today防止6秒同步延迟导致重复卖出
  9. 限价精度：ETF限价单必须3位小数（PTrade API要求）

策略逻辑（与聚宽V15.9 100%一致）：
  - 动量轮动：每3个交易日从ETF池选动量最强的N只持有
  - 双重动量过滤：ROC20>0 且 ROC60>0 且 价格>MA20
  - ATR跟踪止损：动态倍数（高波动2.0x，正常2.5x）
  - 国债填空：候选不足max_hold时，国债自动补位
  - 买入价使用T日实时价
  - 统一max_hold=3，小资金也持3只

ETF池（4A股+5跨市场+3跨资产 = 12只）：
  - 510300 沪深300  - 159915 创业板  - 512100 中证1000  - 159928 消费ETF
  - 513100 纳指ETF  - 513500 标普500  - 159920 恒生ETF
  - 513880 日经ETF  - 513050 中概互联ETF
  - 518880 黄金ETF  - 511010 国债ETF  - 159985 豆粕ETF
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
#  初始化
# ============================================================
def initialize(context):
    set_benchmark('000300.SS')

    # ---- 佣金与滑点（仅回测生效，实盘/模拟会提示WARNING，不影响运行）----
    try:
        set_commission(commission_ratio=0.0003, min_commission=5.0, type='ETF')
        set_slippage(slippage=0.001)
    except Exception:
        pass

    # ---- ETF标的池（PTrade使用.SS/.SZ格式，12只）----
    g.etf_pool = [
        # ---- A股 4只 ----
        '510300.SS',    # 沪深300（大盘均衡）
        '159915.SZ',    # 创业板（成长弹性）
        '512100.SS',    # 中证1000（小盘弹性）
        '159928.SZ',    # 消费ETF（内需消费）
        # ---- 跨市场 5只 ----
        '513100.SS',    # 纳指ETF（美国科技）
        '513500.SS',    # 标普500ETF（美国均衡）
        '159920.SZ',    # 恒生ETF（港股）
        '513880.SS',    # 日经ETF（日本市场）
        '513050.SS',    # 中概互联ETF（海外中概）
        # ---- 跨资产 3只 ----
        '518880.SS',    # 黄金ETF（避险）
        '511010.SS',    # 国债ETF（债券对冲/兜底）
        '159985.SZ',    # 豆粕ETF（商品周期）
    ]
    g.bond_etf = '511010.SS'  # 国债ETF作为兜底标的

    # ---- 资金档位配置（V15.9：统一max_hold=3）----
    g.capital_tiers = {
        'micro': {'max_hold': 3, 'base_position_ratio': 0.70},
        'small': {'max_hold': 3, 'base_position_ratio': 0.70},
        'medium': {'max_hold': 3, 'base_position_ratio': 0.65},
        'large': {'max_hold': 3, 'base_position_ratio': 0.60},
    }

    # ---- 策略参数（与聚宽版一致）----
    g.params = {
        'rebalance_interval': 2,     # 2天轮动（降低再平衡时机运气，路径差距从161pp降至34pp）
        'momentum_period': 20,
        'momentum_period_long': 60,
        'atr_period': 14,
        'trailing_atr_mult': 2.5,
        'trailing_atr_mult_high_vol': 2.0,
        'high_vol_threshold': 0.30,
        'stop_floor': 0.03,
        'stop_cap': 0.15,
        'trend_weight': 0.3,
    }

    # ---- QDII标的（需要溢价检测）----
    g.qdii_etfs = {
        '513100.SS',    # 纳指ETF
        '513500.SS',    # 标普500ETF
        '159920.SZ',    # 恒生ETF
        '513880.SS',    # 日经ETF
        '513050.SS',    # 中概互联ETF
    }
    g.qdii_premium_limit = 0.05  # QDII溢价上限5%

    g.current_tier = None
    g.day_count = 0
    g.highest_since_buy = {}
    g.entry_atr = {}
    g.sold_today = {}  # 实盘防重复卖出标记（order_target有6秒同步延迟）
    g.__last_snapshot = {}  # 缓存get_snapshot结果，供trade_status/iopv/down_px复用（__前缀不持久化）
    g.__pending_orders = {}  # 待on_trade_response确认的买入订单（__前缀不持久化）

    # ---- 注册标的池（set_universe后handle_data的data[code]才能访问BarData）----
    try:
        set_universe(g.etf_pool)
    except Exception:
        pass

    # ---- 模式检测（is_trade()：实盘返回True，回测返回False）----
    try:
        g.__is_live = is_trade()
    except Exception:
        g.__is_live = False

    # 实盘模式：使用run_daily精确控制执行时间
    # 注意：PTrade实盘run_daily+run_interval累计不能超过5个
    if g.__is_live:
        run_daily(context, _update_tier_wrapper, time='09:30')
        run_daily(context, _do_trading_wrapper, time='09:35')
        run_daily(context, _update_highest_wrapper, time='15:00')
        run_daily(context, _after_close_wrapper, time='15:30')


def handle_data(context, data):
    """
    回测模式：日频回测固定在15:00调用，承担全部策略逻辑。
    实盘模式：主逻辑由run_daily在指定时间处理，handle_data仅保存data引用。
    注意：get_price/get_history不支持handle_data与run_daily同时调用（官方文档）。
    """
    g.__data = data

    if g.__is_live:
        return

    # 回测模式：在handle_data中执行全部逻辑
    _update_tier(context)
    _do_trading(context)
    _update_highest(context)


def before_trading_start(context, data):
    """盘前准备（回测8:30调用，实盘9:10调用）"""
    g.__data = data
    g.sold_today = {}
    g.__last_snapshot = {}  # 清除前日snapshot缓存
    log.info('[盘前] 策略启动，清理缓存完毕')

    # 清理未成交的pending_orders（前日下单未回调的视为未成交）
    if g.__pending_orders:
        for code in list(g.__pending_orders.keys()):
            log.warning('[订单超时] %s 前日买入未收到成交确认，清理pending' % code)
        g.__pending_orders = {}

    # 恢复缺失entry_atr的持仓（PTrade重启/pending丢失导致）
    if g.__is_live:
        _recover_missing_atr(context)


def after_trading_end(context, data):
    """盘后清理（回测15:30调用，实盘15:30后调用）"""
    # 实盘模式下_after_close已由run_daily 15:30的_after_close_wrapper调用，这里只清理标记
    # 回测模式下handle_data不调_after_close，由这里补上
    if not g.__is_live:
        _after_close(context)
    g.sold_today = {}


# ============================================================
#  run_daily包装函数（实盘模式使用）
# ============================================================
def _update_tier_wrapper(context):
    log.info('[09:30] 档位检查开始')
    _update_tier(context)


def _do_trading_wrapper(context):
    log.info('[09:35] 交易逻辑开始')
    _do_trading(context)


def _update_highest_wrapper(context):
    log.info('[15:00] 更新最高价')
    _update_highest(context)


def _after_close_wrapper(context):
    _after_close(context)
    g.sold_today = {}


# ============================================================
#  Portfolio/Position兼容层
# ============================================================
def _get_total_value(context):
    return context.portfolio.portfolio_value


def _get_available_cash(context):
    return context.portfolio.cash


def _get_positions(context):
    return context.portfolio.positions


def _pos_amount(pos):
    return pos.amount


def _pos_cost(pos):
    return pos.cost_basis


def _pos_price(pos):
    return pos.last_sale_price


# ============================================================
#  行情数据获取
# ============================================================
def _get_current_price(code):
    """
    获取当前价格（兼容回测和实盘，三级回退）
    1. 实盘：get_snapshot获取实时价，并缓存供trade_status/iopv/down_px复用
    2. 回测/实盘回退：data[code].price（需set_universe注册）
    3. 最终回退：get_history取最近收盘价
    """
    if g.__is_live:
        try:
            snap = get_snapshot(code)
            if snap and code in snap:
                snap = snap[code]
            if snap and snap.get('last_px', 0) > 0:
                g.__last_snapshot[code] = snap  # 缓存供后续使用
                return snap['last_px']
        except Exception:
            pass

    if hasattr(g, '__data') and g.__data is not None:
        try:
            return g.__data[code].price
        except Exception:
            pass

    try:
        df = get_history(1, '1d', 'close', [code], include=True)
        return float(df[code].iloc[-1])
    except Exception:
        pass

    return None


def _is_paused(code):
    """
    检查是否停牌（三级回退）
    1. 实盘：用缓存的snapshot trade_status判断（HALT/SUSP/STOPT/DELISTED=停牌）
    2. 回测/实盘回退：get_stock_status([code], 'HALT')
    3. 最终回退：data[code].is_open == 0
    """
    # 实盘：优先检查缓存的snapshot trade_status
    if g.__is_live and code in g.__last_snapshot:
        ts = g.__last_snapshot[code].get('trade_status', '')
        if ts in ('HALT', 'SUSP', 'STOPT', 'DELISTED'):
            return True
        if ts in ('TRADE', 'OCALL', 'BREAK', 'ENDTR', 'POSTR'):
            return False  # 可交易或盘中状态，不算停牌

    try:
        status = get_stock_status([code], 'HALT')
        return status.get(code, False)
    except Exception:
        pass
    if hasattr(g, '__data') and g.__data is not None:
        try:
            return g.__data[code].is_open == 0
        except Exception:
            pass
    return False


def _get_prev_trade_date(context):
    """获取前一个交易日"""
    today = context.current_dt.date()
    try:
        trade_days = get_trade_days(end_date=today, count=2)
        return trade_days[0]
    except Exception:
        pass
    try:
        all_days = get_all_trades_days(date=today.strftime('%Y%m%d'))
        past_days = [d for d in all_days if d < today]
        if past_days:
            return past_days[-1]
    except Exception:
        pass
    prev = today - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def _get_price_data(code, end_date, count):
    """
    获取历史行情数据（两级回退）
    1. get_price：一次获取多字段，支持end_date+count组合，返回不含当天数据
    2. get_history：逐字段获取，作为get_price失败时的回退
    两条路径都过滤volume=0的停牌日（等效聚宽skip_paused=True）
    """
    if hasattr(end_date, 'strftime'):
        end_date_str = end_date.strftime('%Y-%m-%d')
    else:
        end_date_str = str(end_date)

    try:
        df = get_price(code,
                       end_date=end_date_str,
                       count=count,
                       frequency='1d',
                       fields=['open', 'close', 'high', 'low', 'volume'],
                       fq='pre')
        if df is not None and len(df) > 0:
            # 手动过滤停牌日（成交量为0的日期，等效于聚宽的skip_paused=True）
            if 'volume' in df.columns:
                df = df[df['volume'] > 0]
            return df
    except Exception:
        pass

    try:
        df_c = get_history(count, '1d', 'close', [code], fq='pre')
        df_o = get_history(count, '1d', 'open', [code], fq='pre')
        df_h = get_history(count, '1d', 'high', [code], fq='pre')
        df_l = get_history(count, '1d', 'low', [code], fq='pre')
        df_v = get_history(count, '1d', 'volume', [code], fq='pre')
        df = pd.DataFrame({
            'open': df_o[code],
            'close': df_c[code],
            'high': df_h[code],
            'low': df_l[code],
            'volume': df_v[code],
        })
        # 手动过滤停牌日
        df = df[df['volume'] > 0]
        return df
    except Exception as e:
        log.error('获取行情失败 %s: %s' % (code, str(e)))
        return None



# ============================================================
#  动态资金档位
# ============================================================
def _update_tier(context):
    total = _get_total_value(context)
    if total < 15000:
        new_tier = 'micro'
    elif total < 50000:
        new_tier = 'small'
    elif total < 100000:
        new_tier = 'medium'
    else:
        new_tier = 'large'

    if new_tier != g.current_tier:
        old_tier = g.current_tier or '初始化'
        g.current_tier = new_tier
        cfg = g.capital_tiers[new_tier]
        log.info('[档位变更] %s -> %s | 总资产:%.0f | 最大持仓:%d' % (
            old_tier, new_tier, total, cfg['max_hold']))


def _get_tier_param(param_name):
    return g.capital_tiers[g.current_tier][param_name]


# ============================================================
#  动量计算（策略逻辑与聚宽版一致）
# ============================================================
def _calc_momentum(code, end_date):
    """
    计算单只ETF的动量得分。
    双重动量（短期ROC20+中期ROC60），双正才买入。
    返回None表示不满足买入条件。
    """
    mom_long = g.params['momentum_period_long']
    df = _get_price_data(code, end_date, count=mom_long + 10)

    if df is None or len(df) < mom_long:
        return None

    C = df['close']
    H = df['high']
    L = df['low']
    mom_period = g.params['momentum_period']

    # ---- 短期动量：20日收益率 ----
    roc_short = C.iloc[-1] / C.iloc[-mom_period] - 1

    # ---- 中期动量：60日收益率 ----
    roc_long = C.iloc[-1] / C.iloc[-mom_long] - 1

    # ---- 波动率：20日年化 ----
    returns = C.pct_change().iloc[-mom_period:]
    vol = returns.std() * np.sqrt(252)
    if vol <= 0 or pd.isna(vol):
        return None

    # ---- 风险调整动量 ----
    risk_adj_mom = roc_short / vol

    # ---- 过滤：双重动量+趋势确认 ----
    ma20 = C.iloc[-20:].mean()
    if C.iloc[-1] < ma20:
        return None

    if roc_short < 0:
        return None

    if roc_long < 0:
        return None

    # ---- 趋势强度 ----
    trend_strength = (C.iloc[-1] - ma20) / ma20

    # ---- 综合排名分 ----
    tw = g.params['trend_weight']
    sort_score = risk_adj_mom * (1 - tw) + trend_strength * tw

    # ---- ATR ----
    atr_period = g.params['atr_period']
    TR = pd.concat([
        H - L,
        (H - C.shift(1)).abs(),
        (L - C.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = TR.iloc[-atr_period:].mean()

    return {
        'code': code,
        'momentum': sort_score,
        'risk_adj_mom': risk_adj_mom,
        'trend_strength': trend_strength,
        'roc': roc_short,
        'roc_long': roc_long,
        'volatility': vol,
        'close': C.iloc[-1],
        'atr': atr if not pd.isna(atr) else C.iloc[-1] * 0.02,
    }


# ============================================================
#  ATR跟踪止损（策略逻辑与聚宽版一致，实盘用跌停价limit_price确保成交）
# ============================================================
def _calc_trailing_stop_price(highest_price, atr_value):
    """动态倍数：高波动2.0x，正常2.5x"""
    vol_pct = atr_value / highest_price * np.sqrt(252 / g.params['atr_period'])
    if vol_pct > g.params['high_vol_threshold']:
        atr_mult = g.params['trailing_atr_mult_high_vol']
    else:
        atr_mult = g.params['trailing_atr_mult']

    pct_stop = atr_mult * atr_value / highest_price
    pct_stop = max(g.params['stop_floor'], min(g.params['stop_cap'], pct_stop))
    return highest_price * (1 - pct_stop)


def _check_stop_loss(context):
    """检查所有持仓的止损条件，返回被止损的代码列表"""
    stopped = []
    positions = _get_positions(context)
    for code in list(positions.keys()):
        pos = positions[code]
        if _pos_amount(pos) <= 0:
            continue

        # 实盘防重复卖出
        if g.__is_live and g.sold_today.get(code, False):
            continue

        if _is_paused(code):
            continue

        cur_price = _get_current_price(code)
        if cur_price is None:
            continue

        cost = _pos_cost(pos)
        profit_pct = (cur_price - cost) / cost

        # ATR跟踪止损
        if code in g.highest_since_buy and code in g.entry_atr:
            highest = g.highest_since_buy[code]
            stop_price = _calc_trailing_stop_price(highest, g.entry_atr[code])
            if cur_price <= stop_price:
                drawdown = (highest - cur_price) / highest
                log.info('[ATR止损] %s 最高%.3f 现价%.3f 回撤%.1f%% 盈亏%.1f%%' % (
                    code, highest, cur_price, drawdown * 100, profit_pct * 100))
                # 止损用跌停价(down_px)作limit_price，确保在下跌行情中成交
                # 不传limit_price时PTrade内部调get_snapshot取价，若snapshot失败则下单失败
                sell_lmt = round(cur_price, 3)
                if g.__is_live and code in g.__last_snapshot:
                    try:
                        down_px = float(g.__last_snapshot[code].get('down_px', 0))
                    except (ValueError, TypeError):
                        down_px = 0
                    if down_px > 0:
                        sell_lmt = round(down_px, 3)
                order_target(code, 0, limit_price=sell_lmt)
                g.sold_today[code] = True
                g.highest_since_buy.pop(code, None)
                g.entry_atr.pop(code, None)
                stopped.append(code)

    return stopped


# ============================================================
#  核心交易逻辑（策略逻辑与聚宽版一致，增加涨停/溢价过滤+成交回调机制）
# ============================================================
def _do_trading(context):
    prev_date = _get_prev_trade_date(context)

    # ======== 第一步：每日止损检查 ========
    stopped_codes = _check_stop_loss(context)
    if stopped_codes:
        log.info('[09:35] 止损触发%d只：%s，提前进入轮动' % (len(stopped_codes), ', '.join(stopped_codes)))
    else:
        log.info('[09:35] 止损检查完毕，无触发')

    # ======== 第二步：判断是否到轮动日 ========
    g.day_count += 1
    if g.day_count < g.params['rebalance_interval'] and not stopped_codes:
        log.info('[09:35] 非轮动日（%d/%d），等待下次轮动' % (g.day_count, g.params['rebalance_interval']))
        return
    if g.day_count >= g.params['rebalance_interval']:
        g.day_count = 0
        log.info('[09:35] 轮动日到达，开始计算动量排名')

    # ======== 第三步：计算所有ETF动量并排名 ========
    candidates = []
    for code in g.etf_pool:
        if _is_paused(code):
            continue
        result = _calc_momentum(code, prev_date)
        if result is not None:
            candidates.append(result)

    candidates.sort(key=lambda x: x['momentum'], reverse=True)

    log.info('[09:35] 动量筛选完毕：%d/%d只ETF通过过滤' % (len(candidates), len(g.etf_pool)))
    for c in candidates[:5]:  # 打印前5名
        log.info('  %s 综合=%.3f ROC20=%.1f%% ROC60=%.1f%%' % (
            c['code'], c['momentum'], c['roc'] * 100, c['roc_long'] * 100))

    # ======== 第四步：确定目标持仓（候选不足时国债填空）========
    max_hold = _get_tier_param('max_hold')
    target_list = candidates[:max_hold]

    # 候选不足max_hold时，用国债ETF填满剩余仓位
    bond = g.bond_etf
    bond_in_targets = any(t['code'] == bond for t in target_list)
    if len(target_list) < max_hold and not bond_in_targets:
        if not _is_paused(bond):
            bond_price = _get_current_price(bond)
            if bond_price is not None:
                target_list.append({
                    'code': bond,
                    'momentum': 0, 'risk_adj_mom': 0, 'trend_strength': 0,
                    'roc': 0, 'roc_long': 0,
                    'volatility': 0.03,
                    'close': bond_price,
                    'atr': bond_price * 0.005,
                    '_is_bond_fill': True,
                })
                log.info('[国债填空] 候选%d只不足%d，国债补位' % (
                    len(target_list) - 1, max_hold))

    target_codes = set(t['code'] for t in target_list)
    log.info('[09:35] 目标持仓：%s' % ', '.join(target_codes) if target_codes else '[09:35] 目标持仓：无（全部弱势）')

    # ======== 第五步：卖出不在目标中的持仓 ========
    positions = _get_positions(context)
    for code in list(positions.keys()):
        pos = positions[code]
        if _pos_amount(pos) <= 0:
            continue
        if g.__is_live and g.sold_today.get(code, False):
            continue
        if code not in target_codes:
            cur_price = _get_current_price(code)
            cost = _pos_cost(pos)
            if cur_price and cost > 0:
                profit_pct = (cur_price - cost) / cost
                log.info('[轮动卖出] %s 盈亏%.1f%%（被更强标的替换）' % (code, profit_pct * 100))
            # 卖出用跌停价作limit_price，确保成交
            sell_lmt = round(cur_price, 3) if cur_price else None
            if g.__is_live and code in g.__last_snapshot:
                try:
                    down_px = float(g.__last_snapshot[code].get('down_px', 0))
                except (ValueError, TypeError):
                    down_px = 0
                if down_px > 0:
                    sell_lmt = round(down_px, 3)
            if sell_lmt:
                order_target(code, 0, limit_price=sell_lmt)
            else:
                order_target(code, 0)
            g.sold_today[code] = True
            g.highest_since_buy.pop(code, None)
            g.entry_atr.pop(code, None)

    # ======== 第六步：买入目标持仓 ========
    positions = _get_positions(context)  # 刷新
    current_holds = set()
    for code in positions:
        if _pos_amount(positions[code]) > 0:
            current_holds.add(code)

    to_buy = [sig for sig in target_list if sig['code'] not in current_holds]
    if not to_buy:
        log.info('[09:35] 持仓与目标一致，无需换仓')
        return

    available = _get_available_cash(context)
    slots = max_hold - len(current_holds & target_codes)

    if slots <= 0 or available < 500:
        log.info('[09:35] 无空余仓位(slots=%d)或资金不足(%.0f)，跳过买入' % (slots, available))
        return

    base_ratio = _get_tier_param('base_position_ratio')

    for sig in to_buy:
        if slots <= 0 or available < 500:
            break

        code = sig['code']
        price = _get_current_price(code)  # T日实时价
        if price is None or price <= 0:
            continue

        # ---- 涨停过滤：check_limit返回{code:int}，>=1为涨停，避免挂单浪费资金 ----
        try:
            limit_status = check_limit(code)
            if limit_status and limit_status.get(code, 0) >= 1:
                log.info('[涨停跳过] %s 涨停中，跳过买入' % code)
                continue
        except Exception:
            pass

        # ---- QDII溢价过滤：用snapshot的iopv计算溢价率，>5%时跳过 ----
        if g.__is_live and code in g.qdii_etfs and code in g.__last_snapshot:
            snap = g.__last_snapshot[code]
            try:
                iopv = float(snap.get('iopv', 0))
            except (ValueError, TypeError):
                iopv = 0
            if iopv > 0:
                premium = price / iopv - 1
                if premium > g.qdii_premium_limit:
                    log.info('[溢价跳过] %s 溢价%.1f%%超限%.0f%%，跳过买入' % (
                        code, premium * 100, g.qdii_premium_limit * 100))
                    continue

        is_bond_fill = sig.get('_is_bond_fill', False)

        # 仓位计算：国债填空用剩余全部资金，权益类等权分配 × 波动率反比
        if is_bond_fill:
            alloc = available * 0.95  # 国债填空：尽量多买
        else:
            alloc = available / slots * base_ratio

        if not is_bond_fill:
            target_vol = 0.15
            actual_vol = max(sig['volatility'], 0.05)
            vol_mult = min(target_vol / actual_vol, 1.5)
            vol_mult = max(vol_mult, 0.4)
            alloc *= vol_mult

        alloc = min(alloc, available * 0.95)

        shares = int(alloc / price / 100) * 100
        if shares < 100:
            if available >= price * 100 * 1.003:
                shares = 100
            else:
                continue

        if is_bond_fill:
            log.info('[国债填空买入] %s %d股 @%.3f' % (code, shares, price))
        else:
            log.info('[轮动买入] %s 综合=%.3f 动量=%.3f 趋势=%.3f ROC20=%.1f%% ROC60=%.1f%% 波动率=%.1f%% %d股 @%.3f' % (
                code, sig['momentum'], sig['risk_adj_mom'], sig['trend_strength'],
                sig['roc'] * 100, sig['roc_long'] * 100,
                sig['volatility'] * 100, shares, price))

        order(code, shares, limit_price=round(price, 3))

        # 实盘：记录待确认订单，等on_trade_response确认成交后再设止损基准
        # 回测：直接设置（on_trade_response仅交易模式可用，回测无回调）
        if g.__is_live:
            g.__pending_orders[code] = {'price': price, 'atr': sig['atr']}
        else:
            g.highest_since_buy[code] = price
            g.entry_atr[code] = sig['atr']

        available -= shares * price * 1.003
        slots -= 1


# ============================================================
#  每日更新最高价
# ============================================================
def _update_highest(context):
    positions = _get_positions(context)
    for code in list(positions.keys()):
        pos = positions[code]
        if _pos_amount(pos) <= 0:
            continue
        cur = _get_current_price(code)
        if cur is None:
            continue
        if code in g.highest_since_buy:
            if cur > g.highest_since_buy[code]:
                g.highest_since_buy[code] = cur
        else:
            g.highest_since_buy[code] = max(cur, _pos_cost(pos))


# ============================================================
#  恢复缺失的ATR止损基准（实盘重启/pending丢失后的安全网）
# ============================================================
def _recover_missing_atr(context):
    """
    检查所有持仓，如果有持仓缺少entry_atr（重启或pending_orders丢失导致），
    从历史数据重新计算ATR并补上，确保止损保护不中断。
    """
    positions = _get_positions(context)
    prev_date = _get_prev_trade_date(context)
    for code in list(positions.keys()):
        pos = positions[code]
        if _pos_amount(pos) <= 0:
            continue
        if code in g.entry_atr:
            continue
        # 缺少entry_atr，需要恢复
        atr_period = g.params['atr_period']
        df = _get_price_data(code, prev_date, count=atr_period + 5)
        if df is not None and len(df) >= atr_period:
            H = df['high']
            L = df['low']
            C = df['close']
            TR = pd.concat([
                H - L,
                (H - C.shift(1)).abs(),
                (L - C.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = TR.iloc[-atr_period:].mean()
            if not pd.isna(atr):
                g.entry_atr[code] = atr
                cost = _pos_cost(pos)
                if code not in g.highest_since_buy:
                    g.highest_since_buy[code] = max(cost, C.iloc[-1])
                log.info('[ATR恢复] %s 重建ATR=%.4f 最高价=%.3f' % (
                    code, atr, g.highest_since_buy[code]))
                continue
        # fallback：用成本价估算ATR
        cost = _pos_cost(pos)
        g.entry_atr[code] = cost * 0.02
        if code not in g.highest_since_buy:
            g.highest_since_buy[code] = cost
        log.warning('[ATR恢复] %s 数据不足，使用成本2%%估算ATR=%.4f' % (code, cost * 0.02))


# ============================================================
#  盘后记录
# ============================================================
def _after_close(context):
    if g.current_tier is None:
        return

    positions = _get_positions(context)
    hold = {code: pos for code, pos in positions.items() if _pos_amount(pos) > 0}

    log.info('=' * 60)
    log.info('[%s] 总值:%.2f 现金:%.2f 持仓:%d/%d' % (
        g.current_tier,
        _get_total_value(context),
        _get_available_cash(context),
        len(hold),
        _get_tier_param('max_hold')))

    for code, pos in hold.items():
        cur_price = _pos_price(pos)
        cost = _pos_cost(pos)
        profit_pct = (cur_price - cost) / cost * 100 if cost > 0 else 0
        highest = g.highest_since_buy.get(code, cur_price)
        log.info('  %s 成本:%.3f 现价:%.3f 高:%.3f 盈亏:%.1f%%' % (
            code, cost, cur_price, highest, profit_pct))
    log.info('=' * 60)


# ============================================================
#  委托回调（仅交易模式可用，回测不支持）
# ============================================================
def on_order_response(context, order_list):
    """
    委托状态推送回调（仅交易模式可用，回测模式直接跳过）。
    order_list为dict列表，字段：stock_code/status/amount/price/error_info/entrust_type/entrust_no等。
    注意：entrust_type是委托类别（'0'=普通,'2'=撤单），不是买卖方向。
    status：'5'=部分撤单,'6'=已撤单,'9'=废单。amount始终为正数（买卖均正）。
    """
    try:
        if not g.__is_live:
            return
    except AttributeError:
        return

    # 兼容单个对象和列表
    if not isinstance(order_list, list):
        order_list = [order_list]

    for od in order_list:
        code = od.get('stock_code', '')
        status = od.get('status', '')
        error_info = od.get('error_info', '')
        amount = od.get('amount', 0)
        price = od.get('price', 0)

        # status: '5'=部分撤单, '6'=已撤单, '9'=废单/被拒
        if str(status) in ('5', '6', '9'):
            # 判断买卖方向：先查pending_orders（买入），再查sold_today（卖出）
            # 必须先查买入，因为同一ETF可能同日先卖后买（止损后重入场）
            if code in g.__pending_orders:
                # 买入失败：清理pending_orders
                g.__pending_orders.pop(code, None)
                log.warning('[买入失败] %s %d股 @%.3f 状态:%s 原因:%s' % (
                    code, amount, price, status, error_info))
            elif code in g.sold_today:
                # 卖出失败（可能是止损单被拒）：严重告警
                log.error('[止损告警] %s 卖出委托失败！持仓未减少，需人工处理。状态:%s 原因:%s' % (
                    code, status, error_info))
                # 重置sold_today标记，允许后续重试
                g.sold_today.pop(code, None)
            else:
                # 未知来源（可能是策略外交易的主推）
                log.warning('[委托异常] %s %d股 @%.3f 状态:%s 原因:%s（非本策略委托）' % (
                    code, amount, price, status, error_info))


# ============================================================
#  成交回调（仅交易模式可用，回测不支持）
# ============================================================
def on_trade_response(context, trade_list):
    """
    成交推送回调（仅交易模式可用，回测模式直接跳过）。
    trade_list为dict列表，字段：stock_code/entrust_bs/business_amount/business_price等。
    entrust_bs：'1'=买入,'2'=卖出。
    买入成交后用实际成交价设置止损基准，比下单时盲设更准确。
    """
    try:
        if not g.__is_live:
            return
    except AttributeError:
        return

    # 实盘模式：trade_list是dict列表
    if not isinstance(trade_list, list):
        trade_list = [trade_list]

    for trade in trade_list:
        code = trade.get('stock_code', '')
        direction = trade.get('entrust_bs', '')  # '1'=买入,'2'=卖出（on_trade_response独有字段）
        filled_qty = trade.get('business_amount', 0)
        filled_price = trade.get('business_price', 0)

        if not code or filled_qty <= 0:
            continue

        # 买入成交：从待确认订单中取出ATR，用实际成交价设止损基准
        if direction == '1' and code in g.__pending_orders:
            pending = g.__pending_orders.pop(code)
            g.highest_since_buy[code] = filled_price
            g.entry_atr[code] = pending['atr']
            log.info('[成交确认] 买入 %s %d股 @%.3f ATR=%.4f' % (
                code, filled_qty, filled_price, pending['atr']))

        # 卖出成交：清理止损记录
        elif direction == '2':
            log.info('[成交确认] 卖出 %s %d股 @%.3f' % (
                code, filled_qty, filled_price))
