# -*- coding: utf-8 -*-
"""
智能ETF量化交易策略 V15.9 - PTrade版（回测+实盘兼容）
=============================================
从聚宽V15.9逐行移植，所有交易逻辑、参数、信号条件100%不变。
根据PTrade官方API文档适配，同时兼容回测和实盘两种模式。

V15.9相对V15.7的改动：
  1. ETF池从10只扩展到12只（+日经ETF+中概互联ETF）
  2. 统一所有资金档位max_hold=3

适配要点（根据PTrade-API.html官方文档）：
  1. 代码格式：.XSHG/.XSHE → .SS/.SZ
  2. Portfolio属性：portfolio_value/cash 替代 total_value/available_cash
  3. Position属性：amount/cost_basis/last_sale_price 替代 total_amount/avg_cost/price
  4. 当前行情：data[code].price / get_snapshot 替代 get_current_data()
  5. 停牌检测：get_stock_status 替代 paused属性
  6. run_daily签名：run_daily(context, func, time) 需传context
  7. 回测兼容：日频回测run_daily固定15:00，改用handle_data驱动全部逻辑
  8. 实盘防重复：sold_today防止6秒同步延迟导致重复卖出

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

    # ---- 佣金与滑点（仅回测生效，实盘由券商设置）----
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

    # ---- 策略参数（与聚宽版100%一致）----
    g.params = {
        'rebalance_interval': 3,
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

    g.current_tier = None
    g.day_count = 0
    g.highest_since_buy = {}
    g.entry_atr = {}
    g.sold_today = {}  # 实盘防重复卖出标记

    # ---- 模式检测 ----
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
    回测模式：日频在15:00调用，承担全部逻辑。
    实盘模式：主逻辑由run_daily处理，这里不做操作。
    """
    g.__data = data

    if g.__is_live:
        return

    # 回测模式：在handle_data中执行全部逻辑
    _update_tier(context)
    _do_trading(context)
    _update_highest(context)


def before_trading_start(context, data):
    """盘前准备"""
    g.__data = data
    g.sold_today = {}


def after_trading_end(context, data):
    """盘后记录"""
    _after_close(context)
    g.sold_today = {}


# ============================================================
#  run_daily包装函数（实盘模式使用）
# ============================================================
def _update_tier_wrapper(context):
    _update_tier(context)


def _do_trading_wrapper(context):
    _do_trading(context)


def _update_highest_wrapper(context):
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
    获取当前价格（兼容回测和实盘）
    实盘：优先get_snapshot获取实时价
    回测：从handle_data的data参数获取
    """
    if g.__is_live:
        try:
            snap = get_snapshot(code)
            if snap and code in snap:
                snap = snap[code]
            if snap and snap.get('last_px', 0) > 0:
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
    """检查是否停牌"""
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
        all_days = get_all_trades_days(date=today)
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
    """获取历史行情数据"""
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
        df = pd.DataFrame({
            'open': df_o[code],
            'close': df_c[code],
            'high': df_h[code],
            'low': df_l[code],
        })
        return df
    except Exception as e:
        log.error('获取行情失败 %s: %s' % (code, str(e)))
        return None



# ============================================================
#  动态资金档位（与聚宽版100%一致）
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
#  动量计算（与聚宽版100%一致）
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
#  ATR跟踪止损（与聚宽版100%一致）
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
                order_target(code, 0)
                g.sold_today[code] = True
                g.highest_since_buy.pop(code, None)
                g.entry_atr.pop(code, None)
                stopped.append(code)

    return stopped


# ============================================================
#  核心交易逻辑（与聚宽版100%一致）
# ============================================================
def _do_trading(context):
    today = context.current_dt.date()
    prev_date = _get_prev_trade_date(context)

    # ======== 第一步：每日止损检查 ========
    stopped_codes = _check_stop_loss(context)

    # ======== 第二步：判断是否到轮动日 ========
    g.day_count += 1
    if g.day_count < g.params['rebalance_interval'] and not stopped_codes:
        return
    if g.day_count >= g.params['rebalance_interval']:
        g.day_count = 0

    # ======== 第三步：计算所有ETF动量并排名 ========
    candidates = []
    for code in g.etf_pool:
        if _is_paused(code):
            continue
        result = _calc_momentum(code, prev_date)
        if result is not None:
            candidates.append(result)

    candidates.sort(key=lambda x: x['momentum'], reverse=True)

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
        return

    available = _get_available_cash(context)
    slots = max_hold - len(current_holds & target_codes)

    if slots <= 0 or available < 500:
        return

    base_ratio = _get_tier_param('base_position_ratio')

    for sig in to_buy:
        if slots <= 0 or available < 500:
            break

        code = sig['code']
        price = _get_current_price(code)  # T日实时价
        if price is None or price <= 0:
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

        order(code, shares)
        g.highest_since_buy[code] = price
        g.entry_atr[code] = sig['atr']
        available -= shares * price * 1.003
        slots -= 1


# ============================================================
#  每日更新最高价（与聚宽版100%一致）
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
