# -*- coding: utf-8 -*-
"""
小牛量化选股策略 V3.0 - 单信号精简+熊市过滤
=============================================
V2回测结果（2万起始，bug修复后）：-41.5%总收益，64.2%最大回撤
  - high_vol累计+136%（唯一盈利信号），lizhuang累计-30%
  - 2022年-40.5%（无熊市保护）

V3改动：
  1. 高量柱为主信号，lizhuang降级为备选（仅score=3入场）
  2. 熊市过滤：沪深300低于MA60超2%且MA60近10日下行→不开新仓
     （加2%偏离+10日窗口避免MA60附近频繁切换）
  3. 均线粘合保留为辅助加分（不单独触发买入）
  4. 评分门槛降至1.5（single signal策略门槛不能太高）

股票池：沪深800成分股（沪深300+中证500）主板（60/00）
核心信号：高量柱（主）+ 定庄起爆score=3（辅）+ 量价确认 + 均线粘合加分
风控：ATR止损(2.0x,3%-6%) + 形态止损 + 时间止损(8天) + 熊市空仓

回测建议：2020-01-01 ~ 2025-12-31，20,000元
"""

import numpy as np
import pandas as pd
from jqdata import *


# ============================================================
#  初始化
# ============================================================
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)

    # 个股交易成本（比ETF高：有印花税）
    set_slippage(PriceRelatedSlippage(0.002))
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0.001,        # 印花税0.1%（卖出时收取）
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5
    ), type='stock')

    # ---- 资金档位配置（2万起步，资金可能增减）----
    g.capital_tiers = {
        'micro':  {'max_hold': 2, 'base_position_ratio': 0.80},  # <1.5万：集中2只
        'small':  {'max_hold': 3, 'base_position_ratio': 0.70},  # 1.5-5万
        'medium': {'max_hold': 3, 'base_position_ratio': 0.60},  # 5-15万
        'large':  {'max_hold': 4, 'base_position_ratio': 0.55},  # >15万
    }

    # ---- 策略参数 ----
    g.params = {
        # 选股宇宙（沪深800成分股，无需市值/成交额筛选）
        'pre_filter_count': 150,         # 预筛选股票数量上限

        # 定庄起爆
        'lizhuang_vol_mult': 2.0,        # 立庄K线：量>前日2倍
        'lizhuang_body_pct': 0.05,       # 立庄K线：实体>5%
        'lizhuang_lookback': 30,         # 立庄K线回看30天
        'lizhuang_pullback_vol': 0.8,    # 回调缩量：量<5日均量80%

        # 均线粘合（V3保留为辅助加分）
        'ma_converge_pct': 0.03,         # MA收敛范围<3%
        'ma_vol_expand': 1.2,            # 量能扩张：5日均量>90日均量*1.2

        # 扫描频率（降低回测开销）
        'scan_interval': 1,             # 每天扫描（批量get_price已足够提速）

        # 高量柱（V3主信号）
        'high_vol_mult': 2.5,            # 高量柱：量>=前日2.5倍
        'high_vol_lookback': 20,         # 高量柱回看20天

        # V3新增：熊市过滤（参考V13经验，2022年-40.5%的教训）
        'bear_index': '000300.XSHG',     # 用沪深300判断大盘状态
        'bear_ma_period': 60,            # MA60作为牛熊分界

        # 风控
        'atr_period': 14,
        'trailing_atr_mult': 2.0,        # 跟踪止损2.0x ATR
        'stop_floor': 0.03,              # 止损下限3%
        'stop_cap': 0.06,                # 止损上限6%
        'max_hold_days': 8,              # 时间止损8天
        'cooldown_days': 3,              # 冷却期3天
        'max_single_pct': 0.35,          # 单只最大35%

        # 评分门槛（V3：high_vol score=2即可入场，无需其他信号凑分）
        'min_composite_score': 1.5,      # 降至1.5，让score=2的high_vol也能入场
    }

    # ---- 状态变量 ----
    g.current_tier = None
    g.highest_since_buy = {}     # {code: highest_close} ATR跟踪止损
    g.entry_atr = {}             # {code: atr_at_entry}
    g.entry_date = {}            # {code: date} 时间止损
    g.pattern_stop = {}          # {code: stop_price} 形态止损
    g.pattern_type = {}          # {code: str} 持仓对应的形态类型
    g.cooldown = {}              # {code: date} 冷却期
    g.is_bear = False            # V3：熊市标记
    g.scan_day = 0               # V3：扫描日计数器

    # ---- 定时任务 ----
    run_daily(update_tier, time='09:30')
    run_daily(do_trading, time='09:35')
    run_daily(update_highest, time='15:00')
    run_daily(after_close, time='15:30')


def get_prev_trade_date(context):
    today = context.current_dt.date()
    return get_trade_days(end_date=today, count=2)[0]


# ============================================================
#  动态资金档位
# ============================================================
def update_tier(context):
    total = context.portfolio.total_value
    if total < 15000:
        new_tier = 'micro'
    elif total < 50000:
        new_tier = 'small'
    elif total < 150000:
        new_tier = 'medium'
    else:
        new_tier = 'large'

    if new_tier != g.current_tier:
        old_tier = g.current_tier or '初始化'
        g.current_tier = new_tier
        cfg = g.capital_tiers[new_tier]
        log.info('[档位变更] %s -> %s | 总资产:%.0f | 最大持仓:%d' % (
            old_tier, new_tier, total, cfg['max_hold']))


def get_tier_param(param_name):
    return g.capital_tiers[g.current_tier][param_name]


# ============================================================
#  V3：熊市检测（参考V13经验）
# ============================================================
def check_bear_market(prev_date):
    """
    沪深300低于MA60且MA60下行 → 熊市，不开新仓。
    V13经验：同等收益但回撤减半（27.9%→12.5%）。
    """
    p = g.params
    index_code = p['bear_index']
    ma_period = p['bear_ma_period']

    df = get_price(index_code, end_date=prev_date, count=ma_period + 10,
                   frequency='daily', fields=['close'], skip_paused=True)
    if df is None or len(df) < ma_period + 5:
        return False

    C = df['close']
    ma60 = C.rolling(ma_period).mean()

    # 条件1：沪深300收盘价低于MA60
    below_ma = C.iloc[-1] < ma60.iloc[-1]
    # 条件2：MA60在下行（近10日MA60斜率为负，拉长窗口避免频繁切换）
    ma_declining = ma60.iloc[-1] < ma60.iloc[-10]
    # 条件3：偏离MA60超过2%（避免MA60附近反复穿越的噪音）
    deviation = (C.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1]
    far_below = deviation < -0.02

    is_bear = far_below and ma_declining

    if is_bear != g.is_bear:
        g.is_bear = is_bear
        status = '熊市（暂停开仓）' if is_bear else '非熊市（正常交易）'
        log.info('[大盘研判] %s | 沪深300:%.2f MA60:%.2f' % (
            status, C.iloc[-1], ma60.iloc[-1]))

    return is_bear


# ============================================================
#  选股宇宙过滤
# ============================================================
def calc_max_affordable_price(context):
    """
    根据当前资金动态计算能买得起的最高股价。
    逻辑：每只仓位分到的金额 / 100股（A股最小交易单位）。
    留10%余量应对波动率调整和手续费。
    """
    total = context.portfolio.total_value
    max_hold = get_tier_param('max_hold')
    base_ratio = get_tier_param('base_position_ratio')
    per_slot = total * base_ratio / max_hold
    # 留10%余量（波动率反比可能缩减仓位，手续费等）
    max_price = per_slot * 0.90 / 100
    return max_price


def get_stock_universe(context, prev_date):
    """
    V2股票池：沪深300 + 中证500 = 沪深800成分股。
    成分股天然满足市值和流动性要求，省去市值/成交额筛选。
    过滤：非ST + 非停牌 + 买得起（动态价格上限）+ 量价异动预筛。
    """
    p = g.params
    current_data = get_current_data()

    # 动态价格上限：根据当前资金计算
    max_price = calc_max_affordable_price(context)
    log.info('[选股] 当前资金:%.0f 单仓上限:%.0f 最高可买股价:%.1f元' % (
        context.portfolio.total_value,
        context.portfolio.total_value * get_tier_param('base_position_ratio') / get_tier_param('max_hold'),
        max_price))

    # 沪深800 = 沪深300 + 中证500
    hs300 = get_index_stocks('000300.XSHG', date=prev_date)
    zz500 = get_index_stocks('000905.XSHG', date=prev_date)
    pool = list(set(hs300 + zz500))

    # 仅保留主板（60/00开头）
    pool = [s for s in pool if s[:2] in ('60', '00')]

    # ST过滤
    if pool:
        st_data = get_extras('is_st', pool, start_date=prev_date, end_date=prev_date, df=True)
        if not st_data.empty:
            pool = st_data.columns[~st_data.iloc[0]].tolist()

    # 停牌过滤
    pool = [s for s in pool if not current_data[s].paused]

    # 价格过滤 + 量价异动预筛
    pre_filter = []
    batch_size = 200
    for i in range(0, len(pool), batch_size):
        batch = pool[i:i + batch_size]
        panel = get_price(batch, end_date=prev_date, count=10,
                         frequency='daily', fields=['close', 'volume', 'money'],
                         skip_paused=True, fq='pre', panel=False)
        if panel is None or panel.empty:
            continue

        for code in batch:
            df_code = panel[panel['code'] == code]
            if len(df_code) < 5:
                continue

            close = df_code['close'].values
            vol = df_code['volume'].values

            # 价格过滤：最新收盘价必须买得起100股
            if close[-1] > max_price:
                continue

            # 预筛：近5日内有量能放大（当日量>前日1.5倍）或价格接近10日新高
            has_vol_spike = any(vol[j] > vol[j - 1] * 1.5 for j in range(-5, 0) if j - 1 >= -len(vol))
            near_high = close[-1] >= np.max(close) * 0.95

            if has_vol_spike or near_high:
                pre_filter.append(code)

    # 控制扫描数量
    if len(pre_filter) > p['pre_filter_count']:
        pre_filter = pre_filter[:p['pre_filter_count']]

    return pre_filter


# ============================================================
#  技术指标计算辅助
# ============================================================
def calc_ema(series, period):
    """指数移动平均"""
    return series.ewm(span=period, adjust=False).mean()


def calc_atr(high, low, close, period=14):
    """ATR计算"""
    TR = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return TR.rolling(period).mean()


# ============================================================
#  形态检测引擎（5大信号）
# ============================================================

def detect_lizhuang(df):
    """
    定庄起爆（PDF P369-397）
    寻找立庄K线 → 缩量回调 → 放量突破
    返回: {'score': 0-3, 'stop_price': float, 'pattern': 'lizhuang'} 或 None
    """
    p = g.params
    if len(df) < 60:
        return None

    C = df['close'].values
    O = df['open'].values
    H = df['high'].values
    L = df['low'].values
    V = df['volume'].values

    # EXPMA(10) 和 EXPMA(32)
    ema10 = calc_ema(df['close'], 10).values
    ema32 = calc_ema(df['close'], 32).values

    # 在最近30天内寻找立庄K线
    lookback = min(p['lizhuang_lookback'], len(df) - 30)
    lizhuang_idx = None
    lizhuang_score = 0

    for i in range(-lookback, -2):
        idx = len(df) + i
        if idx < 1:
            continue

        # 条件1：倍量（量>前日2倍）
        if V[idx] < V[idx - 1] * p['lizhuang_vol_mult']:
            continue

        # 条件2：阳线实体>5%
        body_pct = (C[idx] - O[idx]) / C[idx] if C[idx] > 0 else 0
        if body_pct < p['lizhuang_body_pct']:
            continue

        # 条件3：EXPMA(10)在EXPMA(32)上方或即将金叉（±5日）
        ema_bull = False
        for j in range(max(0, idx - 5), min(len(df), idx + 6)):
            if ema10[j] > ema32[j]:
                ema_bull = True
                break
        if not ema_bull:
            continue

        # 找到立庄K线
        lizhuang_idx = idx
        lizhuang_score = 1
        break

    if lizhuang_idx is None:
        return None

    # 检查立庄K后的回调状态
    lk_high = H[lizhuang_idx]
    lk_low = L[lizhuang_idx]
    lk_vol_avg = np.mean(V[max(0, lizhuang_idx - 4):lizhuang_idx + 1])

    # 回调期间成交量萎缩
    post_vols = V[lizhuang_idx + 1:]
    if len(post_vols) >= 2:
        recent_vol_avg = np.mean(post_vols[-min(5, len(post_vols)):])
        vol_shrink = recent_vol_avg < lk_vol_avg * p['lizhuang_pullback_vol']
    else:
        vol_shrink = False

    # 回调期间价格守住EXPMA(32)
    held_ema32 = all(L[j] > ema32[j] * 0.98 for j in range(lizhuang_idx + 1, len(df)))

    if vol_shrink and held_ema32:
        lizhuang_score = 2

    # 突破信号：最新收盘突破立庄K最高价 + 放量
    if C[-1] > lk_high and V[-1] > np.mean(V[-6:-1]) * 1.2:
        lizhuang_score = 3

    if lizhuang_score < 1:
        return None

    return {
        'pattern': 'lizhuang',
        'score': lizhuang_score,
        # V2：收紧止损，用立庄K实体底部而非最低价，EMA容差从2%收到1%
        'stop_price': max(min(O[lizhuang_idx], C[lizhuang_idx]), ema32[-1] * 0.99),
    }


def detect_ma_convergence(df):
    """
    均线粘合发散（PDF P34-37）
    5/10/20/60MA收敛 → 向上发散 → 量能扩张
    返回: {'score': 0-3, 'stop_price': float, 'pattern': 'ma_converge'} 或 None
    """
    p = g.params
    if len(df) < 70:
        return None

    C = df['close']
    V = df['volume']

    ma5 = C.rolling(5).mean()
    ma10 = C.rolling(10).mean()
    ma20 = C.rolling(20).mean()
    ma60 = C.rolling(60).mean()

    # 最新值
    mas = [ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1], ma60.iloc[-1]]
    if any(pd.isna(m) for m in mas):
        return None

    price = C.iloc[-1]
    if price <= 0:
        return None

    # 收敛度：最高MA与最低MA的差 / 价格
    spread = (max(mas) - min(mas)) / price

    # 需要最近10天内出现过收敛
    was_converged = False
    for i in range(-10, -1):
        idx = len(df) + i
        if idx < 60:
            continue
        m = [ma5.iloc[idx], ma10.iloc[idx], ma20.iloc[idx], ma60.iloc[idx]]
        if any(pd.isna(x) for x in m):
            continue
        sp = (max(m) - min(m)) / C.iloc[idx]
        if sp < p['ma_converge_pct']:
            was_converged = True
            break

    if not was_converged and spread >= p['ma_converge_pct']:
        return None

    score = 0

    # 收敛确认
    if was_converged:
        score = 1

    # 发散确认：MA10 > MA20 且 MA20 > MA60
    if ma10.iloc[-1] > ma20.iloc[-1] and ma20.iloc[-1] > ma60.iloc[-1]:
        score = 2

    # 量能扩张：5日均量 > 90日均量 * 1.2
    vol_5 = V.iloc[-5:].mean()
    vol_90 = V.iloc[-90:].mean() if len(V) >= 90 else V.mean()
    if vol_5 > vol_90 * p['ma_vol_expand']:
        score = min(score + 1, 3)

    # MA60斜率为正（不在下降趋势中收敛）
    if ma60.iloc[-1] < ma60.iloc[-5]:
        score = max(score - 1, 0)

    if score < 1:
        return None

    return {
        'pattern': 'ma_converge',
        'score': score,
        # V2：收紧止损，用MA20而非MA20和MA60中的更低者
        'stop_price': ma20.iloc[-1] * 0.99,
    }


def detect_dragon_pullback(df):
    """
    龙回头（PDF P314-316）
    首波涨25%+ → 缩量A字回调 → 长下影线转折 → 小K线确认
    返回: {'score': 0-3, 'stop_price': float, 'pattern': 'dragon'} 或 None
    """
    p = g.params
    if len(df) < 40:
        return None

    C = df['close'].values
    H = df['high'].values
    L = df['low'].values
    O = df['open'].values
    V = df['volume'].values

    # 在最近60天内寻找首波涨幅>=25%的波段
    wave_start = None
    wave_peak = None
    wave_peak_idx = None

    # 滑窗找首波
    lookback = min(50, len(df) - 10)
    for start_i in range(len(df) - lookback, len(df) - 10):
        for peak_i in range(start_i + 3, min(start_i + 25, len(df) - 3)):
            wave_gain = H[peak_i] / L[start_i] - 1
            if wave_gain >= p['dragon_wave_pct']:
                # 找到首波，取最高点
                if wave_peak is None or H[peak_i] > wave_peak:
                    wave_start = start_i
                    wave_peak = H[peak_i]
                    wave_peak_idx = peak_i

    if wave_peak is None or wave_peak_idx is None:
        return None

    # 峰值后需要回调
    post_peak = df.iloc[wave_peak_idx:]
    if len(post_peak) < 4:
        return None

    # 回调幅度
    post_low = min(L[wave_peak_idx:])
    pullback_pct = 1 - post_low / wave_peak
    if pullback_pct < p['dragon_pullback_pct']:
        return None  # 回调不够深

    score = 0

    # 回调期间成交量萎缩
    peak_vol = np.mean(V[max(0, wave_peak_idx - 2):wave_peak_idx + 1])
    pullback_vol = np.mean(V[wave_peak_idx + 1:])
    if pullback_vol < peak_vol * 0.7:
        score = 1

    # 寻找长下影线K线（近5日内）
    for i in range(-5, 0):
        idx = len(df) + i
        if idx <= wave_peak_idx:
            continue
        body = abs(C[idx] - O[idx])
        lower_shadow = min(C[idx], O[idx]) - L[idx]
        if body > 0 and lower_shadow >= body * p['dragon_shadow_ratio']:
            score = 2
            # 确认日：下影线后一天是小K线（涨跌幅<2%）
            if idx + 1 < len(df):
                change = abs(C[idx + 1] / C[idx] - 1)
                if change < 0.02:
                    score = 3
            break

    if score < 1:
        return None

    return {
        'pattern': 'dragon',
        'score': score,
        'stop_price': post_low * 0.98,  # 略低于回调最低点
    }


def detect_high_volume_bar(df):
    """
    高量柱战法（PDF P288-311, P375-393）
    倍量柱定义实底实顶 → 价格回踩实底 → 突破实顶
    返回: {'score': 0-3, 'stop_price': float, 'pattern': 'high_vol'} 或 None
    """
    p = g.params
    if len(df) < 25:
        return None

    C = df['close'].values
    O = df['open'].values
    H = df['high'].values
    L = df['low'].values
    V = df['volume'].values

    # 在最近20天内寻找高量柱
    lookback = min(p['high_vol_lookback'], len(df) - 5)
    hv_idx = None
    hv_support = None  # 实底
    hv_resist = None   # 实顶

    for i in range(-lookback, -2):
        idx = len(df) + i
        if idx < 1:
            continue

        # 高量：量>=前日2.5倍
        if V[idx] < V[idx - 1] * p['high_vol_mult']:
            continue

        # 记录实底和实顶
        if C[idx] > O[idx]:  # 阳线：开=实底，收=实顶
            hv_support = O[idx]
            hv_resist = C[idx]
        else:  # 阴线：收=实底，开=实顶
            hv_support = C[idx]
            hv_resist = O[idx]

        hv_idx = idx
        break  # 取最近的一根

    if hv_idx is None:
        return None

    score = 0
    cur_close = C[-1]

    # 高量柱找到
    score = 1

    # 价格在实底之上（回踩守住）
    held_support = all(L[j] >= hv_support * 0.98 for j in range(hv_idx + 1, len(df)))
    if held_support:
        score = 2

    # 突破实顶 + 放量
    vol_5 = np.mean(V[-5:]) if len(V) >= 5 else V[-1]
    if cur_close > hv_resist and V[-1] > vol_5 * 1.1:
        score = 3

    if score < 1:
        return None

    return {
        'pattern': 'high_vol',
        'score': score,
        # V2：高量柱止损收紧，从97%实底提高到99%实底
        'stop_price': hv_support * 0.99,
    }


def detect_volume_price(df):
    """
    量价确认（PDF P361-368 量价关系16种模型精华）
    辅助加分项，不单独触发买入。
    返回: {'vp_score': 0-1.5, 'vp_type': str}
    """
    if len(df) < 10:
        return {'vp_score': 0, 'vp_type': 'none'}

    C = df['close'].values
    V = df['volume'].values
    H = df['high'].values

    vol_5 = np.mean(V[-6:-1]) if len(V) >= 6 else V[-2]
    price_change = C[-1] / C[-2] - 1 if C[-2] > 0 else 0
    high_20 = np.max(H[-20:]) if len(H) >= 20 else H[-1]

    vp_score = 0
    vp_type = 'none'

    # 缩量上涨（看涨延续）：涨>1%但量<5日均量80%
    if price_change > 0.01 and V[-1] < vol_5 * 0.8:
        vp_score = 0.5
        vp_type = 'shrink_rally'

    # 放量突破（最强信号）：破20日最高+量>5日均量1.5倍
    if C[-1] >= high_20 and V[-1] > vol_5 * 1.5:
        vp_score = 1.0
        vp_type = 'vol_breakout'

    # 缩量回调（健康回调）：跌1-3%但量<5日均量60%
    if -0.03 < price_change < -0.01 and V[-1] < vol_5 * 0.6:
        vp_score = 0.5
        vp_type = 'shrink_pullback'

    return {'vp_score': vp_score, 'vp_type': vp_type}


# ============================================================
#  综合评分与候选排名
# ============================================================
def scan_and_rank(universe, prev_date, current_data):
    """
    对选股宇宙中的每只股票跑信号检测器，综合评分排名。
    V3优化：批量获取K线数据，避免逐只调用get_price。
    """
    candidates = []

    # 过滤冷却期和停牌
    scan_list = []
    for code in universe:
        if current_data[code].paused:
            continue
        if code in g.cooldown:
            if (pd.Timestamp(prev_date) - pd.Timestamp(g.cooldown[code])).days < g.params['cooldown_days']:
                continue
        scan_list.append(code)

    if not scan_list:
        return candidates

    # 批量获取120日K线（一次调用，代替逐只请求）
    batch_size = 50  # JoinQuant批量上限
    all_data = {}
    for i in range(0, len(scan_list), batch_size):
        batch = scan_list[i:i + batch_size]
        panel = get_price(batch, end_date=prev_date, count=120,
                         frequency='daily',
                         fields=['open', 'close', 'high', 'low', 'volume', 'money'],
                         skip_paused=True, fq='pre', panel=False)
        if panel is None or panel.empty:
            continue
        for code in batch:
            df_code = panel[panel['code'] == code].copy()
            if len(df_code) >= 60:
                df_code = df_code.set_index('time') if 'time' in df_code.columns else df_code
                all_data[code] = df_code

    for code, df in all_data.items():
        try:

            # V3：高量柱为主 + lizhuang为辅（仅score=3）+ 量价确认 + 均线粘合加分
            sig_highvol = detect_high_volume_bar(df)
            sig_lizhuang = detect_lizhuang(df)
            sig_vp = detect_volume_price(df)
            sig_ma = detect_ma_convergence(df)

            # 高量柱优先，lizhuang仅score=3时作为备选
            if sig_highvol:
                best = sig_highvol
            elif sig_lizhuang and sig_lizhuang['score'] >= 3:
                best = sig_lizhuang
            else:
                continue

            # 综合评分 = 形态分 + 量价加分 + 均线粘合加分
            composite = best['score']
            composite += sig_vp['vp_score']
            if sig_ma and sig_ma['score'] >= 2:
                composite += 0.5  # 均线多头发散加分
            # 高量柱额外+0.3（优先选择）
            if best['pattern'] == 'high_vol':
                composite += 0.3

            if composite < g.params['min_composite_score']:
                continue

            # 计算ATR和波动率
            atr_series = calc_atr(df['high'], df['low'], df['close'], g.params['atr_period'])
            atr = atr_series.iloc[-1]
            if pd.isna(atr):
                atr = df['close'].iloc[-1] * 0.02

            returns = df['close'].pct_change().iloc[-20:]
            vol = returns.std() * np.sqrt(252)
            if pd.isna(vol) or vol <= 0:
                vol = 0.30

            candidates.append({
                'code': code,
                'composite_score': composite,
                'best_pattern': best['pattern'],
                'pattern_score': best['score'],
                'stop_price': best['stop_price'],
                'vp_type': sig_vp['vp_type'],
                'vp_score': sig_vp['vp_score'],
                'multi_signal': len(signals),
                'close': df['close'].iloc[-1],
                'atr': atr,
                'volatility': vol,
            })

        except Exception as e:
            continue

    # 按综合评分降序排列
    candidates.sort(key=lambda x: x['composite_score'], reverse=True)
    return candidates


# ============================================================
#  ATR跟踪止损
# ============================================================
def calc_trailing_stop_price(highest_price, atr_value):
    """基于ATR的跟踪止损价（比ETF策略更紧：2.0x，3%-8%夹紧）"""
    p = g.params
    pct_stop = p['trailing_atr_mult'] * atr_value / highest_price
    pct_stop = max(p['stop_floor'], min(p['stop_cap'], pct_stop))
    return highest_price * (1 - pct_stop)


def repair_missing_state(context, code, pos, cur_price, today):
    """
    修复持仓缺失的状态变量。
    V2 bug修复：JoinQuant回测中g变量可能因pickle/重启丢失，
    导致持仓绕过所有止损检查。检测到缺失时用保守默认值补填。
    """
    repaired = False

    if code not in g.entry_date:
        # 无法确定真实入场日，用今天作为起点（下一个周期就会触发时间止损）
        g.entry_date[code] = str(today)
        repaired = True

    if code not in g.entry_atr:
        # 用成本的2%作为默认ATR（保守估计）
        g.entry_atr[code] = pos.avg_cost * 0.02
        repaired = True

    if code not in g.highest_since_buy:
        # 用当前价和成本的较大值
        g.highest_since_buy[code] = max(cur_price, pos.avg_cost)
        repaired = True

    if code not in g.pattern_stop:
        # 用成本的95%作为默认形态止损（即亏5%触发）
        g.pattern_stop[code] = pos.avg_cost * 0.95
        repaired = True

    if code not in g.pattern_type:
        g.pattern_type[code] = 'repaired'
        repaired = True

    if repaired:
        log.info('[状态修复] %s 成本:%.2f 现价:%.2f 补填止损:%.2f ATR:%.3f' % (
            code, pos.avg_cost, cur_price,
            g.pattern_stop[code], g.entry_atr[code]))

    return repaired


def check_stop_loss(context, current_data, today):
    """
    三重止损检查：ATR跟踪 + 形态止损 + 时间止损。
    返回被止损的代码列表。
    """
    stopped = []
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0:
            continue
        if current_data[code].paused:
            continue

        cur_price = current_data[code].last_price

        # V2 bug修复：检测并修复缺失的状态变量
        repair_missing_state(context, code, pos, cur_price, today)

        profit_pct = (cur_price - pos.avg_cost) / pos.avg_cost
        stop_reason = None

        # 1. 时间止损：持仓超过max_hold_days个交易日
        hold_days = len(get_trade_days(
            start_date=g.entry_date[code], end_date=today))
        if hold_days > g.params['max_hold_days']:
            stop_reason = '时间止损(%d天)' % hold_days

        # 2. 形态止损：跌破形态关键支撑位
        if stop_reason is None and cur_price <= g.pattern_stop[code]:
            stop_reason = '形态止损(破%.2f)' % g.pattern_stop[code]

        # 3. ATR跟踪止损
        if stop_reason is None:
            highest = g.highest_since_buy[code]
            atr_stop = calc_trailing_stop_price(highest, g.entry_atr[code])
            if cur_price <= atr_stop:
                drawdown = (highest - cur_price) / highest
                stop_reason = 'ATR止损(高%.2f回撤%.1f%%)' % (highest, drawdown * 100)

        if stop_reason:
            pattern = g.pattern_type.get(code, '?')
            log.info('[%s] %s(%s) 现价%.2f 盈亏%.1f%%' % (
                stop_reason, code, pattern, cur_price, profit_pct * 100))
            order_target(code, 0)
            # 清理状态 + 设置冷却
            g.highest_since_buy.pop(code, None)
            g.entry_atr.pop(code, None)
            g.entry_date.pop(code, None)
            g.pattern_stop.pop(code, None)
            g.pattern_type.pop(code, None)
            g.cooldown[code] = str(today)
            stopped.append(code)

    return stopped


# ============================================================
#  核心交易逻辑
# ============================================================
def do_trading(context):
    today = context.current_dt.date()
    prev_date = get_prev_trade_date(context)
    current_data = get_current_data()

    # ======== 第一步：每日止损检查（每天都执行，熊市也要止损）========
    stopped_codes = check_stop_loss(context, current_data, today)

    # ======== 第二步：扫描间隔控制（降低回测开销）========
    g.scan_day += 1
    need_scan = (g.scan_day >= g.params['scan_interval']) or len(stopped_codes) > 0
    if not need_scan:
        return  # 非扫描日且无止损，只做止损检查
    g.scan_day = 0

    # ======== 第三步：V3熊市过滤 ========
    is_bear = check_bear_market(prev_date)
    if is_bear:
        return  # 熊市不开新仓，等待止损自然清仓

    # ======== 第四步：检查是否有空仓位需要填充 ========
    max_hold = get_tier_param('max_hold')
    current_holds = sum(1 for code in context.portfolio.positions
                       if context.portfolio.positions[code].total_amount > 0)

    if current_holds >= max_hold:
        return  # 满仓跳过

    # ======== 第五步：选股宇宙过滤 ========
    universe = get_stock_universe(context, prev_date)
    if not universe:
        return

    # ======== 第五步：综合评分扫描 ========
    candidates = scan_and_rank(universe, prev_date, current_data)
    if not candidates:
        return

    # ======== 第六步：卖出持仓（仅在超限时替换）========
    # V1的"形态失效卖出"胜率仅49%均盈+1.1%，过于激进
    # V2：只在持仓数>max_hold且有评分>=4.0的强候选时才替换最弱持仓
    if current_holds > max_hold and candidates and candidates[0]['composite_score'] >= 4.0:
        # 找到盈亏最差的持仓卖出
        worst_code = None
        worst_pnl = float('inf')
        for code in context.portfolio.positions:
            pos = context.portfolio.positions[code]
            if pos.total_amount <= 0:
                continue
            pnl = (current_data[code].last_price - pos.avg_cost) / pos.avg_cost
            if pnl < worst_pnl:
                worst_pnl = pnl
                worst_code = code
        if worst_code:
            pattern = g.pattern_type.get(worst_code, '?')
            log.info('[替换卖出] %s(%s) 盈亏%.1f%% 让位给%.1f分候选' % (
                worst_code, pattern, worst_pnl * 100, candidates[0]['composite_score']))
            order_target(worst_code, 0)
            g.highest_since_buy.pop(worst_code, None)
            g.entry_atr.pop(worst_code, None)
            g.entry_date.pop(worst_code, None)
            g.pattern_stop.pop(worst_code, None)
            g.pattern_type.pop(worst_code, None)
            current_holds -= 1

    # ======== 第七步：买入新候选 ========
    available = context.portfolio.available_cash
    slots = max_hold - current_holds
    base_ratio = get_tier_param('base_position_ratio')

    for sig in candidates:
        if slots <= 0 or available < 2000:
            break

        code = sig['code']

        # 已持有则跳过
        if code in context.portfolio.positions and \
           context.portfolio.positions[code].total_amount > 0:
            continue

        price = current_data[code].last_price

        # 仓位计算：等权 × 波动率反比
        alloc = available / max(slots, 1) * base_ratio

        # 波动率反比调整
        target_vol = 0.25  # 个股目标波动率（比ETF的0.15高）
        actual_vol = max(sig['volatility'], 0.10)
        vol_mult = min(target_vol / actual_vol, 1.3)
        vol_mult = max(vol_mult, 0.5)
        alloc *= vol_mult

        # 评分强度调整：分越高仓位越大（70%-100%）
        strength = min(sig['composite_score'] / 5.0, 1.0)
        alloc *= (0.7 + 0.3 * strength)

        # 单只上限35%
        total_value = context.portfolio.total_value
        alloc = min(alloc, total_value * g.params['max_single_pct'])
        alloc = min(alloc, available * 0.95)

        shares = int(alloc / price / 100) * 100
        if shares < 100:
            continue

        log.info('[买入] %s 形态:%s 综合:%.1f 形态分:%d 量价:%s(+%.1f) 信号数:%d 波动率:%.0f%% %d股@%.2f' % (
            code, sig['best_pattern'], sig['composite_score'],
            sig['pattern_score'], sig['vp_type'], sig['vp_score'],
            sig['multi_signal'], sig['volatility'] * 100,
            shares, price))

        order(code, shares)

        # 记录状态
        g.highest_since_buy[code] = price
        g.entry_atr[code] = sig['atr']
        g.entry_date[code] = str(today)
        g.pattern_stop[code] = sig['stop_price']
        g.pattern_type[code] = sig['best_pattern']

        available -= shares * price * 1.003
        slots -= 1


# ============================================================
#  每日更新最高价（收盘价，不用盘中最高）
# ============================================================
def update_highest(context):
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0:
            continue
        # 使用收盘价而非盘中最高价（CLAUDE.md设计规则）
        cur = get_current_data()[code].last_price
        if code in g.highest_since_buy:
            if cur > g.highest_since_buy[code]:
                g.highest_since_buy[code] = cur
        else:
            g.highest_since_buy[code] = max(cur, pos.avg_cost)


# ============================================================
#  盘后记录
# ============================================================
def after_close(context):
    today = context.current_dt.date()
    current_data = get_current_data()
    positions = context.portfolio.positions
    hold = {code: pos for code, pos in positions.items() if pos.total_amount > 0}

    # V2 bug修复：盘后也检查状态完整性，确保不会有持仓漏掉止损
    for code, pos in hold.items():
        cur_price = current_data[code].last_price
        repair_missing_state(context, code, pos, cur_price, today)

    log.info('=' * 60)
    log.info('[%s] 总值:%.2f 现金:%.2f 持仓:%d/%d' % (
        g.current_tier,
        context.portfolio.total_value,
        context.portfolio.available_cash,
        len(hold),
        get_tier_param('max_hold')))

    for code, pos in hold.items():
        profit_pct = (pos.price - pos.avg_cost) / pos.avg_cost * 100
        highest = g.highest_since_buy.get(code, pos.price)
        pattern = g.pattern_type.get(code, 'repaired')
        entry = g.entry_date.get(code, str(today))
        stop = g.pattern_stop.get(code, pos.avg_cost * 0.95)
        log.info('  %s(%s) 入:%s 成本:%.2f 现:%.2f 高:%.2f 止:%.2f 盈亏:%.1f%%' % (
            code, pattern, entry, pos.avg_cost, pos.price, highest, stop, profit_pct))
    log.info('=' * 60)
