# -*- coding: utf-8 -*-
"""
小牛量化选股策略 V1.0 - 基于《小牛特训营学习精要》的个股短线策略
=============================================
将PDF中最具量化潜力的5大战法提炼为量化信号，组合打分选股。

交易范围：仅沪深主板（60xxxx/00xxxx），不含创业板/科创板/北交所

核心信号源（来自PDF）：
  1. 定庄起爆（立庄K线识别+缩量回调+放量突破）    — 权重最高
  2. 均线粘合发散（5/10/20/60MA收敛后向上发散）    — 中期趋势信号
  3. 龙回头（首波涨30%+缩量A字回调+长下影转折）   — 二波主升信号
  4. 高量柱（倍量柱实底实顶+价格收复实顶）         — 量价博弈信号
  5. 量价确认（缩量上涨/放量突破/缩量回调）        — 辅助加分项

风控特点（比ETF策略更紧）：
  - ATR跟踪止损：2.0x ATR，止损夹紧3%-8%（ETF为2.5x/3%-15%）
  - 形态止损：跌破各形态关键支撑位直接出局
  - 时间止损：持仓超10个交易日强制清仓
  - 单只上限30%仓位，最多5只持仓

回测建议：
  起始日期：2020-01-01（覆盖疫情底/牛市/震荡）
  结束日期：2025-12-31
  初始资金：100,000元
  基准：000300.XSHG
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

    # ---- 资金档位配置 ----
    g.capital_tiers = {
        'small':  {'max_hold': 3, 'base_position_ratio': 0.28},
        'medium': {'max_hold': 4, 'base_position_ratio': 0.22},
        'large':  {'max_hold': 5, 'base_position_ratio': 0.18},
    }

    # ---- 策略参数（全部使用学术默认值，不做参数优化）----
    g.params = {
        # 选股宇宙
        'min_market_cap': 30e8,          # 最低市值30亿（过滤壳股/微盘股）
        'min_avg_amount': 5e6,           # 近5日日均成交额 > 500万
        'ipo_min_days': 120,             # 上市满120天（过滤次新股）
        'pre_filter_count': 150,         # 预筛选股票数量上限

        # 定庄起爆
        'lizhuang_vol_mult': 2.0,        # 立庄K线：量>前日2倍
        'lizhuang_body_pct': 0.05,       # 立庄K线：实体>5%
        'lizhuang_lookback': 30,         # 立庄K线回看30天
        'lizhuang_pullback_vol': 0.8,    # 回调缩量：量<5日均量80%

        # 均线粘合
        'ma_converge_pct': 0.03,         # MA收敛范围<3%
        'ma_vol_expand': 1.2,            # 量能扩张：5日均量>90日均量*1.2

        # 龙回头
        'dragon_wave_pct': 0.25,         # 首波涨幅>=25%
        'dragon_pullback_pct': 0.10,     # 回调幅度>=10%
        'dragon_shadow_ratio': 2.0,      # 下影线>实体2倍

        # 高量柱
        'high_vol_mult': 2.5,            # 高量柱：量>=前日2.5倍
        'high_vol_lookback': 20,         # 高量柱回看20天

        # 风控
        'atr_period': 14,
        'trailing_atr_mult': 2.0,        # 跟踪止损2.0x ATR（比ETF更紧）
        'stop_floor': 0.03,              # 止损下限3%
        'stop_cap': 0.08,                # 止损上限8%（比ETF的15%紧很多）
        'max_hold_days': 10,             # 时间止损：10个交易日
        'cooldown_days': 3,              # 冷却期3天
        'max_single_pct': 0.30,          # 单只最大仓位30%

        # 评分门槛
        'min_composite_score': 2.0,      # 综合评分>=2.0才买入
    }

    # ---- 状态变量 ----
    g.current_tier = None
    g.highest_since_buy = {}     # {code: highest_close} ATR跟踪止损
    g.entry_atr = {}             # {code: atr_at_entry}
    g.entry_date = {}            # {code: date} 时间止损
    g.pattern_stop = {}          # {code: stop_price} 形态止损
    g.pattern_type = {}          # {code: str} 持仓对应的形态类型
    g.cooldown = {}              # {code: date} 冷却期

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
    if total < 50000:
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
#  选股宇宙过滤
# ============================================================
def get_stock_universe(context, prev_date):
    """
    多重过滤：非ST、非停牌、上市满120天、市值>30亿、日均成交>500万。
    然后从中预筛出有近期量价异动的标的（减少全量扫描开销）。
    """
    p = g.params

    # 基础过滤：仅主板（沪市60xxxx + 深市00xxxx），排除创业板(30)/科创板(68)/北交所(8/4)
    all_stocks = get_all_securities('stock', date=prev_date)
    all_stocks = all_stocks[all_stocks.index.str[:2].isin(['60', '00'])]
    # 上市满120天
    cutoff_date = pd.Timestamp(prev_date) - pd.Timedelta(days=p['ipo_min_days'])
    stocks = all_stocks[all_stocks['start_date'] <= cutoff_date.date()].index.tolist()

    # ST过滤
    st_data = get_extras('is_st', stocks, start_date=prev_date, end_date=prev_date, df=True)
    if not st_data.empty:
        non_st = st_data.columns[~st_data.iloc[0]].tolist()
    else:
        non_st = stocks

    # 停牌过滤
    current_data = get_current_data()
    non_paused = [s for s in non_st if not current_data[s].paused]

    # 市值过滤
    q = query(
        valuation.code,
        valuation.market_cap,           # 总市值（亿元）
    ).filter(
        valuation.code.in_(non_paused),
        valuation.market_cap > p['min_market_cap'] / 1e8,   # 转为亿元
    )
    df_val = get_fundamentals(q, date=prev_date)
    if df_val.empty:
        return []
    valid_codes = df_val['code'].tolist()

    # 日均成交额过滤 + 量价异动预筛
    # 获取近5日价格数据用于预筛
    pre_filter = []
    # 分批获取避免单次请求过大
    batch_size = 200
    for i in range(0, len(valid_codes), batch_size):
        batch = valid_codes[i:i + batch_size]
        panel = get_price(batch, end_date=prev_date, count=10,
                         frequency='daily', fields=['close', 'volume', 'money'],
                         skip_paused=True, fq='pre', panel=False)
        if panel is None or panel.empty:
            continue

        for code in batch:
            df_code = panel[panel['code'] == code]
            if len(df_code) < 5:
                continue

            # 日均成交额（近5日）
            avg_amount = df_code['money'].iloc[-5:].mean()
            if avg_amount < p['min_avg_amount']:
                continue

            # 预筛：近5日内有量能放大（当日量>前日1.5倍）或价格创10日新高
            vol = df_code['volume'].values
            close = df_code['close'].values

            has_vol_spike = any(vol[j] > vol[j - 1] * 1.5 for j in range(-5, 0) if j - 1 >= -len(vol))
            near_high = close[-1] >= np.max(close) * 0.95  # 接近10日最高价

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
        'stop_price': max(lk_low, ema32[-1] * 0.98),
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
        'stop_price': min(ma20.iloc[-1], ma60.iloc[-1]),
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
        'stop_price': hv_support * 0.97,
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
    对选股宇宙中的每只股票跑5大信号检测器，综合评分排名。
    返回按评分降序排列的候选列表。
    """
    candidates = []

    for code in universe:
        try:
            if current_data[code].paused:
                continue

            # 冷却期检查
            if code in g.cooldown:
                if (pd.Timestamp(prev_date) - pd.Timestamp(g.cooldown[code])).days < g.params['cooldown_days']:
                    continue

            # 获取120日K线数据（T-1，绝不用当日数据生成信号）
            df = get_price(code, end_date=prev_date, count=120,
                          frequency='daily',
                          fields=['open', 'close', 'high', 'low', 'volume', 'money'],
                          skip_paused=True, fq='pre')

            if df is None or len(df) < 60:
                continue

            # 运行5大检测器
            sig_lizhuang = detect_lizhuang(df)
            sig_ma = detect_ma_convergence(df)
            sig_dragon = detect_dragon_pullback(df)
            sig_highvol = detect_high_volume_bar(df)
            sig_vp = detect_volume_price(df)

            # 取最强形态信号
            signals = []
            if sig_lizhuang:
                signals.append(sig_lizhuang)
            if sig_ma:
                signals.append(sig_ma)
            if sig_dragon:
                signals.append(sig_dragon)
            if sig_highvol:
                signals.append(sig_highvol)

            if not signals:
                continue

            # 最强信号
            best = max(signals, key=lambda s: s['score'])

            # 综合评分 = 最强形态分 + 量价加分 + 多信号加分
            composite = best['score']
            composite += sig_vp['vp_score']
            if len(signals) >= 2:
                composite += 0.5  # 多信号共振加分

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
        profit_pct = (cur_price - pos.avg_cost) / pos.avg_cost
        stop_reason = None

        # 1. 时间止损：持仓超过max_hold_days个交易日
        if code in g.entry_date:
            hold_days = len(get_trade_days(
                start_date=g.entry_date[code], end_date=today))
            if hold_days > g.params['max_hold_days']:
                stop_reason = '时间止损(%d天)' % hold_days

        # 2. 形态止损：跌破形态关键支撑位
        if stop_reason is None and code in g.pattern_stop:
            if cur_price <= g.pattern_stop[code]:
                stop_reason = '形态止损(破%.2f)' % g.pattern_stop[code]

        # 3. ATR跟踪止损
        if stop_reason is None:
            if code in g.highest_since_buy and code in g.entry_atr:
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

    # ======== 第一步：每日止损检查（每天都执行）========
    stopped_codes = check_stop_loss(context, current_data, today)

    # ======== 第二步：检查是否有空仓位需要填充 ========
    max_hold = get_tier_param('max_hold')
    current_holds = sum(1 for code in context.portfolio.positions
                       if context.portfolio.positions[code].total_amount > 0)

    # 每天扫描（个股机会稍纵即逝，不做轮动间隔）
    if current_holds >= max_hold and not stopped_codes:
        return  # 满仓且无止损，跳过

    # ======== 第三步：选股宇宙过滤 ========
    universe = get_stock_universe(context, prev_date)
    if not universe:
        return

    # ======== 第四步：综合评分扫描 ========
    candidates = scan_and_rank(universe, prev_date, current_data)
    if not candidates:
        return

    # ======== 第五步：卖出评分过低的持仓（形态已破坏）========
    # 对已持仓股重新评估，如果综合分<1.0则卖出
    held_codes = {c['code'] for c in candidates}
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0:
            continue
        if code in held_codes:
            continue  # 仍在候选中，保留

        # 持仓不在候选中，检查是否需要卖出
        # 只在有更好候选时才替换
        if current_holds > max_hold or (candidates and candidates[0]['composite_score'] > 3.0):
            profit_pct = (current_data[code].last_price - pos.avg_cost) / pos.avg_cost
            pattern = g.pattern_type.get(code, '?')
            log.info('[形态失效卖出] %s(%s) 盈亏%.1f%%' % (code, pattern, profit_pct * 100))
            order_target(code, 0)
            g.highest_since_buy.pop(code, None)
            g.entry_atr.pop(code, None)
            g.entry_date.pop(code, None)
            g.pattern_stop.pop(code, None)
            g.pattern_type.pop(code, None)
            current_holds -= 1

    # ======== 第六步：买入新候选 ========
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

        # 单只上限30%
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
    positions = context.portfolio.positions
    hold = {code: pos for code, pos in positions.items() if pos.total_amount > 0}

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
        pattern = g.pattern_type.get(code, '?')
        entry = g.entry_date.get(code, '?')
        stop = g.pattern_stop.get(code, 0)
        log.info('  %s(%s) 入:%s 成本:%.2f 现:%.2f 高:%.2f 止:%.2f 盈亏:%.1f%%' % (
            code, pattern, entry, pos.avg_cost, pos.price, highest, stop, profit_pct))
    log.info('=' * 60)
