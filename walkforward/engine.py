# -*- coding: utf-8 -*-
"""
回测引擎核心 —— 复刻聚宽多因子V2.6的 do_trading / after_close / update_tier / 熊市检测。

关键对齐约定（与用户确认）：
- 只用日线；T日成交价用 T日开盘价 近似聚宽的9:35价。
- 信号用 T-1 日线计算（与聚宽完全一致）。
- 手续费 0.0003，min 5；滑点 PriceRelatedSlippage(0.001) → 买价×(1+0.0005)，卖价×(1-0.0005)。
- 前复权数据。
- 熊市检测用 000300 指数（与聚宽一致），需单独提供。
"""
import numpy as np
import pandas as pd
from scoring import calc_multi_factor_score
from indicators import calc_atr

# ETF池（6位代码） + 中文名 + 是否A股(熊市减仓对象)
ETF_POOL = ['510300', '159915', '512100', '159928', '510880',
            '513100', '513500', '159920', '513880', '513050',
            '518880', '159985']
A_SHARE_CODES = {'510300', '159915', '512100', '159928', '510880'}
# 模块级评分缓存，跨 Engine 实例共享，大幅加速参数扫描
_score_cache = {}
def clear_score_cache():
    _score_cache.clear()

DEFAULT_PARAMS = {
    'lookback': 120, 'rebalance_weekdays': [1, 3], 'min_hold_days': 5,
    'smooth_days': 3, 'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26,
    'macd_signal': 9, 'bb_period': 25, 'bb_std': 1.8, 'kdj_n': 9,
    'kdj_m1': 3, 'kdj_m2': 3, 'momentum_period': 20, 'vol_ma_period': 20,
    'atr_period': 14, 'trailing_atr_mult': 2.5, 'trailing_atr_mult_high_vol': 2.0,
    'high_vol_threshold': 0.30, 'stop_floor': 0.05, 'stop_cap': 0.15,
    'score_buy_threshold': 60, 'switch_threshold': 8.0,
    'profit_floor_enabled': False,
    'profit_floor_tiers': [(0.15, 0.08), (0.10, 0.05)],
    'hold_threshold': 55,   # 与买入门槛差5分惯性保护（WF验证：OOS +1.4pp，微弱正贡献）
}
BASE_WEIGHTS = {'rsi': 0.108, 'macd': 0.161, 'bollinger': 0.089, 'momentum': 0.223,
                'volume': 0.071, 'kdj': 0.108, 'ma_trend': 0.24}

# 品种级止损参数（对齐聚宽 g.code_stop_params）
DEFAULT_CODE_PARAMS = {
    '518880': {'stop_floor': 0.03, 'trailing_atr_mult': 2.0},  # 黄金ETF：均值回复型，宽止损不适用
}

COMMISSION = 0.0003
MIN_COMMISSION = 5.0
SLIP_HALF = 0.001 / 2  # PriceRelatedSlippage(0.001)


def get_tier(total):
    if total < 15000:
        return 'micro'
    elif total < 50000:
        return 'small'
    elif total < 100000:
        return 'medium'
    return 'large'


TIER_CFG = {
    'micro':  {'max_hold': 3, 'base_ratio': 0.75},
    'small':  {'max_hold': 3, 'base_ratio': 0.75},
    'medium': {'max_hold': 3, 'base_ratio': 0.75},
    'large':  {'max_hold': 3, 'base_ratio': 0.75},
}


def calc_profit_floor_price(entry_cost, highest, params):
    if not params.get('profit_floor_enabled', False):
        return None
    if entry_cost is None or highest is None or entry_cost <= 0 or highest <= 0:
        return None
    peak_profit = highest / entry_cost - 1
    for trigger_profit, locked_profit in params.get('profit_floor_tiers', []):
        if peak_profit >= trigger_profit:
            return entry_cost * (1 + locked_profit)
    return None


def calc_stop_price(highest, atr_val, params, atr_mult_override=None, profit_pct=None, entry_cost=None):
    p = params
    if atr_mult_override is not None:
        atr_mult = atr_mult_override
    else:
        vol_pct = atr_val / highest * np.sqrt(252.0 / p['atr_period'])
        if vol_pct > p['high_vol_threshold']:
            atr_mult = p['trailing_atr_mult_high_vol']
        else:
            atr_mult = p['trailing_atr_mult']
        if profit_pct is not None and profit_pct > 0:
            if profit_pct > 0.15:
                atr_mult *= 0.6
            elif profit_pct > 0.05:
                atr_mult *= 0.8
    pct_stop = atr_mult * atr_val / highest
    pct_stop = max(p['stop_floor'], min(p['stop_cap'], pct_stop))
    atr_stop = highest * (1 - pct_stop)
    floor_price = calc_profit_floor_price(entry_cost, highest, p)
    if floor_price is not None:
        return max(atr_stop, floor_price)
    return atr_stop


class Engine:
    def __init__(self, data, bench, params=None, init_cash=20000.0, verbose=False,
                 code_params=None, code_weights=None, score_map=None):
        """
        data: {code: DataFrame(index=date, cols open/close/high/low/volume)}  前复权
        bench: Series(index=date) 000300指数收盘价，用于熊市检测
        code_params: {code: {param_overrides}}  按品种覆盖参数（如 {'513100': {'momentum_period': 15}}）
        code_weights: {code: {factor_weights}}  按品种覆盖因子权重
        score_map: {(code, date): {final_score, atr, volatility, roc, close, ...}}
                   外部注入评分，跳过内部计算和缓存
        """
        self.data = data
        self.bench = bench
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.code_params = dict(DEFAULT_CODE_PARAMS)
        if code_params:
            self.code_params.update(code_params)
        self.code_weights = code_weights or {}
        self.score_map = score_map
        self.init_cash = init_cash
        self.verbose = verbose

        self.cash = init_cash
        self.positions = {}          # code -> {'amount':int,'cost':float}
        self.highest = {}            # code -> 最高价(收盘)
        self.entry_atr = {}
        self.buy_date = {}           # code -> date
        self.holding_scores = {}
        self.market_bearish = False
        self.portfolio_high = init_cash

        # 统一交易日历：用基准指数的日期（始终存在）
        self.calendar = list(bench.index)
        self._cal_idx = {d: i for i, d in enumerate(self.calendar)}
        # 每个 code 的日期→行号 索引，加速切片
        self._idx = {c: {d: i for i, d in enumerate(df.index)}
                     for c, df in data.items()}

        self.daily_value = []        # [(date, total_value)]
        self.trades = []             # 交易流水

    # ---- 取截至 end_date(含) 最后 n 根 ----
    def _hist(self, code, end_date, n):
        df = self.data.get(code)
        if df is None:
            return None
        pos = self._idx[code].get(end_date)
        if pos is None:
            # end_date 当天该ETF无数据（停牌/未上市）→ 取 <= end_date 的最后位置
            sub = df.loc[:end_date]
            if len(sub) == 0:
                return None
            pos = len(sub) - 1
        start = pos - n + 1
        if start < 0:
            return df.iloc[:pos + 1]
        return df.iloc[start:pos + 1]

    def _price(self, code, date, field):
        df = self.data.get(code)
        if df is None:
            return None
        pos = self._idx[code].get(date)
        if pos is None:
            return None
        return float(df.iloc[pos][field])

    def _has_bar(self, code, date):
        return self._idx[code].get(date) is not None

    def total_value(self, date, price_field='open'):
        tv = self.cash
        for code, p in self.positions.items():
            px = self._price(code, date, price_field)
            if px is None:  # 停牌，用最近可得收盘
                hist = self._hist(code, date, 1)
                px = float(hist['close'].iloc[-1]) if hist is not None and len(hist) else p['cost']
            tv += p['amount'] * px
        return tv

    def _trade_days_between(self, d0, d1):
        """[d0, d1] 内交易日数（含两端），用基准日历。"""
        i0 = self._cal_idx.get(d0)
        i1 = self._cal_idx.get(d1)
        if i0 is None or i1 is None:
            return len([d for d in self.calendar if d0 <= d <= d1])
        return i1 - i0 + 1

    def _detect_bear(self, prev_date):
        b = self.bench.loc[:prev_date]
        if len(b) < 61:
            self.market_bearish = False
            return
        close = b.iloc[-1]
        ma = b.iloc[-60:].mean()
        ma_prev = b.iloc[-61:-1].mean()
        self.market_bearish = (close < ma) and (ma < ma_prev)

    def _buy(self, code, shares, price):
        cost = shares * price
        comm = max(cost * COMMISSION, MIN_COMMISSION)
        self.cash -= (cost + comm)
        if code in self.positions:
            old = self.positions[code]
            tot = old['amount'] + shares
            old['cost'] = (old['cost'] * old['amount'] + cost) / tot
            old['amount'] = tot
        else:
            self.positions[code] = {'amount': shares, 'cost': price}

    def _sell_all(self, code, price):
        p = self.positions.get(code)
        if not p:
            return
        proceeds = p['amount'] * price
        comm = max(proceeds * COMMISSION, MIN_COMMISSION)
        self.cash += (proceeds - comm)
        del self.positions[code]
        for d in (self.highest, self.entry_atr, self.buy_date, self.holding_scores):
            d.pop(code, None)

    # ============ 每日主流程 ============
    def run(self):
        cal = self.calendar
        for i, today in enumerate(cal):
            if i == 0:
                continue
            prev_date = cal[i - 1]

            # --- 9:30 update_tier + 熊市检测 ---
            self._detect_bear(prev_date)

            # --- 9:35 do_trading ---
            self._do_trading(today, prev_date)

            # --- 15:30 after_close：更新最高价/ATR ---
            self._after_close(today)

            # 记录净值（收盘）
            self.daily_value.append((today, self.total_value(today, 'close')))

    def _exec_price(self, code, today, side):
        """T日开盘价近似9:35价，叠加滑点。side: 'buy'/'sell'"""
        op = self._price(code, today, 'open')
        if op is None:
            return None
        return op * (1 + SLIP_HALF) if side == 'buy' else op * (1 - SLIP_HALF)

    def _do_trading(self, today, prev_date):
        p = self.p

        # 1. 止损检测（用T日开盘价近似9:35价；停牌跳过）
        stop_triggered = []
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            if pos['amount'] <= 0 or pos['cost'] <= 0:
                continue
            if not self._has_bar(code, today):  # 停牌
                continue
            cur = self._price(code, today, 'open')
            if code in self.highest and code in self.entry_atr:
                pnl = (cur - pos['cost']) / pos['cost'] if pos['cost'] > 0 else 0
                cp = self.code_params.get(code, {})
                ep = dict(p); ep.update(cp)
                sp = calc_stop_price(
                    self.highest[code], self.entry_atr[code], ep, None, pnl,
                    pos['cost'])
                if cur <= sp:
                    stop_triggered.append(code)

        # 2. 是否轮动日
        is_rebalance = today.weekday() in p['rebalance_weekdays']
        if not is_rebalance and not stop_triggered:
            return

        # 4. 全池评分（T-1数据），带缓存；支持按品种覆盖参数/权重
        all_results = []
        if self.score_map is not None:
            for code in ETF_POOL:
                if not self._has_bar(code, today):
                    continue
                r = self.score_map.get((code, prev_date))
                if r is not None:
                    r = dict(r)
                    r['code'] = code
                    all_results.append(r)
        else:
            for code in ETF_POOL:
                if not self._has_bar(code, today):
                    continue
                cp = self.code_params.get(code, {})
                cw = self.code_weights.get(code, None)
                eff_p = dict(p)
                eff_w = BASE_WEIGHTS if cw is None else cw
                eff_p.update(cp)
                mp = eff_p['momentum_period']
                # 有品种级覆盖时不走缓存（参数/权重可能不同）
                if not cp and cw is None:
                    ck = (code, prev_date, mp)
                    if ck in _score_cache:
                        r = _score_cache[ck]
                        if r is not None:
                            r['code'] = code
                            all_results.append(r)
                        continue
                hist = self._hist(code, prev_date, eff_p['lookback'])
                r = calc_multi_factor_score(hist, eff_p, eff_w)
                if not cp and cw is None:
                    _score_cache[(code, prev_date, mp)] = r
                if r is not None:
                    r['code'] = code
                    all_results.append(r)

        if not all_results:
            for code in stop_triggered:
                px = self._exec_price(code, today, 'sell')
                if px is not None:
                    self._record_sell(code, today, px, 'stop_empty')
            return

        all_results.sort(key=lambda x: x['final_score'], reverse=True)
        score_map = {r['code']: r['final_score'] for r in all_results}
        sig_map = {r['code']: r for r in all_results}

        current_holds = {c: True for c, pos in self.positions.items() if pos['amount'] > 0}

        # 更新持仓得分
        for code in current_holds:
            if code in score_map:
                self.holding_scores[code] = score_map[code]

        # 5. 换仓逻辑
        threshold = p['score_buy_threshold']
        switch_th = p['switch_threshold']
        min_hold = p['min_hold_days']
        hold_th = p['hold_threshold']
        max_hold = TIER_CFG[get_tier(self.total_value(today, 'open'))]['max_hold']

        candidates = [r for r in all_results if r['final_score'] > threshold]

        target_codes = set()
        protected_codes = set()
        for code in list(current_holds.keys()):
            if code in self.buy_date:
                days_held = self._trade_days_between(self.buy_date[code], today)
                if days_held <= min_hold:
                    target_codes.add(code)
                    protected_codes.add(code)
                    continue
            if self.holding_scores.get(code, 0) > hold_th:
                target_codes.add(code)

        for r in candidates:
            if len(target_codes) >= max_hold:
                break
            if r['code'] not in target_codes:
                target_codes.add(r['code'])

        # 换仓门槛
        if len(target_codes) >= max_hold:
            removable = [(c, self.holding_scores.get(c, 0))
                         for c in target_codes
                         if c in current_holds and c not in protected_codes]
            removable.sort(key=lambda x: x[1])
            for r in candidates:
                if r['code'] in target_codes or not removable:
                    continue
                worst_code, worst_score = removable[0]
                if r['final_score'] > worst_score + switch_th:
                    target_codes.discard(worst_code)
                    target_codes.add(r['code'])
                    removable.pop(0)

        # 6. 执行止损（在target且MA10趋势未破→豁免；否则止损）
        force_stopped = set()
        for code in stop_triggered:
            if code in target_codes:
                cur = self._price(code, today, 'open')
                trend_broken = False
                ma_hist = self._hist(code, prev_date, 15)
                if ma_hist is not None and len(ma_hist) >= 11:
                    ma10 = ma_hist['close'].iloc[-10:].mean()
                    ma10_prev = ma_hist['close'].iloc[-11:-1].mean()
                    if cur < ma10 and ma10 < ma10_prev:
                        trend_broken = True
                if trend_broken:
                    px = self._exec_price(code, today, 'sell')
                    self._record_sell(code, today, px, 'trend_stop')
                    force_stopped.add(code)
                # else 豁免，保留
            else:
                px = self._exec_price(code, today, 'sell')
                self._record_sell(code, today, px, 'stop')

        # 7. 轮动卖出
        for code in list(current_holds.keys()):
            if code not in target_codes and code not in stop_triggered:
                if not self._has_bar(code, today):  # 停牌跳过
                    continue
                px = self._exec_price(code, today, 'sell')
                self._record_sell(code, today, px, 'rotate')

        # 8. 买入
        to_buy = [c for c in target_codes
                  if c not in current_holds and c not in force_stopped]
        if not to_buy:
            return
        to_buy.sort(key=lambda c: sig_map.get(c, {}).get('final_score', 0), reverse=True)

        available = self.cash
        actual_hold = len([c for c, pos in self.positions.items() if pos['amount'] > 0])
        slots = max_hold - actual_hold
        if slots <= 0 or available < 500:
            return

        base_ratio = TIER_CFG[get_tier(self.total_value(today, 'open'))]['base_ratio']

        for code in to_buy:
            if slots <= 0 or available < 500:
                break
            if code not in sig_map:
                continue
            sig = sig_map[code]
            price = self._exec_price(code, today, 'buy')
            if price is None:
                continue

            alloc = available / slots * base_ratio
            actual_vol = max(sig['volatility'], 0.05)
            alloc *= max(0.4, min(1.5, 0.15 / actual_vol))
            alloc = min(alloc, available * 0.95)
            if self.market_bearish and code in A_SHARE_CODES:
                alloc *= 0.5

            shares = int(alloc / price / 100) * 100
            if shares < 100:
                if available >= price * 100 * 1.003:
                    shares = 100
                else:
                    continue

            self._buy(code, shares, price)
            self.highest[code] = price
            self.entry_atr[code] = sig['atr']
            self.buy_date[code] = today
            self.holding_scores[code] = sig['final_score']
            self.trades.append((today, code, 'buy', shares, price, sig['final_score']))
            available -= shares * price * 1.003
            slots -= 1

    def _record_sell(self, code, today, px, reason):
        pos = self.positions.get(code)
        if not pos:
            return
        pnl = (px - pos['cost']) / pos['cost'] if pos['cost'] > 0 else 0
        self.trades.append((today, code, 'sell_' + reason, pos['amount'], px, pnl))
        self._sell_all(code, px)

    def _after_close(self, today):
        p = self.p
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            if pos['amount'] <= 0:
                continue
            cur = self._price(code, today, 'close')
            if cur is None:  # 停牌，跳过更新
                continue
            # 更新最高价
            if code in self.highest:
                if cur > self.highest[code]:
                    self.highest[code] = cur
            else:
                self.highest[code] = max(cur, pos['cost'])
            # 动态更新ATR
            if code in self.entry_atr:
                atr_df = self._hist(code, today, p['atr_period'] + 5)
                if atr_df is not None and len(atr_df) >= p['atr_period']:
                    new_atr = calc_atr(atr_df['high'], atr_df['low'],
                                       atr_df['close'], p['atr_period']).iloc[-1]
                    if not pd.isna(new_atr) and new_atr > 0:
                        self.entry_atr[code] = new_atr

        tv = self.total_value(today, 'close')
        if tv > self.portfolio_high:
            self.portfolio_high = tv

    # ============ 绩效统计 ============
    def stats(self):
        if not self.daily_value:
            return {}
        dates = [d for d, _ in self.daily_value]
        vals = np.array([v for _, v in self.daily_value])
        total_ret = vals[-1] / self.init_cash - 1
        years = (dates[-1] - dates[0]).days / 365.25
        cagr = (vals[-1] / self.init_cash) ** (1 / years) - 1 if years > 0 else 0
        # 最大回撤
        peak = np.maximum.accumulate(vals)
        dd = (peak - vals) / peak
        max_dd = dd.max()
        # 夏普（日频）
        rets = np.diff(vals) / vals[:-1]
        sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
        return {
            'total_return': total_ret, 'cagr': cagr, 'max_dd': max_dd,
            'sharpe': sharpe, 'final_value': vals[-1],
            'n_trades': len([t for t in self.trades if t[2] == 'buy']),
            'start': dates[0], 'end': dates[-1],
        }

    def yearly_returns(self):
        df = pd.DataFrame(self.daily_value, columns=['date', 'value']).set_index('date')
        df.index = pd.to_datetime(df.index)
        yearly = df['value'].resample('YE').last()
        start_val = pd.Series([self.init_cash],
                              index=[df.index[0] - pd.Timedelta(days=1)])
        full = pd.concat([start_val, yearly])
        return full.pct_change().dropna()
