# -*- coding: utf-8 -*-
"""
预计算所有评分并缓存到模块级 dict，大幅加速参数扫描。
一次计算覆盖所有 ETF × 交易日 × momentum_period → engine 只做查表。
"""
import os, time
import pandas as pd
import numpy as np
from scoring import calc_multi_factor_score
from engine import ETF_POOL, DEFAULT_PARAMS, BASE_WEIGHTS, _score_cache, clear_score_cache

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

MOMENTUM_VALUES = [10, 15, 20, 25, 30]


def precompute_all(data, bench, start='2015-01-01', end='2026-06-08'):
    """
    预计算全部评分，写入模块级 _score_cache。
    key = (code, date, momentum_period)
    """
    cal = data['510300'].loc[pd.Timestamp(start):pd.Timestamp(end)].index
    params = dict(DEFAULT_PARAMS)
    base_weights = dict(BASE_WEIGHTS)

    total = len(cal) * len(ETF_POOL) * len(MOMENTUM_VALUES)
    done = 0
    t0 = time.time()

    for mp in MOMENTUM_VALUES:
        params['momentum_period'] = mp
        for code in ETF_POOL:
            df = data[code]
            for d in cal:
                ck = (code, d, mp)
                if ck in _score_cache:
                    done += 1
                    continue
                pos = _find_pos(df, d)
                if pos is None or pos < params['lookback'] - 10:
                    done += 1
                    continue
                start_i = pos - params['lookback'] + 1
                if start_i < 0:
                    hist = df.iloc[:pos + 1]
                else:
                    hist = df.iloc[start_i:pos + 1]
                r = calc_multi_factor_score(hist, params, base_weights)
                _score_cache[ck] = r
                done += 1
                if done % 5000 == 0:
                    elapsed = time.time() - t0
                    pct = done / total * 100
                    eta = elapsed / done * (total - done)
                    print(f'  [{done}/{total}] {pct:.0f}% 已耗时{elapsed:.0f}s 预计剩余{eta:.0f}s')

    elapsed = time.time() - t0
    print(f'预计算完成: {len(_score_cache)}条, 耗时{elapsed:.0f}s')
    return _score_cache


def _find_pos(df, date):
    """在 DataFrame 索引中查找 date 的位置"""
    idx = df.index
    if date in idx:
        return idx.get_loc(date)
    # date 不在索引中（如该 ETF 当日停牌/未上市），取 <= date 的最近位置
    sub = idx[idx <= date]
    return len(sub) - 1 if len(sub) > 0 else None


if __name__ == '__main__':
    print('加载数据...')
    data = {}
    for code in ETF_POOL:
        path = os.path.join(DATA_DIR, f'{code}.csv')
        data[code] = pd.read_csv(path, parse_dates=['date']).set_index('date')
    bench = pd.read_csv(os.path.join(DATA_DIR, '000300_index.csv'),
                        parse_dates=['date']).set_index('date')['close']

    print(f'开始预计算: {len(MOMENTUM_VALUES)} momentum值 x {len(ETF_POOL)} ETF x ~2800交易日')
    precompute_all(data, bench)
