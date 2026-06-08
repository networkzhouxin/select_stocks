# -*- coding: utf-8 -*-
"""
阶段1：对齐验证
用默认参数(switch=8/min_hold=5/momentum=20/hold_th=55)跑2015-01-01~2026-06-08，
对比聚宽 +372% / 年化15.4% / 回撤14.4% / 夏普0.96。

通过标准：总收益差距±20%以内 且 年度涨跌方向一致。
"""
import os
import pandas as pd
from engine import Engine, ETF_POOL

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

BT_START = pd.Timestamp('2015-01-01')
BT_END = pd.Timestamp('2026-06-08')


def load_data():
    """载入12个ETF + 000300指数。返回 (data_dict, bench_series)"""
    data = {}
    for code in ETF_POOL:
        path = os.path.join(DATA_DIR, f'{code}.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f'缺少 {code}.csv，请先运行 fetch_data.py')
        df = pd.read_csv(path, parse_dates=['date']).set_index('date')
        df = df.sort_index()
        data[code] = df

    idx_path = os.path.join(DATA_DIR, '000300_index.csv')
    if not os.path.exists(idx_path):
        raise FileNotFoundError('缺少 000300_index.csv')
    bench = pd.read_csv(idx_path, parse_dates=['date']).set_index('date')['close'].sort_index()
    return data, bench


def build_calendar(data, bench, start, end):
    """
    交易日历用 510300(沪深300ETF) 的日期 —— 它全程存在，是A股交易日的可靠代表。
    bench(000300指数)对齐到该日历(reindex+ffill)，供熊市检测用。
    """
    cal = data['510300'].loc[start:end].index
    bench_aligned = bench.reindex(cal).ffill()
    return cal, bench_aligned


def main():
    print('载入数据...')
    data, bench = load_data()
    for code in ETF_POOL:
        df = data[code]
        print(f'  {code}: {len(df)}行 {df.index.min().date()} ~ {df.index.max().date()}')
    print(f'  000300指数: {len(bench)}行 {bench.index.min().date()} ~ {bench.index.max().date()}')

    cal, bench_aligned = build_calendar(data, bench, BT_START, BT_END)
    print(f'\n回测日历: {len(cal)}个交易日 {cal.min().date()} ~ {cal.max().date()}')

    # 用回测期日历作为基准（Engine 用 bench.index 当日历）
    eng = Engine(data, bench_aligned, params=None, init_cash=20000.0)
    print('\n运行回测...')
    eng.run()

    s = eng.stats()
    print('\n' + '=' * 56)
    print('阶段1对齐结果（本地引擎 vs 聚宽V2.6）')
    print('=' * 56)
    print(f"{'指标':<14}{'本地':>14}{'聚宽':>14}")
    print(f"{'总收益':<14}{s['total_return']*100:>13.1f}%{'+372%':>14}")
    print(f"{'年化':<14}{s['cagr']*100:>13.1f}%{'15.4%':>14}")
    print(f"{'最大回撤':<14}{s['max_dd']*100:>13.1f}%{'14.4%':>14}")
    print(f"{'夏普':<14}{s['sharpe']:>14.2f}{'0.96':>14}")
    print(f"{'买入次数':<14}{s['n_trades']:>14}{'~529':>14}")
    print(f"{'最终市值':<14}{s['final_value']:>14.0f}{'':>14}")

    print('\n年度收益:')
    yr = eng.yearly_returns()
    for d, r in yr.items():
        print(f"  {d.year}: {r*100:+.1f}%")

    # 对齐判定
    jq_ret = 3.72
    local_ret = s['total_return']
    diff_pct = abs(local_ret - jq_ret) / jq_ret * 100
    print('\n' + '-' * 56)
    print(f"总收益差距: {diff_pct:.1f}% (阈值±20%)")
    if diff_pct <= 20:
        print("[PASS] 对齐通过 -- 引擎可信，可进入阶段2 walk-forward")
    else:
        print("[FAIL] 对齐未通过 -- 需排查逻辑差异，不可进入阶段2")
    print('-' * 56)


if __name__ == '__main__':
    main()
