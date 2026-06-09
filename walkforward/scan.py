# -*- coding: utf-8 -*-
"""
阶段2：Walk-Forward 参数扫描

滚动窗口：训练2015-2018测2019 ... 训练2015-2024测2025（7个窗口）
4个载荷参数，各5个取值，每次只变1个、固定其余3个→ 140次回测
多线程并行，输出参数稳定性+敏感性分析

指标选择：用训练集总收益选最优，也记录夏普/回撤作为参考
"""
import os, sys, time, itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

from engine import Engine, ETF_POOL, DEFAULT_PARAMS, BASE_WEIGHTS, clear_score_cache
from precompute import precompute_all

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

WINDOWS = [
    ('2015-01-01', '2018-12-31', '2019-01-01', '2019-12-31'),
    ('2015-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
    ('2015-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('2015-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2015-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2015-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ('2015-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),
    ('2015-01-01', '2025-12-31', '2026-01-01', '2026-03-11'),
]

# 单参数扫描：每次只变1个参数
PARAM_SWEEPS = {
    'switch_threshold': [4, 6, 8, 10, 12],
    'min_hold_days':    [2, 3, 5, 7, 10],
    'momentum_period':  [10, 15, 20, 25, 30],
    'hold_threshold':   [50, 52, 55, 58, 60],
}

# ---- 全局数据（只读，线程安全） ----
_global_data = None
_global_bench = None


def load_global():
    global _global_data, _global_bench
    if _global_data is not None:
        return _global_data, _global_bench
    _global_data = {}
    for code in ETF_POOL:
        path = os.path.join(DATA_DIR, f'{code}.csv')
        _global_data[code] = pd.read_csv(path, parse_dates=['date']).set_index('date')
    _global_bench = pd.read_csv(os.path.join(DATA_DIR, '000300_index.csv'),
                                parse_dates=['date']).set_index('date')['close']
    return _global_data, _global_bench


def run_one(params, train_start, train_end):
    """单次回测，返回 (总收益, 年化, 回撤, 夏普, 买入次数)"""
    data, bench = _global_data, _global_bench
    cal = data['510300'].loc[pd.Timestamp(train_start):pd.Timestamp(train_end)].index
    bench_slice = bench.reindex(cal).ffill()

    eng = Engine(data, bench_slice, params=params, init_cash=20000.0)
    eng.run()
    s = eng.stats()
    return (s['total_return'], s['cagr'], s['max_dd'], s['sharpe'], s['n_trades'])


def scan_window(window, base_params, n_workers=4):
    """扫描一个窗口：对4个参数各自扫描5个值，得到最优参数"""
    train_s, train_e, test_s, test_e = window
    window_label = f'Train{train_s[:4]}-{train_e[:4]} Test{test_s[:4]}'

    results = {}
    all_tasks = []

    for param_name, values in PARAM_SWEEPS.items():
        for val in values:
            p = dict(base_params)
            p[param_name] = val
            all_tasks.append((param_name, val, p))

    # 多线程并行跑
    train_results = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for param_name, val, p in all_tasks:
            key = (param_name, val)
            f = ex.submit(run_one, p, train_s, train_e)
            futures[f] = key

        done = 0
        for f in as_completed(futures):
            key = futures[f]
            try:
                ret, cagr, dd, sharpe, ntr = f.result()
                train_results[key] = (ret, cagr, dd, sharpe, ntr)
            except Exception as e:
                print(f'  ERR {key}: {e}')
            done += 1
            if done % 10 == 0:
                print(f'  [{done}/{len(all_tasks)}] {done/len(all_tasks)*100:.0f}%')

    # 对每个参数，找训练集最优值，然后在测试集上验证
    window_output = {}
    for param_name in PARAM_SWEEPS:
        best_val = None
        best_ret = -999
        param_table = []
        for val in PARAM_SWEEPS[param_name]:
            key = (param_name, val)
            if key not in train_results:
                continue
            ret, cagr, dd, sharpe, ntr = train_results[key]
            param_table.append((val, ret, cagr, dd, sharpe, ntr))
            if ret > best_ret:
                best_ret = ret
                best_val = val

        # 用最优参数跑测试集
        test_p = dict(base_params)
        test_p[param_name] = best_val
        test_ret, test_cagr, test_dd, test_sharpe, test_ntr = run_one(test_p, test_s, test_e)

        window_output[param_name] = {
            'best_value': best_val,
            'train_ret': best_ret,
            'test_ret': test_ret,
            'test_cagr': test_cagr,
            'test_dd': test_dd,
            'test_sharpe': test_sharpe,
            'param_table': param_table,
        }

    return window_label, window_output


def analyze(all_results):
    """汇总所有窗口结果，输出参数稳定性+敏感性分析"""
    param_names = list(PARAM_SWEEPS.keys())

    print('\n' + '=' * 72)
    print('Walk-Forward 参数扫描报告')
    print('=' * 72)

    # ---- 1. 参数稳定性表 ----
    print('\n--- 1. 参数稳定性：各窗口训练集最优值 ---')
    header = f"{'窗口':<24}"
    for pn in param_names:
        header += f' {pn:>16}'
    print(header)
    print('-' * (24 + 17 * len(param_names)))

    stability = {pn: [] for pn in param_names}
    for wl, wo in all_results:
        row = f'{wl:<24}'
        for pn in param_names:
            bv = wo[pn]['best_value']
            tr = wo[pn]['train_ret']
            stability[pn].append(bv)
            row += f' {bv:>4}({tr*100:+.0f}%)'
        print(row)

    # 统计每个参数的最优值分布
    print('\n--- 参数最优值分布 ---')
    for pn in param_names:
        vals = stability[pn]
        from collections import Counter
        cnt = Counter(vals)
        dominant = cnt.most_common(1)[0]
        pct = dominant[1] / len(vals) * 100
        status = 'STABLE' if pct >= 70 else ('UNSTABLE' if pct <= 40 else 'MODERATE')
        print(f'  {pn:20}: 最频值={dominant[0]} ({dominant[1]}/{len(vals)}窗={pct:.0f}%) [{status}]')
        print(f'    {"":20}  各窗选择: {vals}')

    # ---- 2. 样本外测试收益 ----
    print('\n--- 2. 样本外(OOS)测试期收益（用训练集最优参数） ---')
    header = '{:<24}'.format('窗口')
    for pn in param_names:
        header += ' {:>14}'.format(pn)
    print(header)
    print('-' * (24 + 15 * len(param_names)))
    oos_returns = {pn: [] for pn in param_names}
    for wl, wo in all_results:
        row = '{:<24}'.format(wl)
        for pn in param_names:
            tr = wo[pn]['test_ret']
            oos_returns[pn].append(tr)
            row += ' {:>+13.1f}%'.format(tr * 100)
        print(row)

    # 平均OOS
    row = '{:<24}'.format('平均OOS')
    for pn in param_names:
        avg_oos = np.mean(oos_returns[pn])
        row += ' {:>+13.1f}%'.format(avg_oos * 100)
    print('-' * (24 + 15 * len(param_names)))
    print(row)

    # ---- 3. 单参数敏感性曲线（汇总所有窗口） ----
    print('\n--- 3. 单参数敏感性（汇总所有窗口） ---')
    for pn in param_names:
        all_vals = PARAM_SWEEPS[pn]
        print(f'\n  {pn} (默认值: {DEFAULT_PARAMS[pn]}):')
        # 每窗口每参数值的训练收益
        for val in all_vals:
            rets = []
            for wl, wo in all_results:
                for v, r, c, d, s, n in wo[pn]['param_table']:
                    if abs(v - val) < 0.01:
                        rets.append(r)
                        break
            if rets:
                avg_ret = np.mean(rets)
                std_ret = np.std(rets)
                marker = ' <-- default' if abs(val - DEFAULT_PARAMS[pn]) < 0.01 else ''
                print(f'    {val:>5}: avg={avg_ret*100:+5.1f}% std={std_ret*100:.1f}% (n={len(rets)}){marker}')

    print('\n' + '=' * 72)
    print('分析完成')
    print('=' * 72)


def main():
    t0 = time.time()
    data, bench = load_global()
    base = dict(DEFAULT_PARAMS)

    # 预计算全部评分（一次性），之后所有窗口查表即用
    print('预计算评分...')
    precompute_all(data, bench, end='2026-03-11')

    all_results = []
    window_times = []
    for i, win in enumerate(WINDOWS):
        wl = f'Train{win[0][:4]}-{win[1][:4]} Test{win[2][:4]}'
        print(f'\n[{i+1}/8] {wl}')
        t1 = time.time()
        wl, wo = scan_window(win, base)
        elapsed = time.time() - t1
        window_times.append(elapsed)
        all_results.append((wl, wo))
        avg_time = np.mean(window_times)
        remaining = avg_time * (8 - i - 1)
        print(f'  耗时: {elapsed:.0f}s | 预计剩余: {remaining/60:.0f}min')

    analyze(all_results)
    print(f'\n总耗时: {(time.time()-t0)/60:.1f}分钟')


if __name__ == '__main__':
    main()
