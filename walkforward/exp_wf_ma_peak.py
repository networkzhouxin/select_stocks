# -*- coding: utf-8 -*-
"""
Walk-Forward MA权重峰值验证（单线程优化版）

优化：
1. 复用 precompute_all() 预计算 _score_cache（已含独立因子得分）
2. 单线程执行，避免 GIL 竞争（实测单线程比多线程快3倍）
3. 预构建各窗口日历，避免重复 pandas 切片
"""
import os, sys, time
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine, ETF_POOL, DEFAULT_PARAMS, BASE_WEIGHTS
from scoring import calc_multi_factor_score

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

MA_WEIGHTS = [round(float(x), 2) for x in np.arange(0.08, 0.36, 0.01)]
FACTOR_KEYS = ['rsi', 'macd', 'bollinger', 'momentum', 'volume', 'kdj', 'ma_trend']
SCORE_KEYS = ['rsi_score', 'macd_score', 'bollinger_score', 'momentum_score',
              'volume_score', 'kdj_score', 'ma_trend_score']
AUX_KEYS = ['atr', 'volatility', 'roc', 'close']


def rebalance_weights(ma_weight, base_weights):
    other_total = sum(base_weights[k] for k in FACTOR_KEYS if k != 'ma_trend')
    scale = (1.0 - ma_weight) / other_total
    return {k: ma_weight if k == 'ma_trend' else base_weights[k] * scale
            for k in FACTOR_KEYS}


def build_score_map(factor_cache, weights):
    score_map = {}
    for (code, date), entry in factor_cache.items():
        final_score = sum(entry[sk] * weights[fk]
                          for sk, fk in zip(SCORE_KEYS, FACTOR_KEYS))
        score_map[(code, date)] = {
            'final_score': final_score,
            'atr': entry['atr'], 'volatility': entry['volatility'],
            'roc': entry['roc'], 'close': entry['close'],
        }
    return score_map


def run_engine(data, bench_slice, score_map):
    eng = Engine(data, bench_slice, params=DEFAULT_PARAMS, init_cash=20000.0,
                 score_map=score_map)
    eng.run()
    return eng.stats()


def main():
    t0 = time.time()

    # 加载数据
    print('加载数据...')
    data = {}
    for code in ETF_POOL:
        path = os.path.join(DATA_DIR, f'{code}.csv')
        data[code] = pd.read_csv(path, parse_dates=['date']).set_index('date')
    bench = pd.read_csv(os.path.join(DATA_DIR, '000300_index.csv'),
                        parse_dates=['date']).set_index('date')['close']

    # Step 1: 预计算因子得分（仅 mp=20，单线程）
    print('预计算因子得分 (12 ETFs × ~2700天, mp=20)...')
    params = dict(DEFAULT_PARAMS)
    full_cal = data['510300'].loc['2015-01-01':'2026-03-11'].index
    factor_cache = {}
    total = len(ETF_POOL) * len(full_cal)
    done = 0
    t_pre = time.time()
    for code in ETF_POOL:
        df = data[code]
        for d in full_cal:
            if d in df.index:
                pos = df.index.get_loc(d)
            else:
                sub = df.index[df.index <= d]
                pos = len(sub) - 1 if len(sub) > 0 else None
            if pos is None or pos < params['lookback'] - 10:
                done += 1
                continue
            start_i = pos - params['lookback'] + 1
            if start_i < 0:
                hist = df.iloc[:pos + 1]
            else:
                hist = df.iloc[start_i:pos + 1]
            r = calc_multi_factor_score(hist, params, BASE_WEIGHTS)
            if r is not None:
                entry = {}
                for sk in SCORE_KEYS:
                    entry[sk] = r[sk]
                for ak in AUX_KEYS:
                    entry[ak] = r[ak]
                factor_cache[(code, d)] = entry
            done += 1
            if done % 5000 == 0:
                elapsed = time.time() - t_pre
                eta = elapsed / done * (total - done)
                print(f'  [{done}/{total}] {done/total*100:.0f}% {elapsed:.0f}s ETA{eta:.0f}s')
    print(f'预计算完成: {len(factor_cache)}条, 耗时{time.time()-t_pre:.0f}s')

    # Step 3: 构建所有 score_map
    print(f'构建 {len(MA_WEIGHTS)} 组 score_map...')
    t_build = time.time()
    score_maps = {}
    for ma_w in MA_WEIGHTS:
        weights = rebalance_weights(ma_w, BASE_WEIGHTS)
        score_maps[ma_w] = build_score_map(factor_cache, weights)
    print(f'  耗时: {time.time()-t_build:.0f}s')

    # Step 4: 预构建各窗口的 bench_slice
    print('预构建窗口日历...')
    window_cals = {}
    for win in WINDOWS:
        for train_flag in [True, False]:
            start, end = (win[0], win[1]) if train_flag else (win[2], win[3])
            cal = data['510300'].loc[pd.Timestamp(start):pd.Timestamp(end)].index
            bs = bench.reindex(cal).ffill()
            window_cals[(win, train_flag)] = (cal, bs)

    # Step 5: 单线程扫描
    total_runs = len(MA_WEIGHTS) * len(WINDOWS) * 2
    print(f'\n单线程扫描: {len(MA_WEIGHTS)} MA权重 × {len(WINDOWS)} 窗口 × 2 = {total_runs} 次回测')

    all_results = {ma_w: {} for ma_w in MA_WEIGHTS}
    t_scan = time.time()
    done = 0

    for ma_w in MA_WEIGHTS:
        sm = score_maps[ma_w]
        for win in WINDOWS:
            for train_flag in [True, False]:
                _, bs = window_cals[(win, train_flag)]
                s = run_engine(data, bs, sm)

                wl = f'Train{win[0][:4]}-{win[1][:4]} Test{win[2][:4]}'
                if wl not in all_results[ma_w]:
                    all_results[ma_w][wl] = {}
                key_prefix = 'train' if train_flag else 'test'
                all_results[ma_w][wl][f'{key_prefix}_ret'] = s['total_return']
                all_results[ma_w][wl][f'{key_prefix}_cagr'] = s['cagr']
                all_results[ma_w][wl][f'{key_prefix}_dd'] = s['max_dd']
                all_results[ma_w][wl][f'{key_prefix}_sharpe'] = s['sharpe']

                done += 1
                if done % 20 == 0:
                    elapsed = time.time() - t_scan
                    eta = elapsed / done * (total_runs - done)
                    print(f'  [{done}/{total_runs}] {done/total_runs*100:.0f}% '
                          f'{elapsed:.0f}s ETA{eta:.0f}s')

    elapsed_scan = time.time() - t_scan
    print(f'扫描完成: {elapsed_scan:.0f}s ({elapsed_scan/done:.1f}s/次)')

    # Step 6: 分析
    window_labels = [f'Train{w[0][:4]}-{w[1][:4]} Test{w[2][:4]}' for w in WINDOWS]

    print('\n' + '=' * 80)
    print('Walk-Forward MA权重峰值验证报告')
    print('=' * 80)

    print('\n--- 1. 各窗口训练集最优MA权重 ---')
    print(f'{"窗口":<26} {"最优MA":>8} {"训练收益":>10} {"对应OOS":>10}')
    print('-' * 56)
    optimal_per_window = {}
    optimal_vals = []
    for wl in window_labels:
        best_w, best_ret, best_oos = None, -999, 0
        for ma_w in MA_WEIGHTS:
            ret = all_results[ma_w][wl]['train_ret']
            if ret > best_ret:
                best_ret = ret
                best_w = ma_w
                best_oos = all_results[ma_w][wl]['test_ret']
        optimal_per_window[wl] = (best_w, best_ret, best_oos)
        optimal_vals.append(best_w)
        print(f'{wl:<26} {best_w:>7.0%} {best_ret*100:>+9.1f}% {best_oos*100:>+9.1f}%')

    print(f'\n--- 2. MA权重最优值分布 ---')
    cnt = Counter(optimal_vals)
    for w, count in cnt.most_common():
        pct = count / len(optimal_vals) * 100
        print(f'  {w:.0%}: {count}/{len(optimal_vals)}窗口 ({pct:.0f}%)')
    dominant = cnt.most_common(1)[0]
    stability = 'STABLE' if dominant[1] / len(optimal_vals) >= 0.7 else \
                ('MODERATE' if dominant[1] / len(optimal_vals) >= 0.5 else 'UNSTABLE')
    print(f'  稳定性: [{stability}]  各窗选择: {[f"{v:.0%}" for v in optimal_vals]}')
    matches_21 = sum(1 for v in optimal_vals if abs(v - 0.21) < 0.005)
    print(f'  21%匹配: {matches_21}/{len(optimal_vals)}')

    print(f'\n--- 3. 样本外(OOS)测试收益 ---')
    header = f'{"MA权重":<8}'
    for wl in window_labels:
        header += f' {wl:>22}'
    header += f' {"平均OOS":>10}'
    print(header)
    print('-' * (8 + 23 * len(window_labels) + 10))

    oos_by_weight = {}
    for ma_w in MA_WEIGHTS:
        oos_by_weight[ma_w] = [all_results[ma_w][wl]['test_ret'] for wl in window_labels]

    ranked = sorted(MA_WEIGHTS, key=lambda w: np.mean(oos_by_weight[w]), reverse=True)
    for ma_w in ranked:
        avg = np.mean(oos_by_weight[ma_w])
        marker = ' <-- V2.9' if abs(ma_w - 0.21) < 0.005 else ''
        row = f'{ma_w:<8.0%}'
        for r in oos_by_weight[ma_w]:
            row += f' {r*100:>+21.1f}%'
        row += f' {avg*100:>+9.1f}%{marker}'
        print(row)

    print(f'\n--- 4. 训练集最优 vs 21% OOS对比 ---')
    print(f'{"窗口":<26} {"训练最优MA":>10} {"最优MA的OOS":>12} {"21%的OOS":>12} {"差值":>10}')
    print('-' * 72)
    diffs = []
    for wl in window_labels:
        best_w, _, best_oos = optimal_per_window[wl]
        oos_21 = all_results[0.21][wl]['test_ret']
        diff = best_oos - oos_21
        diffs.append(diff)
        print(f'{wl:<26} {best_w:>9.0%} {best_oos*100:>+11.1f}% {oos_21*100:>+11.1f}% {diff*100:>+9.1f}pp')
    print(f'{"平均":<26} {"":>10} {"":>12} {"":>12} {np.mean(diffs)*100:>+9.1f}pp')

    print(f'\n--- 5. 结论 ---')
    v21_avg = np.mean(oos_by_weight[0.21])
    best_overall = max(ranked, key=lambda w: np.mean(oos_by_weight[w]))
    best_avg = np.mean(oos_by_weight[best_overall])
    print(f'  V2.9 21%均线权重平均OOS: {v21_avg*100:+.1f}%')
    print(f'  全局最优MA权重: {best_overall:.0%} (avg OOS={best_avg*100:+.1f}%)')
    if abs(best_overall - 0.21) < 0.02:
        print(f'  结论: 21%接近全局最优（差距{abs(best_overall-0.21):.0%}），Walk-Forward验证通过。')
    else:
        print(f'  结论: 21%并非最优，全局最优为{best_overall:.0%}（偏差{abs(best_overall-0.21):.0%}）。')

    print(f'\n总耗时: {(time.time()-t0)/60:.1f}分钟')
    print('=' * 80)


if __name__ == '__main__':
    main()
