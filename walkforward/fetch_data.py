# -*- coding: utf-8 -*-
"""
步骤1：数据获取模块（腾讯接口版）
用腾讯行情接口(web.ifzq.gtimg.cn)拉取12个ETF的前复权(qfq)日线+000300指数。
腾讯服务器独立于东方财富，不受eastmoney限流影响。

数据格式：[date, open, close, high, low, volume]
缓存到 walkforward/data/*.csv
"""
import os, time
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# 12个ETF: 腾讯代码 -> CSV文件名
ETF_MAP = {
    'sh510300': '510300',   # 沪深300
    'sz159915': '159915',   # 创业板
    'sh512100': '512100',   # 中证1000
    'sz159928': '159928',   # 消费ETF
    'sh510880': '510880',   # 红利ETF
    'sh513100': '513100',   # 纳指ETF
    'sh513500': '513500',   # 标普500ETF
    'sz159920': '159920',   # 恒生ETF
    'sh513880': '513880',   # 日经ETF
    'sh513050': '513050',   # 中概互联ETF
    'sh518880': '518880',   # 黄金ETF
    'sz159985': '159985',   # 豆粕ETF
}

COLUMNS = ['date', 'open', 'close', 'high', 'low', 'volume']

session = requests.Session()
session.trust_env = False
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
})


def fetch_one_tx(tx_code, start_date, end_date):
    """拉一段日期范围的前复权日线，优先qfqday，回退到day（部分ETF无复权数据）"""
    url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
    params = {
        'param': f'{tx_code},day,{start_date},{end_date},500,qfq',
        '_': str(int(time.time() * 1000)),
    }
    r = session.get(url, params=params, timeout=30)
    data = r.json()
    if isinstance(data.get('data'), dict) and tx_code in data['data']:
        inner = data['data'][tx_code]
        return inner.get('qfqday') or inner.get('day') or []
    return []


def fetch_full(tx_code, name):
    """按年分段拉取全量数据，返回 DataFrame"""
    csv_path = os.path.join(DATA_DIR, f'{name}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'])
        return df, 'cached'

    years = range(2014, 2027)  # 2014作为lookback缓冲
    all_rows = []
    for yr in years:
        start = f'{yr}-01-01'
        end = f'{yr}-12-31'
        klines = fetch_one_tx(tx_code, start, end)
        all_rows.extend(klines)
        if klines:
            print(f'    {yr}: {len(klines)}根')
        time.sleep(0.3)  # 礼貌间隔

    if not all_rows:
        return None, 'empty'

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    return df, 'fetched'


def fetch_index_tx(symbol, start_date, end_date):
    """拉指数日线（指数无复权概念）"""
    url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
    params = {
        'param': f'{symbol},day,{start_date},{end_date},500,',
        '_': str(int(time.time() * 1000)),
    }
    r = session.get(url, params=params, timeout=30)
    data = r.json()
    if isinstance(data.get('data'), dict) and symbol in data['data']:
        return data['data'][symbol].get('day', [])
    return []


def fetch_index_full():
    """拉000300指数全量"""
    csv_path = os.path.join(DATA_DIR, '000300_index.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'])
        return df, 'cached'

    years = range(2014, 2027)
    all_rows = []
    for yr in years:
        start = f'{yr}-01-01'
        end = f'{yr}-12-31'
        klines = fetch_index_tx('sh000300', start, end)
        all_rows.extend(klines)
        if klines:
            print(f'    {yr}: {len(klines)}根')
        time.sleep(0.3)

    if not all_rows:
        return None, 'empty'

    # 指数字段可能只有 date/open/close/high/low/volume
    df = pd.DataFrame(all_rows)
    ncol = len(df.columns)
    col_map = {0: 'date', 1: 'open', 2: 'close', 3: 'high', 4: 'low', 5: 'volume'}
    df = df.rename(columns=col_map)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    return df, 'fetched'


def fetch_all():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 先拉指数
    print('[指数] 000300')
    df_idx, status = fetch_index_full()
    if df_idx is not None:
        print(f'  [{status}] 000300指数 {len(df_idx)}行 {df_idx.date.min().date()} ~ {df_idx.date.max().date()}')
    else:
        print('  [FAIL] 000300指数')
    time.sleep(1)

    # 拉ETF
    for tx_code, name in ETF_MAP.items():
        print(f'[ETF] {tx_code} -> {name}')
        try:
            df, status = fetch_full(tx_code, name)
            if df is not None:
                print(f'  [{status}] {name} {len(df)}行 {df.date.min().date()} ~ {df.date.max().date()}')
            else:
                print(f'  [FAIL] {name} empty')
        except Exception as e:
            print(f'  [FAIL] {name} {repr(e)}')


if __name__ == '__main__':
    print('数据目录:', DATA_DIR)
    print('-' * 50)
    fetch_all()
    print('-' * 50)
    print('完成')
