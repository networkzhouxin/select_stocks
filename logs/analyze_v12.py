import sys, io, re, os
from collections import defaultdict
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

log_path = os.path.join(os.path.dirname(__file__), '201511-2026312.txt')
with open(log_path, 'r', encoding='gbk') as f:
    lines = f.readlines()

buys = []
sells = []
portfolio_values = []
order_executions = []

for i, line in enumerate(lines):
    line_s = line.strip()

    buy_match = re.search(r'\[(强买|中买|弱买)\]\s+(\S+)\s+买分=(\S+)\s+趋势=(\d+)\s+动量=(\S+)\s+波动率=(\S+)%?\s+ATR=(\S+)\s+(\d+)股\s+@(\S+)', line_s)
    if buy_match:
        bu_str = ''
        if i+1 < len(lines):
            bu_match = re.search(r'BU:\s*\[(.+)\]', lines[i+1])
            if bu_match:
                bu_str = bu_match.group(1)
        buys.append({
            'date': line_s[:10],
            'signal': buy_match.group(1),
            'code': buy_match.group(2),
            'buy_score': float(buy_match.group(3)),
            'trend': int(buy_match.group(4)),
            'momentum': float(buy_match.group(5)),
            'volatility': buy_match.group(6),
            'atr': float(buy_match.group(7)),
            'shares': int(buy_match.group(8)),
            'price': float(buy_match.group(9)),
            'bu': bu_str,
        })
        continue

    sell_match = re.search(r'\[(ATR跟踪止损|ATR最大止损|信号卖出)\]\s+(\S+)\s+最高(\S+)\s+现价(\S+)\s+ATR=(\S+)\s+回撤(\S+)%\s+盈亏(\S+)%', line_s)
    if sell_match:
        sells.append({
            'date': line_s[:10],
            'type': sell_match.group(1),
            'code': sell_match.group(2),
            'high': float(sell_match.group(3)),
            'price': float(sell_match.group(4)),
            'atr': float(sell_match.group(5)),
            'drawdown': float(sell_match.group(6)),
            'pnl': float(sell_match.group(7)),
        })
        continue

    order_match = re.search(r'trade price:\s*(\S+),\s*amount:(\S+),\s*commission:\s*(\S+)', line_s)
    if order_match:
        sec_match = re.search(r'security=(\S+)\s', line_s)
        action_match = re.search(r'action=(\w+)', line_s)
        if sec_match and action_match:
            order_executions.append({
                'date': line_s[:10],
                'code': sec_match.group(1),
                'action': action_match.group(1),
                'price': float(order_match.group(1)),
                'amount': int(order_match.group(2)),
                'commission': float(order_match.group(3)),
            })
        continue

    pv_match = re.search(r'\[(\w+)\]\s+总值:(\S+)\s+现金:(\S+)\s+持仓:(\d+)/(\d+)', line_s)
    if pv_match:
        portfolio_values.append({
            'date': line_s[:10],
            'tier': pv_match.group(1),
            'total': float(pv_match.group(2)),
        })

# ============================================================
# MATCH BUYS TO SELLS
# ============================================================
open_positions = defaultdict(list)
trades = []

events = []
for b in buys:
    events.append(('buy', b))
for s in sells:
    events.append(('sell', s))
# Sells before buys on same day (sell first, then buy)
events.sort(key=lambda x: (x[1]['date'], 0 if x[0]=='sell' else 1))

for event_type, event in events:
    code = event['code']
    if event_type == 'buy':
        exec_price = None
        for o in order_executions:
            if o['date'] == event['date'] and o['code'] == code and o['action'] == 'open':
                exec_price = o['price']
                break
        event['exec_price'] = exec_price if exec_price else event['price']
        open_positions[code].append(event)
    else:
        if open_positions[code]:
            buy_event = open_positions[code].pop(0)
            exec_sell = None
            for o in order_executions:
                if o['date'] == event['date'] and o['code'] == code and o['action'] == 'close':
                    exec_sell = o['price']
                    break

            buy_date = datetime.strptime(buy_event['date'], '%Y-%m-%d')
            sell_date = datetime.strptime(event['date'], '%Y-%m-%d')
            holding_days = (sell_date - buy_date).days

            trades.append({
                'buy_date': buy_event['date'],
                'sell_date': event['date'],
                'code': code,
                'buy_signal': buy_event['signal'],
                'buy_score': buy_event['buy_score'],
                'trend': buy_event['trend'],
                'buy_price': buy_event['exec_price'],
                'sell_price': exec_sell if exec_sell else event['price'],
                'sell_type': event['type'],
                'pnl_pct': event['pnl'],
                'drawdown': event['drawdown'],
                'holding_days': holding_days,
                'shares': buy_event['shares'],
            })
        else:
            print("WARNING: Sell without matching buy: {} {} pnl={}%".format(event['date'], code, event['pnl']))

# ============================================================
# OUTPUT
# ============================================================
print('=' * 110)
print('ALL BUY SIGNALS ({} total)'.format(len(buys)))
print('=' * 110)
for i, b in enumerate(buys, 1):
    print('{:3d}. {}  {}  {:<14s}  买分={:.1f}  趋势={}  动量={:.2f}  波动率={}%  {:5d}股 @{:.3f}'.format(
        i, b['date'], b['signal'], b['code'], b['buy_score'], b['trend'], b['momentum'], b['volatility'], b['shares'], b['price']))

print()
print('=' * 110)
print('ALL SELL SIGNALS ({} total)'.format(len(sells)))
print('=' * 110)
for i, s in enumerate(sells, 1):
    print('{:3d}. {}  {:<10s}  {:<14s}  盈亏={:+.1f}%  回撤={:.1f}%  最高={:.3f}  现价={:.3f}'.format(
        i, s['date'], s['type'], s['code'], s['pnl'], s['drawdown'], s['high'], s['price']))

print()
print('=' * 110)
print('COMPLETE TRADE LIST ({} matched trades)'.format(len(trades)))
print('=' * 110)
for i, t in enumerate(trades, 1):
    win = 'W' if t['pnl_pct'] > 0 else 'L'
    print('{:3d}. [{}] {} -> {}  {:<14s}  {}  买@{:.3f} 卖@{:.3f}  P&L={:+.1f}%  {:<10s}  {:3d}天'.format(
        i, win, t['buy_date'], t['sell_date'], t['code'], t['buy_signal'], t['buy_price'], t['sell_price'], t['pnl_pct'], t['sell_type'], t['holding_days']))

print()
print('OPEN POSITIONS (unmatched buys at end):')
for code, positions in open_positions.items():
    for p in positions:
        print('  {} {} {} @{}'.format(p['date'], code, p['signal'], p['price']))

# ============================================================
# PORTFOLIO VALUES
# ============================================================
if portfolio_values:
    # Deduplicate by date (take last value per date)
    pv_by_date = {}
    for pv in portfolio_values:
        pv_by_date[pv['date']] = pv['total']

    dates_sorted = sorted(pv_by_date.keys())
    start_val = pv_by_date[dates_sorted[0]]
    end_val = pv_by_date[dates_sorted[-1]]

    peak_val = 0
    peak_date = ''
    trough_val = float('inf')
    trough_date = ''
    max_dd = 0
    max_dd_peak_date = ''
    max_dd_trough_date = ''
    running_peak = 0
    running_peak_date = ''

    for d in dates_sorted:
        v = pv_by_date[d]
        if v > peak_val:
            peak_val = v
            peak_date = d
        if v < trough_val:
            trough_val = v
            trough_date = d
        if v > running_peak:
            running_peak = v
            running_peak_date = d
        dd = (running_peak - v) / running_peak * 100
        if dd > max_dd:
            max_dd = dd
            max_dd_peak_date = running_peak_date
            max_dd_trough_date = d

    print()
    print('=' * 110)
    print('PORTFOLIO SUMMARY')
    print('=' * 110)
    print('Start: {} = {:.2f}'.format(dates_sorted[0], start_val))
    print('End:   {} = {:.2f}'.format(dates_sorted[-1], end_val))
    print('Total Return: {:.1f}%'.format((end_val - start_val) / start_val * 100))
    print()
    print('Peak:   {} = {:.2f}'.format(peak_date, peak_val))
    print('Trough: {} = {:.2f}'.format(trough_date, trough_val))
    print('Max Drawdown: {:.2f}% (from {} to {})'.format(max_dd, max_dd_peak_date, max_dd_trough_date))

# ============================================================
# TRADE STATISTICS
# ============================================================
print()
print('=' * 110)
print('TRADE STATISTICS')
print('=' * 110)

wins = [t for t in trades if t['pnl_pct'] > 0]
losses = [t for t in trades if t['pnl_pct'] <= 0]

print('Total Trades: {}'.format(len(trades)))
print('Wins: {} ({:.1f}%)'.format(len(wins), len(wins)/len(trades)*100 if trades else 0))
print('Losses: {}'.format(len(losses)))
print()

if wins:
    avg_win = sum(t['pnl_pct'] for t in wins) / len(wins)
    print('Average Win: {:.1f}%'.format(avg_win))
    print('Median Win: {:.1f}%'.format(sorted(t['pnl_pct'] for t in wins)[len(wins)//2]))
if losses:
    avg_loss = sum(t['pnl_pct'] for t in losses) / len(losses)
    print('Average Loss: {:.1f}%'.format(avg_loss))
    print('Median Loss: {:.1f}%'.format(sorted(t['pnl_pct'] for t in losses)[len(losses)//2]))

# Profit factor
gross_profit = sum(t['pnl_pct'] for t in wins) if wins else 0
gross_loss = abs(sum(t['pnl_pct'] for t in losses)) if losses else 1
print()
print('Gross Profit: {:.1f}%'.format(gross_profit))
print('Gross Loss: {:.1f}%'.format(gross_loss))
print('Profit Factor: {:.3f}'.format(gross_profit / gross_loss if gross_loss > 0 else float('inf')))
print('盈亏比 (avg win / avg loss): {:.2f}'.format(abs(avg_win / avg_loss) if losses and avg_loss != 0 else float('inf')))

# Best and worst trades
best = max(trades, key=lambda t: t['pnl_pct'])
worst = min(trades, key=lambda t: t['pnl_pct'])
print()
print('Best Trade:  {} {} {} P&L={:+.1f}% ({} -> {}, {}天)'.format(
    best['code'], best['buy_signal'], best['sell_type'], best['pnl_pct'], best['buy_date'], best['sell_date'], best['holding_days']))
print('Worst Trade: {} {} {} P&L={:+.1f}% ({} -> {}, {}天)'.format(
    worst['code'], worst['buy_signal'], worst['sell_type'], worst['pnl_pct'], worst['buy_date'], worst['sell_date'], worst['holding_days']))

# Holding period
if wins:
    avg_hold_win = sum(t['holding_days'] for t in wins) / len(wins)
    print()
    print('Avg Holding (wins): {:.0f} days'.format(avg_hold_win))
if losses:
    avg_hold_loss = sum(t['holding_days'] for t in losses) / len(losses)
    print('Avg Holding (losses): {:.0f} days'.format(avg_hold_loss))

# Winning/losing streaks
print()
streak_w = 0
streak_l = 0
max_streak_w = 0
max_streak_l = 0
for t in trades:
    if t['pnl_pct'] > 0:
        streak_w += 1
        streak_l = 0
    else:
        streak_l += 1
        streak_w = 0
    max_streak_w = max(max_streak_w, streak_w)
    max_streak_l = max(max_streak_l, streak_l)
print('Longest Winning Streak: {}'.format(max_streak_w))
print('Longest Losing Streak: {}'.format(max_streak_l))

# ============================================================
# BREAKDOWN BY ETF
# ============================================================
print()
print('=' * 110)
print('BREAKDOWN BY ETF')
print('=' * 110)
etf_names = {'510500.XSHG': '中证500', '159915.XSHE': '创业板', '512100.XSHG': '中证1000'}
by_etf = defaultdict(list)
for t in trades:
    by_etf[t['code']].append(t)

for code in sorted(by_etf.keys()):
    ts = by_etf[code]
    w = [t for t in ts if t['pnl_pct'] > 0]
    l = [t for t in ts if t['pnl_pct'] <= 0]
    total_pnl = sum(t['pnl_pct'] for t in ts)
    name = etf_names.get(code, '')
    print('{} ({})'.format(code, name))
    print('  Trades: {}  Wins: {} ({:.1f}%)  Losses: {}'.format(
        len(ts), len(w), len(w)/len(ts)*100 if ts else 0, len(l)))
    print('  Total P&L sum: {:+.1f}%  Avg P&L: {:+.1f}%'.format(total_pnl, total_pnl/len(ts)))
    if w:
        print('  Avg Win: {:.1f}%  Best: {:+.1f}%'.format(sum(t['pnl_pct'] for t in w)/len(w), max(t['pnl_pct'] for t in w)))
    if l:
        print('  Avg Loss: {:.1f}%  Worst: {:+.1f}%'.format(sum(t['pnl_pct'] for t in l)/len(l), min(t['pnl_pct'] for t in l)))
    print()

# ============================================================
# BREAKDOWN BY YEAR
# ============================================================
print('=' * 110)
print('BREAKDOWN BY YEAR')
print('=' * 110)
by_year = defaultdict(list)
for t in trades:
    yr = t['sell_date'][:4]
    by_year[yr].append(t)

for yr in sorted(by_year.keys()):
    ts = by_year[yr]
    w = [t for t in ts if t['pnl_pct'] > 0]
    total_pnl = sum(t['pnl_pct'] for t in ts)
    print('{}: {} trades, {} wins ({:.1f}%), sum P&L={:+.1f}%'.format(
        yr, len(ts), len(w), len(w)/len(ts)*100, total_pnl))

# ============================================================
# BREAKDOWN BY EXIT TYPE
# ============================================================
print()
print('=' * 110)
print('BREAKDOWN BY EXIT TYPE')
print('=' * 110)
by_exit = defaultdict(list)
for t in trades:
    by_exit[t['sell_type']].append(t)

for exit_type in sorted(by_exit.keys()):
    ts = by_exit[exit_type]
    w = [t for t in ts if t['pnl_pct'] > 0]
    total_pnl = sum(t['pnl_pct'] for t in ts)
    print('{}: {} trades, {} wins ({:.1f}%), sum P&L={:+.1f}%, avg P&L={:+.1f}%'.format(
        exit_type, len(ts), len(w), len(w)/len(ts)*100, total_pnl, total_pnl/len(ts)))

# ============================================================
# BREAKDOWN BY BUY SIGNAL LEVEL
# ============================================================
print()
print('=' * 110)
print('BREAKDOWN BY BUY SIGNAL LEVEL')
print('=' * 110)
by_signal = defaultdict(list)
for t in trades:
    by_signal[t['buy_signal']].append(t)

for sig in ['强买', '中买', '弱买']:
    if sig in by_signal:
        ts = by_signal[sig]
        w = [t for t in ts if t['pnl_pct'] > 0]
        total_pnl = sum(t['pnl_pct'] for t in ts)
        print('{}: {} trades, {} wins ({:.1f}%), sum P&L={:+.1f}%, avg P&L={:+.1f}%'.format(
            sig, len(ts), len(w), len(w)/len(ts)*100, total_pnl, total_pnl/len(ts)))

# ============================================================
# COMPARISON WITH V10
# ============================================================
print()
print('=' * 110)
print('V12 vs V10 COMPARISON')
print('=' * 110)
total_pnl_sum = sum(t['pnl_pct'] for t in trades)
win_rate = len(wins)/len(trades)*100 if trades else 0
avg_w = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
avg_l = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0
profit_loss_ratio = abs(avg_w / avg_l) if avg_l != 0 else float('inf')

print('{:<25s} {:>12s} {:>12s}'.format('Metric', 'V12', 'V10'))
print('-' * 50)
print('{:<25s} {:>12d} {:>12d}'.format('Total Trades', len(trades), 103))
print('{:<25s} {:>11.1f}% {:>11.1f}%'.format('Win Rate', win_rate, 37.9))
total_return = (end_val - 20000) / 20000 * 100 if portfolio_values else 0
print('{:<25s} {:>11.1f}% {:>11.1f}%'.format('Total Return', total_return, 58.0))
print('{:<25s} {:>12.2f} {:>12.3f}'.format('盈亏比', profit_loss_ratio, 1.675))
print('{:<25s} {:>11.2f}% {:>11.2f}%'.format('Max Drawdown', max_dd, 27.89))
print('{:<25s} {:>12.3f} {:>12s}'.format('Profit Factor', gross_profit/gross_loss if gross_loss>0 else 0, 'N/A'))
print()
print('V10 by ETF: 510500 +88.3%, 159915 +20.6%, 510300 +5.4%')
print('V12 by ETF:')
for code in sorted(by_etf.keys()):
    ts = by_etf[code]
    total_pnl = sum(t['pnl_pct'] for t in ts)
    name = etf_names.get(code, '')
    print('  {} ({}) {:+.1f}% ({} trades)'.format(code, name, total_pnl, len(ts)))
print()
print('V10 signal sells: 9 trades, 11.1% win rate, -17.3% total')
print('V12 signal sells: ', end='')
if '信号卖出' in by_exit:
    ss = by_exit['信号卖出']
    sw = [t for t in ss if t['pnl_pct'] > 0]
    print('{} trades, {:.1f}% win rate, {:+.1f}% total'.format(len(ss), len(sw)/len(ss)*100, sum(t['pnl_pct'] for t in ss)))
else:
    print('0 trades (signal selling removed in V12)')
