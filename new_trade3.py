# 克隆自聚宽文章：https://www.joinquant.com/post/1399
# 标题：【量化课堂】多因子策略入门
# 作者：JoinQuant量化课堂

# ETF轮动策略 - 独立版本
# 拆分自：FF的三策略50-400W-171.py

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import datetime
import math
from datetime import time
import talib
from prettytable import PrettyTable
import warnings
warnings.filterwarnings("ignore")

# ==================== ETF轮动策略配置 ====================
ETF_CONFIG = {
    'etf_pool': [
        "513100.XSHG", "513520.XSHG", "513030.XSHG",
        "518880.XSHG", "159980.XSHE", "501018.XSHG",
        "511090.XSHG", "512890.XSHG", '159915.XSHE'
    ],
    'target_num': 1,
    'm25_days': 25,
    'auto_day': True,
    'min_days': 20,
    'max_days': 60,
    'premium_threshold': 5.0
}

# ==================== 初始化函数 ====================
def initialize(context):
    set_option('avoid_future_data', True)
    set_option('use_real_price', True)
    set_benchmark('000300.XSHG')
    
    set_slippage(FixedSlippage(0.0001), type='fund')
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0,
        open_commission=0.0002, close_commission=0.0002,
        close_today_commission=0, min_commission=1
    ), type='fund')
    
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'info')
    
    # 全局变量
    g.positions = {}
    
    # 调度任务
    run_daily(etf_trade, '9:50')
    run_daily(daily_settlement, '15:00')
    
    log.info("=" * 60)
    log.info("ETF轮动策略 - 独立版本 初始化完成")
    log.info("=" * 60)

# ==================== 主交易函数 ====================
def etf_trade(context):
    """ETF轮动主交易函数"""
    config = ETF_CONFIG
    
    # 获取目标ETF列表
    if config['auto_day']:
        target_list, rank_info = etf_get_rank_auto(context, return_info=True)
        target_list = target_list[:config['target_num']]
    else:
        target_list, rank_info = etf_get_rank_fixed(context, return_info=True)
        target_list = target_list[:config['target_num']]
    
    if rank_info:
        log.info("[ETF轮动] 动量评分排行:")
        for i, (etf, score) in enumerate(rank_info[:5], 1):
            etf_name = get_security_info(etf).display_name
            log.info(f"  {i}. {etf} {etf_name:10s} 得分:{score:.4f}")
    
    if not target_list:
        log.warning("[ETF轮动] 无符合条件的标的")
        return
    
    current_holdings = list(context.portfolio.positions.keys())
    
    # 卖出不在目标列表的持仓
    for stock in current_holdings:
        if stock not in target_list:
            order_target_value(stock, 0)
            log.info(f"[ETF轮动] 卖出 {stock}")
    
    # 买入新标的
    holding_count = sum(1 for s in target_list if s in context.portfolio.positions)
    if holding_count < config['target_num']:
        available = context.portfolio.available_cash
        buy_count = config['target_num'] - holding_count
        if available > 0 and buy_count > 0:
            value_per_stock = available / buy_count
            for stock in target_list:
                if stock not in context.portfolio.positions:
                    order_target_value(stock, value_per_stock)
                    log.info(f"[ETF轮动] 买入 {stock} 金额:{value_per_stock:.0f}")

# ==================== 动量评分函数 ====================
def etf_get_rank_fixed(context, return_info=False):
    """固定周期动量评分"""
    config = ETF_CONFIG
    data = pd.DataFrame(index=config['etf_pool'],
                       columns=["annualized_returns", "r2", "score"])
    current_data = get_current_data()
    
    for etf in config['etf_pool']:
        try:
            df = attribute_history(etf, config['m25_days'], "1d", ["close", "high"])
            if len(df) < config['m25_days']:
                continue
            
            prices = np.append(df["close"].values, current_data[etf].last_price)
            y = np.log(prices)
            x = np.arange(len(y))
            weights = np.linspace(1, 2, len(y))
            
            slope, intercept = np.polyfit(x, y, 1, w=weights)
            data.loc[etf, "annualized_returns"] = math.exp(slope * 250) - 1
            
            ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
            ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
            data.loc[etf, "r2"] = 1 - ss_res / ss_tot if ss_tot else 0
            data.loc[etf, "score"] = data.loc[etf, "annualized_returns"] * data.loc[etf, "r2"]
            
            if len(prices) >= 4:
                if min(prices[-1]/prices[-2], prices[-2]/prices[-3], prices[-3]/prices[-4]) < 0.95:
                    data.loc[etf, "score"] = 0
            
            premium_rate = get_etf_premium_rate(context, etf)
            if premium_rate >= config['premium_threshold']:
                data.loc[etf, "score"] -= 1
                
        except Exception as e:
            continue
    
    data = data.query("0 < score < 6").sort_values(by="score", ascending=False)
    
    if return_info:
        rank_info = [(idx, row['score']) for idx, row in data.iterrows()]
        return data.index.tolist(), rank_info
    return data.index.tolist()

def etf_get_rank_auto(context, return_info=False):
    """动态周期动量评分（基于波动率）"""
    config = ETF_CONFIG
    data = pd.DataFrame(index=config['etf_pool'],
                       columns=["annualized_returns", "r2", "score"])
    current_data = get_current_data()
    
    for etf in config['etf_pool']:
        try:
            df = attribute_history(etf, config['max_days']+10, "1d", ["close", "high", "low"])
            
            if len(df) < (config['max_days']+10) or \
               df["low"].isna().sum() > config['max_days'] or \
               df["close"].isna().sum() > config['max_days'] or \
               df["high"].isna().sum() > config['max_days']:
                continue
            
            long_atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=config['max_days'])
            short_atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=config['min_days'])
            
            lookback = int(config['min_days'] +
                          (config['max_days'] - config['min_days']) *
                          (1 - min(0.9, short_atr[-1]/long_atr[-1])))
            
            prices = np.append(df["close"].values, current_data[etf].last_price)
            prices = prices[-lookback:]
            
            y = np.log(prices)
            x = np.arange(len(y))
            weights = np.linspace(1, 2, len(y))
            
            slope, intercept = np.polyfit(x, y, 1, w=weights)
            data.loc[etf, "annualized_returns"] = math.exp(slope * 250) - 1
            
            ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
            ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
            data.loc[etf, "r2"] = 1 - ss_res / ss_tot if ss_tot else 0
            data.loc[etf, "score"] = data.loc[etf, "annualized_returns"] * data.loc[etf, "r2"]
            
            con1 = min(prices[-1]/prices[-2], prices[-2]/prices[-3], prices[-3]/prices[-4]) < 0.95
            con2 = (prices[-1] < prices[-2]) & (prices[-2] < prices[-3]) & \
                   (prices[-3] < prices[-4]) & (prices[-1]/prices[-4] < 0.95)
            
            if con1 or con2:
                data.loc[etf, "score"] = 0
            
            premium_rate = get_etf_premium_rate(context, etf)
            if premium_rate >= config['premium_threshold']:
                data.loc[etf, "score"] -= 1
                
        except Exception as e:
            continue
    
    data = data.sort_values(by="score", ascending=False).reset_index()
    data = data.query("0 < score < 6").sort_values(by="score", ascending=False)
    
    if return_info:
        rank_info = [(row['index'], row['score']) for _, row in data.iterrows()]
        return data['index'].tolist(), rank_info
    return data['index'].tolist()

def get_etf_premium_rate(context, etf_code):
    """计算ETF溢价率"""
    try:
        etf_price = get_price(etf_code, start_date=context.previous_date,
                             end_date=context.previous_date).iloc[-1]['close']
        iopv = get_extras('unit_net_value', etf_code,
                         start_date=context.previous_date,
                         end_date=context.previous_date).iloc[-1].values[0]
        
        if iopv is not None and iopv != 0:
            premium_rate = (etf_price - iopv) / iopv * 100
        else:
            premium_rate = 0
        
        return premium_rate
    except Exception as e:
        return 0

# ==================== 结算 ====================
def daily_settlement(context):
    """尾盘结算"""
    current_data = get_current_data()
    
    log.info("=" * 60)
    log.info("ETF轮动策略 - 收盘结算")
    log.info("=" * 60)
    
    holdings = list(context.portfolio.positions.keys())
    if holdings:
        table = PrettyTable()
        table.field_names = ["代码", "名称", "数量", "成本", "现价", "市值", "盈亏", "盈亏率"]
        for col in ["数量", "成本", "现价", "市值", "盈亏", "盈亏率"]:
            table.align[col] = "r"
        table.align["代码"] = "l"
        table.align["名称"] = "l"
        
        total_value = 0
        total_pnl = 0
        for stock in holdings:
            pos = context.portfolio.positions[stock]
            price = current_data[stock].last_price
            value = pos.total_amount * price
            pnl = value - pos.total_amount * pos.avg_cost
            pnl_ratio = (price - pos.avg_cost) / pos.avg_cost * 100 if pos.avg_cost > 0 else 0
            total_value += value
            total_pnl += pnl
            
            table.add_row([
                stock,
                get_security_info(stock).display_name[:8],
                f"{pos.total_amount:.0f}",
                f"{pos.avg_cost:.3f}",
                f"{price:.3f}",
                f"{value:.0f}",
                f"{pnl:+.0f}",
                f"{pnl_ratio:+.2f}%"
            ])
        
        log.info("\n" + str(table))
    else:
        log.info("  空仓")
    
    returns = (context.portfolio.total_value / context.portfolio.starting_cash - 1) * 100
    log.info(f"总收益: {returns:+.2f}% | 总资产: {context.portfolio.total_value:.0f} | 可用资金: {context.portfolio.available_cash:.0f}")
    log.info("=" * 60 + "\n")
