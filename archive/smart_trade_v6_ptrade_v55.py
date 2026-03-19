# -*- coding: utf-8 -*-
"""
智能买卖策略 V6.0 - V55 性能优化版
====================================
资金规模：2 万 | 市场：主板 + 创业板 | 股票池：沪深 300+ 中证 500+ 创业板指
核心：财报新鲜度 + 现金流质量 + 行业估值 + 技术面动量
优化：性能优先，精简第二
"""
import numpy as np, pandas as pd

def initialize(context):
    """初始化 - 性能优化：预计算常量"""
    log.info('='*60 + '\n智能买卖策略 V6.0 - V55 性能优化版\n' + '='*60)
    
    # === 核心参数（2万资金适配：2只持仓、45%单只仓位）===
    context.max_stocks = 2
    context.single_max_pct = 0.45
    context.reserve_pct = 0.05
    context.per_buy_pct = 0.45
    context.top_n = 10

    # === 止损止盈（2万资金收紧）===
    context.stop_loss_pct = 0.08         # 普通止损：8%（原15%过宽）
    context.max_loss_pct = 0.12          # 硬止损：12%（原20%过宽）
    context.take_profit_pct = 0.25       # 止盈触发：25%（原40%太高难触发）
    context.take_profit_sell_pct = 0.50  # 止盈卖出比例：50%（原30%太少）
    context.trailing_start = 0.20        # 移动止盈启动：20%（原35%太高）
    context.trailing_stop = 0.08         # 移动止盈回落：8%（原15%回吐太多）
    context.partial_profit_dd = 0.10     # 部分止盈回撤：10%（原15%）
    context.dynamic_stop_days = 20       # 动态止损天数：20天（原30天太长）
    context.dynamic_stop_loss = 0.04     # 动态止损阈值：4%（原5%）
    
    # === 选股条件 ===
    context.min_price = 3.0
    context.max_price = 100.0
    context.min_daily_amount = 1e7
    context.max_rsi = 75
    context.min_momentum_60 = 0.05
    context.min_vol_ratio = 1.0
    
    # === 持仓时间 ===
    context.min_hold_days = 10
    context.max_hold_days = 60
    context.signal_cooldown = 10
    
    # === 牛熊自适应（2万资金适配）===
    context.bull_stop_loss = 0.10        # 牛市止损：10%（原20%）
    context.bear_stop_loss = 0.06        # 熊市止损：6%（原12%）
    context.bull_max_stocks = 2          # 牛市最大持仓：2只（原4只）
    context.bear_max_stocks = 1          # 熊市最大持仓：1只（原2只）
    context.bull_single_max = 0.45       # 牛市单只上限：45%（原25%）
    context.bear_single_max = 0.50       # 熊市单只上限：50%（原30%，集中持仓）
    
    # === 风控参数 ===
    context.max_total_drawdown = 0.15
    context.max_monthly_drawdown = 0.08
    context.max_industry_weight = 0.50
    context.max_consecutive_losses = 3
    context.loss_cooldown_days = 5
    
    # === 质量参数 ===
    context.goodwill_limit = 0.30
    context.pb_roe_multiplier = 10.0
    context.pb_roe_tolerance = 2.0
    context.min_roe_for_pb = 0.05
    
    # === 成长股标准 ===
    context.growth_revenue_threshold = 0.20
    context.growth_profit_threshold = 0.25
    context.growth_roe_threshold = 0.15
    context.growth_pe_relax = 1.2
    context.growth_peg_max = 1.5
    
    # === 行业估值限制 ===
    context.industry_pe_limits = {
        '科技': 60, '电子': 60, '计算机': 60, '通信': 60, '半导体': 60,
        '消费': 50, '食品饮料': 50, '家电': 50,
        '医药': 55, '生物医药': 55,
        '周期': 25, '钢铁': 25, '煤炭': 25, '有色': 25, '化工': 25,
        '金融': 12, '银行': 12, '保险': 12,
        '房地产': 8, '建筑': 12, '电力': 15, '交运': 15,
        '传媒': 35, '农业': 25, '综合': 40,
    }
    
    context.industry_pb_limits = {
        '科技': 6, '电子': 6, '计算机': 6, '通信': 6, '半导体': 6,
        '消费': 5, '食品饮料': 5, '家电': 5,
        '医药': 6, '生物医药': 6,
        '周期': 2.5, '钢铁': 2.5, '煤炭': 2.5, '有色': 2.5, '化工': 2.5,
        '金融': 1.2, '银行': 1.2, '保险': 1.2,
        '房地产': 1, '建筑': 1.5, '电力': 2, '交运': 2,
        '传媒': 3.5, '农业': 3, '综合': 4,
    }
    
    # === 现金流行业阈值 ===
    context.cashflow_thresholds = {
        '消费': 0.8, '食品饮料': 0.8, '医药': 0.8,
        '金融': 0.6, '银行': 0.6,
        '科技': 0.3, '电子': 0.3, '计算机': 0.3,
        '周期': 0.3, '机械': 0.3, '汽车': 0.3,
        '建筑': 0.1, '房地产': 0.1,
    }
    
    # === 行业新鲜度调整 ===
    context.industry_freshness_adjustment = {
        '科技': 10, '医药': 8, '电子': 8,
        '消费': 5,
        '金融': -5, '银行': -5,
        '周期': 3, '综合': 0,
    }
    
    # === 状态追踪 - 性能优化：使用局部变量 ===
    context.buy_prices = {}
    context.highest_since_buy = {}
    context.buy_dates = {}
    context.last_signal_date = {}
    context.partial_profit_taken = set()
    context.remaining_costs = {}
    context.remaining_amounts = {}
    context.partial_profit_high = {}
    context.consecutive_losses = 0
    context.cooldown_end_date = None
    context.last_trade_was_loss = False
    context.total_trades = 0
    context.winning_trades = 0
    context.losing_trades = 0
    context.industry_positions = {}
    context.stock_industry = {}
    context.peak_value = context.portfolio.portfolio_value
    context.monthly_peak = context.portfolio.portfolio_value
    context.current_month = None
    context.trading_halted = False
    context.ind_cache = {}
    context.stock_status_cache = {}
    context.freshness_cache = {}
    context.industry_cache = {}
    context.stock_pool = None
    context.last_pool_update_month = None
    context.target_position = 1.0
    context.market_bull = True
    context.market_strength = 1.0
    context.daily_values = []

    # === 负面清单开关 ===
    context.negative_list_st = True
    context.negative_list_loss = True
    context.negative_list_goodwill = True

    # === 调仓日 ===
    context.rebalance_weekday = 1  # 周一选股
    context.weekly_candidates = []  # 本周候选股

    # === 场景判断 ===
    context.is_real_trading = is_trade()
    context.merge_type = None if not context.is_real_trading else 1
    log.info(f'当前环境：{"实盘" if context.is_real_trading else "回测"} | merge_type={context.merge_type}')
    
    # 季节性调仓
    _init_seasonal(context)
    
    # 基准和费率
    try: set_benchmark('000300.XBHS')
    except: pass
    if not context.is_real_trading:
        set_commission(0.0003)
        set_slippage(0.001)

def _init_seasonal(context):
    """季节性参数 - 性能优化：减少重复计算"""
    m = context.current_dt.month if hasattr(context, 'current_dt') else 1
    
    # 预定义规则
    rules = {
        4: (75, 0.85, 0.85, '年报披露，利好出尽'),
        8: (75, 0.85, 0.90, '中报披露，观望为主'),
        10: (65, 0.75, 1.0, '三季报 + 估值切换，黄金窗口'),
        1: (55, 0.65, 0.85, '年报预告，谨慎为主'),
    }
    
    if m in rules:
        score, weight, pos, desc = rules[m]
    elif m in [2, 3, 9]:
        score, weight, pos, desc = 60, 0.7, 0.9, '财报披露中后期'
    else:
        score, weight, pos, desc = 45, 0.55, 0.9, '财报真空期'
    
    context.min_freshness_score = score
    context.freshness_weight_min = weight
    context.seasonal_position_adjustment = pos
    log.info(f'📊 {m}月：{desc} (新鲜度≥{score}, 仓位{pos*100:.0f}%)')

def before_trading_start(context, data):
    """盘前准备 - 性能优化：缓存股票池"""
    # 官方API：context.blotter.current_dt，设置别名供全局使用
    context.current_dt = context.blotter.current_dt
    log.info('=== 盘前准备 ===')
    
    # 每月更新股票池（性能优化：避免每日更新）
    if context.last_pool_update_month != context.current_dt.month:
        try:
            # 性能优化：使用集合去重
            pool = set()
            idx_date = context.previous_date.strftime('%Y-%m-%d') if hasattr(context, 'previous_date') else context.current_dt.strftime('%Y-%m-%d')
            pool.update(get_index_stocks('000300.XBHS', idx_date))
            pool.update(get_index_stocks('000905.XBHS', idx_date))
            pool.update(get_index_stocks('399006.XBHS', idx_date))
            
            # 性能优化：列表推导式过滤
            context.stock_pool = [s for s in pool 
                                 if not (s.startswith('688') or s.startswith('8') or s.startswith('4'))]
            log.info(f'✅ 股票池更新：{len(context.stock_pool)}只')
        except Exception as e:
            log.info(f'⚠️  股票池更新失败：{e}')
        context.last_pool_update_month = context.current_dt.month
    
    # 清理缓存（性能优化：只清理必要缓存）
    context.ind_cache.clear()
    context.stock_status_cache.clear()
    
    check_market_state(context)
    
    # 季节性检查（性能优化：每月只检查一次）
    m = context.current_dt.month
    if not hasattr(context, 'last_seasonal_month') or context.last_seasonal_month != m:
        context.last_seasonal_month = m
        _init_seasonal(context)
    
    # 市场转熊减仓
    if context.target_position < 0.7 and len(context.portfolio.positions) > context.bear_max_stocks:
        log.info(f'市场转熊（目标仓位{context.target_position*100:.0f}%），主动减仓至 {context.bear_max_stocks} 只')
        _sell_lowest(context, data, force=True)

def after_trading_end(context, data):
    """盘后总结"""
    context.current_dt = context.blotter.current_dt
    cv = context.portfolio.portfolio_value
    if cv > context.peak_value: context.peak_value = cv
    if cv > context.monthly_peak: context.monthly_peak = cv

def on_trade_response(context, trade_list):
    """成交回调（仅实盘生效，回测不触发）- 官方API签名"""
    try:
        for t in trade_list:
            stock = t.get('stock_code', '')
            bs = t.get('entrust_bs', '')  # 1=买, 2=卖
            amt = t.get('business_amount', 0)
            price = t.get('business_price', 0)
            log.info(f'成交回报: {stock} {"买" if bs=="1" else "卖"} {amt}股 @ {price:.2f}')
    except Exception as e:
        log.info(f'on_trade_response回调异常（不影响交易）: {e}')

def _track_trade_result(context, pnl):
    """统一追踪交易盈亏（回测+实盘均可用）"""
    context.total_trades += 1
    if pnl > 0:
        context.winning_trades += 1
        context.consecutive_losses = 0
        context.last_trade_was_loss = False
    else:
        context.losing_trades += 1
        if context.last_trade_was_loss:
            context.consecutive_losses += 1
        else:
            context.consecutive_losses = 1
        context.last_trade_was_loss = True

        if context.consecutive_losses >= context.max_consecutive_losses:
            try:
                td = get_trade_days(context.current_dt, context.current_dt + pd.Timedelta(days=30))
                context.cooldown_end_date = td[context.loss_cooldown_days] if len(td) > context.loss_cooldown_days else context.current_dt + pd.Timedelta(days=context.loss_cooldown_days * 1.5)
            except:
                context.cooldown_end_date = context.current_dt + pd.Timedelta(days=context.loss_cooldown_days * 1.5)
            log.info(f'冷却：连续{context.consecutive_losses}次亏损，停止{context.loss_cooldown_days}天')
            context.consecutive_losses = 0
            context.last_trade_was_loss = False

# ==================== 核心函数 - 性能优化 ====================

def calc_indicators(context, stock):
    """技术指标 - 性能优化：使用局部变量，减少字典访问"""
    if stock in context.ind_cache:
        return context.ind_cache[stock]
    
    try:
        # 性能优化：一次性获取所有数据
        h = get_history(250, '1d', ['close','volume','open','high','low'], security_list=stock, fq='pre')
        if h is None or len(h) < 250:
            return None
        
        # 性能优化：直接访问 values，避免重复字典查找
        c = h['close'].values
        v = h['volume'].values
        
        if len(c) < 250:
            return None
        
        # 预计算常用值
        c_last = c[-1]
        c_5 = c[-5:]
        c_10 = c[-10:]
        c_20 = c[-20:]
        c_60 = c[-60:]
        c_250 = c[-250:]
        
        # 性能优化：使用局部变量
        ma5 = c_5.mean()
        ma10 = c_10.mean()
        ma20 = c_20.mean()
        ma60 = c_60.mean()
        ma250 = c_250.mean()
        
        # 动量指标
        mom5 = c_last/c_5[0]-1 if c_5[0]!=0 else 0
        mom10 = c_last/c_10[0]-1 if c_10[0]!=0 else 0
        mom20 = c_last/c_20[0]-1 if c_20[0]!=0 else 0
        mom60 = c_last/c_60[0]-1 if c_60[0]!=0 else 0
        
        # 成交量指标
        vol_ma5 = v[-5:].mean() if len(v)>=5 else v[-1]
        vol_ma20 = v[-20:].mean() if len(v)>=20 else v[-1]
        vol_ratio = v[-1]/(vol_ma20+1e-10) if len(v)>=21 else 1.0
        
        # RSI 计算（性能优化：避免重复计算）
        if len(c) >= 100:
            d = np.diff(c[-100:])
            up = np.maximum(d, 0)
            down = np.maximum(-d, 0)
            up_ema = pd.Series(up).ewm(span=14, adjust=False).mean().iloc[-1]
            down_ema = pd.Series(down).ewm(span=14, adjust=False).mean().iloc[-1]
            rsi = 100 if down_ema==0 else 100-(100/(1+up_ema/down_ema))
        else:
            rsi = 50
        
        # 构建指标字典
        ind = {
            'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60, 'ma250': ma250,
            'momentum_5': mom5, 'momentum_10': mom10, 'momentum_20': mom20, 'momentum_60': mom60,
            'vol_ma5': vol_ma5, 'vol_ma20': vol_ma20, 'vol_ratio': vol_ratio,
            'close': c_last,
            'above_ma20': c_last > ma20,
            'above_ma60': c_last > ma60,
            'above_ma250': c_last > ma250,
            'rsi14': rsi,
        }
        
        # 均线关系
        ind['ma20_above_ma60'] = ma20 > ma60
        ind['ma20_above_ma250'] = ma20 > ma250
        ind['ma60_above_ma250'] = ma60 > ma250
        
        context.ind_cache[stock] = ind
        return ind
    except Exception as e:
        return None

def get_financial_freshness(context, stock):
    """财报新鲜度 - 性能优化：周缓存"""
    wk = f'{stock}_{context.current_dt.strftime("%Y-%W")}'
    if wk in context.freshness_cache:
        return context.freshness_cache[wk]
    
    try:
        cds = get_trading_day(-1).strftime('%Y%m%d')
        best, best_score = None, -1
        weights = {'年报':1.2, '中报':1.1, '一季报':1.0, '三季报':1.0}
        type_map = {'4':'年报','1':'一季报','2':'中报','3':'三季报'}
        
        # 性能优化：预定义循环
        for rt in ['4','1','2','3']:
            try:
                d = get_fundamentals(stock, 'growth_ability', 
                                   fields=['secu_code','end_date','publ_date'], 
                                   date=cds, report_types=rt, is_dataframe=True)
                if d is None or len(d)==0:
                    continue
                
                row = d.iloc[0]
                end_d = pd.to_datetime(row.get('end_date')).date()
                pub_d = pd.to_datetime(row.get('publ_date')).date()
                days = (context.current_dt.date() - pub_d).days
                
                if days < 0:
                    continue
                
                # 新鲜度计算
                score = max(0, 100 - (days/180*100))
                if days > 270:
                    score = max(0, score-40)
                
                tw = weights.get(type_map[rt], 1.0)
                ws = score * tw
                
                if ws > best_score:
                    best_score = ws
                    best = {
                        'report_type': type_map[rt],
                        'report_type_code': rt,
                        'end_date': end_d,
                        'publ_date': pub_d,
                        'days_since_disclosure': days,
                        'freshness_score': score,
                        'type_weight': tw,
                        'weight': 1.0 if score>=80 else 0.8 if score>=60 else 0.6 if score>=40 else 0.5,
                    }
            except:
                continue
        
        if best is None:
            best = {
                'report_type':'默认',
                'report_type_code':'0',
                'end_date':context.current_dt.date(),
                'publ_date':context.current_dt.date(),
                'days_since_disclosure':0,
                'freshness_score':50,
                'type_weight':1.0,
                'weight':0.6,
            }
        
        # 行业调整
        ind = get_stock_industry(context, stock)
        adj = context.industry_freshness_adjustment.get(ind, 0)
        best['adjusted_freshness_score'] = best['freshness_score']*best['type_weight'] + adj
        best['industry'] = ind
        best['industry_adjustment'] = adj
        
        context.freshness_cache[wk] = best
        return best
    except:
        return {
            'report_type':'默认', 'report_type_code':'0',
            'end_date':context.current_dt.date(), 'publ_date':context.current_dt.date(),
            'days_since_disclosure':0, 'freshness_score':50,
            'type_weight':1.0, 'weight':0.6,
            'adjusted_freshness_score':50, 'industry':'综合', 'industry_adjustment':0,
        }

def get_fundamental_data(context, stock, fi=None):
    """基本面数据 - 性能优化：批量获取"""
    try:
        cds = get_trading_day(-1).strftime('%Y%m%d')
        
        # 估值数据
        vd = get_fundamentals(stock, 'valuation', date=cds, is_dataframe=True)
        if vd is None or len(vd)==0:
            return None,None,None,None,None,None,1.0
        
        # PE 获取（性能优化：遍历一次）
        pe = None
        for f in ['pe_ttm','pe_static']:
            if f in vd:
                val = vd[f].values[0]
                if val is not None and 0 < val <= 10000:
                    pe = val
                    break
        
        pb = vd['pb'].values[0] if 'pb' in vd else None
        
        # 成长数据
        rg, npg = None, None
        try:
            gd = get_fundamentals(stock, 'growth_ability',
                                fields=['operating_revenue_grow_rate','net_profit_grow_rate'],
                                date=cds, is_dataframe=True)
            if gd is not None and len(gd)>0:
                if 'operating_revenue_grow_rate' in gd:
                    rg = gd['operating_revenue_grow_rate'].values[0]
                if 'net_profit_grow_rate' in gd:
                    npg = gd['net_profit_grow_rate'].values[0]
        except:
            pass
        
        # 盈利数据
        roe = None
        try:
            pd_ = get_fundamentals(stock, 'profit_ability', fields=['roe'], date=cds, is_dataframe=True)
            if pd_ is not None and len(pd_)>0 and 'roe' in pd_:
                roe = pd_['roe'].values[0]
        except:
            pass
        
        # 商誉数据
        gwr = None
        try:
            fd = get_fundamentals(stock, 'balance_statement',
                                fields=['good_will','total_assets'],
                                date=cds, merge_type=context.merge_type, is_dataframe=True)
            if fd is not None and len(fd)>0:
                gw = fd['good_will'].values[0]
                ta = fd['total_assets'].values[0]
                if gw is not None and ta is not None and ta>0:
                    gwr = gw/ta
        except:
            pass
        
        # 数据转换：官方API返回百分比值（如15.5表示15.5%），统一转为小数
        if rg is not None: rg /= 100
        if npg is not None: npg /= 100
        if roe is not None:
            roe /= 100
            if roe < 0: return None,None,None,None,None,None,1.0
        
        return pe, pb, rg, npg, roe, gwr, fi.get('weight',1.0) if fi else 1.0
    except:
        return None,None,None,None,None,None,1.0

def get_stock_industry(context, stock):
    """行业分类 - 性能优化：缓存"""
    if stock in context.industry_cache:
        return context.industry_cache[stock]
    
    try:
        b = get_stock_blocks(stock)
        if b and 'HY' in b and len(b['HY'])>0 and len(b['HY'][0])>=2:
            ind = _map_ind(b['HY'][0][1])
            context.industry_cache[stock] = ind
            return ind
    except:
        pass
    
    ind = _get_ind_name(stock)
    context.industry_cache[stock] = ind
    return ind

def _map_ind(n):
    """行业映射 - 性能优化：快速匹配"""
    if any(k in n for k in ['计算机','软件','电子','通信','半导体','芯片','信息','网络','互联网','人工智能']): return '科技'
    elif any(k in n for k in ['医药','医疗','生物','健康','药业','器械']): return '医药'
    elif any(k in n for k in ['银行','保险','证券','金融','信托','期货']): return '金融'
    elif any(k in n for k in ['食品','饮料','酒','旅游','酒店','餐饮','零售','商贸','服装','家电','家具','化妆']): return '消费'
    elif any(k in n for k in ['钢铁','煤炭','有色','化工','建材','机械','汽车','军工','国防','航空','航天']): return '周期'
    elif any(k in n for k in ['地产','建筑','工程','装修','物业']): return '房地产'
    elif any(k in n for k in ['电力','能源','石油','燃气','新能源','光伏','风电']): return '电力'
    elif any(k in n for k in ['铁路','公路','航运','港口','物流','运输','快递']): return '交运'
    elif any(k in n for k in ['传媒','广告','影视','出版','印刷','游戏']): return '传媒'
    elif any(k in n for k in ['造纸','包装','印刷','轻工','制造','制品']): return '轻工'
    elif any(k in n for k in ['农业','林业','牧业','渔业','饲料','养殖','农产品']): return '农业'
    elif any(k in n for k in ['纺织','服装','服饰']): return '纺织'
    elif any(k in n for k in ['环保','环境','治理','水务']): return '环保'
    else: return '综合'

def _get_ind_name(stock):
    """从名称获取行业"""
    try:
        n = get_stock_name(stock).get(stock, '')
        if not n: return '综合'
        if any(k in n for k in ['科技','软件','信息','网络','电子','通信','计算机','半导体','芯片']): return '科技'
        elif any(k in n for k in ['药','医疗','生物','健康','药业']): return '医药'
        elif any(k in n for k in ['银行','保险','证券','金融','信托']): return '金融'
        elif any(k in n for k in ['酒','食品','饮料','消费','旅游','酒店','餐饮']): return '消费'
        elif any(k in n for k in ['钢铁','煤炭','有色','化工','建材','机械','汽车','航运','航空']): return '周期'
        elif any(k in n for k in ['地产','物业','建筑','工程']): return '房地产'
        elif any(k in n for k in ['电力','能源','石油','燃气']): return '电力'
        else: return '综合'
    except: return '综合'

def check_market_state(context):
    """市场状态 - 性能优化：减少计算"""
    try:
        h = get_history(61, '1d', ['close'], security_list='000300.XBHS', fq='pre')
        if h and len(h)>=61:
            c = h['close'].values if 'close' in h else h.get('close',[])
            if len(c)>0:
                bc = c[-2] if len(c)>61 else c[-1]
                ma60 = sum(c[-61:-1])/60 if len(c)>61 else sum(c[-60:])/60
                context.market_bull = bc > ma60*0.95
                context.market_strength = bc/ma60
                
                if context.market_strength >= 1.05:
                    context.target_position, context.stop_loss_pct = 1.0, context.bull_stop_loss
                    context.max_stocks, context.single_max_pct = context.bull_max_stocks, context.bull_single_max
                elif context.market_strength >= 0.95:
                    context.target_position, context.stop_loss_pct = 0.7, (context.bull_stop_loss+context.bear_stop_loss)/2
                    context.max_stocks = context.bull_max_stocks
                    context.single_max_pct = (context.bull_single_max + context.bear_single_max) / 2
                else:
                    context.target_position, context.stop_loss_pct = 0.5, context.bear_stop_loss
                    context.max_stocks, context.single_max_pct = context.bear_max_stocks, context.bear_single_max
    except:
        context.market_bull, context.market_strength, context.target_position = True, 1.0, 1.0

def check_valuation(context, ind, pe, pb, rg=None, npg=None, roe=None):
    """估值检查 - 性能优化：快速路径"""
    if ind is None or ind not in context.industry_pe_limits:
        ind = '综合'
    
    pel = context.industry_pe_limits.get(ind, 40)
    pbl = context.industry_pb_limits.get(ind, 4)
    
    # 周期股逆向投资（快速路径）
    if ind in ['周期','钢铁','煤炭','有色','化工','建材','机械','汽车','航运','航空','养殖']:
        if pb is not None and roe is not None:
            if pb>pbl and roe>0.15: return False
            if pb<pbl*0.7: return True
            return pb<=pbl
        elif pe is not None and pe>0: return pe<=100
        return False
    
    # 成长股 PEG
    is_g = (rg and rg>context.growth_revenue_threshold and
            npg and npg>context.growth_profit_threshold and
            roe and roe>context.growth_roe_threshold)
    
    if is_g and pe and npg and npg>0:
        peg = pe/(npg*100)
        if peg>context.growth_peg_max: return False
        pel = pel*context.growth_pe_relax
    
    # PB-ROE 匹配
    if pb and roe and roe>context.min_roe_for_pb:
        if pb > roe*context.pb_roe_multiplier*context.pb_roe_tolerance: return False
    
    if pe and pel and pe>pel: return False
    if pb and pbl and pb>pbl: return False
    return True

def check_cash_flow(context, stock):
    """现金流检查 - 性能优化：提前返回"""
    try:
        ds = get_trading_day(-1).strftime('%Y%m%d')
        
        id_ = get_fundamentals(stock, 'income_statement',
                              fields=['net_profit','np_parent_company_owners'],
                              date=ds, merge_type=context.merge_type, is_dataframe=True)
        if id_ is None or len(id_)==0: return True
        
        np = id_['net_profit'].values[0] if 'net_profit' in id_ else id_['np_parent_company_owners'].values[0]
        
        cd = get_fundamentals(stock, 'cashflow_statement',
                             fields=['net_operate_cash_flow'],
                             date=ds, merge_type=context.merge_type, is_dataframe=True)
        if cd is None or len(cd)==0: return True
        
        ocf = cd['net_operate_cash_flow'].values[0] if 'net_operate_cash_flow' in cd else None
        
        if ocf is None: return True
        if np is not None and np<0: return False
        if np is None or np==0 or ocf==0: return True
        
        r = ocf/np
        ind = get_stock_industry(context, stock)
        return r >= context.cashflow_thresholds.get(ind, 0.5)
    except:
        return True

def check_liquidity(context, stock):
    """流动性检查 - 性能优化：numpy 向量化"""
    try:
        h = get_history(20, '1d', ['volume','close'], security_list=stock, fq='pre')
        if h is None or len(h)<20: return True
        
        v = h['volume'].values
        c = h['close'].values
        
        if len(v)<20: return True
        
        # 性能优化：numpy 向量化计算
        return np.mean(v*c) >= context.min_daily_amount
    except:
        return True

def check_limit_status(context, stock):
    """涨跌停 - 性能优化：直接调用"""
    try:
        s = check_limit(stock).get(stock, 0)
        return s not in [1,2], s not in [-1,-2]
    except:
        return True, True

def is_suspended(context, stock, data):
    """停牌 - 性能优化：缓存"""
    try:
        if stock in context.stock_status_cache:
            return context.stock_status_cache[stock]
        h = get_stock_status([stock], 'HALT').get(stock, False)
        context.stock_status_cache[stock] = h
        return h
    except:
        return True

def check_negative_list(context, stock, fund=None):
    """负面清单 - 性能优化：提前返回"""
    if context.negative_list_st:
        try:
            n = get_stock_name(stock).get(stock, '')
            if n and ('ST' in n.upper() or '*' in n or '退' in n): return False
        except: pass
    
    if context.negative_list_loss and fund and fund[3] is not None and fund[3]<-0.30: return False
    if context.negative_list_goodwill and fund and fund[6] is not None and fund[6]>context.goodwill_limit: return False
    return True

# ==================== 交易执行 ====================

def execute_buy(context, data, stock, tag, ind=None, pe=None, pb=None, rg=None, npg=None, roe=None):
    """买入 - 性能优化：减少重复计算"""
    # 快速检查
    if (stock in context.portfolio.positions or context.trading_halted or in_cooldown(context) or
        is_suspended(context, stock, data) or has_recent_suspension(stock)):
        return
    if not check_negative_list(context, stock): return
    
    # 行业集中度检查
    if ind:
        iw = sum((context.portfolio.positions[s].last_sale_price*context.portfolio.positions[s].amount)/context.portfolio.portfolio_value
                for s in context.portfolio.positions if context.stock_industry.get(s)==ind)
        if iw+context.per_buy_pct > context.max_industry_weight: return
    
    if not check_limit_status(context, stock)[0]: return
    if not check_valuation(context, ind, pe, pb, rg, npg, roe): return
    
    i = calc_indicators(context, stock)
    if i is None: return
    if not (i['above_ma250'] and i['ma20_above_ma60'] and i['ma20_above_ma250']): return
    if i['momentum_20']<0 or i['momentum_60']<context.min_momentum_60: return
    if i['rsi14']>context.max_rsi or i['vol_ratio']<context.min_vol_ratio: return
    
    # 计算买入数量
    sa = getattr(context, 'seasonal_position_adjustment', 1.0)
    bm = context.target_position * sa
    t = context.portfolio.portfolio_value
    cash = context.portfolio.cash
    res = t*context.reserve_pct
    ac = max(cash-res, 0)
    pbv = t*context.per_buy_pct*bm
    abv = min(pbv, ac)
    dp = i['close']
    bs = max(int(abv / dp / 100) * 100, 100)  # 按手取整
    ms = int(t * context.single_max_pct / dp / 100) * 100  # 单只上限
    if bs>ms: bs=ms
    if bs<100 or bs*dp>ac: return
    
    order(stock, bs)
    context.buy_prices[stock] = dp
    context.buy_dates[stock] = context.current_dt
    context.remaining_costs[stock] = dp*bs
    context.remaining_amounts[stock] = bs
    
    if ind:
        if ind not in context.industry_positions: context.industry_positions[ind] = []
        context.industry_positions[ind].append(stock)
        context.stock_industry[stock] = ind
    
    log.info(f'[{tag}] {stock} 买:{bs}股 价:{dp:.2f} 行业:{ind}')

def execute_sell(context, data, stock, reason='', force=False):
    """卖出 - 性能优化：减少分支"""
    if stock not in context.portfolio.positions: return False
    hold = hold_days(context, stock)
    if is_suspended(context, stock, data): return False
    
    pos = context.portfolio.positions[stock]
    try:
        h = get_history(2, '1d', ['open','high','low','close'], security_list=stock, fq='pre')
        if h is None or len(h)<2: return False
        p = h['close'].values[-1]
        ph = h['high'].values[-2]
        ch = h['high'].values[-1]
    except:
        return False
    
    rc = context.remaining_costs.get(stock, context.buy_prices.get(stock, p)*pos.amount)
    ra = context.remaining_amounts.get(stock, pos.amount)
    cp = rc/ra if ra>0 else p
    pnl = (p-cp)/cp if cp!=0 else 0
    
    # 更新最高价
    if stock not in context.highest_since_buy:
        context.highest_since_buy[stock] = max(ph, ch)
    else:
        context.highest_since_buy[stock] = max(context.highest_since_buy[stock], ph, ch)
    
    if not check_limit_status(context, stock)[1]: return False
    
    ss, sr, fs = 0, '', False
    
    # 止损止盈判断（性能优化：按优先级排序）
    if pnl <= -context.max_loss_pct:
        ss, sr, fs = ra, '立即止损', True
    elif pnl >= context.trailing_start:
        hg = context.partial_profit_high.get(stock, context.highest_since_buy[stock]) if stock in context.partial_profit_taken else context.highest_since_buy[stock]
        dd = (p-hg)/hg
        if dd <= -context.trailing_stop:
            ss, sr, fs = ra, f'移动止盈 回落{dd:.1%}', True
    elif pnl >= context.take_profit_pct and stock not in context.partial_profit_taken:
        ts = int(ra*context.take_profit_sell_pct)
        ss = (ts//100)*100
        if ss<100 and ra>=100: ss=100
        if ss>=ra:
            ss, sr = ra, '止盈 (全仓)'
        else:
            sr = f'止盈 ({context.take_profit_sell_pct*100:.0f}%)'
        context.partial_profit_taken.add(stock)
        context.partial_profit_high[stock] = p
        if stock in context.remaining_costs:
            ratio = ss/ra
            context.remaining_costs[stock] = rc*(1-ratio)
            context.remaining_amounts[stock] = ra*(1-ratio)
    elif stock in context.partial_profit_taken and pnl>=0.20:
        hg = context.partial_profit_high.get(stock, context.highest_since_buy[stock])
        dd = (p-hg)/hg
        if dd <= -context.partial_profit_dd:
            ss, sr, fs = ra, f'回撤止盈 回落{dd:.1%}', True
    elif pnl <= -context.stop_loss_pct:
        ss, sr, fs = ra, f'止损 ({context.stop_loss_pct*100:.0f}%)', True
    elif hold>=context.dynamic_stop_days and pnl<=-context.dynamic_stop_loss:
        ss, sr, fs = ra, f'动态止损 持{hold}天', True
    elif hold >= context.max_hold_days:
        ss, sr, fs = ra, '超时卖出', True
    
    if ss==0: return False
    if ss<100 and ra>=100: ss=ra
    if ss>0 and hold<context.min_hold_days and not fs and not force: return False
    
    if ss>=ra: order_target(stock, 0)
    else: order(stock, -ss)
    
    log.info(f'[{sr}] {stock} 盈亏:{pnl:.1%} 持{hold}天 卖:{ss}股')
    if ss>=ra:
        _track_trade_result(context, pnl)
        clean_stock_data(context, stock)
    return True

def select_stocks(context, data):
    """选股 - 性能优化：批量处理"""
    scored = []
    stocks = context.stock_pool if context.stock_pool else []
    if not stocks:
        log.warning('股票池为空，跳过选股')
        return []
    tc = len(stocks)
    log.info(f'📊 选股：{tc}只')
    
    for i, s in enumerate(stocks):
        if (i+1)%max(1,tc//10)==0:
            log.info(f'📈 进度：{(i+1)/tc*100:.0f}%')
        
        # 快速过滤
        if (s in context.buy_dates or s.startswith('688') or s.startswith('8') or s.startswith('4') or
            not (s.startswith('000') or s.startswith('001') or s.startswith('002') or s.startswith('003') or s.startswith('300') or s.startswith('301'))):
            continue
        
        try:
            # 用 calc_indicators 统一获取数据（复用缓存，避免重复 get_history）
            ind_data = calc_indicators(context, s)
            if ind_data is None: continue
            p = ind_data['close']
            if p < context.min_price or p > context.max_price: continue

            fi = get_financial_freshness(context, s)
            if fi.get('adjusted_freshness_score', fi['freshness_score']) < getattr(context, 'min_freshness_score', 45): continue

            ind = get_stock_industry(context, s)
            pe, pb, rg, npg, roe, gw, fw = get_fundamental_data(context, s, fi)
            if pb is None: continue

            fund = (ind, pe, pb, rg, npg, roe, gw, fw)
            if not check_negative_list(context, s, fund) or not check_liquidity(context, s): continue

            if not (ind_data['above_ma250'] and ind_data['ma20_above_ma60'] and ind_data['ma20_above_ma250']): continue
            if ind_data['momentum_20'] < 0 or ind_data['momentum_60'] < context.min_momentum_60: continue
            if ind_data['rsi14'] > context.max_rsi: continue
            if not check_valuation(context, ind, pe, pb, rg, npg, roe): continue
            if not check_cash_flow(context, s): continue

            # === 评分（增加基本面权重，减少纯动量依赖）===
            # 技术面动量分（满分40）
            ts = min(20, ind_data['momentum_20'] * 80) + min(20, ind_data['momentum_60'] * 40)
            # 趋势确认分（满分15）
            ts += (5 if ind_data['above_ma250'] else 0)
            ts += (5 if ind_data['ma20_above_ma60'] else 0)
            ts += (5 if ind_data['ma20_above_ma250'] else 0)
            # 量能分（满分10）
            ts += min(10, ind_data['vol_ratio'] * 5)
            # RSI适中度分（满分5，RSI=50最佳）
            ts += max(0, 5 - abs(ind_data['rsi14'] - 50) * 0.1)

            # 基本面分（满分30）
            fs = 0
            if roe: fs += min(15, roe * 100) * fw        # ROE得分（满分15）
            if pe and pe > 0: fs += max(0, 8 - pe / 10)  # 低PE加分（满分8）
            if pb and pb > 0: fs += max(0, 7 - pb)        # 低PB加分（满分7）
            if rg and rg > 0: fs += min(5, rg * 10)      # 营收增长加分（上限5）
            if npg and npg > 0: fs += min(5, npg * 8)    # 净利增长加分（上限5）

            # 新鲜度加分（满分10）
            freshness_score = fi.get('adjusted_freshness_score', fi['freshness_score'])
            fresh_s = min(10, freshness_score / 10)

            scored.append((s, ts + fs + fresh_s, fund))
        except:
            continue

    scored.sort(key=lambda x: x[1], reverse=True)
    log.info(f'选出{len(scored)}只')
    # 简化返回结构，避免额外 calc_indicators 调用
    result = []
    for x in scored[:context.top_n]:
        cached = context.ind_cache.get(x[0])
        close_price = cached['close'] if cached else 0
        obj = type('obj', (object,), {'code': x[0], 'close': close_price})
        result.append((obj, x[1], x[2]))
    return result

def hold_days(context, stock):
    """持仓天数"""
    if stock not in context.buy_dates: return 999
    try:
        td = get_trade_days(context.buy_dates[stock], context.current_dt)
        return len(td)-1 if td else 0
    except:
        return int((context.current_dt-context.buy_dates[stock]).days*0.7)

def in_cooldown(context):
    """冷却期：在冷却结束日期之前返回True，到期后自动重置"""
    if context.cooldown_end_date is None:
        return False
    if context.current_dt < context.cooldown_end_date:
        return True
    # 冷却期已过，重置
    context.cooldown_end_date = None
    return False

def clean_stock_data(context, stock):
    """清理数据 - 性能优化：批量清理"""
    attrs = ['buy_prices','highest_since_buy','buy_dates','remaining_costs','remaining_amounts','partial_profit_high']
    for a in attrs:
        getattr(context,a).pop(stock,None)
    context.partial_profit_taken.discard(stock)
    if stock in context.stock_industry:
        ind = context.stock_industry[stock]
        if ind in context.industry_positions and stock in context.industry_positions[ind]:
            context.industry_positions[ind].remove(stock)

def check_risk_control(context, data):
    """风控 - 性能优化：提前返回"""
    if context.trading_halted: return False
    cv = context.portfolio.portfolio_value
    
    # 总回撤
    if cv<context.peak_value:
        td = (context.peak_value-cv)/context.peak_value
        if td>context.max_total_drawdown:
            log.info(f'⚠️  总回撤{td*100:.1f}% > {context.max_total_drawdown*100:.0f}%')
            context.trading_halted = True
            return False
    
    # 月度回撤
    cms = context.current_dt.strftime('%Y-%m')
    if cms != context.current_month:
        context.current_month = cms
        context.monthly_peak = cv
    if cv<context.monthly_peak:
        md = (context.monthly_peak-cv)/context.monthly_peak
        if md>context.max_monthly_drawdown:
            log.info(f'⚠️  月度回撤{md*100:.1f}% > {context.max_monthly_drawdown*100:.0f}%')
            context.trading_halted = True
            return False
    
    # 行业集中度
    iw, ist = {}, {}
    for s in context.portfolio.positions:
        ind = context.stock_industry.get(s)
        if ind is None:
            ind = get_stock_industry(context, s)
            context.stock_industry[s] = ind
        pos = context.portfolio.positions[s]
        w = (pos.last_sale_price*pos.amount)/cv
        iw[ind] = iw.get(ind,0)+w
        if ind not in ist: ist[ind] = []
        ist[ind].append((s,w))
    
    for ind, w in iw.items():
        if w>context.max_industry_weight:
            log.info(f'⚠️  行业超标：{ind} 占比{w*100:.1f}%')
            if ind in ist:
                ist[ind].sort(key=lambda x:x[1], reverse=True)
                execute_sell(context, data, ist[ind][0][0], '行业集中度')
    
    if cv>context.peak_value: context.peak_value=cv
    if cv>context.monthly_peak: context.monthly_peak=cv
    return True

def _sell_lowest(context, data, force=False):
    """卖最低分"""
    if not context.portfolio.positions: return
    ss = []
    for s in context.portfolio.positions:
        i = calc_indicators(context,s)
        if i:
            score = i['momentum_60']*100+(1 if i['above_ma250'] else 0)*50
            ss.append((s, score))
    if ss:
        ss.sort(key=lambda x:x[1])
        execute_sell(context, data, ss[0][0], '减仓', force=force)

def has_recent_suspension(stock, days=10):
    """近期停牌"""
    try: return get_stock_status([stock], 'HALT').get(stock, False)
    except: return False

# ==================== 主函数 ====================

def handle_data(context, data):
    """主函数 - 每天检查卖出，每周一选股买入"""
    context.current_dt = context.blotter.current_dt
    cv = context.portfolio.portfolio_value
    context.daily_values.append((context.current_dt, cv))
    weekday = context.current_dt.isoweekday()
    log.info(f'\n=== {context.current_dt.strftime("%Y-%m-%d")} ===')
    log.info(f'资产：{cv:,.0f} | 现金：{context.portfolio.cash:,.0f}')

    check_market_state(context)

    # 卖出检查（每天执行）
    for s in list(context.portfolio.positions.keys()):
        execute_sell(context, data, s)

    # 风控检查
    if not check_risk_control(context, data):
        log.info('风控触发，暂停交易')
        return
    if in_cooldown(context):
        log.info('冷却期，暂停买入')
        return

    # 仅在调仓日（周一）选股
    if weekday == context.rebalance_weekday:
        context.weekly_candidates = select_stocks(context, data)
        if context.weekly_candidates:
            log.info(f'本周候选：{len(context.weekly_candidates)}只')
            for i, (s, sc, f) in enumerate(context.weekly_candidates[:5]):
                log.info(f'  Top{i+1}: {s.code} (评分: {sc:.1f})')

    # 买入逻辑（调仓日执行，循环候选列表直到仓位填满）
    if weekday == context.rebalance_weekday and context.weekly_candidates:
        for s, sc, fund in context.weekly_candidates:
            if len(context.portfolio.positions) >= context.max_stocks:
                break
            stock = s.code
            if stock in context.portfolio.positions:
                continue
            ind, pe, pb, rg, npg, roe, gw, fw = fund
            rg_ = f'{rg:.2%}' if rg else 'N/A'
            npg_ = f'{npg:.2%}' if npg else 'N/A'
            roe_ = f'{roe:.2%}' if roe else 'N/A'
            log.info(f'【候选】{stock} 评分={sc:.1f} 行业={ind} PE={pe} PB={pb} 营收={rg_} 净利={npg_} ROE={roe_}')
            context.last_signal_date[f'buy_{stock}'] = context.current_dt
            execute_buy(context, data, stock, '精选', ind, pe, pb, rg, npg, roe)
