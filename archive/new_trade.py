# 克隆自聚宽文章：https://www.joinquant.com/post/17194
# 标题：年化64%的市值选股策略（有止损模块）
# 作者：lsydmn

# 克隆自聚宽文章：https://www.joinquant.com/post/1399
# 标题：【量化课堂】多因子策略入门
# 作者：JoinQuant量化课堂

# 克隆自聚宽文章：https://www.joinquant.com/post/61069
# 标题：多策略学习版2.3，19年至今年化55，回撤7.6
# 作者：Friday_
# 重构版本：单独测试搅屎棍策略（子策略1）

from jqdata import *
import pandas as pd
import numpy as np
import math
import datetime

# -------------------- 运行调度函数 --------------------
# 初始化函数
def initialize(context):
    set_benchmark("000300.XSHG")  # 设定沪深300作为基准
    set_option("avoid_future_data", True)  # 打开防未来函数
    set_option("use_real_price", True)  # 开启动态复权模式(真实价格)
    set_trading_costs() # 设定交易成本
    log.set_level("order", "error")  # 过滤掉order系列API产生的比error级别低的log

    # 全局变量（简化，只针对搅屎棍策略）
    g.positions = {}  # 记录持仓股票 {股票代码: 持仓数量}

    # 搅屎棍策略调度
    run_daily(jsg_check, "14:45")  # 每日14:45检查
    run_weekly(jsg_adjust, 1, "11:00")  # 每周第1个交易日11:00调仓

    run_daily(check_stop_loss, "14:50") # 每日止损检查
    run_daily(end_trade, "14:59") # 每日尾盘处理

    # 直接初始化策略实例
    process_initialize(context)

# 创建策略实例
def process_initialize(context):
    g.strategy = JSG_Strategy(context, name="搅屎棍策略")
    log.info(f"初始化策略: 搅屎棍策略, 资金占比: 100%")

# 全局止损检查（简化，只针对这个策略）
def check_stop_loss(context):
    if hasattr(g, 'strategy'):        
        try:
            stop_loss_stocks = g.strategy.check_stop_loss()
            if stop_loss_stocks: 
                pass
        except Exception as e:
            log.error(f"搅屎棍策略 止损检查出错: {str(e)}")

# 尾盘处理
def end_trade(context):
    marked = {s for s in g.positions}
    for stock in context.portfolio.positions: 
        if stock not in marked:
            if order_target_value(stock, 0): 
                log.info(f"end_trade清仓未记录持仓: {stock}")

def jsg_check(context):
    if hasattr(g, 'strategy'):
        g.strategy.check()

def jsg_adjust(context):
    if hasattr(g, 'strategy'):
        g.strategy.adjust()

# -------------------- 策略基类 --------------------
class Strategy:

    def __init__(self, context, name):
        self.context = context
        self.name = name
        self.stock_sum = 1
        self.hold_list = []
        self.def_stocks = ["511260.XSHG", "518880.XSHG", "512800.XSHG"]  # 债券ETF、黄金ETF、银行ETF

        # 移动止损 
        self.stop_loss_rate = 0.12   # 移动止损比率（12%）  

        # 初始化移动止损跟踪字典
        if not hasattr(g, 'stop_loss_tracking'):
            g.stop_loss_tracking = {}  # {股票代码: 最高价}
        self.stop_loss_tracking = g.stop_loss_tracking

    # 获取持仓市值
    def get_total_value(self):
        if not g.positions:
            return 0
        return sum(self.context.portfolio.positions[key].price * value for key, value in g.positions.items())

    # 计算动态最小交易阈值
    def get_min_trade_value(self):
      strategy_total_value = self.context.portfolio.total_value   

      if strategy_total_value <= 100000:
          return 2000
      elif strategy_total_value <= 500000:
          ratio = (strategy_total_value - 100000) / 400000
          return int(2000 + ratio * 5000)
      else:
          threshold = strategy_total_value * 0.012
          return min(50000, max(8000, int(threshold)))

    # 调仓(targets为字典，key为股票代码，value为目标权重)
    def _adjust(self, targets):

        current_data = get_current_data()

        # 输出信息
        strategy_data = {
            "strategy_name": self.name,
            "stocks": [{"stock_name": current_data[stock].name, "stock_code": stock, "weight": weight} for stock, weight in targets.items()],
        }

        # 获取已持有列表
        self.hold_list = list(g.positions.keys())
        portfolio = self.context.portfolio

        # 获取目标策略市值
        target_value = self.context.portfolio.total_value

        # 清仓被调出的
        for stock in self.hold_list:
            if stock not in targets:
                self.order_target_value_(stock, 0)

        # 获取动态阈值
        min_trade_value = self.get_min_trade_value()

        # 先卖出
        for stock, weight in targets.items():
            target = target_value * weight
            price = current_data[stock].last_price
            value = g.positions.get(stock, 0) * price
            if value - target > max(min_trade_value, price * 100):
                self.order_target_value_(stock, target)

        # 后买入
        for stock, weight in targets.items():
            target = target_value * weight
            price = current_data[stock].last_price
            value = g.positions.get(stock, 0) * price
            if min(target - value, portfolio.available_cash) > max(min_trade_value, price * 100):
                self.order_target_value_(stock, target)

    # 自定义下单(涨跌停不交易)
    def order_target_value_(self, security, value):
        current_data = get_current_data()

        if current_data[security].paused:
            return False

        if current_data[security].last_price == current_data[security].high_limit:
            return False

        if current_data[security].last_price == current_data[security].low_limit:
            return False

        price = current_data[security].last_price
        current_position = g.positions.get(security, 0)
        target_position = (int(value / price) // 100) * 100 if price != 0 else 0
        
        if target_position == 0 and value > 0:
            return False

        adjustment = target_position - current_position
        closeable_amount = self.context.portfolio.positions[security].closeable_amount if security in self.context.portfolio.positions else 0
        if adjustment < 0 and closeable_amount == 0:
            return False

        if adjustment != 0:
            o = order(security, adjustment)
            if o:
                if adjustment > 0:
                    price = current_data[security].last_price
                    if security not in self.stop_loss_tracking:
                        self.stop_loss_tracking[security] = price
                filled = o.filled if o.is_buy else -o.filled
                g.positions[security] = filled + current_position
                if g.positions[security] == 0:
                    g.positions.pop(security, None)
                self.hold_list = list(g.positions.keys())
                return True
        return False
    
    # 基础过滤(过滤创业科创北交、ST、停牌、次新股)
    def filter_basic_stock(self, stock_list):
        current_data = get_current_data()
        return [
            stock
            for stock in stock_list
            if not current_data[stock].paused
            and not current_data[stock].is_st
            and "ST" not in current_data[stock].name
            and "*" not in current_data[stock].name
            and "退" not in current_data[stock].name
            and not (stock[0] == "4" or stock[0] == "8" or stock[:2] == "68" or stock[:2] == "30")
            and not self.context.previous_date - get_security_info(stock).start_date < datetime.timedelta(375)
        ]

    # 过滤当前时间涨跌停的股票
    def filter_limitup_limitdown_stock(self, stock_list):
        current_data = get_current_data()
        return [
            stock
            for stock in stock_list
            if current_data[stock].last_price < current_data[stock].high_limit and current_data[stock].last_price > current_data[stock].low_limit
        ]

    # 过滤近几日涨停过的股票
    def filter_limitup_stock(self, stock_list, days):
        df = get_price(
            stock_list,
            end_date=self.context.previous_date,
            frequency="daily",
            fields=["close", "high_limit"],
            count=days,
            panel=False,
        )
        df = df[df["close"] == df["high_limit"]]
        filterd_stocks = df.code.drop_duplicates().tolist()
        return [stock for stock in stock_list if stock not in filterd_stocks]

    # 检查持仓中曾经涨停但当前未涨停的股票
    def _check(self):
        hold = list(g.positions.keys())
        if not hold:
            return []
        current_data = get_current_data()
        filtered = self.filter_limitup_stock(hold, 3)
        return [s for s in hold if s not in filtered and current_data[s].last_price < current_data[s].high_limit]

    # 检查移动止损
    def check_stop_loss(self):
        current_data = get_current_data()
        stop_loss_stocks = []

        for stock in list(g.positions.keys()):
            if current_data[stock].paused:
                continue

            current_price = current_data[stock].last_price
            position = self.context.portfolio.positions[stock]

            if stock not in self.stop_loss_tracking:
                try:
                    hist_data = attribute_history(stock, 20, '1d', ['high'])
                    if not hist_data.empty:
                        self.stop_loss_tracking[stock] = max(position.avg_cost, current_price, hist_data['high'].max())
                    else:
                        self.stop_loss_tracking[stock] = max(position.avg_cost, current_price)
                except:
                    self.stop_loss_tracking[stock] = max(position.avg_cost, current_price)
            else:
                self.stop_loss_tracking[stock] = max(self.stop_loss_tracking[stock], current_price) 

            highest_price = self.stop_loss_tracking[stock]
            if current_price <= highest_price * (1 - self.stop_loss_rate):
                stop_loss_stocks.append(stock)      

        for stock in stop_loss_stocks:
            if self.order_target_value_(stock, 0):
                del self.stop_loss_tracking[stock]

        return stop_loss_stocks

    # 识别无法交易的股票（停牌、涨跌停）
    def filter_untradeable_stock(self, stocks):
        current_data = get_current_data()
        return [
            stock
            for stock in stocks
            if current_data[stock].paused or current_data[stock].last_price in (current_data[stock].high_limit, current_data[stock].low_limit)
        ]

    # 根据调仓逻辑计算最终保留的股票列表
    def get_adjusted_stocks(self, selected, sell):
        fixed = self.filter_untradeable_stock(list(g.positions.keys()))
        sum_val = len(self.def_stocks) if selected == self.def_stocks else self.stock_sum - len(fixed)
        return fixed + [s for s in selected if s not in fixed and s not in sell][:sum_val]

# -------------------- 子策略：搅屎棍策略 --------------------
class JSG_Strategy(Strategy):

    def __init__(self, context, name):
        super().__init__(context, name)

        self.stock_sum = 6
        # 判断买卖点的行业数量
        self.num = 1
        # 空仓的月份
        self.pass_months = [1, 4]

    def getStockIndustry(self, stocks):
        industry = get_industry(stocks)
        return pd.Series({stock: info["sw_l1"]["industry_name"] for stock, info in industry.items() if "sw_l1" in info})

    # 获取市场宽度
    def get_market_breadth(self):
        # 指定日期防止未来数据
        yesterday = self.context.previous_date
        # 获取初始列表
        stocks = get_index_stocks("000985.XSHG")
        count = 3
        h = get_price(
            stocks,
            end_date=yesterday,
            frequency="1d",
            fields=["close"],
            count=count + 20,
            panel=False,
        )
        h["date"] = pd.DatetimeIndex(h.time).date
        df_close = h.pivot(index="code", columns="date", values="close").dropna(axis=0)
        # 计算20日均线
        df_ma20 = df_close.rolling(window=20, axis=1).mean().iloc[:, -count:]
        # 计算偏离程度
        df_bias = df_close.iloc[:, -count:] > df_ma20
        df_bias["industry_name"] = self.getStockIndustry(stocks)
        # 计算行业偏离比例
        df_ratio = ((df_bias.groupby("industry_name").sum() * 100.0) / df_bias.groupby("industry_name").count()).round()
        # 获取偏离程度最高的行业
        top_values = df_ratio.loc[:, yesterday].nlargest(self.num)
        I = top_values.index.tolist()
        return I

    # 过滤股票
    def filter(self):
        stocks = get_index_stocks("399101.XSHE")
        stocks = self.filter_basic_stock(stocks)
        stocks = self.filter_limitup_stock(stocks, 10)
        stocks = get_fundamentals(
            query(
                valuation.code,
            )
            .filter(
                valuation.code.in_(stocks),
                indicator.adjusted_profit > 0,
            )
            .order_by(valuation.market_cap.asc())
        ).code
        stocks = self.filter_limitup_limitdown_stock(stocks)
        return stocks

    # 择时
    def select(self):
        I = self.get_market_breadth()
        industries = {"银行I", "煤炭I", "交通运输I", "钢铁I"}
        if not industries.intersection(I) and not self.context.current_dt.month in self.pass_months:
            return self.filter()[: self.stock_sum * 2]
        return self.def_stocks

    # 调仓
    def adjust(self):
        stocks = self.get_adjusted_stocks(self.select(), [])
        self._adjust({stock: round(1 / len(stocks), 3) for stock in stocks})

    # 检查昨日涨停票
    def check(self):
        banner = self._check()
        if banner:
            stocks = self.get_adjusted_stocks(self.select(), banner)
            self._adjust({s: round(1 / len(stocks), 3) for s in stocks})

# -------------------- 辅助函数 --------------------
# 设置交易成本
def set_trading_costs():
      # 固定滑点设置
      set_slippage(FixedSlippage(0.002),type="stock")
      set_slippage(FixedSlippage(0.001),type="fund")

      # 设置股票交易佣金
      set_order_cost(OrderCost(
          open_tax=0,
          close_tax=0.0005,
          open_commission=0.85 / 10000,
          close_commission=0.85 / 10000,
          close_today_commission=0,
          min_commission=5,
      ), type="stock")

      # 设置ETF交易佣金
      set_order_cost(OrderCost(
          open_tax=0,
          close_tax=0,  # ETF印花税为0
          open_commission=0.5 / 10000,
          close_commission=0.5 / 10000,
          close_today_commission=0,
          min_commission=5
      ), type='fund')

      # 设置货币ETF交易佣金
      set_order_cost(OrderCost(
          open_tax=0,
          close_tax=0,
          open_commission=0,
          close_commission=0,
          close_today_commission=0,
          min_commission=0
      ), type='mmf')