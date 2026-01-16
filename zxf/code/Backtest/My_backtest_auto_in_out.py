import os
import sys
import shutil
try:
    import fire
except ModuleNotFoundError:
    fire = None  # type: ignore
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from typing import Union, Optional
from qlib import init
from qlib.constant import REG_CN

import warnings
import pandas.errors
warnings.filterwarnings('ignore', category=pandas.errors.SettingWithCopyWarning)    # 过滤 SettingWithCopyWarning 警告

dirname = "~/notebook/zxf/code"
if os.path.exists(dirname):
    sys.path.append(dirname)


class Backtest:
    def __init__(
            self,
            exp_no: str = f'master_20251020_csi1000_20150101_20251017',     # 预测值目录
            qlib_path: str = f'/home/idc2/notebook/qlib_bin/cn_data_backtest',     # qlib路径
            top_k: int = 6,                                                 # 每日最大持仓个数
            n_drop: int = 2,                                                # 每日卖出个数
            hold_p: int = 5,                                                # 最小持仓天数(非止损止盈)
            account: float = 100_000_000,  # 100_000_000,                       # 初始资金总额
            buy_cost_rate: float = 0.0015,                                  # 买入费率
            sell_cost_rate: float = 0.0015,                                 # 卖出费率
            min_cost: float = 5,                                            # 交易最小成本
            slippage_ratio: float = 0.001,                                  # 滑动率-暂未启用
            is_set_loss_profit: bool = True,                                # 是否止损止盈
            stop_loss_ratio: float = -0.05,                                  # 止损比例
            take_profit_ratio: float = 1,                                 # 止盈比例
            printing_info: bool = False,                                     # 是否打印过程明细
            market_name: str = "csi800",                                    # 市场
            pred_filename: Union[str, None] = None,                         # 预测文件名
            backtest_start_date: Union[str, None] = None,                   # 回测开始日期
            rf_annual: float = 0.03,                                        # 无风险收益率（年化）
            annual_factor: int = 252,                                       # 年化系数
            output_dir: str = ".",                                          # 输出目录
            output_basename: Optional[str] = None,                          # 输出文件名后缀（不含 account_details_/position_details_ 前缀）
            save_position_details: bool = True                              # 是否保存持仓明细
    ):
        self._exp_no = exp_no
        self._qlib_path = qlib_path
        self._top_k = top_k
        self._n_drop = n_drop
        self._hold_p = hold_p
        self._account = account
        self._buy_cost_rate = buy_cost_rate
        self._sell_cost_rate = sell_cost_rate
        self._min_cost = min_cost
        self._slippage_ratio = slippage_ratio
        self._is_set_loss_profit = is_set_loss_profit
        self._stop_loss_ratio = stop_loss_ratio
        self._take_profit_ratio = take_profit_ratio
        self._printing_info = printing_info
        self._rf_annual = rf_annual
        self._annual_factor = annual_factor


        self._market_name = market_name
        index_map = {
            "csi300": "SH000300",
            "csi500": "SH000905",
            "csi800": "SH000906",
            "csi1000": "SH000852",
            "csiall": "SH000985",
        }
        self._benchmark = index_map.get(market_name, "SH000906")            # 基板

        self._pred_filename = pred_filename
        self._backtest_start_date = backtest_start_date
        self._output_dir = output_dir
        self._output_basename = output_basename
        self._save_position_details = save_position_details

        self._position_details = pd.DataFrame({
            'tradedate': [],                        # 交易日期
            'instrument': [],                       # 编码
            'status': [],                           # 状态: hold,sell,buy
            'holding_period': [],                   # 持仓天数
            'volume': [],                           # 持仓量
            'curr_price': [],                       # 当天价格
            'value': [],                            # 当前市值
            'yield': [],                            # 当前收益率

            'buy_positive_ratio': [],               # 买单预测时positive_ratio值
            'buy_pred_median': [],                  # 买单预测时当天预测值平均值

            'buy_pred': [],                         # 买入预测值
            'buy_price': [],                        # 买入价格
            'buy_cost': [],                         # 买入成本
            'buy_date': [],                         # 买入日期

            'sell_positive_ratio': [],              # 卖单预测时positive_ratio值
            'sell_pred_median': [],                 # 卖单预测时当天预测值平均值

            'sell_pred': [],                        # 卖出预测值
            'sell_price': [],                       # 卖出价格
            'sell_cost': [],                        # 卖出成本

            'sell_profit': [],                      # 销售单收益
            'sell_profit_rate': [],                 # 销售单收益率
            'sell_profit_rate_daily': [],           # 销售单日均收益率
        })                      # 定义交易明细表
        self._account_details = pd.DataFrame({
            'tradedate': [],                        # 交易日期
            'opening_account': [],                  # 期初总额: 上个交易日的closing_account
            'sell_amount': [],                      # 卖出金额
            'sell_cost': [],                        # 卖出成本
            'net_sell_amount': [],                  # 卖出净额: sell_amount - sell_cost = net_sell_amount
            'buy_amount': [],                       # 买入金额
            'buy_cost': [],                         # 买入成本
            'total_buy_amount': [],                 # 买入总额: buy_amount + buy_cost = total_buy_amount
            'closing_account': [],                  # 期末总额: opening_account + net_sell_amount - total_buy_amount = closing_account
            'values': [],                           # 持仓市值
            'estimated_sell_cost': [],              # 持仓卖出预估成本
            'total_assets': [],                     # 总价值(扣除成本)

            'total_return': [],                    # 预估总收益（扣除成本） overall_yield -> total_return
            'annual_return': [],                    # 年化收益（扣除成本）
            'max_drawdown': [],                     # 最大回撤
            'annual_turnover': [],                  # 换手率（年化）
            'volatility': [],                       # 每日收益波动率（年化）
            'sharpe': [],                           # 夏普
            'information_ratio': [],                # 信息率
            'alpha': [],                            # 阿尔法
            'beta': [],                             # 贝塔

            'benchmark_return': [],                 # 基准收益率
            'cumulative_benchmark_return': [],      # 累计基准收益率
            'annual_benchmark_return': [],          # 年化基准收益率

            'positive_ratio_today': [],             # 今日正预测值占比
            'sell_order_number_today':[],           # 今日卖出订单数
            'winning_number_today':[],              # 今日胜出卖单数
            'winning_rate_today': [],               # 今日胜率
            'sell_order_number_cumulant': [],       # 累计卖出订单数
            'winning_number_cumulant':[],           # 累计胜出卖单数
            'winning_rate_cumulant': [],            # 累计胜率
        })                       # 定义账户明细表


    ### 获取个股最大涨跌幅 ------------------------------------------------------------------------------------------------
    def get_stock_info(self):
        s_info = self._df_pred.copy()
        s_info['code'] = s_info['instrument'].str[2:]
        s_info = s_info[['instrument', 'code']]

        s_info =  s_info[['instrument', 'code']].drop_duplicates().reset_index(drop=True)
        s_info['code'] = s_info['code'].astype(int)

        s_info['max_change'] = np.where(s_info['code'] < 300000, 0.095,
                                        np.where(s_info['code'] < 400000, 0.195,
                                                 np.where(s_info['code'] < 680000, 0.095,
                                                          np.where(s_info['code'] < 800000, 0.195, 0.295))))
        s_info = s_info.sort_values(by='code')
        return s_info[['instrument', 'max_change']]


    ### 读取预测值 -------------------------------------------------------------------------------------------------------
    def read_pred(self):
        if self._pred_filename is None:
            predfile_path = f'../../Data/Results/{self._exp_no}/Backtest_results/predictions/master_predictions_backday_8_csi1000_3_29.csv'
        else:
            predfile_path = self._pred_filename

        try:
            df_pred = pd.read_csv(predfile_path)
            logger.info(f"Pred_data_from_file: {predfile_path}")
            df_pred['datetime'] = pd.to_datetime(df_pred['datetime'])
            if self._backtest_start_date is not None:
                self._backtest_start_date = pd.to_datetime(self._backtest_start_date)
                df_pred = df_pred[df_pred['datetime'] >= self._backtest_start_date]
            df_pred = df_pred[~df_pred['score'].isna()]
            df_pred = df_pred.sort_values(by=['datetime','score'])
            return df_pred
        except Exception as e:
            logger.error(f"Pred_data_read_failed: {str(e)}")
            sys.exit(1)


    ### 获取qlib close数据 ----------------------------------------------------------------------------------------------
    def get_qlib_data(self):
        """"""
        try:
            # qlib初始化
            init(provider_uri=self._qlib_path, region=REG_CN)
            from qlib.data import D

            # 获取stock close
            instruments = self._df_pred['instrument'].drop_duplicates().tolist()
            df_close = D.features(instruments, ["$close", "Ref($close, 1)"],
                                  start_time=self._df_pred['datetime'].min(),
                                  end_time=self._df_pred['datetime'].max())
            df_close = df_close.reset_index()
            df_close = df_close.rename(columns={'$close': 'close', 'Ref($close, 1)': 'preclose'})
            df_close['preclose'] = df_close['preclose'].fillna(df_close['close'])
            df_close['change'] = df_close['close'] / df_close['preclose'] - 1


            # 获取基板的close,生成基板的涨跌幅
            benchmark_profit_rate = D.features([self._benchmark], ["$close"],
                                               start_time=self._df_pred['datetime'].min(),
                                               end_time=self._df_pred['datetime'].max())

            benchmark_profit_rate = benchmark_profit_rate.reset_index()
            benchmark_profit_rate = benchmark_profit_rate.rename(columns={'datetime': 'tradedate', '$close': 'close'})

            # 基准收益率(每日收益率)
            benchmark_profit_rate['benchmark_return'] = benchmark_profit_rate['close'].pct_change()

            # 累计基准收益率
            first_close = benchmark_profit_rate['close'].iloc[0]
            benchmark_profit_rate['cumulative_benchmark_return'] = benchmark_profit_rate['close'] / first_close - 1
            benchmark_profit_rate = benchmark_profit_rate[['tradedate', 'benchmark_return', 'cumulative_benchmark_return']]

            return df_close, benchmark_profit_rate
        except Exception as e:
            logger.error(f"Qlib_init_failed: {str(e)}")
            sys.exit(1)


    ### 预测明日买卖 -----------------------------------------------------------------------------------------------------
    def predition(self,trade_date,df_pred_today,df_close_today,positive_ratio_today,pred_median_today):
        ### 初始化数据
        intending_sell = intending_buy = pd.DataFrame()

        # 获取今日最新仓位信息
        try:
            position_details_today = self._position_details[(self._position_details['tradedate'] == trade_date) & (self._position_details['status'].isin(['hold', 'buy']))]
        except Exception as e:
            logger.error(f"Position_details_data_faild: {str(e)}")
            return intending_sell, intending_buy

        ### 拟买
        intending_buy_check = pd.merge(df_pred_today, df_close_today, on=['datetime', 'instrument'], how='left')
        intending_buy_check = intending_buy_check[~intending_buy_check['close'].isna()]        # 排除train补齐数据造成的假象！
        intending_buy_check = intending_buy_check[~intending_buy_check['instrument'].isin(position_details_today['instrument'])]        # 剔除已持仓
        intending_buy_check = intending_buy_check[-self._top_k:]                               # 今日预测值最大的self._top_k只

        if not intending_buy_check.empty:
            # 生成拟购买信息
            intending_buy = intending_buy_check.rename(columns={'datetime': 'tradedate', 'score': 'buy_pred', 'close': 'buy_price'})
            intending_buy['amount_limited'] = 0
            intending_buy['status'] = "buy"
            intending_buy['holding_period'] = 0

            intending_buy['buy_positive_ratio'] = positive_ratio_today
            intending_buy['buy_pred_median'] = pred_median_today

            intending_buy['buy_date'] = ""
            intending_buy[
                ['curr_price', 'value', 'yield', 'buy_cost', 'sell_pred', 'sell_price', 'sell_cost',
                 'sell_positive_ratio', 'sell_pred_median', 'sell_profit', 'sell_profit_rate', 'sell_profit_rate_daily'
                 ]] = 0.0
            intending_buy = intending_buy.drop(['preclose', 'change'], axis=1)


        ### 拟卖
        if not position_details_today.empty:
            # 拟卖基础数据
            position_details_sell_list = pd.merge(position_details_today, df_pred_today, on='instrument', how='left')
            position_details_sell_list = pd.merge(position_details_sell_list, df_close_today, on=['instrument', 'datetime'], how='left')
            position_details_sell_list['holding_period'] += 1
            position_details_sell_list = position_details_sell_list.sort_values(by='score', ascending=True)

            # 拟卖筛选
            # 排除前top位和拟买的股票
            df_pred_today_top = df_pred_today[-self._top_k:]
            combined_set = set(df_pred_today_top['instrument']) | set(intending_buy['instrument'])
            position_details_sell_check = position_details_sell_list[~position_details_sell_list['instrument'].isin(combined_set)]
            # 持仓不小于hold_p
            position_details_sell_check = position_details_sell_check[position_details_sell_check['holding_period'] >= self._hold_p]
            # n_drop
            position_details_sell_check = position_details_sell_check[:self._n_drop]

            # 是否止损止盈
            if self._is_set_loss_profit:
                set_loss_con = (position_details_sell_list['close'] / position_details_sell_list['buy_price'] - 1) <= self._stop_loss_ratio # 止损
                set_profit_con = (position_details_sell_list['close'] / position_details_sell_list['buy_price'] - 1) >= self._take_profit_ratio # 止盈
                condition_loss_profit = set_loss_con | set_profit_con
                position_details_loss_profit = position_details_sell_list[condition_loss_profit]

                position_details_sell_check = pd.concat([position_details_sell_check, position_details_loss_profit],ignore_index=True)
                position_details_sell_check = position_details_sell_check.drop_duplicates()

            # 处理拟卖信息
            if len(position_details_sell_check) > 0:
                position_details_sell_check['sell_pred'] = position_details_sell_check['score']
                position_details_sell_check['sell_price'] = np.where(position_details_sell_check['close'].isna(),
                                                                     position_details_sell_check['curr_price'],
                                                                     position_details_sell_check['close'])
                position_details_sell_check['value'] = position_details_sell_check['sell_price'] * \
                                                       position_details_sell_check['volume']
                position_details_sell_check['sell_cost'] = ((self._sell_cost_rate * position_details_sell_check['value']).
                                                            clip(lower=self._min_cost).round(2))
                position_details_sell_check['status'] = "sell"
                position_details_sell_check['sell_positive_ratio'] = positive_ratio_today
                position_details_sell_check['sell_pred_median'] = pred_median_today
                intending_sell = position_details_sell_check.drop(['datetime', 'score', 'close', 'change', 'preclose'], axis=1)
                intending_sell = intending_sell.drop_duplicates(ignore_index=True)

        return intending_sell, intending_buy


    ### 执行卖出 --------------------------------------------------------------------------------------------------------
    def sell(self, trade_date, previous_trade_date, sell_intending_yesterday, df_close_today, df_pred_today, positive_ratio_today, pred_median_today):
        ### 初始化参数
        sell_details = pd.DataFrame()
        sell_amount = sell_cost = net_sell_amount = 0

        ### 是否止损止盈
        set_loss_profit = pd.DataFrame()
        if self._is_set_loss_profit:
            position_details_yesterday = self._position_details[
                (self._position_details['tradedate'] == previous_trade_date) & (self._position_details['status'].isin(['hold', 'buy']))]
            position_details_yesterday['holding_period'] += 1
            set_loss_profit = pd.merge(position_details_yesterday, df_close_today, on='instrument', how='left')
            set_loss_profit = set_loss_profit[~set_loss_profit['close'].isna()]

            set_loss_con = (set_loss_profit['close'] / set_loss_profit['buy_price'] - 1) <= self._stop_loss_ratio  # 止损
            set_profit_con = (set_loss_profit['close'] / set_loss_profit['buy_price'] - 1) >= self._take_profit_ratio  # 止盈
            condition_loss_profit = set_loss_con | set_profit_con
            set_loss_profit = set_loss_profit[condition_loss_profit]

            # 更新止损止盈卖单信息
            set_loss_profit = pd.merge(set_loss_profit, df_pred_today, on=['instrument','datetime'], how='left')
            set_loss_profit['status'] = "sell"
            set_loss_profit['sell_pred'] = set_loss_profit['score']
            set_loss_profit = set_loss_profit.drop(['score'], axis=1)

            set_loss_profit['sell_positive_ratio'] = positive_ratio_today
            set_loss_profit['sell_pred_median'] = pred_median_today

        ### 预测卖单
        sell_details_temp = pd.DataFrame()
        if not sell_intending_yesterday.empty:
            sell_details_temp = pd.merge(sell_intending_yesterday, df_close_today, on='instrument', how='left')
            sell_details_temp = sell_details_temp[~sell_details_temp['close'].isna()]

        ### 合并预测卖单+止损盈卖单,合并重复数据,保留最后一行数据
        sell_details_temp = pd.concat([sell_details_temp, set_loss_profit], ignore_index=True)
        sell_details_temp = sell_details_temp.drop_duplicates(
            subset=['tradedate', 'instrument', 'status', 'holding_period', 'buy_date'],
            keep='last',
            ignore_index=True)
        if sell_details_temp.empty:
            return sell_details, sell_amount, sell_cost, net_sell_amount

        ### 跌停不卖
        sell_details_temp = pd.merge(sell_details_temp, self._stock_max_change, on='instrument', how='left')
        sell_details_temp = sell_details_temp[sell_details_temp['change'] >= -sell_details_temp['max_change']]
        if sell_details_temp.empty:
            return sell_details, sell_amount, sell_cost, net_sell_amount

        ### 更新卖单信息
        sell_details = sell_details_temp.copy()
        sell_details['curr_price'] = sell_details['close']
        sell_details['sell_price'] = sell_details['close']
        sell_details['value'] = sell_details['curr_price'] * sell_details['volume']
        sell_details['yield'] = sell_details['curr_price'] / sell_details['buy_price'] - 1
        sell_details['sell_cost'] = (self._sell_cost_rate * sell_details['value']).clip(lower=self._min_cost).round(2)
        sell_details['tradedate'] = trade_date

        sell_details['sell_profit'] = (sell_details['sell_price'] -  sell_details['buy_price']) * sell_details['volume'] - sell_details['buy_cost'] - sell_details['sell_cost']
        sell_details['sell_profit_rate'] = sell_details['sell_profit'] / (sell_details['buy_price'] * sell_details['volume'] + sell_details['buy_cost'])
        sell_details['sell_profit_rate_daily'] = sell_details['sell_profit_rate'] / sell_details['holding_period']

        sell_details = sell_details.drop(['datetime', 'close', 'change', 'preclose', 'max_change'], axis=1)

        sell_amount = sell_details['value'].sum()
        sell_cost = sell_details['sell_cost'].sum()
        net_sell_amount = sell_amount - sell_cost

        return sell_details, sell_amount, sell_cost, net_sell_amount


    ### 执行买入 --------------------------------------------------------------------------------------------------------
    def buy(self, trade_date, buy_intending_yesterday, df_close_today, opening_account, sell_net_today, sell_order_number, position_details_yesterday):
        ### 初始化参数
        buy_details = pd.DataFrame()
        buy_amount = buy_cost = total_buy_amount = 0

        ### 无拟买信息,跳过
        if buy_intending_yesterday.empty:
            return buy_details, buy_amount, buy_cost, total_buy_amount

        ### 判断可买股票只数
        buy_number = self._top_k - len(position_details_yesterday) + sell_order_number
        if buy_number <= 0:
            return buy_details, buy_amount, buy_cost, total_buy_amount

        ### 判断实际可用资金
        actual_account = (opening_account + sell_net_today)
        if actual_account <= 0:
            return buy_details, buy_amount, buy_cost, total_buy_amount

        ### 今日拟买
        buy_details_temp = pd.merge(buy_intending_yesterday, df_close_today, on='instrument', how='left')
        buy_details_temp = buy_details_temp[~buy_details_temp['close'].isna()]         # 过滤今日未开盘
        buy_details_temp = buy_details_temp[-buy_number:]

        if buy_details_temp.empty:
            return buy_details, buy_amount, buy_cost, total_buy_amount

        ### 若当天涨停,不买
        buy_details_temp = pd.merge(buy_details_temp, self._stock_max_change, on='instrument', how='left')
        buy_details_temp = buy_details_temp[buy_details_temp['change'] <= buy_details_temp['max_change']]
        if buy_details_temp.empty:
            return buy_details, buy_amount, buy_cost, total_buy_amount

        ### 检查资金是否可购买
        for x in range(len(buy_details_temp), 0, -1):  # 倒计数循环,不包括0
            buy_details_check = buy_details_temp[-x:]
            amount_of_stock = actual_account * 0.99 / x             # 根据实际资金缩放拟买金额（保留1%现金垫）

            buy_details_check['volume'] = (np.floor((amount_of_stock / buy_details_check['close']) / 100) * 100).astype(int)  # 以手为单位购买
            buy_details_check['amount_limited'] = amount_of_stock
            if (buy_details_check['volume'] >= 100).all():
                break
            else:
                buy_details_check = pd.DataFrame()

        if buy_details_check.empty:
            return buy_details, buy_amount, buy_cost, total_buy_amount
        else:
            buy_details = buy_details_check.copy()

        ### 更新买入订单信息
        buy_details['tradedate'] = buy_details['datetime']
        buy_details['buy_price'] = buy_details['close']

        buy_details['value'] = buy_details['volume'] * buy_details['buy_price']
        buy_details['buy_cost'] = (self._buy_cost_rate * buy_details['value']).clip(lower=self._min_cost).round(2)
        buy_details['curr_price'] = buy_details['buy_price']

        buy_details['holding_period'] = 0
        buy_details['buy_date'] = trade_date
        buy_details['yield'] = 0.0

        buy_amount = buy_details['value'].sum()
        buy_cost = buy_details['buy_cost'].sum()
        total_buy_amount = buy_amount + buy_cost

        buy_details = buy_details.drop(['datetime', 'close', 'amount_limited', 'preclose', 'change','max_change'], axis=1)

        return buy_details, buy_amount, buy_cost, total_buy_amount


    ### 更新持仓 --------------------------------------------------------------------------------------------------------
    def update_position(
            self, trade_date,previous_trade_date,df_close_today,position_details_yesterday,opening_account,sell_details_today,sell_amount_today,
            sell_cost_today,sell_net_today,buy_details_today,buy_amount_today, buy_cost_today,buy_total_amount_today,positive_ratio_today
    ):
        ### 更新持仓信息
        position_details_today = position_details_yesterday.copy()

        # 去除已卖掉部分
        if not sell_details_today.empty:
            sell_details_check = sell_details_today.copy()
            sell_details_check = sell_details_check[['instrument', 'buy_date']]
            sell_details_check['sell'] = True
            position_details_today = pd.merge(position_details_today, sell_details_check, on=['instrument', 'buy_date'], how='left')
            position_details_today = position_details_today[position_details_today['sell'].isna()]
            position_details_today = position_details_today.drop(['sell'], axis=1)

        # 更新持仓信息
        position_details_today['tradedate'] = trade_date
        position_details_today['holding_period'] += 1
        position_details_today['status'] = 'hold'

        # 刷新价格
        position_details_today = pd.merge(
            position_details_today, df_close_today,
            left_on=['tradedate', 'instrument'],
            right_on=['datetime', 'instrument'],
            how='left')
        position_details_today['curr_price'] = np.where(
            position_details_today['close'].isna(),
            position_details_today['curr_price'],
            position_details_today['close']
        )

        position_details_today = position_details_today.drop(['datetime', 'close', 'preclose', 'change'], axis=1)
        position_details_today['value'] = position_details_today['curr_price'] * position_details_today['volume']
        position_details_today['yield'] = position_details_today['curr_price'] / position_details_today['buy_price'] - 1

        # 并入新购信息&卖出信息
        position_details_today = pd.concat([position_details_today, buy_details_today], ignore_index=True)
        position_details_today = pd.concat([position_details_today, sell_details_today], ignore_index=True)

        values_today = position_details_today.loc[position_details_today['status'].isin(['hold', 'buy']), 'value'].sum()
        if not position_details_today.empty:
            self._position_details = pd.concat([self._position_details, position_details_today], ignore_index=True)


        ### 更新账户信息
        account_details_today = pd.DataFrame([{
            'tradedate': trade_date,
            'opening_account': opening_account,
            'sell_amount': sell_amount_today,
            'sell_cost': sell_cost_today,
            'net_sell_amount': sell_net_today,
            'buy_amount': buy_amount_today,
            'buy_cost': buy_cost_today,
            'total_buy_amount': buy_total_amount_today,
        }])
        account_details_today['closing_account'] = account_details_today['opening_account'] + \
                                                   account_details_today['net_sell_amount'] - \
                                                   account_details_today['total_buy_amount']
        account_details_today['values'] = values_today

        account_details_today['estimated_sell_cost'] = (self._sell_cost_rate * account_details_today['values']).clip(
            lower=self._min_cost).round(2)
        account_details_today['total_assets'] = account_details_today['closing_account'] + account_details_today[
            'values'] - account_details_today['estimated_sell_cost']
        account_details_today['total_return'] = account_details_today['total_assets'] / self._account - 1

        account_details_today['positive_ratio_today'] = positive_ratio_today

        # # 年化收益率扣成本
        # date_num = pd.to_datetime(trade_date) - pd.to_datetime(self._first_date)
        # account_details_today['annual_return'] = account_details_today['total_return'] * (365 / date_num.days)

        account_details_yesterday = self._account_details[self._account_details['tradedate'] == previous_trade_date]

        if len(account_details_yesterday) > 0:
            account_details_today['sell_order_number_cumulant'] = account_details_yesterday['sell_order_number_cumulant'].iloc[0]
            account_details_today['winning_number_cumulant'] = account_details_yesterday['winning_number_cumulant'].iloc[0]
            account_details_today['winning_rate_cumulant'] = account_details_yesterday['winning_rate_cumulant'].iloc[0]

            account_details_today['sell_order_number_cumulant'] = account_details_today['sell_order_number_cumulant'].fillna(0)
            account_details_today['winning_number_cumulant'] = account_details_today['winning_number_cumulant'].fillna(0)
            account_details_today['winning_rate_cumulant'] = account_details_today['winning_rate_cumulant'].fillna(0)
        else:
            account_details_today['sell_order_number_cumulant'] = 0
            account_details_today['winning_number_cumulant'] = 0
            account_details_today['winning_rate_cumulant'] = 0

        _sell_order_number_today  = len(sell_details_today)
        if _sell_order_number_today > 0:
            _winning_order_number_today = len(sell_details_today[sell_details_today['sell_profit_rate'] > 0])

            account_details_today['sell_order_number_today'] = _sell_order_number_today
            account_details_today['winning_number_today'] = _winning_order_number_today
            account_details_today['winning_rate_today'] = _winning_order_number_today / _sell_order_number_today

            account_details_today['sell_order_number_cumulant'] += _sell_order_number_today
            account_details_today['winning_number_cumulant'] += _winning_order_number_today
            account_details_today['winning_rate_cumulant'] = account_details_today['winning_number_cumulant'] / \
                                                               account_details_today['sell_order_number_cumulant']
        else:
            account_details_today['sell_order_number_today'] = 0
            account_details_today['winning_number_today'] = 0
            account_details_today['winning_rate_today'] = 0

        # 基准收益率(每日)
        _benchmark_return = self._benchmark_profit_rate.loc[self._benchmark_profit_rate['tradedate'] == trade_date,'benchmark_return'].iloc[0]
        account_details_today['benchmark_return'] = _benchmark_return

        # 累计基准收益率
        _cumulative_benchmark_return = self._benchmark_profit_rate.loc[self._benchmark_profit_rate['tradedate'] == trade_date,'cumulative_benchmark_return'].iloc[0]
        account_details_today['cumulative_benchmark_return'] = _cumulative_benchmark_return

        # 回测结果评估数据
        df_evaluation = pd.concat([self._account_details, account_details_today], ignore_index=True)
        # print(df_evaluation)

        if len(df_evaluation) == 1:
        # if len(df_evaluation) < 2:
            account_details_today['annual_return'] = account_details_today['total_return']
            account_details_today['annual_benchmark_return'] = _benchmark_return
            account_details_today[
                ['volatility', 'sharpe', 'alpha', 'beta', 'information_ratio', 'max_drawdown', 'annual_turnover']] = 0
        else:
            metrics = self.evaluation(df_evaluation)
            account_details_today = pd.concat([account_details_today, pd.DataFrame([metrics])], axis=1)

        # 记录到账户明细表中
        self._account_details = pd.concat([self._account_details, account_details_today], ignore_index=True)

    ### 运行回测  --------------------------------------------------------------------------------------------------------
    def run(self):
        ### 初始化数据
        self._df_pred = self.read_pred()        # 先预测值
        self._df_close, self._benchmark_profit_rate = self.get_qlib_data()
        self._stock_max_change = self.get_stock_info()

        ### 回测日历
        backtest_date_list = self._df_pred['datetime'].drop_duplicates().tolist()
        self._first_date = backtest_date_list[0]

        ### 按日回测
        previous_trade_date = None
        for trade_date in backtest_date_list:
            ### 预制数据
            df_close_today = self._df_close[self._df_close['datetime'] == trade_date]
            df_pred_today = self._df_pred[self._df_pred['datetime'] == trade_date]
            positive_ratio_today = len(df_pred_today[df_pred_today['score'] > 0]) / len(df_pred_today)
            pred_median_today = df_pred_today['score'].median()

            position_details_yesterday = self._position_details[
                (self._position_details['tradedate'] == previous_trade_date) & (self._position_details['status'].isin(['hold', 'buy']))]

            ### 回测过程 -- 首日
            if self._first_date == trade_date:
                # 预测首日拟买
                opening_account = self._account
                sell_intending_today, buy_intending_today = self.predition(
                    trade_date=trade_date,
                    df_pred_today=df_pred_today,
                    df_close_today=df_close_today,
                    positive_ratio_today=positive_ratio_today,
                    pred_median_today=pred_median_today
                )
                buy_details_today = sell_details_today = pd.DataFrame()

            ### 回测过程 -- 非首日
            else:
                # 账户余额更新
                if self._account_details.empty:
                    opening_account = self._account
                else:
                    opening_account = \
                    self._account_details.loc[self._account_details['tradedate'] == previous_trade_date, 'closing_account'].values[0]

                # 执行卖单
                sell_details_today, sell_amount_today, sell_cost_today, sell_net_today = self.sell(
                        trade_date=trade_date,
                        previous_trade_date=previous_trade_date,
                        sell_intending_yesterday=sell_intending_yesterday,
                        df_close_today=df_close_today,
                        df_pred_today=df_pred_today,
                        positive_ratio_today=positive_ratio_today,
                        pred_median_today=pred_median_today,
                    )

                # 执行买单
                sell_order_number = len(sell_details_today)
                buy_details_today, buy_amount_today, buy_cost_today, buy_total_amount_today = self.buy(
                        trade_date=trade_date,
                        buy_intending_yesterday=buy_intending_yesterday,
                        df_close_today=df_close_today,
                        opening_account=opening_account,
                        sell_net_today=sell_net_today,
                        sell_order_number=sell_order_number,
                        position_details_yesterday=position_details_yesterday
                    )

                # 更新持仓
                self.update_position(
                    trade_date=trade_date,
                    previous_trade_date=previous_trade_date,
                    df_close_today=df_close_today,
                    position_details_yesterday=position_details_yesterday,
                    opening_account=opening_account,
                    sell_details_today=sell_details_today,
                    sell_amount_today=sell_amount_today,
                    sell_cost_today=sell_cost_today,
                    sell_net_today=sell_net_today,
                    buy_details_today=buy_details_today,
                    buy_amount_today=buy_amount_today,
                    buy_cost_today=buy_cost_today,
                    buy_total_amount_today=buy_total_amount_today,
                    positive_ratio_today=positive_ratio_today
                )

                # 预测明日: 拟买/拟卖
                sell_intending_today, buy_intending_today = self.predition(
                    trade_date=trade_date,
                    df_pred_today=df_pred_today,
                    df_close_today=df_close_today,
                    positive_ratio_today=positive_ratio_today,
                    pred_median_today=pred_median_today,
                )


            ### 往下个交易日 传递参数
            previous_trade_date = trade_date
            sell_intending_yesterday = sell_intending_today
            buy_intending_yesterday = buy_intending_today

            ### 打印
            if self._printing_info:
                print("\n" + f"*** Trade_date: [{trade_date.date()}] " + "*" * 150)
                print("--- 上一日持仓明细： --- ")
                print(f"{None if position_details_yesterday.empty else position_details_yesterday.to_string(max_cols=None)}")

                print("--- 今日卖： ---")
                print(f"{None if sell_details_today.empty else sell_details_today.to_string(max_cols=None)}")

                print("--- 今日买： --- ")
                print(f"{None if buy_details_today.empty else buy_details_today.to_string(max_cols=None)}")

                print("--- 交易后持仓： --- ")
                position_details_t = self._position_details[self._position_details['tradedate'] == trade_date]
                print(f"{None if position_details_t.empty else position_details_t.to_string(max_cols=None)}")

                print(f"--- 期初账户余额:{round(opening_account,2)} ---")
                print("--- 账户信息： ---")
                account_details_today_t = self._account_details[self._account_details['tradedate'] == trade_date]
                print(f"{None if account_details_today_t.empty else account_details_today_t.to_string(max_cols=None)}")

                print("--- 明日拟卖： --- ")
                print(f"{None if sell_intending_today.empty else sell_intending_today.to_string(max_cols=None)}")
                print("--- 明日可买： --- ")
                print(f"{None if buy_intending_today.empty else buy_intending_today.to_string(max_cols=None)}")
                print(f"--- 行情趋势： {round(positive_ratio_today, 2)} ---")

        # 保存最终结果
        os.makedirs(self._output_dir, exist_ok=True)

        if self._output_basename is not None and str(self._output_basename).strip() != "":
            basename = str(self._output_basename).strip()
            account_path = os.path.join(self._output_dir, f"account_details_{basename}.csv")
            position_path = os.path.join(self._output_dir, f"position_details_{basename}.csv")
        else:
            account_path = os.path.join(self._output_dir, f"account_details_{self._top_k}_{self._n_drop}_{self._hold_p}.csv")
            position_path = os.path.join(self._output_dir, f"position_details_{self._top_k}_{self._n_drop}_{self._hold_p}.csv")

        self._account_details.to_csv(
            account_path, index=False, date_format="%Y-%m-%d", float_format="%.3f"
        )
        logger.info(f"Saved account_details: {account_path}")

        if self._save_position_details:
            self._position_details.to_csv(
                position_path, index=False, date_format="%Y-%m-%d", float_format="%.3f"
            )
            logger.info(f"Saved position_details: {position_path}")

        return self._account_details

        


    ### 回测结果评估 ------------------------------------------------------------------------------------------------------
    def evaluation(self,df_evaluation):

        # 无风险利率年化换算为日化
        rf = (1 + self._rf_annual) ** (1 / 252) - 1

        df = df_evaluation.copy()

        # 计算每日收益率 ret_t
        df['ret'] = df['total_assets'].pct_change()
        df = df.dropna(subset=['ret', 'benchmark_return'])

        # 计算超额收益：ret - benchmark
        df['excess_ret'] = df['ret'] - df['benchmark_return']
        df['ret_rf'] = df['ret'] - rf
        df['excess_ret_rf'] = df['ret'] - rf - df['benchmark_return']

        # 收益波动率（年化）
        volatility = df['ret'].std() * np.sqrt(self._annual_factor)

        # 夏普率
        sharpe = df['ret_rf'].mean() / df['ret_rf'].std() * np.sqrt(self._annual_factor)

        # alpha, beta              
        # cov = np.cov(df['ret'], df['benchmark_return'])[0][1]
        # var_bench = np.var(df['benchmark_return'])
        # beta = cov / var_bench
        # alpha = (df['ret'].mean() - beta * df['benchmark_return'].mean()) * self._annual_factor

        beta = np.cov(df['ret_rf'], df['benchmark_return'] - rf, ddof=0)[0, 1] / np.var(df['benchmark_return'] - rf, ddof=0)
        alpha = (df['ret_rf'].mean() - beta * (df['benchmark_return'] - rf).mean()) * self._annual_factor

        # 信息比率 IR
        ir = df['excess_ret'].mean() / df['excess_ret'].std() * np.sqrt(self._annual_factor)

        # 最大回撤（MDD）
        df['cummax'] = df['total_assets'].cummax()
        df['drawdown'] = (df['total_assets'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()

        # 年化换手率:换手率 = 成交额（买+卖） / 当日资产
        df['turnover'] = (df['total_buy_amount'] + df['sell_amount']) / df['total_assets']
        annual_turnover = df['turnover'].mean() * self._annual_factor


        metrics = {
            "annual_return": (1 + df['ret']).prod() ** (self._annual_factor / len(df)) - 1,
            "annual_benchmark_return": (1 + df['benchmark_return']).prod() ** (self._annual_factor / len(df)) - 1,
            "volatility": volatility,
            "sharpe": sharpe,
            "alpha": alpha,
            "beta": beta,
            "information_ratio": ir,
            "max_drawdown": max_drawdown,
            "annual_turnover": annual_turnover,
        }

        return metrics

DEFAULT_SEED3_FILES = [
    "master_predictions_csi800_5.csv",
    "master_predictions_csi800_7.csv",
    "master_predictions_csi800_8.csv",
    "TimeMixerPP_predictions_csi800b_0.csv",
    "ucast_predictions_csi800c_0.csv",
]


def main(
    src_dir: str = "~/notebook/MASTER",
    dst_dir: str = "~/notebook/zxf/data/Daily_data/Good_seed/seed3",
    output_dir: str = "~/notebook/zxf/data/Daily_data/Good_seed/seed3/backtest_output",
    files: Optional[str] = None,
    qlib_path: str = "~/notebook/qlib_bin/cn_data_backtest",
    top_k: int = 30,
    n_drop: int = 3,
    hold_p: int = 5,
    account: float = 100_000_000,
    buy_cost_rate: float = 0.0015,
    sell_cost_rate: float = 0.0015,
    min_cost: float = 5,
    slippage_ratio: float = 0.001,
    is_set_loss_profit: bool = True,
    stop_loss_ratio: float = -0.05,
    take_profit_ratio: float = 1,
    market_name: str = "csi800",
    printing_info: bool = False,
    backtest_start_date: Optional[str] = None,
):
    """
    功能：
    1) 将 src_dir 下指定预测文件复制到 dst_dir
    2) 逐个文件回测，输出写入 output_dir
    3) 输出文件命名：account_details_<输入文件名去掉.csv>.csv / position_details_<...>.csv

    files 支持逗号分隔，例如：
    --files="a.csv,b.csv,c.csv"
    """

    src_dir = os.path.expanduser(src_dir)
    dst_dir = os.path.expanduser(dst_dir)
    output_dir = os.path.expanduser(output_dir)

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if files is None or str(files).strip() == "":
        filename_list = DEFAULT_SEED3_FILES
    else:
        filename_list = [x.strip() for x in str(files).split(",") if x.strip()]

    results = []
    for filename in filename_list:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source prediction file not found: {src_path}")

        shutil.copy2(src_path, dst_path)
        logger.info(f"Copied: {src_path} -> {dst_path}")

        basename = Path(filename).stem
        backtest = Backtest(
            qlib_path=qlib_path,
            top_k=top_k,
            n_drop=n_drop,
            hold_p=hold_p,
            account=account,
            buy_cost_rate=buy_cost_rate,
            sell_cost_rate=sell_cost_rate,
            min_cost=min_cost,
            slippage_ratio=slippage_ratio,
            is_set_loss_profit=is_set_loss_profit,
            stop_loss_ratio=stop_loss_ratio,
            take_profit_ratio=take_profit_ratio,
            market_name=market_name,
            printing_info=printing_info,
            pred_filename=dst_path,
            backtest_start_date=backtest_start_date,
            output_dir=output_dir,
            output_basename=basename,
            save_position_details=True,
        )
        backtest.run()

        results.append(
            {
                "pred_file": dst_path,
                "account_details": os.path.join(output_dir, f"account_details_{basename}.csv"),
                "position_details": os.path.join(output_dir, f"position_details_{basename}.csv"),
            }
        )

    return results


if __name__ == "__main__":
    fire.Fire(main)
   
