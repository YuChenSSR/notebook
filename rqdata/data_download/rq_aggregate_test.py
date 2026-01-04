import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Union
import time
import fire
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import scipy.stats as st
import multiprocessing

"""1分钟线聚合日线"""

def get_stock_list(stock_list_path):
    stock_list = pd.read_csv(stock_list_path)
    stock_list_s = stock_list[['order_book_id', 'symbol', 'status', 'listed_date', 'de_listed_date']]
    stock_list_s = stock_list_s[stock_list_s['status']!="Unknown"]
    stock_list_s['listed_date'] = pd.to_datetime(stock_list_s['listed_date'], errors='coerce')
    stock_list_s['de_listed_date'] = pd.to_datetime(stock_list_s['de_listed_date'], errors='coerce')
    return stock_list_s


def read_csv_file(csv_path):
    return pd.read_csv(csv_path)

def aggregate_daily_features(df, use_amount=True):
    """ new[24]聚合多种日级别特征:
    1. 日内波动率与基于分钟收益的统计特征[4]
        d_vol：                   分钟收益波动率
        d_ret_skew：              分钟收益偏度
        d_ret_kurt：              分钟收益峰度
        d_realized_vol：          实际波动率 RV（平方收益和）


    2. 成交量结构类[4]
        d_hhi:                     成交量集中度
        d_vol_skew：               成交量偏度
        d_vol_kurt：               成交量峰度
        d_vol_cv：                 成交量变异系数

    3. 微结构特征（交易行为）[5]
        d_corr:	                   close-volume 相关性
        d_dwr:                     成交量加权收益
        d_trades_vol_corr：        num_trades 与 volume 的相关性
        d_trades_close_corr：      num_trades 与 close 的相关性
        d_volatility_of_volume：   1min 成交量波动率

    4. 时间窗口切片特征（开盘/收盘/午盘）[5]
        d_vwap:                    VWAP(全天)
        d_open_vwap：              开盘 30 分钟 VWAP
        d_close_vwap：             收盘 30 分钟 VWAP
        d_am_vwap：                上午 vwap
        d_pm_vwap：                下午 vwap

    5. 价格路径特征[6]
        d_cntp:                     上涨比例
        d_cntn:                     下跌比例
        d_intraday_trend：          日内线性回归趋势 slope
        d_high_time_frac            高点出现的时间占比 
        d_low_time_frac：           低点出现的时间占比
        d_close_rank：              收盘价在日内的 rank（分位数）
    """
    # 聚合结果列名
    agg_results_cols = [
            'd_vol','d_ret_skew','d_ret_kurt','d_realized_vol',
            'd_hhi','d_vol_skew','d_vol_kurt','d_vol_cv',
            'd_corr','d_dwr', 'd_trades_vol_corr','d_trades_close_corr','d_volatility_of_volume',
            'd_vwap','d_open_vwap','d_close_vwap','d_am_vwap','d_pm_vwap', 
            'd_cntp','d_cntn','d_intraday_trend','d_high_time_frac','d_low_time_frac','d_close_rank'
    ]
    
    if df.empty:
        return pd.DataFrame()

    ### 1. 数据加工&清洗
    df = df.copy()
    
    df['date'] = df['datetime'].dt.date
    df['date'] = pd.to_datetime(df['date'])
    
    df = df.dropna(subset=['date', 'datetime', 'order_book_id'])
    

    
    ### 基础列转 float
    for col in ['high', 'low', 'close', 'open', 'volume', 'amount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    ### 排序
    df = df.sort_values(['date', 'datetime'])
   
    # —— 预计算典型价格 —— #
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv_tp'] = df['typical_price'] * df['volume']

    # —— 分组对象 —— #
    groups = df.groupby(['date', 'order_book_id'], sort=False)

    # 核心功能函数
    # ==================================================
    def calc_all_features(g):
        g = g.sort_values('datetime')
        
        out = {}

        # 基础统计
        vol_sum = g['volume'].sum()
        vol_share = g['volume'] / (vol_sum + 1e-12)
        vol = g['volume'].astype(float)
        
        amount_sum = g['amount'].sum() if 'amount' in g else None

        close = g['close'].astype(float)
        r = np.log(close / close.shift(1)).dropna()
        
        ### 聚合计算

        # 1 日内波动率与基于分钟收益的统计特征[4]
        out['d_vol'] = r.std()
        out['d_ret_skew'] = r.skew()
        out['d_ret_kurt'] = r.kurt()
        out['d_realized_vol'] = np.sqrt((r ** 2).sum())


        # 2成交量结构类  
        out['d_hhi'] = (vol_share ** 2).sum()
        out['d_vol_skew'] = vol.skew()
        out['d_vol_kurt'] = vol.kurt()
        out['d_vol_cv'] = vol.std() / (vol.mean() + 1e-12)

        # 3微结构特征（交易行为）    
        if g['close'].std() > 0 and g['volume'].std() > 0:
            out['d_corr'] = g['close'].corr(g['volume'])
        else:
            out['d_corr'] = np.nan

        weight_col = 'amount' if use_amount and 'amount' in g else 'volume'
        w = g[weight_col].iloc[1:].astype(float)
        out['d_dwr'] = (w * r).sum() / (w.sum() + 1e-12) if len(r) > 1 else np.nan

        if 'num_trades' in g.columns and g['num_trades'].std() > 0 and g['volume'].std() > 0:
            out['d_trades_vol_corr'] = g['num_trades'].corr(g['volume'])
        else:
            out['d_trades_vol_corr'] = np.nan        

        if 'num_trades' in g.columns and g['num_trades'].std() > 0 and g['close'].std() > 0:
            out['d_trades_close_corr'] = g['num_trades'].corr(g['close'])
        else:
            out['d_trades_close_corr'] = np.nan

        out['d_volatility_of_volume'] = vol.std()
            
        # 4时间窗口切片特征（开盘/收盘/午盘）
        out['d_vwap'] = g['pv_tp'].sum() / (vol_sum + 1e-12)

        time = g['datetime']
        g_open30 = g[time < (time.iloc[0] + pd.Timedelta(minutes=30))]
        g_close30 = g[time > (time.iloc[-1] - pd.Timedelta(minutes=30))]
        g_am = g[(time.dt.hour < 12)]
        g_pm = g[(time.dt.hour >= 12)]

        def safe_vwap(sub):
            if len(sub) == 0:
                return np.nan
            return (sub['pv_tp'].sum() / (sub['volume'].sum() + 1e-12))

        out['d_open_vwap'] = safe_vwap(g_open30)
        out['d_close_vwap'] = safe_vwap(g_close30)
        out['d_am_vwap'] = safe_vwap(g_am)
        out['d_pm_vwap'] = safe_vwap(g_pm)

        # 5价格路径特征
        'd_cntp','d_cntn','d_intraday_trend','d_high_time_frac','d_low_time_frac','d_close_rank'

        prev = close.shift(1).fillna(g['open'].iloc[0])
        out['d_cntp'] = (close > prev).mean()
        out['d_cntn'] = (close < prev).mean()

        
        x = np.arange(len(close))
        if len(close) > 2:
            slope, _, _, _, _ = st.linregress(x, close)
            out['d_intraday_trend'] = slope
        else:
            out['d_intraday_trend'] = np.nan

        idx_high = g['high'].idxmax()
        idx_low = g['low'].idxmin()
        t_high = g.index.get_loc(idx_high) / len(g)
        t_low = g.index.get_loc(idx_low) / len(g)
        out['d_high_time_frac'] = t_high
        out['d_low_time_frac'] = t_low


        out['d_close_rank'] = (close.iloc[-1] - close.min()) / (close.max() - close.min() + 1e-12)

        return pd.Series(out)

    daily = groups.apply(calc_all_features).reset_index()

    return daily


def process_single_stock():
    
    code = "000001.XSHE"
    data_path = "/home/idc2/notebook/rqdata/Data"
    start_date = "2025-01-01"
    end_date = "2025-10-31"

    try:
        ### 打开1m线
        stock_1m_filename = f"{data_path}/basic_data/stock_klines_1m/{code}.csv"
        stock_1m = read_csv_file(stock_1m_filename)
        if stock_1m.empty:
            logger.error(f"Stock {code} klines 1m data null")
            sys.exite(1)


        stock_1m['datetime'] = pd.to_datetime(stock_1m['datetime'])
        stock_1m = stock_1m[['order_book_id', 'datetime', 'high', 'total_turnover', 'open', 'volume', 'num_trades', 'close', 'low']]
        stock_1m = stock_1m.rename(columns={'total_turnover': 'amount'})
        
        stock_1m_s = stock_1m[(stock_1m['datetime']>=pd.to_datetime(start_date)) & (stock_1m['datetime']<=pd.to_datetime(end_date))]
        # stock_1m_s = stock_1m[stock_1m['datetime']<=pd.to_datetime(end_date)]
        # stock_1m_s = stock_1m.copy()
        # print(stock_1m_s)
        
        ### 聚合
        daily_agg = aggregate_daily_features(df=stock_1m_s)

        print(daily_agg.to_string(max_cols=None))


        

        
        
    except Exception as e:
        logger.error(f"Stock {code} processing failed: {str(e)}")

    

if __name__ == "__main__":
    fire.Fire(process_single_stock)

