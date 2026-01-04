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
    """
    聚合多种日级别特征:
    包括：
        - d_vwap: 成交量加权平均价格
        - d_hhi: 赫芬达尔-赫希曼指数(成交量集中度)
        - d_corr: 成交量与收盘价相关系数
        - d_dwr: 日内成交量加权收益
        - d_cntp, d_cntn: 日内上涨/下跌占比
        - d_trades_skew: 订单数偏度
        - d_trades_corr: 订单数与成交量相关系数
    """
    if df.empty:
        return pd.DataFrame(columns=[
            'date', 'order_book_id', 'd_vwap', 'd_hhi', 'd_corr',
            'd_dwr', 'd_cntp', 'd_cntn', 'd_trades_skew', 'd_trades_corr'
        ])

    ### 1. 数据清洗
    df_temp = df.copy()
    for col in ['high', 'low', 'close', 'open', 'volume']:
        if col in df_temp.columns:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    df_temp = df_temp.dropna(subset=['date', 'datetime', 'order_book_id'])

    ### 2. 成交量加权价格与集中度 (d_vwap, d_hhi, d_corr)
    df_temp['typical_price'] = (df_temp['high'] + df_temp['low'] + df_temp['close']) / 3
    df_temp['pv_tp'] = df_temp['typical_price'] * df_temp['volume']
    df_temp['sum_volume'] = df_temp.groupby(['date', 'order_book_id'])['volume'].transform('sum')
    df_temp['volume_share'] = df_temp['volume'] / (df_temp['sum_volume'] + 1e-12)

    def safe_corr(group):
        if len(group) < 2 or group['close'].std() == 0 or group['volume'].std() == 0:
            return np.nan
        try:
            return group['close'].corr(group['volume'])
        except Exception:
            return np.nan

    daily_base = (
        df_temp.groupby(['date', 'order_book_id'])
        .agg(
            d_vwap=('pv_tp', lambda x: x.sum() / (df_temp.loc[x.index, 'volume'].sum() + 1e-12)),
            d_hhi=('volume_share', lambda x: (x ** 2).sum())
        )
        .reset_index()
    )

    corr_df = (
        df_temp.groupby(['date', 'order_book_id'], group_keys=False)
        .apply(safe_corr)
        .reset_index(name='d_corr')
    )
    daily_base = daily_base.merge(corr_df, on=['date', 'order_book_id'], how='left')


    ### 3. 日内成交量加权收益 (d_dwr)
    weight_col = 'amount' if use_amount and 'amount' in df_temp.columns else 'volume'

    def calc_dwr(g):
        g = g.sort_values('datetime')
        close = g['close'].astype(float)
        if len(close) < 2 or close.std() == 0:
            return np.nan
        r = np.log(close / (close.shift(1) + 1e-12)).dropna()
        w = g[weight_col].iloc[1:].astype(float)
        if w.sum() == 0:
            return np.nan
        return (w * r).sum() / (w.sum() + 1e-12)

    dwr_df = (
        df_temp.groupby(['date', 'order_book_id'], group_keys=False)
        .apply(calc_dwr)
        .reset_index(name='d_dwr')
    )


    ### 4. 日内价格趋势占比 (d_cntp, d_cntn)
    df_temp = df_temp.sort_values(['date', 'datetime'])
    df_temp['close_prev'] = df_temp.groupby(['date', 'order_book_id'])['close'].shift(1)
    df_temp['close_prev'] = np.where(df_temp['close_prev'].isna(), df_temp['open'], df_temp['close_prev'])
    df_temp['close_up_prev'] = (df_temp['close'] > df_temp['close_prev']).astype(int)
    df_temp['close_dw_prev'] = (df_temp['close'] < df_temp['close_prev']).astype(int)

    trend_df = (
        df_temp.groupby(['date', 'order_book_id'])
        .apply(lambda x: pd.Series({
            'd_cntp': x['close_up_prev'].sum() / len(x),
            'd_cntn': x['close_dw_prev'].sum() / len(x),
        }))
        .reset_index()
    )

    ### 5. 日内订单数统计 (d_trades_skew, d_trades_corr)
    if 'num_trades' in df_temp.columns:
        def safe_trade_corr(g):
            if len(g) < 2 or g['num_trades'].std() == 0 or g['volume'].std() == 0:
                return np.nan
            try:
                return g['num_trades'].corr(g['volume'])
            except Exception:
                return np.nan

        trade_skew = (
            df_temp.groupby(['date', 'order_book_id'])
            .agg(d_trades_skew=('num_trades', lambda x: x.skew()))
            .reset_index()
        )

        trade_corr = (
            df_temp.groupby(['date', 'order_book_id'], group_keys=False)
            .apply(safe_trade_corr)
            .reset_index(name='d_trades_corr')
        )

        trade_df = pd.merge(trade_skew, trade_corr, on=['date', 'order_book_id'], how='outer')
    else:
        # 若数据中没有num_trades列
        trade_df = pd.DataFrame(columns=['date', 'order_book_id', 'd_trades_skew', 'd_trades_corr'])

    ### 6. 合并所有特征
    daily_all = (
        daily_base.merge(dwr_df, on=['date', 'order_book_id'], how='outer')
        .merge(trend_df, on=['date', 'order_book_id'], how='outer')
        .merge(trade_df, on=['date', 'order_book_id'], how='outer')
    )

    return daily_all

def process_single_stock(stock_info, data_path, data_cols):
    
    _, row = stock_info
    code = row['order_book_id']

    try:
        ### 打开1m线
        stock_1m_filename = f"{data_path}/basic_data/stock_klines_1m/{code}.csv"
        stock_1m = read_csv_file(stock_1m_filename)
        if stock_1m.empty:
            logger.error(f"Stock {code} klines 1m data null")
            return code, False

        stock_1m['datetime'] = pd.to_datetime(stock_1m['datetime'])
        stock_1m = stock_1m[['order_book_id', 'datetime', 'high', 'total_turnover', 'open', 'volume', 'num_trades', 'close', 'low']]
        stock_1m = stock_1m.rename(columns={'total_turnover': 'amount'})
        stock_1m['date'] = stock_1m['datetime'].dt.date
        stock_1m['date'] = pd.to_datetime(stock_1m['date'])
        stock_1m = stock_1m.sort_values(['date', 'datetime'])
        
        ### 打开已聚合日线,并检查数据
        # 检查数据行
        try:
            stock_agg_filename = f"{data_path}/basic_data/stock_klines_1m_aggregate/{code}.csv"
            stock_agg_file = read_csv_file(stock_agg_filename)

            if stock_agg_file.empty:
                agg_start_date = stock_1m['date'].min()
                is_entirety = True
            else:
                # 检查数据列,缺列全部重生成
                is_cols = set(data_cols).issubset(stock_agg_file.columns)
                if not is_cols:
                    agg_start_date = stock_1m['date'].min()
                    is_entirety = True
                else:
                    stock_agg_file['tradedate'] = pd.to_datetime(stock_agg_file['tradedate'])
                    agg_start_date = stock_agg_file['tradedate'].max() + pd.Timedelta(days=1)
                    is_entirety = False
            
        except Exception as e:
            agg_start_date = stock_1m['date'].min()
            is_entirety = True


        ### 需聚合的数据
        stock_1m_s = stock_1m[stock_1m['date'] >= agg_start_date]

        if stock_1m_s.empty:
            return code, True


        ### 聚合
        daily_agg = aggregate_daily_features(df=stock_1m_s)

        
        if daily_agg.empty:
            return code, False

        daily_agg = daily_agg.rename(columns={'date': 'tradedate','order_book_id': 'symbol'})
        
        if is_entirety:
            stock_agg_file = daily_agg.copy()
        else:
            stock_agg_file = pd.concat([stock_agg_file, daily_agg], ignore_index=True)

        stock_agg_file.to_csv(stock_agg_filename, index=False, date_format='%Y-%m-%d')
            
        return code, True
    except Exception as e:
        logger.error(f"Stock {code} processing failed: {str(e)}")
        return code, False
    

def klines_aggregate(    
    data_path: str = "/home/idc2/notebook/rqdata/Data"
):
    data_path = os.path.expanduser(data_path)  
    max_workers = max(1, multiprocessing.cpu_count() - 1)


    # 调整加工方法后同步调整
    need_cols = ['d_vwap', 'd_hhi', 'd_corr', 'd_dwr', 'd_cntp', 'd_cntn', 'd_trades_skew', 'd_trades_corr']
    
    # 股票列表
    stock_list_path = f"{data_path}/basic_data/stock_info.csv"
    stock_list = get_stock_list(stock_list_path)
    stock_list['listed_date'] = stock_list['listed_date'].fillna(pd.to_datetime('1990-01-01'))

    # 测试用
    # stock_list = stock_list[105:110]

    total_count = len(stock_list)
    successful_count = 0

    logger.info(f"Klines 1m aggregate daily start... {total_count}")

    # 创建进度条
    pbar = tqdm(total=total_count, desc="Klines 1m aggregate daily start...", unit="stock")

    # 使用多进程处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_stock = {
            executor.submit(process_single_stock, stock_info, data_path, need_cols): stock_info
            for stock_info in stock_list.iterrows()
        }

        # 处理完成的任务
        for future in as_completed(future_to_stock):
            stock_info = future_to_stock[future]
            code = stock_info[1]['order_book_id']
            
            try:
                result_code, success = future.result()
                if success:
                    successful_count += 1
                else:
                    logger.warning(f"Stock {code} failed or data null")
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"成功": successful_count, "失败": pbar.n - successful_count, "总数": total_count})
                
            except Exception as e:
                logger.error(f"处理股票 {code} 时发生未捕获的异常: {str(e)}")
                pbar.update(1)
                pbar.set_postfix({"成功": successful_count, "失败": pbar.n - successful_count, "总数": total_count})
    
    # 关闭进度条
    pbar.close()
    logger.success(f"Qlib source data processing completed! Success: {successful_count}, Failure: {total_count - successful_count}, 总计: {total_count}")

    





if __name__ == "__main__":
    fire.Fire(klines_aggregate)

