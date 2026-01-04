import rqdatac
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Union
from tqdm import tqdm
from datetime import date, datetime

import itertools
import time
import fire
import sys
import os


def get_futures_list(futures_list_path):
    futures_list = pd.read_csv(futures_list_path)
    
    # 指数列表
    symbols = futures_list['underlying_symbol'].drop_duplicates().values.astype(str)
    suffixes = ['88', '888', '889', '88A2', '99']

    combinations = list(itertools.product(symbols, suffixes))
    symbols_list = pd.DataFrame(combinations, columns=['underlying_symbol', 'suffix'])
    symbols_list['order_book_id'] = symbols_list['underlying_symbol'].astype(str)  + symbols_list['suffix'].astype(str) 
    symbols_list = symbols_list[['underlying_symbol', 'order_book_id']].sort_values('underlying_symbol')

    # 获取指数列表
    futures_list_s = pd.merge(futures_list, symbols_list, on=['order_book_id','underlying_symbol'], how='inner')
    
    return futures_list_s

def get_futures_rq(code, start_date, end_date):
    ### 获取数据
    try:
        futures_rq = rqdatac.get_price(code, start_date=start_date, end_date=end_date, frequency='1m', fields=None, adjust_type='none', skip_suspended =False, market='cn', expect_df=True,time_slice=None)
        time.sleep(0.1)
        futures_rq.reset_index(inplace=True)
    except Exception as e:
        try:
            futures_rq = rqdatac.get_price(code, start_date=start_date, end_date=end_date, frequency='1m', fields=None, adjust_type='none', skip_suspended =False, market='cn', expect_df=True,time_slice=None)
            time.sleep(0.2)
            futures_rq.reset_index(inplace=True)
        except Exception as e:
            futures_rq = pd.DataFrame()
    return futures_rq

def get_futures_file(futures_file_path):
    futures_klines_file = pd.read_csv(futures_file_path)
    futures_klines_file['trading_date'] = pd.to_datetime(futures_klines_file['trading_date'])
    return futures_klines_file


def futures_klines_run(
    data_path: str = f"/home/idc2/notebook/futures/data",
    end_date: Union[str, None] = None, 
    index_start: Union[int, None] = None,
    index_end: Union[int, None] = None,
):
    # 目录
    today = date.today()
    futures_list_path = f"{data_path}/raw_data/futures_info.csv"
    futures_klines_path = f"{data_path}/raw_data/futures_klines_1m"

    # 初始化
    try:
        rqdatac.init()
    except Exception as e:
        logger.error(f"Rqdatac init error: {str(e)}")
        exit(1)

    if end_date is None:
        end_date = today
    else:
        end_date = pd.to_datetime(end_date)
    
    # 下载列表
    try:
        futures_list = get_futures_list(futures_list_path)
    except Exception as e:
        logger.error(f"Futures List error: {e}")
        exit(1)

    # 限量下载
    if index_start is not None and index_end is not None:
        futures_list = futures_list[index_start:index_end]
    
    # 逐一下载
    for index, row in futures_list.iterrows():
        code = row['order_book_id']

        futures_filename = f"{futures_klines_path}/{code}.csv"
        # 尝试读取本地文件
        is_entirety = True
        try:
            futures_file = get_futures_file(futures_filename)
            start_date = futures_file['trading_date'].iloc[-1] + pd.Timedelta(days=1)
            is_entirety = False
        except Exception as e:
            start_date = pd.to_datetime("2000-01-04").date()

        if start_date >= end_date:
            # 本地文件已最新
            continue

        # 下载数据
        print(f"code:{code} / start_dat:{start_date} / end_date:{end_date}")
        try:
            futures_rq = get_futures_rq(code, start_date, end_date)
        except Exception as e:
            continue
        
        if futures_rq is None or futures_rq.empty:
            # 无下载数据
            continue
        else:
            futures_rq['datetime'] = pd.to_datetime(futures_rq['datetime'])
            futures_rq['trading_date'] = pd.to_datetime(futures_rq['trading_date'])
            
        if is_entirety:
            futures_file = futures_rq.copy()
        else:
            futures_file = pd.concat([futures_file, futures_rq], ignore_index=True)


        futures_file = futures_file.sort_values(by=['trading_date', 'datetime'])
            
        futures_file.to_csv(futures_filename, index=False)   
        
        # print(futures_file)


    rqdatac.reset()


if __name__ == "__main__":
    fire.Fire(futures_klines_run)
    
