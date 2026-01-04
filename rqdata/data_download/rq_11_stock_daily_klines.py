import rqdatac
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Union
from tqdm import tqdm
import time
import fire
import sys
import os

"""下载Stock Klines Daily"""

def get_stock_list(stock_list_path):
    stock_list = pd.read_csv(stock_list_path)

    stock_list_s = stock_list[['order_book_id', 'symbol', 'status', 'listed_date', 'de_listed_date', 'exchange']]
    stock_list_s = stock_list_s[stock_list_s['status']!="Unknown"]
    stock_list_s['listed_date'] = pd.to_datetime(stock_list_s['listed_date'], errors='coerce')
    stock_list_s['de_listed_date'] = pd.to_datetime(stock_list_s['de_listed_date'], errors='coerce')
    
    return stock_list_s

def get_stock_klines_rq(code, start_date, end_date):
    ### 获取数据
    try:
        stock_data = rqdatac.get_price(code, start_date=start_date, end_date=end_date, frequency='1d', fields=None, adjust_type='none', skip_suspended =False, market='cn', expect_df=True,time_slice=None)
        time.sleep(0.1)
        stock_data.reset_index(inplace=True)
    except Exception as e:
        try:
            stock_data = rqdatac.get_price(code, start_date=start_date, end_date=end_date, frequency='1d', fields=None, adjust_type='none', skip_suspended =False, market='cn', expect_df=True,time_slice=None)
            time.sleep(0.2)
            stock_data.reset_index(inplace=True)
        except Exception as e:
            stock_data = pd.DataFrame()
    return stock_data

def get_stock_klines_file(stock_klines_filename):
    stock_klines_file = pd.read_csv(stock_klines_filename)
    stock_klines_file['date'] = pd.to_datetime(stock_klines_file['date'])
    return stock_klines_file

def stock_klines_run(
    data_path: str = "~/notebook/rqdata/Data",
    code: Union[str, None] = None,
    start_date: str = "",
    end_date: Union[str, None] = None,
    market: Union[list[str], None] = None,        # 两组: XSHE | XSHG+BJSE
    index_start: Union[int, None] = None,
    index_end: Union[int, None] = None,
):
    
    logger.info("Stock Klines Daily Start Downloading...")
    
    stock_list_path = f"{data_path}/basic_data/stock_info.csv"
    stock_list = get_stock_list(stock_list_path)
    stock_list['listed_date'] = stock_list['listed_date'].fillna(pd.to_datetime('1990-01-01'))
    
    if market is not None:
        stock_list = stock_list[stock_list['exchange'].isin(market)]

    if index_start is not None and index_end is not None:
        stock_list = stock_list[index_start:index_end]
        
    try:
        rqdatac.init(enable_bjse=True)
    except Exception as e:
        logger.error(f"rqdatac init error: {str(e)}")
        exit(1)
    
    if end_date is None:
        try:
            latest_trading_date = rqdatac.get_latest_trading_date()
            end_date = pd.to_datetime(latest_trading_date)
        except Exception as e:
            logger.error(f"latest_trading_date | error: {str(e)}")
            exit(1)
    else:
        end_date = pd.to_datetime(end_date)

        
    # 全部下载
    if code is None:
        # stock_list = stock_list[0:20]      
        success_count = 0
        fail_count = 0

        with tqdm(total=len(stock_list), desc="Stock Klines Daily Downloading", ncols=100) as pbar:
            
            for _, row in stock_list.iterrows():
                try:
                    ### 生成参数
                    code = row['order_book_id']
        
                    if row['status'] == 'Delisted':
                        # success_count += 1
                        # continue
                        s_start_date = row['listed_date']
                        s_end_date = row['de_listed_date']
                        if end_date > s_end_date + pd.Timedelta(days=3):
                            success_count += 1
                            continue
                    else:
                        s_start_date = row['listed_date']
                        s_end_date = end_date
        
                    ### 读取本地文件
                    stock_klines_filename = f"{data_path}/basic_data/stock_klines_daily/{code}.csv"
                    try:
                        stock_klines_file = get_stock_klines_file(stock_klines_filename)
        
                        ### 判断是否增补数据
                        if stock_klines_file.empty:
                            stock_klines = get_stock_klines_rq(code, s_start_date, s_end_date)
        
                            
                            if stock_klines is None or stock_klines.empty:
                                fail_count += 1
                                logger.error(f"stock:{code}|start_date:{s_start_date.date()}|end_date:{s_end_date.date()}| Data null")
                                continue
                            
                        else:    
                            if stock_klines_file['date'].iloc[-1] < s_end_date:
                                start_date_new = stock_klines_file['date'].iloc[-1] + pd.Timedelta(days=1)
                                stock_klines_rq = get_stock_klines_rq(code, start_date_new, s_end_date)
        
                                    
                                if stock_klines_rq is None or stock_klines_rq.empty:
                                    fail_count += 1
                                    logger.error(f"stock:{code}|start_date:{start_date_new.date()}|end_date:{s_end_date.date()}| Data null")
                                    continue
                                                   
                                stock_klines_rq.reset_index(inplace=True)
                                stock_klines_rq['date'] = pd.to_datetime(stock_klines_rq['date'])
                                stock_klines = pd.concat([stock_klines_file, stock_klines_rq], ignore_index=True)
            
                            else:
                                # logger.info(f"stock:{code}|start_date:{s_start_date.date()}|end_date:{s_end_date.date()}| Data UP-TO-DATE")
                                success_count += 1
                                continue
                        
                    except Exception as e:
                        logger.error(f"stock:{code}|start_date:{s_start_date.date()}|end_date:{s_end_date.date()}| File read error: {str(e)}")
                        stock_klines = get_stock_klines_rq(code, s_start_date, s_end_date)
        
                        if stock_klines is None or stock_klines.empty:
                            logger.error(f"stock:{code}|start_date:{s_start_date.date()}|end_date:{s_end_date.date()}| Data null")
                            fail_count += 1
                            continue
        
                    # 保存
                    if 'index' in stock_klines.columns:
                        # stock_klines.drop('index', axis=1, inplace=True)
                        stock_klines.drop(columns=['index'], inplace=True) 
                    
                    stock_klines['date'] = pd.to_datetime(stock_klines['date'])
                    stock_klines.to_csv(stock_klines_filename, index=False, date_format='%Y-%m-%d')
                    success_count += 1
                    # logger.info(f"stock:{code}|start_date:{s_start_date.date()}|end_date:{s_end_date.date()}| Data saved")
                
                except Exception as e:
                    fail_count += 1
                    logger.error(f"Stock:{code} Klines Daily Error:{str(e)}")
                finally:
                    pbar.update(1)            

        logger.success(f"Stock Adjusted Factor Download Completed | Success: {success_count} | Fail: {fail_count}")
    
    else:

        ### 读取本地文件
        stock_klines_filename = f"{data_path}/basic_data/stock_klines_daily/{code}.csv"
        try:
            stock_klines_file = get_stock_klines_file(stock_klines_filename)
        except Exception as e:
            # logger.error(f"stock:{code}|start_date:{s_start_date.date()}|end_date:{s_end_date.date()}| File read error: {str(e)}")
            stock_klines_file = pd.DataFrame()

        if stock_klines_file.empty:
            stock_list_temp = stock_list[stock_list['order_book_id']==code]

            # stock_list_temp['status'] == 'Delisted'
            
            
            stock_klines = get_stock_klines_rq(code, start_date, end_date)
            time.sleep(0.5)

        
    rqdatac.reset()        


if __name__ == "__main__":
    fire.Fire(stock_klines_run)
