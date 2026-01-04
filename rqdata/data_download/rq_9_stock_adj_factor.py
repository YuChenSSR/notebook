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

"""下载Stock adj factor Daily"""

def get_stock_list(stock_list_path):
    stock_list = pd.read_csv(stock_list_path)
    stock_list_s = stock_list[['order_book_id', 'symbol', 'status', 'listed_date', 'de_listed_date', 'exchange']]
    stock_list_s = stock_list_s[stock_list_s['status']!="Unknown"]
    stock_list_s['listed_date'] = pd.to_datetime(stock_list_s['listed_date'], errors='coerce')
    stock_list_s['de_listed_date'] = pd.to_datetime(stock_list_s['de_listed_date'], errors='coerce')
    return stock_list_s

def get_stock_adj_factor_file(stock_adj_factor_filename):
    stock_adj_factor_file = pd.read_csv(stock_adj_factor_filename)
    stock_adj_factor_file['ex_date'] = pd.to_datetime(stock_adj_factor_file['ex_date'])
    stock_adj_factor_file['ex_end_date'] = pd.to_datetime(stock_adj_factor_file['ex_end_date'])
    
    return stock_adj_factor_file 

def get_stock_adj_factor_rq(code):
    try:
        stock_data = rqdatac.get_ex_factor(code, start_date=None, end_date=None, market='cn')
        time.sleep(0.1)
        stock_data.reset_index(inplace=True)
    except Exception as e:
        try:
            stock_data = rqdatac.get_ex_factor(code, start_date=None, end_date=None, market='cn')
            time.sleep(0.2)
            stock_data.reset_index(inplace=True)
        except Exception as e:
            stock_data = pd.DataFrame()
    return stock_data

def stock_adj_factor_run(
    data_path: str = "~/notebook/rqdata/Data",
    code: Union[str, None] = None,
    start_date: str = "",
    end_date: Union[str, None] = None,
    market: Union[list[str], None] = None,        # 两组: XSHE | XSHG+BJSE
    index_start: Union[int, None] = None,
    index_end: Union[int, None] = None,
):
    logger.info("Stock Adjusted Factor Start Downloading...")
    
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
        logger.error(f"Rqdatac Init Error: {str(e)}")
        exit(1)

    if end_date is None:
        try:
            latest_trading_date = rqdatac.get_latest_trading_date()
            end_date = pd.to_datetime(latest_trading_date) # - pd.Timedelta(days=1)
        except Exception as e:
            logger.error(f"Latest Trading Date Error:{str(e)}")
            exit(1)
    else:
        end_date = pd.to_datetime(end_date)   

    # 全部下载
    if code is None:
        status = len(stock_list)

        success_count = 0
        fail_count = 0

        with tqdm(total=len(stock_list), desc="Stock Adjusted Factor Downloading", ncols=100) as pbar:
            for _, row in stock_list.iterrows():
                try:
                    ### 生成参数
                    status -= 1
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
                    stock_adj_factor_filename = f"{data_path}/basic_data/stock_ex_factor/{code}.csv"    
                    stock_adj_factor = get_stock_adj_factor_rq(code)
                    
                    if stock_adj_factor is None or stock_adj_factor.empty:
                        logger.error(f"Stock:{code} Adjusted Factor Data Null")
                        
                        fail_count += 1
                        continue
        
                    
                    # 保存
                    if 'index' in stock_adj_factor.columns:
                        stock_adj_factor.drop(columns=['index'], inplace=True) 
           
                    stock_adj_factor['ex_date'] = pd.to_datetime(stock_adj_factor['ex_date'])
                    stock_adj_factor['ex_end_date'] = pd.to_datetime(stock_adj_factor['ex_end_date'])
                    stock_adj_factor.to_csv(stock_adj_factor_filename, index=False, date_format='%Y-%m-%d')
                    
                    success_count += 1
                    # logger.info(f"stock:{code} | Data saved")
                except Exception as e:
                    fail_count += 1
                    logger.error(f"Stock:{code} Adjusted Factor Error:{str(e)}")
                finally:
                    pbar.update(1)
                    
    rqdatac.reset()
    logger.success(f"Stock Adjusted Factor Download Completed | Success: {success_count} | Fail: {fail_count}")

if __name__ == "__main__":
    fire.Fire(stock_adj_factor_run)
