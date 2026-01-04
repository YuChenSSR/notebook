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

"""下载Stock turn Daily"""

def get_stock_list(stock_list_path):
    stock_list = pd.read_csv(stock_list_path)

    stock_list_s = stock_list[['order_book_id', 'symbol', 'status', 'listed_date', 'de_listed_date', 'exchange']]
    stock_list_s = stock_list_s[stock_list_s['status']!="Unknown"]
    stock_list_s['listed_date'] = pd.to_datetime(stock_list_s['listed_date'], errors='coerce')
    stock_list_s['de_listed_date'] = pd.to_datetime(stock_list_s['de_listed_date'], errors='coerce')
    
    return stock_list_s

def get_stock_turn_file(stock_turn_filename):
    stock_turn_file = pd.read_csv(stock_turn_filename)
    stock_turn_file['tradedate'] = pd.to_datetime(stock_turn_file['tradedate'])
    return stock_turn_file

def get_stock_turn_rq(code, start_date, end_date):
    try:
        stock_data = rqdatac.get_turnover_rate(code, start_date=start_date, end_date=end_date, fields=None, expect_df=True)
        time.sleep(0.1)
        stock_data.reset_index(inplace=True)
    except Exception as e:
        try:
            stock_data = rqdatac.get_turnover_rate(code, start_date=start_date, end_date=end_date, fields=None, expect_df=True)
            time.sleep(0.2)
            stock_data.reset_index(inplace=True)
        except Exception as e:
            stock_data = pd.DataFrame()
    return stock_data

def stock_turn_run(
    data_path: str = "~/notebook/rqdata/Data",
    code: Union[str, None] = None,
    start_date: str = "",
    end_date: Union[str, None] = None,
    market: Union[list[str], None] = None,
    index_start: Union[int, None] = None,
    index_end: Union[int, None] = None,
):
    logger.info("Stock Turn Daily Start Downloading...")
    
    stock_list_path = f"{data_path}/basic_data/stock_info.csv"
    stock_list = get_stock_list(stock_list_path)
    stock_list['listed_date'] = stock_list['listed_date'].fillna(pd.to_datetime('1990-01-01'))

    if market is not None:
        stock_list = stock_list[stock_list['exchange'].isin(market)]
    # stock_list = stock_list[stock_list['order_book_id']=='600588.XSHG']
    # stock_list = stock_list[0:20]

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
        success_count = 0
        fail_count = 0

        with tqdm(total=len(stock_list), desc="Stock Turn Daily Downloading", ncols=100) as pbar:

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
                    stock_turn_filename = f"{data_path}/basic_data/stock_turn_daily/{code}.csv"
                    try:
                        stock_turn_file = get_stock_turn_file(stock_turn_filename)
        
                        if stock_turn_file.empty:
                            stock_turn = get_stock_turn_rq(code=code, start_date=s_start_date, end_date=s_end_date)
                            if stock_turn is None or stock_turn.empty:
                                fail_count += 1
                                logger.error(f"stock:{code}|start_date:{s_start_date.date()}|end_date:{s_end_date.date()}| Data null")
                                continue
                        else:
                            if stock_turn_file['tradedate'].iloc[-1] < s_end_date:
                                start_date_new = stock_turn_file['tradedate'].iloc[-1] + pd.Timedelta(days=1)
                                stock_turn_rq = get_stock_turn_rq(code, start_date_new, s_end_date)
        
                                if stock_turn_rq is None or stock_turn_rq.empty:
                                    fail_count += 1
                                    logger.error(f"stock:{code}|start_date:{start_date_new.date()}|end_date:{s_end_date.date()}| Data null")
                                    continue
                                               
                                stock_turn_rq.reset_index(inplace=True)
                                stock_turn_rq['tradedate'] = pd.to_datetime(stock_turn_rq['tradedate'])
                                stock_turn = pd.concat([stock_turn_file, stock_turn_rq], ignore_index=True)
        
        
                            else:
                                success_count += 1
                                # logger.info(f"stock:{code}|end_date:{s_end_date.date()}| Data UP-TO-DATE")
                                continue
                                
                    except Exception as e:
                        logger.error(f"stock:{code}|start_date:{s_start_date.date()}|end_date:{s_end_date.date()}| File read error: {str(e)}")
                        stock_turn = get_stock_turn_rq(code=code, start_date=s_start_date, end_date=s_end_date)
        
                        
                        if stock_turn is None or stock_turn.empty:
                            fail_count += 1
                            logger.error(f"stock:{code}|start_date:{s_start_date.date()}|end_date:{s_end_date.date()}| Data null")
                            continue
        
                    # 保存
                    if 'index' in stock_turn.columns:
                        # stock_turn.drop('index', axis=1, inplace=True)
                        stock_turn.drop(columns=['index'], inplace=True) 
                        
                    stock_turn['tradedate'] = pd.to_datetime(stock_turn['tradedate'])
                    stock_turn.to_csv(stock_turn_filename, index=False, date_format='%Y-%m-%d')
                    success_count += 1
                    # logger.info(f"stock:{code}|start_date:{s_start_date.date()}|end_date:{s_end_date.date()}| Data saved")

                except Exception as e:
                    fail_count += 1
                    logger.error(f"Stock:{code} Turn Daily Error:{str(e)}")
                finally:
                    pbar.update(1) 

        logger.success(f"Stock Turn Daily Download Completed | Success: {success_count} | Fail: {fail_count}")

    rqdatac.reset()

if __name__ == "__main__":
    fire.Fire(stock_turn_run)
