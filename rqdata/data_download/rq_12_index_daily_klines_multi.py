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
from multiprocessing import Pool, Manager

"""下载Index Klines Daily"""

def read_index_list(index_list_path):
    return pd.read_csv(index_list_path)


def get_index_klines_file(index_klines_filename):
    index_klines_file = pd.read_csv(index_klines_filename)
    index_klines_file['date'] = pd.to_datetime(index_klines_file['date'])
    return index_klines_file

def get_index_klines_rq(code, start_date, end_date):
    try:
        index_data = rqdatac.get_price(code, start_date=start_date, end_date=end_date, frequency='1d', fields=None, adjust_type='none', skip_suspended =False, market='cn', expect_df=True,time_slice=None)  
        time.sleep(0.1)
        index_data.reset_index(inplace=True)
    except Exception as e:
        try:
            index_data = rqdatac.get_price(code, start_date=start_date, end_date=end_date, frequency='1d', fields=None, adjust_type='none', skip_suspended =False, market='cn', expect_df=True,time_slice=None)  
            time.sleep(0.2)
            index_data.reset_index(inplace=True)
        except Exception as e:
            index_data = pd.DataFrame()
            
    return index_data

def process_index(row, data_path, end_date, success_count, fail_count):
    try:
        code = row['order_book_id']
        if row['status'] == 'Delisted':
            s_start_date = row['listed_date']
            s_end_date = row['de_listed_date']
        else:
            s_start_date = row['listed_date']
            s_end_date = end_date

        index_klines_filename = f"{data_path}/basic_data/index_klines_daily/{code}.csv"    
        
        try:            
            index_klines_file = get_index_klines_file(index_klines_filename)
            if index_klines_file.empty:
                index_klines = get_index_klines_rq(code, s_start_date, s_end_date)
                if index_klines is None or index_klines.empty:
                    fail_count.value += 1
                    logger.warning(f"\nIndex:{code}|start_date:{s_start_date.date()}|end_date:{s_end_date.date()}| Data null")
                    return
            else:
                index_klines_file = index_klines_file[:len(index_klines_file)-3]
                
                if index_klines_file['date'].iloc[-1] < s_end_date:
                    start_date_new = index_klines_file['date'].iloc[-1] + pd.Timedelta(days=1)
                    index_klines_rq_data = get_index_klines_rq(code, start_date_new, s_end_date)
                    if index_klines_rq_data is None or index_klines_rq_data.empty:
                        fail_count.value += 1
                        logger.warning(f"\nIndex:{code}|start_date:{start_date_new.date()}|end_date:{s_end_date.date()}| Data null")
                        return
                    index_klines_rq_data['date'] = pd.to_datetime(index_klines_rq_data['date'])
                    index_klines = pd.concat([index_klines_file, index_klines_rq_data], ignore_index=True)
                else:
                    success_count.value += 1
                    return
        except Exception as e:
            logger.error(f"\nIndex:{code}|File read error: {str(e)}")
            index_klines = get_index_klines_rq(code, s_start_date, s_end_date)
            if index_klines is None or index_klines.empty:
                fail_count.value += 1
                logger.warning(f"\nIndex:{code}|start_date:{s_start_date.date()}|end_date:{s_end_date.date()}| Data null")
                return

        if 'index' in index_klines.columns:
            index_klines.drop(columns=['index'], inplace=True) 
            
        index_klines['date'] = pd.to_datetime(index_klines['date'])
        # index_klines.to_csv(index_klines_filename, index=False, date_format='%Y-%m-%d')
        print("\nindex_klines:")
        print(index_klines)
        success_count.value += 1

    except Exception as e:
        fail_count.value += 1
        logger.error(f"\nIndex:{code} Klines Daily Error:{str(e)}")

def index_klines_run(
    data_path: str = "~/notebook/rqdata/Data",
    start_date: str = "",
    end_date: Union[str, None] = None,
    n_process: int = 5
):

    logger.info("\nIndex Klines Daily Start Downloading...")
    
    index_list_d = [
        '000001.XSHG',	  # 上证
        '399001.XSHE',    # 深证
        '399106.XSHE',    # 深证
        
        '000300.XSHG',    # 中证300  '399300.XSHE'
        '000852.XSHG',    # 中证1000  
        '000905.XSHG',    # 中证500
        '000906.XSHG',    # 中证800
        '000985.XSHG',    # 中证全指

        'CI005001.INDX',   # 石油石化
        'CI005002.INDX',   # 煤炭
        'CI005003.INDX',   # 有色金属
        'CI005004.INDX',   # 电力及公用事业	
        'CI005005.INDX',   # 钢铁	
        'CI005006.INDX',   # 基础化工	
        'CI005007.INDX',   # 建筑	
        'CI005008.INDX',   # 建材	
        'CI005009.INDX',   # 轻工制造
        'CI005010.INDX',   # 机械
        'CI005011.INDX',   # 电力设备及新能源
        'CI005012.INDX',   # 国防军工
        'CI005013.INDX',   # 汽车
        'CI005014.INDX',   # 商贸零售
        'CI005015.INDX',   # 消费者服务
        'CI005016.INDX',   # 家电
        'CI005017.INDX',   # 纺织服装
        'CI005018.INDX',   # 医药
        'CI005019.INDX',   # 食品饮料
        'CI005020.INDX',   # 农林牧渔
        'CI005021.INDX',   # 银行
        'CI005022.INDX',   # 非银行金融
        'CI005023.INDX',   # 房地产
        'CI005030.INDX',   # 综合金融
        'CI005024.INDX',   # 交通运输
        'CI005025.INDX',   # 电子
        'CI005026.INDX',   # 通信
        'CI005027.INDX',   # 计算机
        'CI005028.INDX',   # 传媒
        'CI005029.INDX',   # 综合
    ]

    try:
        index_list_path = f"{data_path}/basic_data/index_info.csv"
        index_list = read_index_list(index_list_path)
        index_list_s = index_list[index_list['order_book_id'].isin(index_list_d)]
        index_list_s['listed_date'] = pd.to_datetime(index_list_s['listed_date'], errors='coerce').fillna(pd.to_datetime('1990-01-01'))
        index_list_s['de_listed_date'] = pd.to_datetime(index_list_s['de_listed_date'], errors='coerce')
    except Exception as e:
        logger.error(f"Index list file error: {str(e)}")
        exit(1)

    try:
        rqdatac.init(enable_bjse=True)
    except Exception as e:
        logger.error(f"rqdatac init error: {str(e)}")
        exit(1)
    
    if end_date is None:
        end_date = pd.to_datetime(rqdatac.get_latest_trading_date())
    else:
        end_date = pd.to_datetime(end_date)

    manager = Manager()
    success_count = manager.Value('i', 0)
    fail_count = manager.Value('i', 0)

    with Pool(processes=n_process) as pool:
        tasks = [
            pool.apply_async(process_index, args=(row, data_path, end_date, success_count, fail_count))
            for _, row in index_list_s.iterrows()
        ]
        for t in tqdm(tasks, desc="Index Klines Daily Downloading", ncols=100):
            t.get()  # 等待任务完成

    logger.success(f"Index Klines Daily Download Completed | Success: {success_count.value} | Fail: {fail_count.value}")

    rqdatac.reset()


if __name__ == "__main__":
    fire.Fire(index_klines_run)
