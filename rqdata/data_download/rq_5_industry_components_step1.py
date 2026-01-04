import rqdatac
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Union
import time
import fire
import sys
import os
from datetime import datetime
from tqdm import tqdm


def get_industry_list(data_path):
    industry_list_path = f"{data_path}/basic_data/industry_info.csv"
    industry_list = pd.read_csv(industry_list_path)
    return industry_list


def get_index_list(data_path):
    index_list_path = f"{data_path}/basic_data/index_info.csv"
    index_list = pd.read_csv(index_list_path)
    index_list['listed_date'] = pd.to_datetime(index_list['listed_date'], errors='coerce')
    index_list['listed_date'] = index_list['listed_date'].fillna(pd.to_datetime('1990-01-01'))
    return index_list


def get_trade_date_list(data_path):
    tradedate_path = f"{data_path}/basic_data/trade_date.csv"
    trade_date_list = pd.read_csv(tradedate_path)
    trade_date_list['trade_date'] = pd.to_datetime(trade_date_list['trade_date'])
    return trade_date_list


def get_industry_components_file(industry_components_path):
    industry_components_file = pd.read_csv(industry_components_path)
    industry_components_file['tradedate'] = pd.to_datetime(industry_components_file['tradedate'])
    return industry_components_file


def get_industry_components_rq(code, start_date, end_date, trade_date_list):
    trade_date_index = trade_date_list[(trade_date_list['trade_date'] >= start_date) & (trade_date_list['trade_date'] <= end_date)]
    trade_date_index = trade_date_index.sort_values(by='trade_date')

    data_list = []
    for _, row in trade_date_index.iterrows():
        query_date = row['trade_date']
        try:
            industry_stock = rqdatac.get_industry(code, source='citics_2019', date=query_date, market='cn')
            time.sleep(0.1)
            for stock in industry_stock:
                data_list.append({
                    'stock_code': stock,
                    'tradedate': query_date
                })
        except Exception as e:
            continue
                
    if data_list: 
        industry_component_list = pd.DataFrame(data_list)
        industry_component_list['industry_code'] = code
    else:
        industry_component_list = pd.DataFrame(columns=['industry_code', 'stock_code', 'tradedate']) # 创建一个空DataFrame，结构一致    
            
    return industry_component_list
    

def industry_components(
    data_path: str = "~/notebook/rqdata/Data"
):
    data_path = os.path.expanduser(data_path) 
    end_date = datetime.now()

    logger.info("Industry Components Start Downloading...")
    
    try:
        rqdatac.init(enable_bjse=True)
    except Exception as e:
        logger.error(f"Rqdatac Init Error: {str(e)}")
        exit(1)

    # 获取交易日历
    try:
        trade_date_list = get_trade_date_list(data_path)
    except Exception as e:
        logger.error(f"trade date list file error: {str(e)}")
        exit(1)
    
    try:
        index_list = get_index_list(data_path)
    except Exception as e:
        logger.error(f"index list file error: {str(e)}")
        exit(1)
        
    # 一级获取行业列表
    industry_list = get_industry_list(data_path)
    industry_list = industry_list['first_industry_code'].drop_duplicates().to_frame(name='industry_code')
    
    industry_code_maps = {
        '10': 'CI005001.INDX',   # 石油石化
        '11': 'CI005002.INDX',   # 煤炭
        '12': 'CI005003.INDX',   # 有色金属
        '20': 'CI005004.INDX',   # 电力及公用事业	
        '21': 'CI005005.INDX',   # 钢铁	
        '22': 'CI005006.INDX',   # 基础化工	
        '23': 'CI005007.INDX',   # 建筑	
        '24': 'CI005008.INDX',   # 建材	
        '25': 'CI005009.INDX',   # 轻工制造
        '26': 'CI005010.INDX',   # 机械
        '27': 'CI005011.INDX',   # 电力设备及新能源
        '28': 'CI005012.INDX',   # 国防军工
        '30': 'CI005013.INDX',   # 汽车
        '31': 'CI005014.INDX',   # 商贸零售
        '32': 'CI005015.INDX',   # 消费者服务
        '33': 'CI005016.INDX',   # 家电
        '34': 'CI005017.INDX',   # 纺织服装
        '35': 'CI005018.INDX',   # 医药
        '36': 'CI005019.INDX',   # 食品饮料
        '37': 'CI005020.INDX',   # 农林牧渔
        '40': 'CI005021.INDX',   # 银行
        '41': 'CI005022.INDX',   # 非银行金融
        '42': 'CI005023.INDX',   # 房地产
        '43': 'CI005030.INDX',   # 综合金融
        '50': 'CI005024.INDX',   # 交通运输
        '60': 'CI005025.INDX',   # 电子
        '61': 'CI005026.INDX',   # 通信
        '62': 'CI005027.INDX',   # 计算机
        '63': 'CI005028.INDX',   # 传媒
        '70': 'CI005029.INDX',   # 综合
    }

    # # 测试用
    # industry_list = industry_list[0:1]
    
    success_count = 0
    fail_count = 0

    with tqdm(total=len(industry_list), desc="Downloading Industry Components", ncols=100) as pbar:
        
        for _, row in industry_list.iterrows():
            try:
                industry_code = row['industry_code'].astype(str)
                index_code = industry_code_maps.get(industry_code, None)
        
                if index_code is None: 
                    logger.error(f"Industry_code:{industry_code}|Index_code:{index_code}|Index_code None")
                    
                    pbar.update(1)
                    fail_count += 1
                    continue
                
                ### 下载数据 ###
                is_entire_download = False
                industry_componens_filename = f"{data_path}/basic_data/industry_componens/{industry_code}.csv"
        
                # 获取本地文件
                try:
                    industry_componens_file = get_industry_components_file(industry_componens_filename)
                    if industry_componens_file.empty:
                        is_entire_download = True
                    else:
                        is_entire_download = False
                except Exception as e:
                    is_entire_download = True
        
                        
                if is_entire_download:
                    # 全量下载
                    start_date = index_list.loc[index_list['order_book_id'] == index_code, 'listed_date'].values[0]
                    start_date = pd.to_datetime(start_date)
                    industry_componens_file = pd.DataFrame(columns=['industry_code', 'stock_code', 'tradedate']) 
                else:
                    # 增量
                    file_max_date = pd.to_datetime(industry_componens_file['tradedate'].max()) 
                    if file_max_date.date() >= end_date.date():
                        logger.info(f"Industry_code:{industry_code}|File_date:{file_max_date.date()} | Data Is Up-To-Date")
                        
                        pbar.update(1)
                        success_count += 1
                        continue
                    else:
                        start_date = file_max_date + pd.Timedelta(days=1)
                
                # 下载数据
                industry_componens_rq = get_industry_components_rq(industry_code, start_date, end_date, trade_date_list)
                industry_componens_file = pd.concat([industry_componens_file, industry_componens_rq], ignore_index=True)
        
                # 保存数据
                industry_componens_file.to_csv(industry_componens_filename, index=False, date_format='%Y-%m-%d')

                success_count += 1
            except Exception as e:
                fail_count += 1
                logger.error(f"industry_code:{industry_code} | Failed Error: {str(e)}")
            finally:
                pbar.update(1)                

    rqdatac.reset()
    logger.success(f"Industry Components Download Completed | Success: {success_count} | Fail: {fail_count}")
                    
if __name__ == "__main__":
    fire.Fire(industry_components)








