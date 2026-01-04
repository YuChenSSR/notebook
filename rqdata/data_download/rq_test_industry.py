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

#######################################

def get_industry_info_rq():
    try:
        rq_data = rqdatac.get_industry_mapping(source='citics_2019', date=None, market='cn')
    except Exception as e:
        return pd.DataFrame()

    return rq_data

def get_industry_stock_rq(industry, date):
    try:
        industry_stock = rqdatac.get_industry(industry, source='citics_2019', date=date, market='cn')
    except Exception as e:
        industry_stock = pd.DataFrame()
    return industry_stock


def get_industry_klines_rq(code, start_date, end_date):
    ### 获取数据
    try:
        stock_data = rqdatac.get_price(code, start_date=start_date, end_date=end_date, frequency='1d', fields=None, adjust_type='none', skip_suspended =False, market='cn', expect_df=True,time_slice=None)
        # stock_data.reset_index(inplace=True)
    except Exception as e:
        return pd.DataFrame()
    return stock_data

def get_index_info():
    try:
        df = rqdatac.all_instruments(type='INDEX', market='cn')
    except Exception as e:
        df = pd.DataFrame()
    return df


def get_industry_turn_rq(code, start_date, end_date):
    try:
        stock_data = rqdatac.get_turnover_rate(code, start_date=start_date, end_date=end_date, fields=None, expect_df=True)
        # stock_data.reset_index(inplace=True)
    except Exception as e:
        print(f"错误:{e}")
        stock_data = pd.DataFrame()
        
######################################################
def data_download(
    code: Union[str, None]       = "600000.XSHG",     # "600588.XSHG"
    start_date: Union[str, None] = "2025-10-01",
    end_date: Union[str, None]   = "2025-10-21"
):

    end_date = datetime.now().date()
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    print(f"Code:{code} || start_date:{start_date.date()} || end_date:{end_date.date()}")


    ### rq init
    try:
        # rqdatac.init(addr='/home/idc2/notebook/rqdata/.rqdatac/license.txt', enable_bjse=True)
        rqdatac.init(enable_bjse=True)
        # rqdatac.info
        
    except Exception as e:
        logger.error(f"rqdatac init error: {str(e)}")
        exit(1)

    # ### 行业信息
    # industry_info = get_industry_info_rq()
    # industry_info_path = "/home/idc2/notebook/rqdata/Data/basic_data"
    # # industry_info.to_csv(f"{industry_info_path}/industry_info_test.csv", index=False, date_format='%Y-%m-%d', encoding='utf-8-sig')
    # print(industry_info.to_string(max_cols=None))
    

    # ### 指数信息
    # index_info = get_index_info()
    # index_info_path = "/home/idc2/notebook/rqdata/Data/basic_data"
    # index_info.to_csv(f"{index_info_path}/index_info_test.csv", index=False, date_format='%Y-%m-%d', encoding='utf-8-sig')
    
    
    # # code = "634020"
    # # i_klines = get_industry_klines_rq(code, start_date, end_date)

    ### 行业股票列表
    industry = "10"
    date = "2000-10-29"
    industry_stock = get_industry_stock_rq(industry, date)

    print(industry_stock)
    
    # # print(industry_stock.to_string(max_cols=None))


    # #first_industry_code, first_industry_index, first_industry_name
    # industry_code_maps = {
    #     '10': 'CI005001.INDX',   # 石油石化
    #     '11': 'CI005002.INDX',   # 煤炭
    #     '12': 'CI005003.INDX',   # 有色金属
    #     '20': 'CI005004.INDX',   # 电力及公用事业	
    #     '21': 'CI005005.INDX',   # 钢铁	
    #     '22': 'CI005006.INDX',   # 基础化工	
    #     '23': 'CI005007.INDX',   # 建筑	
    #     '24': 'CI005008.INDX',   # 建材	
    #     '25': 'CI005009.INDX',   # 轻工制造
    #     '26': 'CI005010.INDX',   # 机械
    #     '27': 'CI005011.INDX',   # 电力设备及新能源
    #     '28': 'CI005012.INDX',   # 国防军工
    #     '30': 'CI005013.INDX',   # 汽车
    #     '31': 'CI005014.INDX',   # 商贸零售
    #     '32': 'CI005015.INDX',   # 消费者服务
    #     '33': 'CI005016.INDX',   # 家电
    #     '34': 'CI005017.INDX',   # 纺织服装
    #     '35': 'CI005018.INDX',   # 医药
    #     '36': 'CI005019.INDX',   # 食品饮料
    #     '37': 'CI005020.INDX',   # 农林牧渔
    #     '40': 'CI005021.INDX',   # 银行
    #     '41': 'CI005022.INDX',   # 非银行金融
    #     '42': 'CI005023.INDX',   # 房地产
    #     '43': 'CI005030.INDX',   # 综合金融
    #     '50': 'CI005024.INDX',   # 交通运输
    #     '60': 'CI005025.INDX',   # 电子
    #     '61': 'CI005026.INDX',   # 通信
    #     '62': 'CI005027.INDX',   # 计算机
    #     '63': 'CI005028.INDX',   # 传媒
    #     '70': 'CI005029.INDX',   # 综合
    # }
    # start_date = "2025-01-01"
    # code = "CI005001.INDX"
    # industry_klines = get_industry_klines_rq(code, start_date, end_date)
    # print(industry_klines.to_string(max_cols=None))
    
    # code = "CI005001.INDX"
    # industry_turn = get_industry_turn_rq(code, start_date, end_date)
    # # print(industry_turn.to_string(max_cols=None))
    # print(industry_turn)
        
if __name__ == "__main__":
    fire.Fire(data_download)
    

