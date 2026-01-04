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


# code = '000001.XSHG'	  # 上证
code = '000001.XSHE'    # 深证
start_date = "2025-11-16"
end_date = "2025-11-17"


rqdatac.init(enable_bjse=True)

# # s列表
# df = rqdatac.all_instruments(type='CS', market='cn')

# # index列表
# df = rqdatac.all_instruments(type='INDEX', market='cn')

# industry 列表
# industry_info = rqdatac.get_industry_mapping(source='citics_2019', date=None, market='cn')


stock_data = rqdatac.get_price(code, start_date=start_date, end_date=end_date, frequency='1m', fields=None, adjust_type='none', skip_suspended =False, market='cn', expect_df=True,time_slice=None)

rqdatac.reset()

print(stock_data)

# index_data = rqdatac.get_price(code, start_date=start_date, end_date=end_date, frequency='1d', fields=None, adjust_type='none', skip_suspended =False, market='cn', expect_df=True,time_slice=None)  
# index_data.reset_index(inplace=True)

# print(index_data)