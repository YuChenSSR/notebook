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

"""下载Stock Klines 1 Minute"""

def get_stock_list(stock_list_path):
    stock_list = pd.read_csv(stock_list_path)
    stock_list_s = stock_list[['order_book_id', 'symbol', 'status', 'listed_date', 'de_listed_date','exchange']]
    stock_list_s = stock_list_s[stock_list_s['status']!="Unknown"]
    stock_list_s['listed_date'] = pd.to_datetime(stock_list_s['listed_date'], errors='coerce')
    stock_list_s['de_listed_date'] = pd.to_datetime(stock_list_s['de_listed_date'], errors='coerce')  
    return stock_list_s


def stock_klines_1m_run(
    data_path: str = "~/notebook/rqdata/Data",
    market: Union[list[str], None] = None,        # 两组: XSHE | XSHG+BJSE

):
    
    stock_list_path = f"{data_path}/basic_data/stock_info.csv"
    stock_list = get_stock_list(stock_list_path)

    part1 = stock_list[stock_list['exchange'].isin(['XSHE'])]
    part1_paragraph1_start = 0
    part1_paragraph1_end = int(len(part1) * 0.35)
    part1_paragraph2_start = part1_paragraph1_end
    part1_paragraph2_end = len(part1)

    part2 = stock_list[stock_list['exchange'].isin(['XSHG', 'BJSE'])]
    part2_paragraph1_start = 0
    part2_paragraph1_end = int(len(part2) * 0.35)
    part2_paragraph2_start = part2_paragraph1_end
    part2_paragraph2_end = len(part2)


    print(f"part1_paragraph1_start={part1_paragraph1_start}")
    print(f"part1_paragraph1_end={part1_paragraph1_end}")

    print(f"part1_paragraph2_start={part1_paragraph2_start}")
    print(f"part1_paragraph2_end={part1_paragraph2_end}")

    print(f"part2_paragraph1_start={part2_paragraph1_start}")
    print(f"part2_paragraph1_end={part2_paragraph1_end}")

    print(f"part2_paragraph2_start={part2_paragraph2_start}")
    print(f"part2_paragraph2_end={part2_paragraph2_end}")    
    
    
    
if __name__ == "__main__":
    fire.Fire(stock_klines_1m_run)
