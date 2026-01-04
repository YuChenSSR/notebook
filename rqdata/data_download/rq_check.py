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


def get_stock_list(stock_list_path):
    stock_list = pd.read_csv(stock_list_path)
    stock_list_s = stock_list[['order_book_id', 'symbol', 'status', 'listed_date', 'de_listed_date', 'exchange']]
    stock_list_s = stock_list_s[stock_list_s['status']!="Unknown"]
    stock_list_s['listed_date'] = pd.to_datetime(stock_list_s['listed_date'], errors='coerce')
    stock_list_s['de_listed_date'] = pd.to_datetime(stock_list_s['de_listed_date'], errors='coerce')
    return stock_list_s

def get_stock_factor_file(stock_factor_filename):
    stock_factor_file = pd.read_csv(stock_factor_filename)
    stock_factor_file['date'] = pd.to_datetime(stock_factor_file['date'])
    return stock_factor_file


def data_check(
    data_path: str = "/home/idc2/notebook/rqdata/Data",
):
    stock_list_path = f"{data_path}/basic_data/stock_info.csv"
    stock_list = get_stock_list(stock_list_path)
    stock_list['listed_date'] = stock_list['listed_date'].fillna(pd.to_datetime('1990-01-01'))

    # stock_list = stock_list[0:10]


    for _, row in stock_list.iterrows():
        code = row['order_book_id']

        ### 读取本地文件
        stock_factor_filename = f"{data_path}/basic_data/stock_factor_daily/{code}.csv"

        ttt = "Revise_Fasle"
        try:
            stock_factor_file = get_stock_factor_file(stock_factor_filename)
            if stock_factor_file.empty:
                continue
            max_tradedate = stock_factor_file['date'].max()
            target_tradedate = pd.to_datetime("2025-11-21")

            if max_tradedate > target_tradedate:
                stock_factor = stock_factor_file[stock_factor_file['date']<=target_tradedate]

                stock_factor.to_csv(stock_factor_filename, index=False, date_format='%Y-%m-%d')
                ttt="Revise_True"
            print(f"\ncode:{code} / max_tradedate:{max_tradedate} / ttt:{ttt}")

        except Exception as e:
            print(print(f"\ncode:{code} / {e}"))
            continue
        
    
    # print(stock_list)

if __name__ == "__main__":
    fire.Fire(data_check)