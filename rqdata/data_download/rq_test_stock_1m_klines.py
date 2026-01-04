import rqdatac
import pandas as pd
from pathlib import Path
from loguru import logger
import time
import fire
import sys
import os

"""下载Stock Klines Daily"""

def read_stock_list(stock_list_path):
    stock_list = pd.read_csv(stock_list_path)
    
    return stock_list
    
def main(skip_exists=True):
    script_path = os.path.dirname(os.path.realpath(__file__))
    stock_list_path = f"{script_path}/Data/stock_info.csv"

    
    stock_list = read_stock_list(stock_list_path)
    stock_list_s = stock_list[['order_book_id', 'symbol', 'status', 'listed_date', 'de_listed_date']]
    stock_list_s = stock_list_s[stock_list_s['status']!="Unknown"]
    stock_list_s['listed_date'] = pd.to_datetime(stock_list_s['listed_date'], errors='coerce')
    stock_list_s['de_listed_date'] = pd.to_datetime(stock_list_s['de_listed_date'], errors='coerce')

    stock_list_s = stock_list_s[stock_list_s['order_book_id']=='600588.XSHG']     # 测试用
    # stock_list_s = stock_list_s[0:100]
    # print(stock_list_s.tail(10).to_string(max_cols=None))


    rqdatac.init(enable_bjse=True)
    latest_trading_date = rqdatac.get_latest_trading_date()
    latest_trading_date = pd.to_datetime(latest_trading_date) - pd.Timedelta(days=1)

    stock_list_s['listed_date'] = stock_list_s['listed_date'].fillna(pd.to_datetime('1990-01-01'))

    status = len(stock_list_s)  
    for _, row in stock_list_s.iterrows():
        status -= 1
        ### 生成参数
        code = row['order_book_id']
        if row['status'] == 'Delisted':
            start_date = row['listed_date']
            end_date = row['de_listed_date']
        else:
            start_date = row['listed_date']
            end_date = latest_trading_date

        ### 已下载跳过
        out_csv_filename = f"{script_path}/Data/stock_klines_1m/{code}.csv"
        if skip_exists and os.path.isfile(out_csv_filename): 
            print(f"stock:{code}|start_date:{start_date}|end_date:{end_date}|skip[{status}]")
            continue

        ### 获取数据
        try:
            stock_data = rqdatac.get_price(code, start_date=start_date, end_date=end_date, frequency='1m', fields=None, adjust_type='none', skip_suspended =False, market='cn', expect_df=True,time_slice=None)
            time.sleep(2)
            # print(stock_data.tail(10).to_string(max_cols=None))

            ### 非空，保存
            if isinstance(stock_data, pd.DataFrame) and (len(stock_data) > 0):
                stock_data.reset_index(inplace=True)
                # stock_data.to_csv(out_csv_filename, index=False, date_format='%Y-%m-%d')
                stock_data.to_csv(out_csv_filename, index=False)
                print(f"stock:{code}|start_date:{start_date}|end_date:{end_date}|data saved[{status}]")
            else:
                print(f"stock:{code}|start_date:{start_date}|end_date:{end_date}|data null[{status}]")

        except Exception as e:
            logger.error(f"stock:{code}|start_date:{start_date}|end_date:{end_date}[{status}]| error: {str(e)}")
            continue
 

if __name__ == "__main__":
    fire.Fire(main)
