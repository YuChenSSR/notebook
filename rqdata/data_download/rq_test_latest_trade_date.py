import rqdatac
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Union
import time
import fire
import sys
import os
from datetime import time as dt_time, datetime


# 交易日历
data_path = "~/notebook/rqdata/Data"
data_path = os.path.expanduser(data_path) 
tradedate_path = f"{data_path}/basic_data/trade_date.csv"
trade_date_list = pd.read_csv(tradedate_path)
trade_date_list['trade_date'] = pd.to_datetime(trade_date_list['trade_date'])

today_l = datetime.now()
today_s = today_l.date()
target_time = dt_time(16, 30)

rqdatac.init(enable_bjse=True)
latest_trading_date = rqdatac.get_latest_trading_date()
latest_trading_date = pd.to_datetime(latest_trading_date)
# latest_trading_date = pd.to_datetime("2025-10-20")

print(f"today_l:{today_l} / today_s:{today_s} / targe_time:{target_time} / latest_trading_date:{latest_trading_date}")


if today_s < latest_trading_date.date():
    mask = trade_date_list['trade_date'] == latest_trading_date
    latest_trading_date = trade_date_list.loc[mask.shift(-1, fill_value=False), 'trade_date'].values[0]
    latest_trading_date = pd.to_datetime(latest_trading_date)
elif today_s == latest_trading_date.date():
    if today_l.time() <= target_time:
        mask = trade_date_list['trade_date'] == latest_trading_date
        latest_trading_date = trade_date_list.loc[mask.shift(-1, fill_value=False), 'trade_date'].values[0]
        latest_trading_date = pd.to_datetime(latest_trading_date)

print(f"latest_trading_date:{latest_trading_date}")

# print(trade_date_list)



