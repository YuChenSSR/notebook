import rqdatac
import pandas as pd
from datetime import datetime
from loguru import logger
from typing import Union
import fire


"""下载交易日历列表"""
def get_calendar(start_date, end_date, data_path):
    start_date = "1990-01-01"
    end_date = datetime.now().strftime("%Y-12-31")
    
    rqdatac.init(enable_bjse=True)
    df = rqdatac.get_trading_dates(start_date=start_date, end_date=end_date, market='cn')
    df = pd.DataFrame(df,columns=['trade_date'])
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    df.to_csv(f'{data_path}/basic_data/trade_date.csv', index=False, date_format='%Y-%m-%d', encoding='utf-8')

    rqdatac.reset()
    
def main(
    start_date: Union[str, None] = "1990-01-01",
    end_date: Union[str, None] = None,
    data_path: str = f"/home/idc2/notebook/rqdata/Data",
):
    logger.info(f"Trading calendar start downloading...")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-12-31")

    try:
        get_calendar(start_date, end_date, data_path)
        logger.success(f"Trading calendar download completed!!!")

    except Exception as e:
        logger.error(f"Trading calendar download failed:{e}")
        

if __name__ == "__main__":
    fire.Fire(main)