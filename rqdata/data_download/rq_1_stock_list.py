import rqdatac
import pandas as pd
from loguru import logger
from typing import Union
import fire

"""下载股票列表"""
def get_stock_list(data_path):
    rqdatac.init(enable_bjse=True)
    df = rqdatac.all_instruments(type='CS', market='cn')
    df = df.drop(['office_address'], axis=1)
    df = df[df['listed_date'] != '2999-12-31']
    df.to_csv(f'{data_path}/basic_data/stock_info.csv', index=False, date_format='%Y-%m-%d', encoding='utf-8')

    rqdatac.reset()

def main(
    data_path: str = f"/home/idc2/notebook/rqdata/Data",
):
    logger.info(f"Stock list start downloading...")
    try:
        get_stock_list(data_path)
        logger.success(f"Stock list download completed!!!")
    except Exception as e:
        logger.error(f"Stock list download failed:{e}")

if __name__ == "__main__":
    fire.Fire(main)