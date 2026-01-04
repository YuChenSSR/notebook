import rqdatac
import pandas as pd
from loguru import logger
import fire

"""下载指数列表"""
def get_index_list(data_path):
    rqdatac.init(enable_bjse=True)
    df = rqdatac.all_instruments(type='INDEX', market='cn')
    if df is not None and not df.empty:
        df.to_csv(f'{data_path}/basic_data/index_info.csv', index=False, date_format='%Y-%m-%d', encoding='utf-8')
    else:
        logger.error(f"Index List Data null")

    rqdatac.reset()

def index_list(
    data_path: str = f"/home/idc2/notebook/rqdata/Data",
):
    logger.info(f"Index list start downloading...")
    
    try:
        get_index_list(data_path)
        logger.success(f"Index List Data Saved")
    except Exception as e:
        logger.error(f"Index List error: {str(e)}")

if __name__ == "__main__":
    fire.Fire(index_list)