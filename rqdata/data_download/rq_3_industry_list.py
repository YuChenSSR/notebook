import rqdatac
import pandas as pd
from loguru import logger
import fire

"""下载行业列表"""

def get_industry_list_rq(data_path):
    rqdatac.init(enable_bjse=True)
    industry_info = rqdatac.get_industry_mapping(source='citics_2019', date=None, market='cn')
    if industry_info is not None and not industry_info.empty:
        industry_info.to_csv(f'{data_path}/basic_data/industry_info.csv', index=False, date_format='%Y-%m-%d', encoding='utf-8')
    else:
        logger.error(f"Industry List Data null")

    rqdatac.reset()

def industry_list(
    data_path: str = f"/home/idc2/notebook/rqdata/Data",
):
    logger.info(f"Industry List start downloading...")
    
    try:
        get_industry_list_rq(data_path)
        logger.success(f"Industry List Data Saved")
    except Exception as e:
        logger.error(f"Industry List error: {str(e)}")

if __name__ == "__main__":
    fire.Fire(industry_list)

