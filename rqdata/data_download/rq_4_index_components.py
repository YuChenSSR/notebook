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
from tqdm import tqdm


def get_index_list(data_path):
    index_list_path = f"{data_path}/basic_data/index_info.csv"
    index_list = pd.read_csv(index_list_path)
    index_list['listed_date'] = pd.to_datetime(index_list['listed_date'], errors='coerce')
    index_list['listed_date'] = index_list['listed_date'].fillna(pd.to_datetime('1990-01-01'))
    return index_list


def get_trade_date_list(data_path):
    tradedate_path = f"{data_path}/basic_data/trade_date.csv"
    trade_date_list = pd.read_csv(tradedate_path)
    trade_date_list['trade_date'] = pd.to_datetime(trade_date_list['trade_date'])
    return trade_date_list


def get_index_components_file(index_components_path):
    index_components_file = pd.read_csv(index_components_path)
    index_components_file['tradedate'] = pd.to_datetime(index_components_file['tradedate'])
    return index_components_file


def index_componens_csi_file(index_components_csi_path):
    index_components_csi_file = pd.read_csv(index_components_csi_path)
    index_components_csi_file['start_date'] = pd.to_datetime(index_components_csi_file['start_date'])
    index_components_csi_file['end_date'] = pd.to_datetime(index_components_csi_file['end_date'])
    return index_components_csi_file


def get_index_components_rq(code, start_date, end_date, trade_date_list):
    trade_date_index = trade_date_list[
        (trade_date_list['trade_date'] >= start_date) & (trade_date_list['trade_date'] <= end_date)
    ]
    trade_date_index = trade_date_index.sort_values(by='trade_date')

    data_list = []
    for _, row in trade_date_index.iterrows():
        query_date = row['trade_date']
        try:
            index_component = rqdatac.index_components(
                order_book_id=code, date=query_date, market='cn', return_create_tm=False
            )
            time.sleep(0.1)
            for stock in index_component:
                data_list.append({
                    'stock_code': stock,
                    'tradedate': query_date
                })
        except Exception as e:
            logger.warning(f"{code} {query_date.date()} failed: {str(e)}")
            continue

    if data_list:
        index_component_list = pd.DataFrame(data_list)
    else:
        index_component_list = pd.DataFrame(columns=['stock_code', 'tradedate'])

    return index_component_list


def index_components(data_path: str = "~/notebook/rqdata/Data"):
    data_path = os.path.expanduser(data_path)
    end_date = datetime.now()

    logger.info("Index Components Start Downloading & Processing ...")

    try:
        rqdatac.init(enable_bjse=True)
    except Exception as e:
        logger.error(f"Rqdatac init error: {str(e)}")
        exit(1)

    # 获取交易日历
    try:
        trade_date_list = get_trade_date_list(data_path)
    except Exception as e:
        logger.error(f"Trade Date List File Error: {str(e)}")
        exit(1)

    try:
        index_list = get_index_list(data_path)
    except Exception as e:
        logger.error(f"index list file error: {str(e)}")
        exit(1)

    index_code = [
        '000300.XSHG',  # 中证300
        '000905.XSHG',  # 中证500
        '000906.XSHG',  # 中证800
        '000852.XSHG',  # 中证1000
        '000985.XSHG',  # 中证全指
    ]

    success_count = 0
    fail_count = 0

    with tqdm(total=len(index_code), desc="Downloading & Processing Index Components", ncols=100) as pbar:
        for code in index_code:
            try:
                index_componens_filename = f"{data_path}/basic_data/index_componens/{code}.csv"

                # 检查文件是否存在
                try:
                    index_componens_file = get_index_components_file(index_componens_filename)
                    if index_componens_file.empty:
                        is_entire_download = True
                    else:
                        is_entire_download = False
                except Exception:
                    is_entire_download = True

                if is_entire_download:
                    # 全量
                    start_date = index_list.loc[index_list['order_book_id'] == code, 'listed_date'].values[0]
                    start_date = pd.to_datetime(start_date)
                    index_componens_file = pd.DataFrame(columns=['stock_code', 'tradedate'])
                else:
                    # 增量
                    file_max_date = pd.to_datetime(index_componens_file['tradedate'].max())
                    if file_max_date.date() >= end_date.date():
                        # logger.info(f"index:{code} | Data is UP-TO-DATE ({file_max_date.date()})")
                        # pbar.update(1)
                        success_count += 1
                        continue
                    else:
                        start_date = file_max_date + pd.Timedelta(days=1)

                # 下载和保存过程数据
                index_componens_rq = get_index_components_rq(code, start_date, end_date, trade_date_list)
                index_componens_file = pd.concat([index_componens_file, index_componens_rq], ignore_index=True)
                index_componens_file.to_csv(index_componens_filename, index=False, date_format='%Y-%m-%d')


                # --- 转换为 csi 文件 ---
                csi_filename = f"{data_path}/basic_data/index_componens/csi_{code}.csv"
                index_componens_csi = pd.DataFrame(columns=['stock_code', 'start_date', 'end_date'])
                i_start_date = pd.to_datetime(index_componens_file['tradedate'].min())
                is_firstday = True
                last_trade_date = ""

                i_end_date = pd.to_datetime(index_componens_file['tradedate'].max())
                if i_start_date > i_end_date:
                    continue

                trade_date_index = trade_date_list[
                    (trade_date_list['trade_date'] >= i_start_date) & (trade_date_list['trade_date'] <= i_end_date)]

                for _, row in trade_date_index.iterrows():
                    query_date = row['trade_date']
                    df_today = index_componens_file.loc[index_componens_file['tradedate'] == query_date]
                    if df_today.empty:
                        continue

                    if is_firstday:
                        # 首日
                        df_today = df_today.rename(columns={'tradedate': 'start_date'})
                        df_today['end_date'] = query_date
                        df_yesterday = df_today
                        is_firstday = False
                    else:
                        # 非首日
                        df_check = pd.merge(df_today, df_yesterday, on='stock_code', how='outer')

                        # 新进
                        df_enter = df_check[df_check['start_date'].isna()].copy()
                        df_enter['start_date'] = df_enter['tradedate']
                        df_enter['end_date'] = df_enter['tradedate']
                        df_enter = df_enter.drop(['tradedate'], axis=1)

                        # 已出
                        df_out = df_check[df_check['tradedate'].isna()].copy()
                        df_out['end_date'] = last_trade_date
                        df_out = df_out.drop(['tradedate'], axis=1)

                        # 持有
                        df_hold = df_check[~(df_check['tradedate'].isna() | df_check['start_date'].isna())].copy()
                        df_hold['end_date'] = df_hold['tradedate']
                        df_hold = df_hold.drop(['tradedate'], axis=1)

                        index_componens_csi = pd.concat([index_componens_csi, df_out], ignore_index=True)
                        df_hold = pd.concat([df_hold, df_enter], ignore_index=True)
                        if query_date == i_end_date:
                            index_componens_csi = pd.concat([index_componens_csi, df_hold], ignore_index=True)
                        df_yesterday = df_hold

                    last_trade_date = query_date

                # 保存文件
                index_componens_csi.to_csv(csi_filename, index=False, date_format='%Y-%m-%d')
                success_count += 1
            except Exception as e:
                fail_count += 1
                logger.error(f"Index:{code} | Failed Error: {str(e)}")
            finally:
                pbar.update(1)

    rqdatac.reset()
    logger.success(f"Index Components Download & Processed Completed | Success: {success_count} | Fail: {fail_count}")


if __name__ == "__main__":
    fire.Fire(index_components)
