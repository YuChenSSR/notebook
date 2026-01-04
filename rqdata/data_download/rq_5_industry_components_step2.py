import os
import sys
import time
import fire
import multiprocessing
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Union
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


### ========================= 基础数据读取函数 ========================= ###
def get_industry_list(data_path):
    """读取行业列表"""
    industry_list_path = f"{data_path}/basic_data/industry_info.csv"
    industry_list = pd.read_csv(industry_list_path)
    return industry_list


def get_trade_date_list(data_path):
    """读取交易日历"""
    tradedate_path = f"{data_path}/basic_data/trade_date.csv"
    trade_date_list = pd.read_csv(tradedate_path)
    trade_date_list["trade_date"] = pd.to_datetime(trade_date_list["trade_date"])
    return trade_date_list


def get_index_list(data_path):
    """读取指数信息"""
    index_list_path = f"{data_path}/basic_data/index_info.csv"
    index_list = pd.read_csv(index_list_path)
    index_list["listed_date"] = pd.to_datetime(index_list["listed_date"], errors="coerce")
    index_list["listed_date"] = index_list["listed_date"].fillna(pd.to_datetime("1990-01-01"))
    return index_list


def get_industry_components_file(industry_components_path):
    """读取行业成分股文件"""
    industry_components_file = pd.read_csv(industry_components_path)
    industry_components_file["tradedate"] = pd.to_datetime(industry_components_file["tradedate"])
    return industry_components_file


### ========================= 并行读取行业成分文件函数 ========================= ###
def load_industry_file(industry_code, data_path, industry_code_maps):
    """并行读取单个行业成分股文件"""
    try:
        index_code = industry_code_maps.get(str(industry_code))
        if index_code is None:
            return pd.DataFrame()

        path = Path(f"{data_path}/basic_data/industry_componens/{industry_code}.csv")
        if not path.exists():
            logger.warning(f"{industry_code} file not found")
            return pd.DataFrame()

        df = get_industry_components_file(path)
        if df.empty:
            return pd.DataFrame()

        return df

    except Exception as e:
        logger.warning(f"{industry_code} read error: {e}")
        return pd.DataFrame()


### ========================= 股票归属行业计算函数 ========================= ###
def process_stock_industry(stock_code, df_stock, industry_code_maps):
    """计算单个股票的行业归属时间段"""
    df_stock = df_stock.sort_values("tradedate")
    data_list = []
    for industry_id, df_group in df_stock.groupby("industry_code"):
        start_date = df_group["tradedate"].min()
        end_date = df_group["tradedate"].max()
        index_code = industry_code_maps.get(str(industry_id))
        data_list.append(
            {
                "stock_code": stock_code,
                "industry_code": industry_id,
                "index_code": index_code,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
    return data_list


# 全局包装函数：修复Pickle错误
def process_stock_industry_wrapper(args):
    """包装函数，用于多进程调用"""
    return process_stock_industry(*args)


### ========================= 主流程 ========================= ###
def industry_components_merge(
    data_path: str = "/home/idc2/notebook/rqdata/Data",
    max_workers: Union[int, None] = 15,
):
    # === 0. 交易日历 ===
    # data_path = os.path.expanduser(data_path)

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 2 )

    # === 1. 交易日历 ===
    try:
        trade_date_list = get_trade_date_list(data_path)
    except Exception as e:
        logger.error(f"trade date list file error: {str(e)}")
        sys.exit(1)

    # === 2. 行业代码列表 ===
    industry_list = get_industry_list(data_path)
    industry_list = industry_list["first_industry_code"].drop_duplicates().to_frame(name="industry_code")

    # 行业与指数映射
    industry_code_maps = {
        "10": "CI005001.INDX",  # 石油石化
        "11": "CI005002.INDX",  # 煤炭
        "12": "CI005003.INDX",  # 有色金属
        "20": "CI005004.INDX",  # 电力及公用事业
        "21": "CI005005.INDX",  # 钢铁
        "22": "CI005006.INDX",  # 基础化工
        "23": "CI005007.INDX",  # 建筑
        "24": "CI005008.INDX",  # 建材
        "25": "CI005009.INDX",  # 轻工制造
        "26": "CI005010.INDX",  # 机械
        "27": "CI005011.INDX",  # 电力设备及新能源
        "28": "CI005012.INDX",  # 国防军工
        "30": "CI005013.INDX",  # 汽车
        "31": "CI005014.INDX",  # 商贸零售
        "32": "CI005015.INDX",  # 消费者服务
        "33": "CI005016.INDX",  # 家电
        "34": "CI005017.INDX",  # 纺织服装
        "35": "CI005018.INDX",  # 医药
        "36": "CI005019.INDX",  # 食品饮料
        "37": "CI005020.INDX",  # 农林牧渔
        "40": "CI005021.INDX",  # 银行
        "41": "CI005022.INDX",  # 非银行金融
        "42": "CI005023.INDX",  # 房地产
        "43": "CI005030.INDX",  # 综合金融
        "50": "CI005024.INDX",  # 交通运输
        "60": "CI005025.INDX",  # 电子
        "61": "CI005026.INDX",  # 通信
        "62": "CI005027.INDX",  # 计算机
        "63": "CI005028.INDX",  # 传媒
        "70": "CI005029.INDX",  # 综合
    }

    # === 3. 并行读取行业成分文件 ===
    logger.info(f"Reading {len(industry_list)} industry component files ...")
    industry_componens_s_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(load_industry_file, code, data_path, industry_code_maps): code
            for code in industry_list["industry_code"]
        }

        for f in tqdm(as_completed(futures), total=len(futures), desc="Reading industry files"):
            df = f.result()
            if not df.empty:
                industry_componens_s_list.append(df)

    if not industry_componens_s_list:
        logger.error("No valid industry components found.")
        sys.exit(1)

    industry_componens_s = pd.concat(industry_componens_s_list, ignore_index=True)
    logger.info(f"Total component records: {len(industry_componens_s):,}")

    # === 4. 预分组股票 ===
    logger.info("Grouping industry components by stock ...")
    grouped = dict(tuple(industry_componens_s.groupby("stock_code")))
    stock_codes = list(grouped.keys())

    logger.info(f"Processing {len(stock_codes)} stocks ...")

    # === 5. 多进程计算行业归属 ===
    data_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(
                    process_stock_industry_wrapper,
                    [(code, grouped[code], industry_code_maps) for code in stock_codes],
                    chunksize=20,
                ),
                total=len(stock_codes),
                desc="Building stock-industry maps",
            )
        )

    for res in results:
        if res:
            data_list.extend(res)

    # === 6. 保存数据 ===
    if not data_list:
        logger.error("No stock-industry mapping data generated.")
        sys.exit(1)

    industry_stock_maps = pd.DataFrame(data_list)
    output_path = Path(f"{data_path}/basic_data/industry_componens/industry_stock_maps.csv")
    industry_stock_maps.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    logger.success(f"Industry Stock Maps saved: {output_path} | {len(industry_stock_maps):,} rows")


### ========================= 主入口 ========================= ###
if __name__ == "__main__":
    fire.Fire(industry_components_merge)
