import requests
import pandas as pd
import json
import sys
import os
from loguru import logger
import fire


class Qds:
    def __init__(self):
        self._name = "msyh_opt"
        self._pwd="msyh_123"
        self._login_url="https://qdstest.csc.com.cn/login"
        self._data_url="https://qdstest.csc.com.cn/load_barra_crowd"
        self._headers={'Content-Type': 'application/json'}
        
        
    def qds_login(self):
        payload = {"name": self._name, "pwd": self._pwd}
        login_info = requests.post(
            self._login_url, 
            json=payload, 
            headers=self._headers
        )
        login_data = login_info.json()
        token=login_data['data']['token']
        self._token = token   

    def qds_data_requests(self, begin_date, end_date, factor_code):

        is_dl = False

        # 登陆
        try:
            self.qds_login()
            logger.success(f"Qds Login succeed{factor_code}")
        except Exception as e:
            logger.error(f"Qds Login Failed: {factor_code} / {begin_date}-{end_date} / {str(e)}")
            return pd.DataFrame(), is_dl

            
        # 参数设置
        param_list = {
            "name": self._name,
            "token": self._token,
            "param": {
                "begin_date": begin_date,
                "end_date": end_date,
                "factor": factor_code
            }
            
        }

        # 获取数据
        try:
            data = requests.post(
                self._data_url, 
                json=param_list, 
                headers=self._headers
            )
            data = data.json()

            # 判断是否获取正确数据
            data_status = data['status']
            if data_status == 1:
                # factor_mom
                data_factor_mom = data['data']['factor_mom']
                data_factor_mom = pd.DataFrame(list(data_factor_mom.items()), columns=['date', 'factor_mom'])
                data_factor_mom['date'] = pd.to_datetime(data_factor_mom['date'])    

                # factor_std
                data_factor_std = data['data']['factor_std']
                data_factor_std = pd.DataFrame(list(data_factor_std.items()), columns=['date', 'factor_std'])
                data_factor_std['date'] = pd.to_datetime(data_factor_std['date'])    

                # pb
                data_pb = data['data']['pb']
                data_pb = pd.DataFrame(list(data_pb.items()), columns=['date', 'pb'])
                data_pb['date'] = pd.to_datetime(data_pb['date'])  

                # turnover
                data_turnover = data['data']['turnover']
                data_turnover = pd.DataFrame(list(data_turnover.items()), columns=['date', 'turnover'])
                data_turnover['date'] = pd.to_datetime(data_turnover['date']) 
                
                # crowd
                data_crowd = data['data']['crowd']
                data_crowd = pd.DataFrame(list(data_crowd.items()), columns=['date', 'crowd'])
                data_crowd['date'] = pd.to_datetime(data_crowd['date'])                         


                # merge
                data_results = pd.merge(data_factor_mom, data_factor_std, on='date', how='outer')
                data_results = pd.merge(data_results, data_pb, on='date', how='outer')
                data_results = pd.merge(data_results, data_turnover, on='date', how='outer')
                data_results = pd.merge(data_results, data_crowd, on='date', how='outer')

                is_dl = True
                return data_results, is_dl

            else:
                logger.error(f"Qds Data Download Failed: {factor_code} / {begin_date} - {end_date}")
                return pd.DataFrame(), is_dl
            
        except Exception as e:
            return pd.DataFrame(), is_dl



            
def qds_run(
    data_path: str = "/home/idc2/notebook/qds/Data",
):
    data_path = os.path.expanduser(data_path)  
    os.makedirs(data_path, exist_ok=True)

    factor_path = f"{data_path}/factor"
    os.makedirs(factor_path, exist_ok=True)
    
    # 读取因子列表
    try:
        factor_list_filename = f"{data_path}/factor_list.csv"
        factor_list = pd.read_csv(factor_list_filename)
    except Exception as e:
        logger.error(f"Qds Factor List Read Failed: {str(e)}")
        sys.exit(1)
        
    # 测试用
    # factor_list = factor_list[-2:]

    begin_date="20150101"
    end_date="20251031"
    
    for _, row in factor_list.iterrows():
        factor_code = (row['code'])

        # 读取已下载数据
        try:
            factor_filename = f"{factor_path}/{factor_code}.csv"
            factor_file = pd.read_csv(factor_filename)
            
            if factor_file.empty:
                s_date = begin_date
                e_date = end_date
                is_entirety = True
            else:
                factor_file['date'] = pd.to_datetime(factor_file['date'])
                s_date = factor_file['date'].max() + pd.Timedelta(days=1)

                if s_date >= pd.to_datetime(end_date):
                    logger.info(f"Qds Factor Data Exists:{factor_code}")
                    # 本地文件已最新
                    continue
                    
                s_date = s_date.strftime("%Y%m%d")
                e_date = end_date
                is_entirety = False
                
        except Exception as e:
            s_date = begin_date
            e_date = end_date
            is_entirety = True
        
              
        data, is_dl = Qds().qds_data_requests(s_date, e_date, factor_code)

        if is_dl and not data.empty:
            data['date'] = pd.to_datetime(data['date'])
        
            if is_entirety:
                factor_file = data.copy()
            else:
                factor_file = pd.concat([factor_file, data], ignore_index=True)

            factor_file.to_csv(factor_filename, index=False, date_format='%Y-%m-%d')
            logger.success(f"Qds Data Download Success: {factor_code} / {begin_date}-{end_date}")

        else:
            logger.error(f"Qds Data Download Failed: {factor_code} / {begin_date}-{end_date}")
            continue
        
        # print("\n" + f"---{factor_code}---"+ "-" * 100)
        # print(is_dl)
        # print(data)


            
if __name__ == "__main__":
    fire.Fire(qds_run)
     
        


    
