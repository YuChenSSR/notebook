import requests
import pandas as pd
import json
import sys
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
        try:
            login_info = requests.post(
                self._login_url, 
                json=payload, 
                headers=self._headers,
                # timeout=30,
            )
            login_data = login_info.json()
            token=login_data['data']['token']

            self._token = token            
        except Exception as e:
            logger.error(f"Qds Login Failed: {str(e)}")
            sys.exit(1)

    def qds_data(self):

        self.qds_login()
        print(self._token)

        
        param_list = {
            "name": self._name,
            "token": self._token,
            "param": {
                "begin_date": "20150101",
                "end_date": "20251031",
                "factor": "CNE5S_BEV"
            }
            
        }
        try:
            data = requests.post(
                self._data_url, 
                json=param_list, 
                headers=self._headers
            )
            data = data.json()
            print("\n" + "-" * 100)
            print(data)
            
            data_status = data['status']
            print("\n" + "-" * 100)
            print(data_status)

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

                print("\n" + "-" * 100)
                print(data_results)
            

            
        except Exception as e:
            logger.error(f"Qds Data Failed: {str(e)}")



            
def qds_run():
    Qds().qds_data()

            
if __name__ == "__main__":
    fire.Fire(qds_run)
     
        


    
