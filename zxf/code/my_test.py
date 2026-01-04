import pandas as pd
import numpy as np


qlib_source_path = f"/home/idc2/notebook/rqdata/Data/qlib_data/raw_data/SH600061.csv"

stock_source = pd.read_csv(qlib_source_path)
stock_source_s = stock_source[(stock_source['volume']==0) & (stock_source['amount']==0)]
print(stock_source_s[['tradedate', 'high', 'low', 'open', 'close', 'volume', 'amount',]])

m_mask = (stock_source['volume']==0) & (stock_source['amount']==0)
stock_source.loc[m_mask, ['high', 'low', 'open']] = 0

stock_source_t = stock_source[stock_source['tradedate']=="2016-10-10"]
print(stock_source_t[['tradedate', 'high', 'low', 'open', 'close', 'volume', 'amount',]])
