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

#######################################

def get_klines_rq(code, start_date, end_date):
    """获取K数据"""
    try:
        klines_data = rqdatac.get_price(code, start_date=start_date, end_date=end_date, frequency='1d', fields=None, adjust_type='none', skip_suspended =False, market='cn', expect_df=True,time_slice=None)
        time.sleep(0.5)
        klines_data.reset_index(inplace=True)
        if klines_data is None or klines_data.empty:
            print("hahaha1")
            return pd.DataFrame()
        return klines_data
    except Exception as e:
        print(f"hahaha2:{e}")
        return pd.DataFrame()

def get_turn_rq(code, start_date, end_date):
    """获取Turn数据"""
    try:
        turn_data = rqdatac.get_turnover_rate(code, start_date=start_date, end_date=end_date, fields=None, expect_df=True)
        time.sleep(0.5)
        turn_data.reset_index(inplace=True)
        if turn_data is None or turn_data.empty:
            return pd.DataFrame()
        return turn_data
    except Exception as e:
        return pd.DataFrame()

def get_adj_rq(code):
    """获取adj数据"""
    try:
        adj_data = rqdatac.get_ex_factor(code, start_date=None, end_date=None, market='cn')
        time.sleep(0.5)
        adj_data.reset_index(inplace=True)
        if adj_data is None or adj_data.empty:
            return pd.DataFrame()
        return adj_data
    except Exception as e:
        return pd.DataFrame()        

def get_stock_factor_rq(code, start_date, end_date):
    """获取factor数据"""
    factor = [
        "pe_ratio_lyr",   # 市盈率lyr
        "pe_ratio_ttm",   # 市盈率ttm
        "ep_ratio_lyr",   # 盈市率lyr
        "ep_ratio_ttm",   # 盈市率ttm
        "pcf_ratio_total_lyr",   # 市现率_总现金流lyr
        "pcf_ratio_total_ttm",   # 市现率_总现金流ttm
        "pcf_ratio_lyr",   # 市现率_经营lyr
        "pcf_ratio_ttm",   # 市现率_经营 ttm
        "cfp_ratio_lyr",   # 现金收益率lyr
        "cfp_ratio_ttm",   # 现金收益率ttm
        "pb_ratio_lyr",   # 市净率lyr
        "pb_ratio_ttm",   # 市净率ttm
        "pb_ratio_lf",   # 市净率lf
        "book_to_market_ratio_lyr",   # 账面市值比lyr
        "book_to_market_ratio_ttm",   # 账面市值比 ttm
        "book_to_market_ratio_lf",   # 账面市值比lf
        "dividend_yield_ttm",   # 股息率ttm
        "peg_ratio_lyr",   # PEG值lyr
        "peg_ratio_ttm",   # PEG值ttm
        "ps_ratio_lyr",   # 市销率lyr
        "ps_ratio_ttm",   # 市销率ttm
        "sp_ratio_lyr",   # 销售收益率lyr
        "sp_ratio_ttm",   # 销售收益率ttm
        "market_cap",   # 总市值1
        "market_cap_2",   # 流通股总市值
        "market_cap_3",   # 总市值
        "a_share_market_val",   # A股市值
        "a_share_market_val_in_circulation",   # 流通A股市值
        "ev_lyr",   # 企业价值lyr
        "ev_ttm",   # 企业价值ttm
        "ev_lf",   # 企业价值lf
        "ev_no_cash_lyr",   # 企业价值(不含货币资金)lyr
        "ev_no_cash_ttm",   # 企业价值(不含货币资金)ttm
        "ev_no_cash_lf",   # 企业价值(不含货币资金)lf
        "ev_to_ebitda_lyr",   # 企业倍数lyr
        "ev_to_ebitda_ttm",   # 企业倍数ttm
        "ev_no_cash_to_ebit_lyr",   # 企业倍数(不含货币资金)lyr
        "ev_no_cash_to_ebit_ttm",   # 企业倍数(不含货币资金)ttm
    
        "diluted_earnings_per_share_lyr",   # 摊薄每股收益lyr
        "diluted_earnings_per_share_ttm",   # 摊薄每股收益ttm
        "adjusted_earnings_per_share_lyr",   # 基本每股收益_扣除lyr
        "adjusted_earnings_per_share_ttm",   # 基本每股收益_扣除ttm
        "adjusted_fully_diluted_earnings_per_share_lyr",   # 稀释每股收益_扣除lyr
        "adjusted_fully_diluted_earnings_per_share_ttm",   # 稀释每股收益_扣除ttm
        "weighted_common_stock_lyr",   # 普通股加权股本lyr
        "weighted_common_stock_ttm",   # 普通股加权股本ttm
        "diluted_common_stock_lyr",   # 稀释普通股lyr
        "diluted_common_stock_ttm",   # 稀释普通股ttm
        "operating_total_revenue_per_share_lyr",   # 每股营业总收入lyr
        "operating_total_revenue_per_share_ttm",   # 每股营业总收入ttm
        "operating_revenue_per_share_lyr",   # 每股营业收入lyr
        "operating_revenue_per_share_ttm",   # 每股营业收入ttm
        "ebit_lyr",   # 息税前利润lyr
        "ebit_ttm",   # 息税前利润ttm
        "ebitda_lyr",   # 息税折旧摊销前利润lyr
        "ebitda_ttm",   # 息税折旧摊销前利润 ttm
        "ebit_per_share_lyr",   # 每股息税前利润lyr
        "ebit_per_share_ttm",   # 每股息税前利润ttm
        "return_on_equity_lyr",   # 净资产收益率lyr
        "return_on_equity_ttm",   # 净资产收益率ttm
        "return_on_equity_diluted_lyr",   # 摊薄净资产收益率lyr
        "return_on_equity_diluted_ttm",   # 摊薄净资产收益率ttm
        "adjusted_return_on_equity_lyr",   # 净资产收益率_扣除lyr
        "adjusted_return_on_equity_ttm",   # 净资产收益率_扣除ttm
        "adjusted_return_on_equity_diluted_lyr",   # 摊薄净资产收益率_扣除lyr
        "adjusted_return_on_equity_diluted_ttm",   # 摊薄净资产收益率_扣除ttm
        "return_on_asset_lyr",   # 总资产报酬率lyr
        "return_on_asset_ttm",   # 总资产报酬率ttm
        "return_on_asset_net_profit_lyr",   # 总资产净利率lyr
        "return_on_asset_net_profit_ttm",   # 总资产净利率ttm
        "return_on_invested_capital_lyr",   # 投入资本回报率lyr
        "return_on_invested_capital_ttm",   # 投入资本回报率ttm
        "net_profit_margin_lyr",   # 销售净利率lyr
        "net_profit_margin_ttm",   # 销售净利率ttm
        "gross_profit_margin_lyr",   # 销售毛利率lyr
        "gross_profit_margin_ttm",   # 销售毛利率ttm
        "cost_to_sales_lyr",   # 销售成本率lyr
        "cost_to_sales_ttm",   # 销售成本率ttm
        "net_profit_to_revenue_lyr",   # 经营净利率lyr
        "net_profit_to_revenue_ttm",   # 经营净利率ttm
        "profit_from_operation_to_revenue_lyr",   # 营业利润率lyr
        "profit_from_operation_to_revenue_ttm",   # 营业利润率ttm
        "ebit_to_revenue_lyr",   # 税前收益率lyr
        "ebit_to_revenue_ttm",   # 税前收益率ttm
        "expense_to_revenue_lyr",   # 经营成本率lyr
        "expense_to_revenue_ttm",   # 经营成本率ttm
        "operating_profit_to_profit_before_tax_lyr",   # 经营活动净收益与利润总额之比lyr
        "operating_profit_to_profit_before_tax_ttm",   # 经营活动净收益与利润总额之比ttm
        "investment_profit_to_profit_before_tax_lyr",   # 价值变动净收益与利润总额之比lyr
        "investment_profit_to_profit_before_tax_ttm",   # 价值变动净收益与利润总额之比ttm
        "non_operating_profit_to_profit_before_tax_lyr",   # 营业外收支净额与利润总额之比lyr
        "non_operating_profit_to_profit_before_tax_ttm",   # 营业外收支净额与利润总额之比ttm
        "income_tax_to_profit_before_tax_lyr",   # 所得税与利润总额之比lyr
        "income_tax_to_profit_before_tax_ttm",   # 所得税与利润总额之比ttm
        "adjusted_profit_to_total_profit_lyr",   # 扣除非经常损益后的净利润与净利润之比lyr
        "adjusted_profit_to_total_profit_ttm",   # 扣除非经常损益后的净利润与净利润之比ttm
        "ebitda_to_debt_lyr",   # 息税折旧摊销前利润/负债总计lyr
        "ebitda_to_debt_ttm",   # 息税折旧摊销前利润/负债总计ttm
        "account_payable_turnover_rate_lyr",   # 应付账款周转率lyr
        "account_payable_turnover_rate_ttm",   # 应付账款周转率ttm
        "account_payable_turnover_days_lyr",   # 应付账款周转天数lyr
        "account_payable_turnover_days_ttm",   # 应付账款周转天数ttm
        "account_receivable_turnover_rate_lyr",   # 应收账款周转率lyr
        "account_receivable_turnover_rate_ttm",   # 应收账款周转率ttm
        "account_receivable_turnover_days_lyr",   # 应收账款周转天数lyr
        "account_receivable_turnover_days_ttm",   # 应收账款周转天数ttm
        "inventory_turnover_lyr",   # 存货周转率lyr
        "inventory_turnover_ttm",   # 存货周转率ttm
        "current_asset_turnover_lyr",   # 流动资产周转率lyr
        "current_asset_turnover_ttm",   # 流动资产周转率ttm
        "fixed_asset_turnover_lyr",   # 固定资产周转率lyr
        "fixed_asset_turnover_ttm",   # 固定资产周转率ttm
        "total_asset_turnover_lyr",   # 总资产周转率lyr
        "total_asset_turnover_ttm",   # 总资产周转率ttm
        "du_profit_margin_lyr",   # 净利率(杜邦分析）lyr
        "du_profit_margin_ttm",   # 净利率(杜邦分析）ttm
        "du_return_on_equity_lyr",   # 净资产收益率ROE(杜邦分析)lyr
        "du_return_on_equity_ttm",   # 净资产收益率ROE(杜邦分析)ttm
        "du_return_on_sales_lyr",   # 息税前利润/营业总收入lyr
        "du_return_on_sales_ttm",   # 息税前利润/营业总收入ttm
        "income_from_main_operations_lyr",   # 主营业务利润lyr
        "income_from_main_operations_ttm",   # 主营业务利润ttm
        "time_interest_earned_ratio_lyr",   # 利息保障倍数lyr
        "time_interest_earned_ratio_ttm",   # 利息保障倍数ttm
        "equity_turnover_ratio_lyr",   # 股东权益周转率lyr
        "equity_turnover_ratio_ttm",   # 股东权益周转率ttm
        "operating_cycle_lyr",   # 营业周期lyr
        "operating_cycle_ttm",   # 营业周期ttm
        "average_payment_period_lyr",   # 应付账款付款期lyr
        "average_payment_period_ttm",   # 应付账款付款期ttm
        "cash_conversion_cycle_lyr",   # 现金转换周期lyr
        "cash_conversion_cycle_ttm",   # 现金转换周期ttm
    
        "cash_flow_per_share_lyr",   # 每股现金流lyr
        "cash_flow_per_share_ttm",   # 每股现金流ttm
        "operating_cash_flow_per_share_lyr",   # 每股经营现金流lyr
        "operating_cash_flow_per_share_ttm",   # 每股经营现金流ttm
        "fcff_lyr",   # 企业自由现金流量lyr
        "fcff_ttm",   # 企业自由现金流量ttm
        "fcfe_lyr",   # 股权自由现金流量lyr
        "fcfe_ttm",   # 股权自由现金流量ttm
        "free_cash_flow_company_per_share_lyr",   # 每股企业自由现金流lyr
        "free_cash_flow_company_per_share_ttm",   # 每股企业自由现金流ttm
        "free_cash_flow_equity_per_share_lyr",   # 每股股东自由现金流lyr
        "free_cash_flow_equity_per_share_ttm",   # 每股股东自由现金流ttm
        "ocf_to_debt_lyr",   # 经营活动产生的现金流量净额/负债合计lyr
        "ocf_to_debt_ttm",   # 经营活动产生的现金流量净额/负债合计ttm
        "surplus_cash_protection_multiples_lyr",   # 盈余现金保障倍数lyr
        "surplus_cash_protection_multiples_ttm",   # 盈余现金保障倍数ttm
        "ocf_to_interest_bearing_debt_lyr",   # 经营活动产生的现金流量净额/带息债务lyr
        "ocf_to_interest_bearing_debt_ttm",   # 经营活动产生的现金流量净额/带息债务ttm
        "ocf_to_current_ratio_lyr",   # 经营活动产生的现金流量净额/流动负债
        "ocf_to_current_ratio_ttm",   # 经营活动产生的现金流量净额/流动负债ttm
        "ocf_to_net_debt_lyr",   # 经营活动产生的现金流量净额/净债务lyr
        "ocf_to_net_debt_ttm",   # 经营活动产生的现金流量净额/净债务ttm
        "depreciation_and_amortization_lyr",   # 当期计提折旧与摊销lyr
        "depreciation_and_amortization_ttm",   # 当期计提折旧与摊销ttm
        "cash_flow_ratio_lyr",   # 现金流量比率lyr
        "cash_flow_ratio_ttm",   # 现金流量比率ttm
    
        "non_interest_bearing_current_debt_lyr",   # 无息流动负债lyr
        "non_interest_bearing_current_debt_ttm",   # 无息流动负债ttm
        "non_interest_bearing_current_debt_lf",   # 无息流动负债lf
        "non_interest_bearing_non_current_debt_lyr",   # 无息非流动负债lyr
        "non_interest_bearing_non_current_debt_ttm",   # 无息非流动负债ttm
        "non_interest_bearing_non_current_debt_lf",   # 无息非流动负债lf
        "interest_bearing_debt_lyr",   # 带息债务lyr
        "interest_bearing_debt_ttm",   # 带息债务ttm
        "interest_bearing_debt_lf",   # 带息债务lf
        "capital_reserve_per_share_lyr",   # 每股资本公积金lyr
        "capital_reserve_per_share_ttm",   # 每股资本公积金ttm
        "capital_reserve_per_share_lf",   # 每股资本公积金lf
        "earned_reserve_per_share_lyr",   # 每股盈余公积金lyr
        "earned_reserve_per_share_ttm",   # 每股盈余公积金ttm
        "earned_reserve_per_share_lf",   # 每股盈余公积金lf
        "undistributed_profit_per_share_lyr",   # 每股未分配利润lyr
        "undistributed_profit_per_share_ttm",   # 每股未分配利润ttm
        "undistributed_profit_per_share_lf",   # 每股未分配利润lf
        "retained_earnings_lyr",   # 留存收益 lyr
        "retained_earnings_ttm",   # 留存收益ttm
        "retained_earnings_lf",   # 留存收益lf
        "retained_earnings_per_share_lyr",   # 每股留存收益lyr
        "retained_earnings_per_share_ttm",   # 每股留存收益ttm
        "retained_earnings_per_share_lf",   # 每股留存收益lf
        "debt_to_asset_ratio_lyr",   # 资产负债率lyr
        "debt_to_asset_ratio_ttm",   # 资产负债率ttm
        "debt_to_asset_ratio_lf",   # 资产负债率lf
        "equity_multiplier_lyr",   # 权益乘数lyr
        "equity_multiplier_ttm",   # 权益乘数ttm
        "equity_multiplier_lf",   # 权益乘数lf
        "capital_to_equity_ratio_lyr",   # 长期资本固定比率lyr
        "capital_to_equity_ratio_ttm",   # 长期资本固定比率ttm
        "capital_to_equity_ratio_lf",   # 长期资本固定比率lf
        "current_asset_to_total_asset_lf",   # 流动资产比率lf
        "non_current_asset_to_total_asset_lyr",   # 非流动资产比率lyr
        "non_current_asset_to_total_asset_ttm",   # 非流动资产比率ttm
        "non_current_asset_to_total_asset_lf",   # 非流动资产比率lf
        "invested_capital_lyr",   # 全部投入资本lyr
        "invested_capital_ttm",   # 全部投入资本ttm
        "invested_capital_lf",   # 全部投入资本lf
        "interest_bearing_debt_to_capital_lyr",   # 带息债务占企业全部投入成本的比重lyr
        "interest_bearing_debt_to_capital_ttm",   # 带息债务占企业全部投入成本的比重ttm
        "interest_bearing_debt_to_capital_lf",   # 带息债务占企业全部投入成本的比重lf
        "current_debt_to_total_debt_lyr",   # 流动负债率lyr
        "current_debt_to_total_debt_ttm",   # 流动负债率ttm
        "current_debt_to_total_debt_lf",   # 流动负债率lf
        "non_current_debt_to_total_debt_lyr",   # 非流动负债率lyr
        "non_current_debt_to_total_debt_ttm",   # 非流动负债率ttm
        "non_current_debt_to_total_debt_lf",   # 非流动负债率lf
        "current_ratio_lyr",   # 流动比率lyr
        "current_ratio_ttm",   # 流动比率ttm
        "current_ratio_lf",   # 流动比率lf
        "quick_ratio_lyr",   # 速动比率lyr
        "quick_ratio_ttm",   # 速动比率ttm
        "quick_ratio_lf",   # 速动比率lf
        "super_quick_ratio_lyr",   # 超速动比率lyr
        "super_quick_ratio_ttm",   # 超速动比率ttm
        "super_quick_ratio_lf",   # 超速动比率lf
        "debt_to_equity_ratio_lyr",   # 产权比率lyr
        "debt_to_equity_ratio_ttm",   # 产权比率ttm
        "debt_to_equity_ratio_lf",   # 产权比率lf
        "equity_to_debt_ratio_lyr",   # 权益负债比率lyr
        "equity_to_debt_ratio_ttm",   # 权益负债比率ttm
        "equity_to_debt_ratio_lf",   # 权益负债比率lf
        "equity_to_interest_bearing_debt_lyr",   # 权益带息负债比率 lyr
        "equity_to_interest_bearing_debt_ttm",   # 权益带息负债比率 ttm
        "equity_to_interest_bearing_debt_lf",   # 权益带息负债比率lf
        "net_debt_lyr",   # 净债务lyr
        "net_debt_ttm",   # 净债务ttm
        "net_debt_lf",   # 净债务lf
        "working_capital_lyr",   # 营运资本lyr
        "working_capital_ttm",   # 营运资本ttm
        "working_capital_lf",   # 营运资本lf
        "net_working_capital_lyr",   # 净营运资本lyr
        "net_working_capital_ttm",   # 净营运资本ttm
        "net_working_capital_lf",   # 净营运资本lf
        "long_term_debt_to_working_capital_lyr",   # 长期债务与营运资金比率lyr
        "long_term_debt_to_working_capital_ttm",   # 长期债务与营运资金比率ttm
        "long_term_debt_to_working_capital_lf",   # 长期债务与营运资金比率lf
        "book_value_per_share_lyr",   # 每股净资产lyr
        "book_value_per_share_ttm",   # 每股净资产ttm
        "book_value_per_share_lf",   # 每股净资产lf
        "du_equity_multiplier_lyr",   # 权益乘数(杜邦分析)lyr
        "du_equity_multiplier_ttm",   # 权益乘数(杜邦分析)ttm
        "du_equity_multiplier_lf",   # 权益乘数(杜邦分析)lf
        "book_leverage_lyr",   # 账面杠杆lyr
        "book_leverage_ttm",   # 账面杠杆ttm
        "book_leverage_lf",   # 账面杠杆lf
        "market_leverage_lyr",   # 市场杠杆lyr
        "market_leverage_ttm",   # 市场杠杆ttm
        "market_leverage_lf",   # 市场杠杆lf
        "equity_ratio_lyr",   # 股东权益比率lyr
        "equity_ratio_ttm",   # 股东权益比率ttm
        "equity_ratio_lf",   # 股东权益比率lf
        "fixed_asset_ratio_lyr",   # 固定资产比率lyr
        "fixed_asset_ratio_ttm",   # 固定资产比率ttm
        "fixed_asset_ratio_lf",   # 固定资产比率lf
        "intangible_asset_ratio_lyr",   # 无形资产比率lyr
        "intangible_asset_ratio_ttm",   # 无形资产比率ttm
        "intangible_asset_ratio_lf",   # 无形资产比率lf
        "equity_fixed_asset_ratio_lyr",   # 股东权益与固定资产比率lyr
        "equity_fixed_asset_ratio_ttm",   # 股东权益与固定资产比率ttm
        "equity_fixed_asset_ratio_lf",   # 股东权益与固定资产比率lf
        "tangible_asset_per_share_lyr",   # 每股有形资产lyr
        "tangible_asset_per_share_ttm",   # 每股有形资产ttm
        "tangible_asset_per_share_lf",   # 每股有形资产lf
        "liabilities_per_share_lyr",   # 每股负债lyr
        "liabilities_per_share_ttm",   # 每股负债ttm
        "liabilities_per_share_lf",   # 每股负债lf
        "depreciation_per_share_lyr",   # 每股折旧和摊销lyr
        "depreciation_per_share_ttm",   # 每股折旧和摊销ttm
        "depreciation_per_share_lf",   # 每股折旧和摊销lf
        "cash_ratio_lyr",   # 现金比率lyr
        "cash_ratio_ttm",   # 现金比率ttm
        "cash_ratio_lf",   # 现金比率lf
        "cash_equivalent_per_share_lyr",   # 每股货币资金lyr
        "cash_equivalent_per_share_ttm",   # 每股货币资金ttm
        "cash_equivalent_per_share_lf",   # 每股货币资金lf
        "dividend_amount_ly0",   # 最近年度分红总额
        "dividend_amount_ly1",   # 最近年度分红总额
        "dividend_amount_ly2",   # 最近年度分红总额
        "dividend_amount_ttm0",   # 最近四个季度分红总额
        "dividend_amount_ttm1",   # 最近四个季度分红总额
        "dividend_amount_ttm2",   # 最近四个季度分红总额
    
        "inc_revenue_lyr",   # 营业总收入同比增长率lyr
        "inc_revenue_ttm",   # 营业总收入同比增长率ttm
        "inc_return_on_equity_lyr",   # 净资产收益率(摊薄）同比增长率lyr
        "inc_return_on_equity_ttm",   # 净资产收益率(摊薄）同比增长率ttm
        "inc_book_per_share_lyr",   # 每股净资产同比增长率lyr
        "inc_book_per_share_ttm",   # 每股净资产同比增长率ttm
        "inc_book_per_share_lf",   # 每股净资产同比增长率lf
        "operating_profit_growth_ratio_lyr",   # 营业利润同比增长率lyr
        "operating_profit_growth_ratio_ttm",   # 营业利润同比增长率ttm
        "net_profit_growth_ratio_lyr",   # 净利润同比增长率lyr
        "net_profit_growth_ratio_ttm",   # 净利润同比增长率ttm
        "profit_growth_ratio_lyr",   # 利润总额同比增长率lyr
        "profit_growth_ratio_ttm",   # 利润总额同比增长率ttm
        "gross_profit_growth_ratio_lyr",   # 毛利润同比增长率lyr
        "gross_profit_growth_ratio_ttm",   # 毛利润同比增长率ttm
        "operating_revenue_growth_ratio_lyr",   # 营业收入同比增长率lyr
        "operating_revenue_growth_ratio_ttm",   # 营业收入同比增长率ttm
        "net_asset_growth_ratio_lyr",   # 净资产同比增长率lyr
        "net_asset_growth_ratio_ttm",   # 净资产同比增长率ttm
        "net_asset_growth_ratio_lf",   # 净资产同比增长率lf
        "total_asset_growth_ratio_lyr",   # 总资产同比增长率lyr
        "total_asset_growth_ratio_ttm",   # 总资产同比增长率ttm
        "total_asset_growth_ratio_lf",   # 总资产同比增长率lf
        "net_profit_parent_company_growth_ratio_lyr",   # 归属母公司所有者的净利润同比增长率lyr
        "net_profit_parent_company_growth_ratio_ttm",   # 归属母公司所有者的净利润同比增长率ttm
        "net_cash_flow_growth_ratio_lyr",   # 净现金流增长率
        "net_cash_flow_growth_ratio_ttm",   # 净现金流增长率
        "net_operate_cash_flow_growth_ratio_lyr",   # 经营现金流量净额同比增长率lyr
        "net_operate_cash_flow_growth_ratio_ttm",   # 经营现金流量净额同比增长率ttm
        "net_investing_cash_flow_growth_ratio_lyr",   # 投资现金流量净额同比增长率lyr
        "net_investing_cash_flow_growth_ratio_ttm",   # 投资现金流量净额同比增长率ttm
        "net_financing_cash_flow_growth_ratio_lyr",   # 筹资现金流量净额同比增长率lyr
        "net_financing_cash_flow_growth_ratio_ttm",   # 筹资现金流量净额同比增长率ttm 
    ]
    try:
        factor_data = rqdatac.get_factor(code, factor, start_date=start_date, end_date=end_date, universe=None, expect_df=True)
        time.sleep(0.5)
        factor_data.reset_index(inplace=True)
        if factor_data is None or factor_data.empty:
            return pd.DataFrame()
        return factor_data
    except Exception as e:
        return pd.DataFrame()  

def data_download(
    code: Union[str, None]       = "600000.XSHG",     # "600588.XSHG"
    start_date: Union[str, None] = "2025-10-01",
    end_date: Union[str, None]   = "2025-10-21"
):

    end_date = datetime.now().date()
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    print(f"Code:{code} || start_date:{start_date.date()} || end_date:{end_date.date()}")


    ### rq init
    try:
        # rqdatac.init(addr='/home/idc2/notebook/rqdata/.rqdatac/license.txt', enable_bjse=True)
        rqdatac.init(enable_bjse=True)
        # rqdatac.info
        
    except Exception as e:
        logger.error(f"rqdatac init error: {str(e)}")
        exit(1)

    klines_data = get_klines_rq(code, start_date, end_date)
    print("-" * 3 + " klines_data " +  "-" * 100)
    print(klines_data)

    adj_data = get_adj_rq(code)
    print("-" * 3 + " adj_data " +  "-" * 100)
    print(adj_data)

    turn_data = get_turn_rq(code, start_date, end_date)
    print("-" * 3 + " turn_data " +  "-" * 100)
    print(turn_data)
    
    factor_data = get_stock_factor_rq(code, start_date, end_date)
    print("-" * 3 + " factor_data " +  "-" * 100)
    print(factor_data.tail(10))

    code = "CI005001.INDX"
    start_date="1990-01-01"
    end_date="2025-10-28"
    industry_data = get_klines_rq(code, start_date, end_date)
    print("-" * 3 + " industry_data " +  "-" * 100)
    print(industry_data)
    
if __name__ == "__main__":
    fire.Fire(data_download)
    

