import pandas as pd

factor_list = [
    # 国家因子
    {"type": "国家因子", "code": "CNE5S_COUNTRY", "name": "国家"},

    # 风格因子
    {"type": "风格因子", "code": "CNE5S_BETA", "name": "BETA"},
    {"type": "风格因子", "code": "CNE5S_SIZE", "name": "规模"},
    {"type": "风格因子", "code": "CNE5S_EARNYILD", "name": "经营收益率"},
    {"type": "风格因子", "code": "CNE5S_RESVOL", "name": "残差波动率"},
    {"type": "风格因子", "code": "CNE5S_GROWTH", "name": "成长"},
    {"type": "风格因子", "code": "CNE5S_BTOP", "name": "账面市值比"},
    {"type": "风格因子", "code": "CNE5S_LEVERAGE", "name": "杠杆"},
    {"type": "风格因子", "code": "CNE5S_LIQUIDTY", "name": "流动性"},
    {"type": "风格因子", "code": "CNE5S_SIZENL", "name": "非线性规模"},
    {"type": "风格因子", "code": "CNE5S_MOMENTUM", "name": "动量"},

    # 行业因子
    {"type": "行业因子", "code": "CNE5S_ENERGY", "name": "能源"},
    {"type": "行业因子", "code": "CNE5S_CHEM", "name": "化学品"},
    {"type": "行业因子", "code": "CNE5S_CONMAT", "name": "建筑材料"},
    {"type": "行业因子", "code": "CNE5S_MTLMIN", "name": "特殊金属"},
    {"type": "行业因子", "code": "CNE5S_MATERIAL", "name": "材料"},
    {"type": "行业因子", "code": "CNE5S_AERODEF", "name": "航空和国防"},
    {"type": "行业因子", "code": "CNE5S_BLDPROD", "name": "建筑产品"},
    {"type": "行业因子", "code": "CNE5S_CNSTENG", "name": "建筑和工程"},
    {"type": "行业因子", "code": "CNE5S_ELECEQP", "name": "电子设备"},
    {"type": "行业因子", "code": "CNE5S_INDCONG", "name": "工业综合类"},
    {"type": "行业因子", "code": "CNE5S_MACH", "name": "工业机械"},
    {"type": "行业因子", "code": "CNE5S_TRDDIST", "name": "贸易公司和分销"},
    {"type": "行业因子", "code": "CNE5S_COMSERV", "name": "商业及专业服务"},
    {"type": "行业因子", "code": "CNE5S_AIRLINE", "name": "航空公司"},
    {"type": "行业因子", "code": "CNE5S_MARINE", "name": "船舶"},
    {"type": "行业因子", "code": "CNE5S_RDRLTRAN", "name": "公路铁路和运输基础设施"},
    {"type": "行业因子", "code": "CNE5S_AUTO", "name": "汽车及其零配件"},
    {"type": "行业因子", "code": "CNE5S_HOUSEDUR", "name": "家庭耐用品"},
    {"type": "行业因子", "code": "CNE5S_LEISLUX", "name": "纺织品服装和奢侈品"},
    {"type": "行业因子", "code": "CNE5S_CONSSERV", "name": "消费服务"},
    {"type": "行业因子", "code": "CNE5S_MEDIA", "name": "媒体"},
    {"type": "行业因子", "code": "CNE5S_RETAIL", "name": "零售"},
    {"type": "行业因子", "code": "CNE5S_PERSPRD", "name": "食品零售家用及个人消费品"},
    {"type": "行业因子", "code": "CNE5S_BEV", "name": "饮料"},
    {"type": "行业因子", "code": "CNE5S_FOODPROD", "name": "食品"},
    {"type": "行业因子", "code": "CNE5S_HEALTH", "name": "健康"},
    {"type": "行业因子", "code": "CNE5S_BANKS", "name": "银行"},
    {"type": "行业因子", "code": "CNE5S_DVFININS", "name": "多元化金融服务"},
    {"type": "行业因子", "code": "CNE5S_REALEST", "name": "房地产"},
    {"type": "行业因子", "code": "CNE5S_SOFTWARE", "name": "软件"},
    {"type": "行业因子", "code": "CNE5S_HDWRSEMI", "name": "硬件和半导体"},
    {"type": "行业因子", "code": "CNE5S_UTILITIE", "name": "公共事业"}
]

factor_list = pd.DataFrame(factor_list)

factor_list_filename = "/home/idc2/notebook/qds/Data/factor_list.csv"
factor_list.to_csv(factor_list_filename, index=False, date_format='%Y-%m-%d')

