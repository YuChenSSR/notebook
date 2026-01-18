from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l

_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class Alpha158(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        print('Using custom alpha + manually defined + [20260115] ...')


        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {},
            "volume": {},
            "feature_filter": {},
            "time_series": {},
            "gating": {},
        }
        return self.parse_config_to_fields(conf)

    def get_label_config(self):
        return ["Ref($adjclose, -5)/Ref($adjclose, -1) - 1"], ["LABEL0"]

    @staticmethod
    def parse_config_to_fields(config):
        col_list = [
            "$adjusted_return_on_equity_ttm",
            "$d_close_rank",
            "$d_cntp",
            "$d_hhi",
            "$d_ret_skew",
            "$d_volatility_of_volume",
            "$high/($close+1e-12)",
            "$i_change_1",
            "$i_change_1_std_30",
            "$inc_book_per_share_ttm",
            "$low/($close+1e-12)",
            "$net_profit_growth_ratio_ttm",
            "$pb_ratio_ttm",
            "$peg_ratio_ttm",
            "$return_on_asset_net_profit_ttm",
            "$turn/($turn_current_year+1e-12)",
            "$volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12))",
            "($close-$open)/$open",
            "($high-Greater($open, $close))/$open",
            "(2*$close-$high-$low)/$open",
            "(Less($open, $close)-$low)/$open",
            "(Less($open, $close)-$low)/($high-$low+1e-12)",
            "Corr($close, Log($D_CORR + 1), 23)",
            "Corr($close, Log($D_CORR + 1), 61)",
            "Corr($close, Log($D_DWR + 1), 23)",
            "Corr($close, Log($D_DWR + 1), 61)",
            "Corr($close, Log($D_REALIZED_VOL + 1), 11)",
            "Corr($close, Log($D_VWAP + 1), 11)",
            "Corr($close, Log($D_VWAP + 1), 23)",
            "Corr($close, Log($D_VWAP + 1), 61)",
            "Corr($close/Ref($close,1), Log($D_CORR/Ref($D_CORR, 1)+1), 23)",
            "Corr($close/Ref($close,1), Log($D_CORR/Ref($D_CORR, 1)+1), 61)",
            "Corr($close/Ref($close,1), Log($D_DWR/Ref($D_DWR, 1)+1), 23)",
            "Corr($close/Ref($close,1), Log($D_REALIZED_VOL/Ref($D_REALIZED_VOL, 1)+1), 11)",
            "Corr($close/Ref($close,1), Log($D_REALIZED_VOL/Ref($D_REALIZED_VOL, 1)+1), 5)",
            "Corr($close/Ref($close,1), Log($D_TRADES_CLOSE_CORR/Ref($D_TRADES_CLOSE_CORR, 1)+1), 23)",
            "Corr($close/Ref($close,1), Log($D_TRADES_CLOSE_CORR/Ref($D_TRADES_CLOSE_CORR, 1)+1), 61)",
            "Corr($close/Ref($close,1), Log($D_TRADES_VOL_CORR/Ref($D_TRADES_VOL_CORR, 1)+1), 23)",
            "Corr($close/Ref($close,1), Log($D_VWAP/Ref($D_VWAP, 1)+1), 11)",
            "Corr($close/Ref($close,1), Log($D_VWAP/Ref($D_VWAP, 1)+1), 61)",
            "Max($high, 5)/$close",
            "Mean($close, 23)/($close + 1e-12)",
            "Mean($d_cntn<Ref($d_cntn, 1), 5)",
            "Mean($d_cntn>Ref($d_cntn, 1), 11)-Mean($d_cntn<Ref($d_cntn, 1), 11)",
            "Mean($d_cntn>Ref($d_cntn, 1), 5)",
            "Mean($d_cntn>Ref($d_cntn, 1), 5)-Mean($d_cntn<Ref($d_cntn, 1), 5)",
            "Mean($d_cntp>Ref($d_cntp, 1), 11)-Mean($d_cntp<Ref($d_cntp, 1), 11)",
            "Mean($d_cntp>Ref($d_cntp, 1), 5)",
            "Mean($d_cntp>Ref($d_cntp, 1), 5)-Mean($d_cntp<Ref($d_cntp, 1), 5)",
            "Mean($d_corr, 23)/($d_corr + 1e-12)",
            "Mean($d_dwr, 61)/($d_dwr + 1e-12)",
            "Mean($d_intraday_trend, 11)/($d_intraday_trend + 1e-12)",
            "Mean($d_intraday_trend, 5)/($d_intraday_trend + 1e-12)",
            "Mean($d_low_time_frac, 11)/($d_low_time_frac + 1e-12)",
            "Mean($d_trades_close_corr, 23)/($d_trades_close_corr + 1e-12)",
            "Mean($turn<Ref($turn, 1), 11)",
            "Mean($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 61)/($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)))",
            "Mean($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12))<Ref($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 1), 11)",
            "Mean($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12))>Ref($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 1), 11)",
            "Mean($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12))>Ref($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 1), 11)-Mean($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12))<Ref($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 1), 11)",
            "Mean($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12))>Ref($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 1), 61)-Mean($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12))<Ref($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 1), 61)",
            "Mean($volume>Ref($volume, 1), 11)",
            "Mean(($buy_volume*2-$volume)/($volume+1e-12)<Ref(($buy_volume*2-$volume)/($volume+1e-12), 1), 5)",
            "Mean(($buy_volume*2-$volume)/($volume+1e-12)>Ref(($buy_volume*2-$volume)/($volume+1e-12), 1), 11)-Mean(($buy_volume*2-$volume)/($volume+1e-12)<Ref(($buy_volume*2-$volume)/($volume+1e-12), 1), 11)",
            "Mean(($buy_volume*2-$volume)/($volume+1e-12)>Ref(($buy_volume*2-$volume)/($volume+1e-12), 1), 23)-Mean(($buy_volume*2-$volume)/($volume+1e-12)<Ref(($buy_volume*2-$volume)/($volume+1e-12), 1), 23)",
            "Mean((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 11)/((($buy_amount*2-$amount)*10000)/($market_cap+1e-12))",
            "Mean((($buy_amount*2-$amount)*10000)/($market_cap+1e-12)>Ref((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 1), 11)",
            "Min($low, 5)/$close",
            "Quantile($d_corr, 23, 0.2)/($d_corr + 1e-12)",
            "Quantile($d_dwr, 23, 0.2)/($d_dwr + 1e-12)",
            "Quantile($d_hhi, 61, 0.2)/($d_hhi + 1e-12)",
            "Quantile($d_realized_vol, 5, 0.8)/($d_realized_vol + 1e-12)",
            "Quantile($d_ret_kurt, 5, 0.8)/($d_ret_kurt + 1e-12)",
            "Quantile($d_ret_skew, 11, 0.2)/($d_ret_skew + 1e-12)",
            "Quantile($d_ret_skew, 5, 0.8)/($d_ret_skew + 1e-12)",
            "Quantile($d_trades_close_corr, 23, 0.2)/($d_trades_close_corr + 1e-12)",
            "Quantile($d_trades_close_corr, 23, 0.8)/($d_trades_close_corr + 1e-12)",
            "Quantile($d_trades_vol_corr, 23, 0.2)/($d_trades_vol_corr + 1e-12)",
            "Rank($d_cntn, 11)",
            "Rank($d_cntp, 11)",
            "Rank($d_ret_kurt, 23)",
            "Rank($d_trades_close_corr, 61)",
            "Rank($d_trades_vol_corr, 23)",
            "Rank($d_vol, 23)",
            "Rank($d_vol_cv, 61)",
            "Rank($d_volatility_of_volume, 61)",
            "Rank(($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12))), 23)",
            "Ref($close, 23)/($close+1e-12)",
            "Ref($d_close_rank, 11)/($d_close_rank+1e-12)",
            "Ref($d_close_rank, 23)/($d_close_rank+1e-12)",
            "Ref($d_corr, 23)/($d_corr+1e-12)",
            "Ref($d_corr, 61)/($d_corr+1e-12)",
            "Ref($d_dwr, 23)/($d_dwr+1e-12)",
            "Ref($d_dwr, 61)/($d_dwr+1e-12)",
            "Ref($d_high_time_frac, 11)/($d_high_time_frac+1e-12)",
            "Ref($d_high_time_frac, 23)/($d_high_time_frac+1e-12)",
            "Ref($d_intraday_trend, 11)/($d_intraday_trend+1e-12)",
            "Ref($d_intraday_trend, 5)/($d_intraday_trend+1e-12)",
            "Ref($d_low_time_frac, 11)/($d_low_time_frac+1e-12)",
            "Ref($d_low_time_frac, 23)/($d_low_time_frac+1e-12)",
            "Ref($d_ret_skew, 11)/($d_ret_skew+1e-12)",
            "Ref($d_ret_skew, 23)/($d_ret_skew+1e-12)",
            "Ref($d_ret_skew, 5)/($d_ret_skew+1e-12)",
            "Ref($d_trades_close_corr, 23) - $d_trades_close_corr",
            "Ref($d_trades_vol_corr, 61) - $d_trades_vol_corr",
            "Ref($d_vol_cv, 23) - $d_vol_cv",
            "Ref($d_vol_cv, 61) - $d_vol_cv",
            "Ref($low, 1)/($close+1e-12)",
            "Ref($volume, 1)/($volume+1e-12)",
            "Ref(($buy_volume*2-$volume)/($volume+1e-12), 11)/(($buy_volume*2-$volume)/($volume+1e-12))",
            "Ref(($buy_volume*2-$volume)/($volume+1e-12), 23)/(($buy_volume*2-$volume)/($volume+1e-12))",
            "Ref(($buy_volume*2-$volume)/($volume+1e-12), 5)/(($buy_volume*2-$volume)/($volume+1e-12))",
            "Ref((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 11)/((($buy_amount*2-$amount)*10000)/($market_cap+1e-12))",
            "Ref((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 23)/((($buy_amount*2-$amount)*10000)/($market_cap+1e-12))",
            "Ref((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 5)/((($buy_amount*2-$amount)*10000)/($market_cap+1e-12))",
            "Resi($close, 5)/($close + 1e-12)",
            "Resi($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 11)/($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)))",
            "Resi($volume, 23)/($volume + 1e-12)",
            "Resi(($buy_volume*2-$volume)/($volume+1e-12), 5)/(($buy_volume*2-$volume)/($volume+1e-12))",
            "Resi((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 5)/((($buy_amount*2-$amount)*10000)/($market_cap+1e-12))",
            "Rsquare($close, 11)/($close + 1e-12)",
            "Rsquare($close, 23)/($close + 1e-12)",
            "Rsquare($close, 5)/($close + 1e-12)",
            "Rsquare($d_hhi, 23)/($d_hhi + 1e-12)",
            "Rsquare($d_hhi, 61)/($d_hhi + 1e-12)",
            "Rsquare($d_intraday_trend, 5)/($d_intraday_trend + 1e-12)",
            "Rsquare($d_realized_vol, 5)/($d_realized_vol + 1e-12)",
            "Rsquare($d_vol, 11)/($d_vol + 1e-12)",
            "Rsquare($d_vol, 23)/($d_vol + 1e-12)",
            "Rsquare($turn, 23)/($turn + 1e-12)",
            "Rsquare($turn, 5)/($turn + 1e-12)",
            "Rsquare($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 11)",
            "Rsquare($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 23)",
            "Rsquare($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 61)",
            "Rsquare($volume, 11)/($volume + 1e-12)",
            "Rsquare($volume, 23)/($volume + 1e-12)",
            "Rsquare($volume, 5)/($volume + 1e-12)",
            "Rsquare((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 11)",
            "Rsquare((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 23)",
            "Rsquare((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 5)",
            "Slope($d_close_rank, 11)/($d_close_rank + 1e-12)",
            "Slope($d_close_rank, 23)/($d_close_rank + 1e-12)",
            "Slope($d_corr, 23)/($d_corr + 1e-12)",
            "Slope($d_corr, 61)/($d_corr + 1e-12)",
            "Slope($d_dwr, 23)/($d_dwr + 1e-12)",
            "Slope($d_dwr, 61)/($d_dwr + 1e-12)",
            "Slope($d_intraday_trend, 11)/($d_intraday_trend + 1e-12)",
            "Slope($d_intraday_trend, 5)/($d_intraday_trend + 1e-12)",
            "Slope($d_open_vwap, 61)/($d_open_vwap + 1e-12)",
            "Slope($d_realized_vol, 11)/($d_realized_vol + 1e-12)",
            "Slope($d_trades_close_corr, 23)/($d_trades_close_corr + 1e-12)",
            "Slope($d_trades_vol_corr, 61)/($d_trades_vol_corr + 1e-12)",
            "Slope($d_vol, 23)/($d_vol + 1e-12)",
            "Slope($d_vol_cv, 23)/($d_vol_cv + 1e-12)",
            "Slope($d_vol_cv, 61)/($d_vol_cv + 1e-12)",
            "Slope($d_volatility_of_volume, 61)/($d_volatility_of_volume + 1e-12)",
            "Slope($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 11)/($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)))",
            "Slope($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 23)/($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)))",
            "Slope($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 61)/($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)))",
            "Slope($volume, 11)/($volume + 1e-12)",
            "Slope($volume, 23)/($volume + 1e-12)",
            "Slope(($buy_volume*2-$volume)/($volume+1e-12), 11)/(($buy_volume*2-$volume)/($volume+1e-12))",
            "Slope(($buy_volume*2-$volume)/($volume+1e-12), 23)/(($buy_volume*2-$volume)/($volume+1e-12))",
            "Slope(($buy_volume*2-$volume)/($volume+1e-12), 5)/(($buy_volume*2-$volume)/($volume+1e-12))",
            "Slope((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 11)/((($buy_amount*2-$amount)*10000)/($market_cap+1e-12))",
            "Slope((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 23)/((($buy_amount*2-$amount)*10000)/($market_cap+1e-12))",
            "Slope((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 5)/((($buy_amount*2-$amount)*10000)/($market_cap+1e-12))",
            "Std($close, 23)/($close + 1e-12)",
            "Std($d_hhi, 23)/($d_hhi + 1e-12)",
            "Std($d_hhi, 61)/($d_hhi + 1e-12)",
            "Std($d_intraday_trend, 5)/($d_intraday_trend + 1e-12)",
            "Std($d_trades_vol_corr, 23)/($d_trades_vol_corr + 1e-12)",
            "Std($d_vol, 11)/($d_vol + 1e-12)",
            "Std($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)), 23)/($volume*10000/($num_trades+1e-12)/($market_cap/($close+1e-12)))",
            "Std($volume, 11)/($volume + 1e-12)",
            "Std($volume, 5)/($volume + 1e-12)",
            "Std((($buy_amount*2-$amount)*10000)/($market_cap+1e-12), 5)/((($buy_amount*2-$amount)*10000)/($market_cap+1e-12))",
            "Std(Abs($close/Ref($close, 1)-1)*$volume, 11)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 11)+1e-12)",
            "Std(Abs($close/Ref($close, 1)-1)*$volume, 23)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 23)+1e-12)",
            "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)",
            "Sum(Greater($close-Ref($close, 1), 0), 11)/(Sum(Abs($close-Ref($close, 1)), 11)+1e-12)",
            "Sum(Greater($close-Ref($close, 1), 0), 5)/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)",
            "Sum(Greater($volume-Ref($volume, 1), 0), 11)/(Sum(Abs($volume-Ref($volume, 1)), 11)+1e-12)",
            "Sum(Greater($volume-Ref($volume, 1), 0), 5)/(Sum(Abs($volume-Ref($volume, 1)), 5)+1e-12)",
            "Sum(Greater(Ref($close, 1)-$close, 0), 5)/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)",
            "Sum(Greater(Ref($volume, 1)-$volume, 0), 23)/(Sum(Abs($volume-Ref($volume, 1)), 23)+1e-12)",
            "Sum(Greater(Ref($volume, 1)-$volume, 0), 5)/(Sum(Abs($volume-Ref($volume, 1)), 5)+1e-12)",
            "$i_change_1_mean_5",
            "$i_change_1_std_5",
            "$i_change_1_mean_10",
            "$i_change_1_std_10",
            "$i_change_1_mean_20",
            "$i_change_1_std_20",
            "$i_change_1_mean_30",
            "$i_change_1_mean_60",
            "$i_change_1_std_60",
        ]       
        fields = col_list
        names = col_list
        
        
        return fields, names


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($adjclose, -5) / Ref($adjclose, -1) - 1"], ["Ref($adjclose, -5) / Ref($adjclose, -1) - 1"]