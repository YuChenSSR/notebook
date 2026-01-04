# -*- coding: utf-8 -*-
"""
在 handler基础上：
- **完整保留 rolling 家族全部特征**；
- **并追加增强特征**：turnover（$turn）、估值 pe_ttm / pb（含 EP/BM 倒数与动量修正）、Amihud 流动性冲击；
- 仅使用 Qlib 原生表达式（Ref/Mean/Std/Log/Quantile/Rank/Slope/Rsquare/Resi/Max/Min/...），不依赖 GroupMean；
- Rank(x, N) 的 N **显式指定**，避免 `Rank.__init__() missing N` 报错；
- 标签仍为未来5日收益：Ref($adjclose,-5)/Ref($adjclose,-1)-1。

说明：
- 若你的环境中原 handler 类名也是 Alpha158，本文件同名覆盖（但实现为“fullrolling + 增强特征”）。
"""

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
                assert (fit_start_time is not None and fit_end_time is not None), \
                    "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update({"fit_start_time": fit_start_time, "fit_end_time": fit_end_time})
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


# def check_transform_proc(proc_l, fit_start_time, fit_end_time):
#     new_l = []
#     for p in proc_l:
#         if not isinstance(p, Processor):
#             klass, pkwargs = get_callable_kwargs(p, processor_module)
#             args = getfullargspec(klass).args
#             if "fit_start_time" in args and "fit_end_time" in args:
#                 assert (
#                     fit_start_time is not None and fit_end_time is not None
#                 ), "Make sure `fit_start_time` and `fit_end_time` are not None."
#                 pkwargs.update(
#                     {
#                         "fit_start_time": fit_start_time,
#                         "fit_end_time": fit_end_time,
#                     }
#                 )
#             proc_config = {"class": klass.__name__, "kwargs": pkwargs}
#             if isinstance(p, dict) and "module_path" in p:
#                 proc_config["module_path"] = p["module_path"]
#             new_l.append(proc_config)
#         else:
#             new_l.append(p)
#     return new_l

_DEFAULT_LEARN_PROCESSORS = [{"class": "DropnaLabel"}, {"class": "ZScoreNorm", "kwargs": {}}]
_DEFAULT_INFER_PROCESSORS = [{"class": "ProcessInf", "kwargs": {}}, {"class": "ZScoreNorm", "kwargs": {}}, {"class": "Fillna", "kwargs": {}}]


class Alpha158(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=None,
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs
    ):
        print("Using Alpha158 (v2 full-rolling + turnover/valuation/Amihud, 无Group依赖 / Rank窗口显式).")
        if infer_processors is None:
            infer_processors = _DEFAULT_INFER_PROCESSORS
        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

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

    def get_label_config(self):
        # 与你之前的 5D 预测保持一致
        return ["Ref($adjclose, -5)/Ref($adjclose, -1) - 1"], ["LABEL0"]

    def get_feature_config(self):
        """配置入口：rolling 全保留 + 增强特征"""
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {
                "windows": [5, 10, 20, 30, 60],
                "include": None,   # None => 全部 rolling 保留
                "exclude": [],
            },
            "enhance": True,       # 开启增强特征
        }
        return self.parse_config_to_fields(conf)

    @staticmethod
    def parse_config_to_fields(config):
        fields, names = [], []

        # ---- kbar ----
        if "kbar" in config:
            fields += [
                "($close-$open)/$open",
                "($high-$low)/$open",
                "($close-$open)/($high-$low+1e-12)",
                "($high-Greater($open, $close))/$open",
                "($high-Greater($open, $close))/($high-$low+1e-12)",
                "(Less($open, $close)-$low)/$open",
                "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "(2*$close-$high-$low)/$open",
                "(2*$close-$high-$low)/($high-$low+1e-12)",
            ]
            names += fields[-9:]

        # ---- price ----
        if "price" in config:
            windows = config["price"].get("windows", [0])
            feature = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for field in feature:
                f = field.lower()
                for d in windows:
                    expr = (f"Ref($%s, %d)/$close" % (f, d)) if d != 0 else (f"$%s/$close" % f)
                    fields.append(expr); names.append(expr)

        # ---- rolling (Full) ----
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = set(config["rolling"].get("exclude", []))

            def use(op):
                return (op not in exclude) and (include is None or op in include)

            # 基础价因子
            if use("ROC"):
                fields += [f"Ref($close, {d})/$close" for d in windows]; names += fields[-len(windows):]
            if use("MA"):
                fields += [f"Mean($close, {d})/$close" for d in windows]; names += fields[-len(windows):]
            if use("STD"):
                fields += [f"Std($close, {d})/$close" for d in windows];  names += fields[-len(windows):]

            # 回归族
            if use("BETA"):
                fields += [f"Slope($close, {d})/$close" for d in windows]; names += fields[-len(windows):]
            if use("RSQR"):
                fields += [f"Rsquare($close, {d})" for d in windows];     names += fields[-len(windows):]
            if use("RESI"):
                fields += [f"Resi($close, {d})/$close" for d in windows];  names += fields[-len(windows):]

            # 通道/分位（中长窗）
            long_w = [d for d in windows if d >= 20]
            if use("MAX") and long_w:
                fields += [f"Max($high, {d})/$close" for d in long_w]; names += fields[-len(long_w):]
            if use("LOW") and long_w:
                fields += [f"Min($low, {d})/$close" for d in long_w];  names += fields[-len(long_w):]
            if use("QTLU") and long_w:
                fields += [f"Quantile($close, {d}, 0.8)/$close" for d in long_w]; names += fields[-len(long_w):]
            if use("QTLD") and long_w:
                fields += [f"Quantile($close, {d}, 0.2)/$close" for d in long_w]; names += fields[-len(long_w):]

            # Rank（显式 N）
            if use("RANK"):
                fields += [f"Rank($close, {d})" for d in windows]; names += fields[-len(windows):]

            # RSV
            if use("RSV"):
                for d in windows:
                    if d in (10, 14, 20, 30):
                        expr = f"($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)"
                        fields.append(expr); names.append(expr)

            # Aroon
            if use("IMAX"):
                fields += [f"IdxMax($high, {d})/{d}" for d in windows]; names += fields[-len(windows):]
            if use("IMIN"):
                fields += [f"IdxMin($low, {d})/{d}" for d in windows];  names += fields[-len(windows):]
            if use("IMXD"):
                fields += [f"(IdxMax($high, {d})-IdxMin($low, {d}))/{d}" for d in windows]; names += fields[-len(windows):]

            # 价量相关性
            if use("CORR"):
                fields += [f"Corr($close, Log($volume+1), {d})" for d in windows]; names += fields[-len(windows):]
            if use("CORD"):
                fields += [f"Corr($close/Ref($close,1), Log($volume/Ref($volume,1)+1), {d})" for d in windows]; names += fields[-len(windows):]

            # 上涨/下跌计数与差
            if use("CNTP"):
                fields += [f"Mean($close>Ref($close, 1), {d})" for d in windows]; names += fields[-len(windows):]
            if use("CNTN"):
                fields += [f"Mean($close<Ref($close, 1), {d})" for d in windows]; names += fields[-len(windows):]
            if use("CNTD"):
                fields += [f"Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})" for d in windows]; names += fields[-len(windows):]

            # RSI 家族
            if use("SUMP"):
                fields += [
                    f"Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
                    for d in windows
                ]; names += fields[-len(windows):]
            if use("SUMN"):
                fields += [
                    f"Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
                    for d in windows
                ]; names += fields[-len(windows):]
            if use("SUMD"):
                fields += [
                    f"(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))"
                    f"/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
                    for d in windows
                ]; names += fields[-len(windows):]

            # 成交量滚动族
            if use("VMA"):
                fields += [f"Mean($volume, {d})/($volume+1e-12)" for d in windows]; names += fields[-len(windows):]
            if use("VSTD"):
                fields += [f"Std($volume, {d})/($volume+1e-12)" for d in windows];  names += fields[-len(windows):]
            if use("WVMA"):
                fields += [
                    f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)"
                    for d in windows
                ]; names += fields[-len(windows):]
            if use("VSUMP"):
                fields += [
                    f"Sum(Greater($volume-Ref($volume, 1), 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
                    for d in windows
                ]; names += fields[-len(windows):]
            if use("VSUMN"):
                fields += [
                    f"Sum(Greater(Ref($volume, 1)-$volume, 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
                    for d in windows
                ]; names += fields[-len(windows):]
            if use("VSUMD"):
                fields += [
                    f"(Sum(Greater($volume-Ref($volume, 1), 0), {d})-Sum(Greater(Ref($volume, 1)-$volume, 0), {d}))"
                    f"/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
                    for d in windows
                ]; names += fields[-len(windows):]

        # ---- 增强特征 ----
        if config.get("enhance", False):
            # 1) 换手率（$turn）深化
            TO = "$turn"
            to_windows = [5, 10, 20, 30, 60]
            to_bases   = [20, 60]
            for w in to_windows:
                fields.append(f"Mean({TO}, {w})"); names.append(f"TO_MEAN_{w}")
                fields.append(f"Std({TO}, {w})");  names.append(f"TO_STD_{w}")
            for b in to_bases:
                for w in to_windows:
                    fields.append(f"(Mean({TO}, {w}))/(Mean({TO}, {b})+1e-12)"); names.append(f"TO_REL_{w}_OVER_{b}")
                    fields.append(f"(Mean({TO}, {w})-Mean({TO}, {b}))/(Std({TO}, {b})+1e-12)"); names.append(f"TO_ZSCORE_{w}_OVER_{b}")
                fields.append(f"{TO}/(Mean({TO}, {b})+1e-12)-1"); names.append(f"TO_CHG_RATE_OVER_{b}")
            fields.append(f"{TO}/(Ref({TO},1)+1e-12) - 1"); names.append("TO_DELTA_1D")
            fields.append(f"Mean({TO}/(Ref({TO},1)+1e-12) - 1, 5)"); names.append("TO_DELTA_MEAN_5D")
            fields.append("Mean($close/Ref($close,1)-1,5) * (1 - Rank($turn,20))"); names.append("LOCK_MOM_5")
            fields.append("($turn/(Ref($turn,1)+1e-12) - 1) * Mean($close/Ref($close,1)-1,5)"); names.append("DELTA_TO_MOM_5")

            # 2) 估值 × 动量（pe_ttm / pb 与倒数 EP/BM）
            def add_val_log(raw, tag):
                expr = f"Log({raw}+1e-12)"
                fields.append(f"Mean({expr},20)"); names.append(f"LOG_{tag}_MEAN_20")
                fields.append(f"Mean({expr},60)"); names.append(f"LOG_{tag}_MEAN_60")
                fields.append(f"(Mean({expr},20))/(Mean({expr},60)+1e-12)"); names.append(f"LOG_{tag}_REL_20_60")
                fields.append(f"(Mean({expr},20)-Mean({expr},60))/(Std({expr},60)+1e-12)"); names.append(f"LOG_{tag}_ZSCORE")
                fields.append(f"{expr}/(Mean({expr},60)+1e-12)-1"); names.append(f"LOG_{tag}_CHG_REL_60")
            add_val_log("$pe_ttm", "PE_TTM")
            add_val_log("$pb", "PB")

            def add_inv(raw, tag):
                expr = f"1.0/({raw}+1e-12)"
                fields.append(f"Mean({expr},20)"); names.append(f"{tag}_MEAN_20")
                fields.append(f"Mean({expr},60)"); names.append(f"{tag}_MEAN_60")
                fields.append(f"Std({expr},20)");  names.append(f"{tag}_STD_20")
                fields.append(f"Std({expr},60)");  names.append(f"{tag}_STD_60")
                fields.append(f"(Mean({expr},20))/(Mean({expr},60)+1e-12)"); names.append(f"{tag}_REL_20_60")
                fields.append(f"(Mean({expr},20)-Mean({expr},60))/(Std({expr},60)+1e-12)"); names.append(f"{tag}_ZSCORE")
                fields.append(f"{expr}/(Mean({expr},60)+1e-12)-1"); names.append(f"{tag}_CHG_REL_60")
            add_inv("$pe_ttm", "EP")
            add_inv("$pb", "BM")

            fields.append("(1/($pe_ttm+1e-12)) / (Mean(1/($pe_ttm+1e-12),60)+1e-12)"); names.append("EP_MKT_REL_60")
            fields.append("(Log($pe_ttm+1e-12) - Mean(Log($pe_ttm+1e-12),60)) * Min(Mean($close/Ref($close,1)-1,5),0)"); names.append("EXPN_MOM_RISK_MKT")

            # 3) 流动性冲击（Amihud proxy）
            ret_abs_amt_5  = "Mean(Abs($close/Ref($close,1)-1)/($amount + 1e-12), 5)"
            ret_abs_amt_20 = "Mean(Abs($close/Ref($close,1)-1)/($amount + 1e-12), 20)"
            fields.append(ret_abs_amt_5);  names.append("AMIH_5")
            fields.append(ret_abs_amt_20); names.append("AMIH_20")
            fields.append(f"Rank({ret_abs_amt_5}, 20)");  names.append("IMPACT_PENALTY_5")
            fields.append(f"Rank({ret_abs_amt_20}, 60)"); names.append("IMPACT_PENALTY_20")

        return fields, names


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($adjclose, -5)/Ref($adjclose, -1) - 1"], ["LABEL0"]
