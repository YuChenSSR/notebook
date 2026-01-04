# -*- coding: utf-8 -*-
# Alpha158 handler (A股 5D 短周期优化&精简版)
# - 根据“保留 / 可选 / 不必保留”清单以及 5D 预测窗口，重构 rolling 特征集合：
#   * 保留：BETA, RSQR, RESI, MAX/LOW, QTLU/QTLD, RSV, CNTP/CNTN/CNTD, SUMP/SUMN/SUMD, VMA/VSTD
#   * 可选：RANK(20), CORR or CORD(10), WVMA(10) —— 默认关闭，可一键开启
#   * 不必保留：IMAX/IMIN/IMXD、过长窗口(>=60)等
# - 其它核心模块（turnover×动量、估值×动量、流动性冲击）按 v2 逻辑保留，并按 5D 窗口收敛
# Licensed under the MIT License.

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
    {"class": "ZScoreNorm", "kwargs": {}},
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

        print("Using Alpha158 (A股 5D 优化&精简版 rolling)")

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
        # 5日超前收益
        return ["Ref($adjclose, -5)/Ref($adjclose, -1) - 1"], ["LABEL0"]

    def get_feature_config(self):
        conf = {
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            # Turnover（贴合 5D）
            "turnover": {
                "use_float_a": False,       #是否使用流通股本计算绝对换手率；若数据集中没有 float_a，请设 False
                "windows": [5, 10],         # 短窗
                "base_windows": [20],       # 行业相对基准
                "use_industry_rel": False,   # 若无行业分组可改 False
                "industry_field": "sw_l1",  # 你的行业字段名（示例：申万一级）
            },
            # Valuation（慢变量，用于稳定与行业对比）
            "valuation": {
                "fields": {"pe": "pe_ttm", "pb": "pb"},
                "log_transform": True,
                "windows": [20],
                "base_windows": [60],
                "use_inverse": False,
                "use_industry_rel": False,   # 若无行业分组可改 False
                "industry_field": "sw_l1",  # 你的行业字段名（示例：申万一级）
            },
            # 核心 2/3/4 指标：换手×动量、估值×动量、流动性冲击（保留）
            "core_2_3_4": {"enable": True, "industry_field": "sw_l1"},
            # Rolling：严格收敛到 5/10/20，并按“保留/可选”清单设置
            "rolling": {
                "windows": [5, 10, 20],
                # 保留集合
                "include": [
                    "MA", "STD", "ROC",
                    "BETA", "RSQR", "RESI",
                    "MAX", "LOW", "QTLU", "QTLD", "RSV",
                    "CNTP", "CNTN", "CNTD",
                    "SUMP", "SUMN", "SUMD",
                    "VMA", "VSTD"
                ],
                # 可选（默认关闭）：想开就把它们加入 include
                "optional": [
                    "RANK",   # 建议只用 d=20
                    "CORR",   # 建议只用 d=10（与 CORD 二选一）
                    "CORD",   # 建议只用 d=10（与 CORR 二选一）
                    "WVMA"    # 建议只用 d=10
                ]
            },
            # 明确添加 3/5/10 短周期动量与波动
            "short_blocks": {"enable": True}
        }
        return self.parse_config_to_fields(conf)

    @staticmethod
    def parse_config_to_fields(config):
        fields, names = [], []

        def add(expr, name):
            fields.append(expr); names.append(name)

        # --- Price snapshots ---
        if "price" in config:
            windows = config["price"].get("windows", [0])
            feature = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for f in feature:
                f_ = f.lower()
                for d in windows:
                    expr = f"Ref(${f_}, {d})/$close" if d != 0 else f"${f_}/$close"
                    add(expr, expr)

        # --- Turnover base ---
        to_expr = None
        ind_field = config.get("turnover", {}).get("industry_field", "sw_l1")
        if "turnover" in config:
            cfg = config["turnover"]
            use_float_a = cfg.get("use_float_a", True)
            win = cfg.get("windows", [5, 10])
            base = cfg.get("base_windows", [20])
            if use_float_a:
                to_expr = "($volume)/(float_a+1e-12)"
                to_name = "TO"
            else:
                to_expr = "($volume)/(Mean($volume, 60)+1e-12)"
                to_name = "TO_PROXY"
            add(to_expr, to_name)
            for w in win:
                add(f"Mean({to_expr}, {w})", f"{to_name}_MEAN_{w}d")
                add(f"Std({to_expr}, {w})",  f"{to_name}_STD_{w}d")
            for b in base:
                for w in win:
                    add(f"(Mean({to_expr}, {w}))/(Mean({to_expr}, {b})+1e-12)", f"{to_name}_REL_{w}d_over_{b}d")
                    add(f"(Mean({to_expr}, {w})-Mean({to_expr}, {b}))/(Std({to_expr}, {b})+1e-12)", f"{to_name}_ZSCORE_{w}d_over_{b}d")
                add(f"({to_expr})/(Mean({to_expr}, {b})+1e-12) - 1", f"{to_name}_CHG_RATE_over_{b}d")
            add(f"{to_expr}/(Ref({to_expr},1)+1e-12) - 1", f"{to_name}_DELTA_1d")
            add(f"Mean({to_expr}/(Ref({to_expr},1)+1e-12) - 1, 5)", f"{to_name}_DELTA_MEAN_5d")
            add(f"{to_expr}/(GroupMean('{ind_field}', Mean({to_expr}, 20)) + 1e-12)", f"{to_name}_IND_REL_over_20d")

        # --- Valuation base ---
        if "valuation" in config:
            cfg = config["valuation"]
            pef = cfg.get("fields", {}).get("pe", "pe_ttm")
            pbf = cfg.get("fields", {}).get("pb", "pb")
            log_tf = cfg.get("log_transform", True)
            win_v  = cfg.get("windows", [20])
            base_v = cfg.get("base_windows", [60])
            use_inv = cfg.get("use_inverse", True)
            ind_val = cfg.get("industry_field", "sw_l1")
            def add_val(raw, base_name):
                expr = f"Log({raw}+1e-12)" if log_tf else raw
                nm   = f"LOG_{base_name}" if log_tf else base_name
                add(expr, nm)
                for w in win_v:
                    add(f"Mean({expr}, {w})", f"{nm}_MEAN_{w}d")
                    add(f"Std({expr}, {w})",  f"{nm}_STD_{w}d")
                for b in base_v:
                    for w in win_v:
                        add(f"(Mean({expr}, {w}))/(Mean({expr}, {b})+1e-12)", f"{nm}_REL_{w}d_over_{b}d")
                        add(f"(Mean({expr}, {w})-Mean({expr}, {b}))/(Std({expr}, {b})+1e-12)", f"{nm}_ZSCORE_{w}d_over_{b}d")
                    add(f"({expr})/(Mean({expr}, {b})+1e-12) - 1", f"{nm}_CHG_RATE_over_{b}d")
                add(f"{expr}/(GroupMean('{ind_val}', Mean({expr}, 60)) + 1e-12)", f"{nm}_IND_REL_over_60d")
            add_val(f"${pef}", "PE_TTM")
            add_val(f"${pbf}", "PB")
            if use_inv:
                add_val(f"1.0/(${pef}+1e-12)", "EP")
                add_val(f"1.0/(${pbf}+1e-12)", "BM")

        # --- 核心 2/3/4 ---
        if config.get("core_2_3_4", {}).get("enable", True):
            ind = config.get("core_2_3_4", {}).get("industry_field", ind_field or "sw_l1")
            if to_expr is None:
                to_expr = "($volume)/(Mean($volume, 60)+1e-12)"
            # 2) 换手×动量×行业相对
            add(f"Mean($close/Ref($close,1)-1,5) * (1 - Rank({to_expr}))", "LOCK_MOM_5")
            add(f"({to_expr}/(Ref({to_expr},1)+1e-12) - 1) * Mean($close/Ref($close,1)-1,5)", "DELTA_TO_MOM_5")
            add(f"{to_expr} / (GroupMean('{ind}', Mean({to_expr},20)) + 1e-12)", "TO_IND_Q")
            # 3) 估值×动量
            add(f"(1/($pe_ttm+1e-12)) / (GroupMean('{ind}', Mean(1/($pe_ttm+1e-12),60)) + 1e-12)", "EP_IND")
            add(f"(Log($pe_ttm+1e-12) - GroupMean('{ind}', Mean(Log($pe_ttm+1e-12),60))) * Min(Mean($close/Ref($close,1)-1,5),0)", "EXPN_MOM_RISK")
            # 4) 流动性冲击
            add("Mean(Abs($close/Ref($close,1)-1)/($amount + 1e-12), 5)",  "AMIH_5")
            add("Mean(Abs($close/Ref($close,1)-1)/($amount + 1e-12), 20)", "AMIH_20")
            add("Rank(Mean(Abs($close/Ref($close,1)-1)/($amount + 1e-12), 5))",  "IMPACT_PENALTY_5")
            add("Rank(Mean(Abs($close/Ref($close,1)-1)/($amount + 1e-12), 20))", "IMPACT_PENALTY_20")

        # --- 短周期专属块 ---
        if config.get("short_blocks", {}).get("enable", True):
            add("Mean($close/Ref($close,1)-1,3)",  "MOM_3")
            add("Mean($close/Ref($close,1)-1,5)",  "MOM_5")
            add("Mean($close/Ref($close,1)-1,10)", "MOM_10")
            add("Std($close/Ref($close,1)-1,5)",   "VOL_5")
            add("Std($close/Ref($close,1)-1,10)",  "VOL_10")

        # --- Rolling（保留清单 + 可选；窗口 5/10/20） ---
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20])
            include = list(config["rolling"].get("include", []))
            include += list(config["rolling"].get("optional", []))

            def use(x): return (include is None) or (x in include)

            for d in windows:
                if use("ROC"):   add(f"Ref($close, {d})/$close", f"Ref($close, {d})/$close")
                if use("MA"):    add(f"Mean($close, {d})/$close", f"Mean($close, {d})/$close")
                if use("STD"):   add(f"Std($close, {d})/$close",  f"Std($close, {d})/$close")

                if use("BETA"):  add(f"Slope($close, {d})/$close", f"Slope($close, {d})/$close")
                if use("RSQR"):  add(f"Rsquare($close, {d})",      f"Rsquare($close, {d})")
                if use("RESI"):  add(f"Resi($close, {d})/$close",  f"Resi($close, {d})/$close")

                if use("MAX") and d==20: add(f"Max($high, {d})/$close", f"Max($high, {d})/$close")
                if use("LOW") and d==20: add(f"Min($low, {d})/$close",   f"Min($low, {d})/$close")

                if use("QTLU") and d==20: add(f"Quantile($close, {d}, 0.8)/$close", f"Quantile($close, {d}, 0.8)/$close")
                if use("QTLD") and d==20: add(f"Quantile($close, {d}, 0.2)/$close", f"Quantile($close, {d}, 0.2)/$close")

                if use("RSV") and d in (10,14,20):
                    # d=14 常见，若 windows 不含 14 则用 10
                    if d==14:
                        add(f"($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)", f"RSV_{d}")
                    elif d==10:
                        add(f"($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)", f"RSV_{d}")

                if use("CNTP"): add(f"Mean($close>Ref($close, 1), {d})", f"Mean($close>Ref($close, 1), {d})")
                if use("CNTN"): add(f"Mean($close<Ref($close, 1), {d})", f"Mean($close<Ref($close, 1), {d})")
                if use("CNTD"): add(f"Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})", f"Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})")

                if use("SUMP"):
                    add(f"Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)", 
                        f"Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)")
                if use("SUMN"):
                    add(f"Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)", 
                        f"Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)")
                if use("SUMD"):
                    add(f"(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)", 
                        f"(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)")

                if use("VMA"):  add(f"Mean($volume, {d})/($volume+1e-12)", f"Mean($volume, {d})/($volume+1e-12)")
                if use("VSTD"): add(f"Std($volume, {d})/($volume+1e-12)",  f"Std($volume, {d})/($volume+1e-12)")

                # 可选增强：只在推荐窗口生成，避免冗余
                if use("RANK") and d==20:  add(f"Rank($close, {d})", f"Rank($close, {d})")
                if use("CORR") and d==10:  add(f"Corr($close, Log($volume+1), {d})", f"Corr($close, Log($volume+1), {d})")
                if use("CORD") and d==10:  add(f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})", f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})")
                if use("WVMA") and d==10:
                    add(f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)", 
                        f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)")

        # 去重（按表达式去重，保持顺序）
        seen = set(); f2, n2 = [], []
        for f, n in zip(fields, names):
            if f not in seen:
                seen.add(f); f2.append(f); n2.append(n)
        return f2, n2


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($adjclose, -5)/Ref($adjclose, -1) - 1"], ["LABEL0"]
