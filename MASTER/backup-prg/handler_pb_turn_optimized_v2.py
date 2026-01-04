# -*- coding: utf-8 -*-
# Alpha158 handler (A股增强): Turnover + Valuation + 三大关键指标(2/3/4)
# - 2) 换手×动量×行业相对：LOCK_MOM_20 / DELTA_TO_MOM_5 / TO_IND_Q
# - 3) 估值×动量：EP_IND / EXPN_MOM_RISK
# - 4) 流动性与冲击成本：AMIH_20 / IMPACT_PENALTY
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

        print("Using Alpha158 (A股增强) with turnover/valuation + 2/3/4关键指标.")

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
        # 5日前瞻收益（相对昨日收盘）
        return ["Ref($adjclose, -5)/Ref($adjclose, -1) - 1"], ["LABEL0"]

    def get_feature_config(self):
        conf = {
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            # Turnover基座（供 2) 模块使用）
            "turnover": {
                "use_float_a": True,
                "windows": [5, 10, 20],
                "base_windows": [20, 60],
                "use_industry_rel": True,
                "industry_field": "sw_l1",
            },
            # Valuation基座（供 3) 模块使用）
            "valuation": {
                "fields": {"pe": "pe_ttm", "pb": "pb"},
                "log_transform": True,
                "windows": [5, 20],
                "base_windows": [60, 120],
                "use_inverse": True,
                "use_industry_rel": True,
                "industry_field": "sw_l1",
            },
            # 2/3/4 三大关键指标开关（默认全开）
            "core_2_3_4": {
                "enable": True,
                "industry_field": "sw_l1"
            },
            # 保留部分经典滚动技术特征（精简）
            "rolling": {"windows": [5, 10, 20, 60], "include": ["MA", "STD", "ROC"]},
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
            win = cfg.get("windows", [5, 10, 20])
            base = cfg.get("base_windows", [20, 60])
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
            add(f"{to_expr}/(GroupMean('{ind_field}', Mean({to_expr}, 20)) + 1e-12)", f"{to_name}_IND_REL_over_20d")

        # --- Valuation base (PE/PB + EP/BM) ---
        if "valuation" in config:
            cfg = config["valuation"]
            pef = cfg.get("fields", {}).get("pe", "pe_ttm")
            pbf = cfg.get("fields", {}).get("pb", "pb")
            log_tf = cfg.get("log_transform", True)
            win_v  = cfg.get("windows", [5, 20])
            base_v = cfg.get("base_windows", [60, 120])
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

        # --- 2/3/4 三大关键指标 ---
        if config.get("core_2_3_4", {}).get("enable", True):
            ind = config.get("core_2_3_4", {}).get("industry_field", ind_field)

            # 2) 换手×动量×行业相对（需要 to_expr）
            if to_expr is None:
                # 若未启用 turnover，使用体量proxy以保证表达式可用
                to_expr = "($volume)/(Mean($volume, 60)+1e-12)"
            # 低换手动量（锁仓动量）
            add(f"Mean($close/Ref($close,1)-1,20) * (1 - Rank({to_expr}))", "LOCK_MOM_20")
            # Δ换手动量风险
            add(f"({to_expr}/(Ref({to_expr},1)+1e-12) - 1) * Mean($close/Ref($close,1)-1,5)", "DELTA_TO_MOM_5")
            # 行业相对低换手
            add(f"{to_expr} / (GroupMean('{ind}', Mean({to_expr},20)) + 1e-12)", "TO_IND_Q")

            # 3) 估值×动量（使用 pe_ttm）
            # 行业相对EP
            add(f"(1/($pe_ttm+1e-12)) / (GroupMean('{ind}', Mean(1/($pe_ttm+1e-12),60)) + 1e-12)", "EP_IND")
            # 估值防踩踏（高估 + 负动量惩罚）
            add(f"(Log($pe_ttm+1e-12) - GroupMean('{ind}', Mean(Log($pe_ttm+1e-12),60))) * Min(Mean($close/Ref($close,1)-1,5),0)", "EXPN_MOM_RISK")

            # 4) 流动性与冲击成本 proxy
            add("Mean(Abs($close/Ref($close,1)-1)/($amount + 1e-12), 20)", "AMIH_20")
            add("Rank(Mean(Abs($close/Ref($close,1)-1)/($amount + 1e-12), 20))", "IMPACT_PENALTY")

        # --- Rolling (精简) ---
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 60])
            include = config["rolling"].get("include", ["MA", "STD", "ROC"])

            def use(x): return (include is None) or (x in include)

            if use("ROC"):
                for d in windows:
                    add(f"Ref($close, {d})/$close", f"Ref($close, {d})/$close")
            if use("MA"):
                for d in windows:
                    add(f"Mean($close, {d})/$close", f"Mean($close, {d})/$close")
            if use("STD"):
                for d in windows:
                    add(f"Std($close, {d})/$close", f"Std($close, {d})/$close")

        # 去重，保持顺序
        seen = set(); f2, n2 = [], []
        for f, n in zip(fields, names):
            if f not in seen:
                seen.add(f); f2.append(f); n2.append(n)
        return f2, n2


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($adjclose, -5)/Ref($adjclose, -1) - 1"], ["LABEL0"]
