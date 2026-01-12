"""
FuturesAlpha158

基于 Qlib 的 Alpha158 思路（OHLCV + rolling 技术指标），并追加期货特有字段：
- open_interest（持仓量）相关 rolling 特征
- roll_flag / roll_count（由 dominant_id 在 1m 聚合时派生）相关 rolling 特征

依赖字段（需在 qlib bin 中存在）：
open, high, low, close, vwap, volume, amount, open_interest, adjclose, roll_flag, roll_count
"""

from __future__ import annotations

from inspect import getfullargspec
from typing import List, Sequence, Tuple

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.data.dataset import processor as processor_module
from qlib.utils import get_callable_kwargs


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert fit_start_time is not None and fit_end_time is not None, (
                    "Make sure `fit_start_time` and `fit_end_time` are not None."
                )
                pkwargs.update({"fit_start_time": fit_start_time, "fit_end_time": fit_end_time})
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


_DEFAULT_LEARN_PROCESSORS = [{"class": "DropnaLabel"}]
_DEFAULT_INFER_PROCESSORS = [{"class": "ProcessInf"}, {"class": "Fillna", "kwargs": {"fields_group": "feature"}}]


class FuturesAlpha158(DataHandlerLP):
    """
    A futures-oriented Alpha158-like handler.

    - Feature group: technical indicators from OHLCV + futures-specific OI/roll features.
    - Label: default 5D return on adjclose.
    """

    def __init__(
        self,
        instruments="all",
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
        **kwargs,
    ):
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
            **kwargs,
        )

    def get_label_config(self):
        # Default: future 5D return (same as stock baseline)
        return ["Ref($adjclose, -5) / Ref($adjclose, -1) - 1"], ["LABEL0"]

    def get_feature_config(self) -> Tuple[List[str], List[str]]:
        windows = [5, 10, 20, 30, 60]
        fields: List[str] = []
        names: List[str] = []

        def add(expr: str, name: str):
            fields.append(expr)
            names.append(name)

        # ---- kbar (9) ----
        add("($close-$open)/$open", "KBAR_RET")
        add("($high-$low)/$open", "KBAR_RANGE")
        add("($close-$open)/($high-$low+1e-12)", "KBAR_BODY_RATIO")
        add("($high-Greater($open, $close))/$open", "KBAR_UPPER_SHADOW")
        add("($high-Greater($open, $close))/($high-$low+1e-12)", "KBAR_UPPER_SHADOW_RATIO")
        add("(Less($open, $close)-$low)/$open", "KBAR_LOWER_SHADOW")
        add("(Less($open, $close)-$low)/($high-$low+1e-12)", "KBAR_LOWER_SHADOW_RATIO")
        add("(2*$close-$high-$low)/$open", "KBAR_CLV")
        add("(2*$close-$high-$low)/($high-$low+1e-12)", "KBAR_CLV_RATIO")

        # ---- price snapshot (relative to close) ----
        add("$open/$close", "OPEN")
        add("$high/$close", "HIGH")
        add("$low/$close", "LOW")
        add("$vwap/$close", "VWAP")

        # ---- volume / amount snapshot (log-scale) ----
        add("Log($volume+1)", "LOG_VOLUME")
        add("Log($amount+1)", "LOG_AMOUNT")

        # ---- rolling: close-based tech indicators ----
        for d in windows:
            add(f"Ref($close, {d})/$close", f"ROC_{d}")
            add(f"Mean($close, {d})/$close", f"MA_{d}")
            add(f"Std($close, {d})/$close", f"STD_{d}")
            add(f"Slope($close, {d})/$close", f"BETA_{d}")
            add(f"Rsquare($close, {d})", f"RSQR_{d}")
            add(f"Resi($close, {d})/$close", f"RESI_{d}")
            add(f"Rank($close, {d})", f"RANK_{d}")
            add(f"Corr($close, Log($volume+1), {d})", f"CORR_{d}")
            add(f"Corr($close/Ref($close,1), Log($volume/Ref($volume,1)+1), {d})", f"CORD_{d}")
            add(f"Mean($close>Ref($close, 1), {d})", f"CNTP_{d}")
            add(f"Mean($close<Ref($close, 1), {d})", f"CNTN_{d}")
            add(
                f"Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})",
                f"CNTD_{d}",
            )
            add(
                f"Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)",
                f"SUMP_{d}",
            )
            add(
                f"Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)",
                f"SUMN_{d}",
            )
            add(
                f"(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))"
                f"/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)",
                f"SUMD_{d}",
            )
            add(f"Mean($volume, {d})/($volume+1e-12)", f"VMA_{d}")
            add(f"Std($volume, {d})/($volume+1e-12)", f"VSTD_{d}")

        # ---- channel/quantile family (long windows) ----
        for d in [w for w in windows if w >= 20]:
            add(f"Max($high, {d})/$close", f"MAX_{d}")
            add(f"Min($low, {d})/$close", f"MIN_{d}")
            add(f"Quantile($close, {d}, 0.8)/$close", f"QTLU_{d}")
            add(f"Quantile($close, {d}, 0.2)/$close", f"QTLD_{d}")
            add(f"IdxMax($high, {d})/{d}", f"IMAX_{d}")
            add(f"IdxMin($low, {d})/{d}", f"IMIN_{d}")
            add(f"(IdxMax($high, {d})-IdxMin($low, {d}))/{d}", f"IMXD_{d}")

        # RSV (selected windows)
        for d in [10, 20, 30]:
            add(
                f"($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)",
                f"RSV_{d}",
            )

        # ---- futures-specific: open_interest ----
        add("Log($open_interest+1)", "OI_LOG")
        add("$open_interest/(Ref($open_interest, 1)+1e-12)-1", "OI_CHG1")
        for d in windows:
            add(f"Mean(Log($open_interest+1), {d})", f"OI_LOG_MA_{d}")
            add(f"Std(Log($open_interest+1), {d})", f"OI_LOG_STD_{d}")
            add(
                f"Corr($close/Ref($close,1), Log($open_interest/Ref($open_interest,1)+1), {d})",
                f"OI_CORD_{d}",
            )

        # ---- futures-specific: roll features ----
        # roll_flag: 0/1; roll_count: integer per day
        add("$roll_flag", "ROLL_FLAG")
        add("$roll_count", "ROLL_COUNT")
        for d in windows:
            add(f"Mean($roll_flag, {d})", f"ROLL_FLAG_MEAN_{d}")
            add(f"Sum($roll_flag, {d})", f"ROLL_FLAG_SUM_{d}")
            add(f"Mean($roll_count, {d})", f"ROLL_COUNT_MEAN_{d}")
            add(f"Max($roll_count, {d})", f"ROLL_COUNT_MAX_{d}")

        return fields, names


class FuturesAlpha158WithGate(FuturesAlpha158):
    """
    在 FuturesAlpha158 的基础上，额外追加一组 gate 特征（放在 feature 列表末尾），用于 MASTER 的 feature gate：
    - MASTER 的输入被切成 [两段：
      - 前 d_feat 维：src features（用于 encoder）
      - 后 d_gate 维：gate input（仅取最后一个 timestep，用于生成对 d_feat 的 softmax 权重）

    注意：
    - gate 特征可以与已有特征表达式重复，但 names 必须唯一（否则列名冲突）。
    - 默认追加 8 个 gate 特征，避免 d_gate=0 导致 Gate(in_features=0) 非法。
    """

    def __init__(self, *args, gate_fields: Sequence[str] | None = None, gate_names: Sequence[str] | None = None, **kwargs):
        self._gate_fields = list(gate_fields) if gate_fields is not None else [
            "Log($volume+1)",
            "Log($amount+1)",
            "Log($open_interest+1)",
            "$open_interest/(Ref($open_interest, 1)+1e-12)-1",
            "$roll_flag",
            "$roll_count",
            "$close/Ref($close, 1)-1",
            "Std($close/Ref($close,1)-1, 5)",
        ]
        self._gate_names = list(gate_names) if gate_names is not None else [
            "GATE_LOG_VOLUME",
            "GATE_LOG_AMOUNT",
            "GATE_OI_LOG",
            "GATE_OI_CHG1",
            "GATE_ROLL_FLAG",
            "GATE_ROLL_COUNT",
            "GATE_RET1",
            "GATE_RET_STD5",
        ]
        if len(self._gate_fields) != len(self._gate_names):
            raise ValueError("gate_fields 与 gate_names 长度必须一致")
        if len(self._gate_fields) == 0:
            raise ValueError("gate_features 不能为空（否则 MASTER gate 输入维度为 0）")
        super().__init__(*args, **kwargs)

    def get_feature_config(self) -> Tuple[List[str], List[str]]:
        fields, names = super().get_feature_config()
        fields = list(fields)
        names = list(names)
        fields.extend(self._gate_fields)
        names.extend(self._gate_names)
        return fields, names

