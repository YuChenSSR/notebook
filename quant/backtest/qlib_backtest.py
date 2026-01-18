import os
import sys
import pickle
import fire
import yaml
import warnings
from pathlib import Path
from loguru import logger
import pandas as pd
from typing import Union, TypeVar, Callable, Optional, Tuple

from dataclasses import dataclass, asdict
from dataclasses_json import DataClassJsonMixin

from qlib import init
from qlib.constant import REG_CN
from qlib.backtest import backtest, executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from qlib.contrib.strategy import TopkDropoutStrategy


_T = TypeVar("_T")

def _parse_seed_step_from_filename(filename: str) -> Tuple[int, int]:
    """
    从文件名末尾提取 seed 与 step，避免 universe/前缀包含 '_' 时 split 固定长度导致报错。

    例如：
    - master_predictions_backday_8_csi800_17_0.csv
    - master_predictions_backday_8_csi800_rank_17_0.csv
    """
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid filename: {filename}")
    seed_str, step_str = parts[-2], parts[-1]
    if not (seed_str.isdigit() and step_str.isdigit()):
        raise ValueError(f"Cannot parse seed/step from filename: {filename}")
    return int(seed_str), int(step_str)


@dataclass
class BacktestResult(DataClassJsonMixin):
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float


class QlibBacktest:
    def __init__(
        self,
        benchmark: str = "SH000906",
        top_k: int = 50,
        n_drop: Optional[int] = 5,   # None
        deal: str = "close",
        open_cost: float = 0.0015,
        close_cost: float = 0.0015,
        min_cost: float = 5,
        account: float = 100_000_000,
        limit_threshold: float = 0.095,
        time_per_step: str = "day",
    ):
        self._benchmark = benchmark
        self._top_k = top_k
        self._n_drop = n_drop if n_drop is not None else top_k
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost
        self._account = account
        self._limit_threshold = limit_threshold
        self._time_per_step = time_per_step


    def run(
        self,
        predfile_path: str
    ) -> Tuple[pd.DataFrame, BacktestResult]:
        # 读取预测值
        try:
            prediction = pd.read_csv(predfile_path)

            prediction['datetime'] = pd.to_datetime(prediction['datetime'])
            prediction = prediction.rename(columns={'score': '0'})
            prediction.set_index(['datetime', 'instrument'], inplace=True)

            logger.info(f"Pred_data_from_file: {predfile_path}")
        except Exception as e:
            logger.error(f"Pred_data_read_failed: {str(e)}")
            sys.exit(1)

        prediction = prediction.sort_index()
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        dates = index.levels[0]

        # print(prediction)
        # print(dates)

        def backtest_impl(last: int = -2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy = TopkDropoutStrategy(
                    signal=prediction,
                    topk=self._top_k,
                    n_drop=self._n_drop,
                    only_tradable=True,
                    forbid_all_trade_at_limit=True
                )
                executor = exec.SimulatorExecutor(
                    time_per_step=self._time_per_step,
                    generate_portfolio_metrics=True
                )

                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=self._account,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": self._limit_threshold,
                        "deal_price": self._deal_price,
                        "open_cost": self._open_cost,
                        "close_cost": self._close_cost,
                        "min_cost": self._min_cost,
                    }
                )[0]

        try:
            portfolio_metric = backtest_impl()
        except IndexError:
            print("Cannot backtest till the last day, trying again with one less day")
            portfolio_metric = backtest_impl(-2)

        report, _ = portfolio_metric["1day"]    # type: ignore
        result = self._analyze_report(report)

        return report, result

    def _analyze_report(self, report: pd.DataFrame) -> BacktestResult:
        excess = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
        returns = risk_analysis(report["return"] - report["cost"])["risk"]

        def loc(series: pd.Series, field: str) -> float:
            return series.loc[field]    # type: ignore

        return BacktestResult(
            sharpe=loc(returns, "information_ratio"),
            annual_return=loc(returns, "annualized_return"),
            max_drawdown=loc(returns, "max_drawdown"),
            information_ratio=loc(excess, "information_ratio"),
            annual_excess_return=loc(excess, "annualized_return"),
            excess_max_drawdown=loc(excess, "max_drawdown"),
        )

def main(
        market_name: str = "csi800",
        qlib_path: str = '/home/idc2/notebook/qlib_bin/cn_data_backtest',
        data_path: str = f"/home/idc2/notebook/zxf/data",
        is_batch: bool = False

):
    ### 1. 参数

    if is_batch:
        pred_folder_path = f'{data_path}/Backtest_Results/predictions' # 预测值目录
        output_path = f"{data_path}/Backtest_Results/results/"      # 回测结果存放目录
        os.makedirs(output_path, exist_ok=True)
    else:
        pred_folder_path = data_path
        output_path = data_path
    
    # experimental_data_path = f"{data_path}/master_results/{folder_name}"             
    # output_path = f"{experimental_data_path}/Backtest_Results/results/"      # 回测结果存放目录
    # os.makedirs(output_path, exist_ok=True)

    # 读取参数
    ### 1.读取配置文件
    with open(f"{data_path}/workflow_config_master_Alpha158_{market_name}.yaml", 'r') as f:
        config = yaml.safe_load(f)

    benchmark = config["port_analysis_config"]["backtest"]["benchmark"]
    deal = config["port_analysis_config"]["backtest"]["exchange_kwargs"]["deal_price"]
    top_k = config["port_analysis_config"]["strategy"]["kwargs"]["topk"]
    n_drop = config["port_analysis_config"]["strategy"]["kwargs"]["n_drop"]
    account = config["port_analysis_config"]["backtest"]["account"]
    open_cost = config["port_analysis_config"]["backtest"]["exchange_kwargs"]["open_cost"]
    close_cost = config["port_analysis_config"]["backtest"]["exchange_kwargs"]["close_cost"]
    min_cost = config["port_analysis_config"]["backtest"]["exchange_kwargs"]["min_cost"]
    time_per_step = config["port_analysis_config"]["backtest"]["freq"]
    limit_threshold = config["port_analysis_config"]["backtest"]["exchange_kwargs"]["limit_threshold"]


    # 读取预测值文件列表
    pred_folder_path = Path(pred_folder_path)
    pred_filename_list = [file.name for file in pred_folder_path.glob("*predictions*.csv")]


    bt_result = pd.DataFrame()
    for filename in pred_filename_list:
        seed, step = _parse_seed_step_from_filename(filename)

        print("\n\n" + "-" * 100)
        # qlib初始化
        init(provider_uri=qlib_path, region=REG_CN)

        # 预测值目录
        predfile_path = Path(pred_folder_path / filename)
        logger.info(f"Backtest： {filename} | Seed:{seed} | Step:{step}")

        # 回测
        qlib_backtest = QlibBacktest(
            benchmark=benchmark,
            top_k=top_k,
            n_drop=n_drop,
            deal=deal,
            open_cost = open_cost,
            close_cost = close_cost,
            min_cost = min_cost,
            account = account,
            limit_threshold = limit_threshold,
            time_per_step = time_per_step,
        )
        report, result = qlib_backtest.run(predfile_path=predfile_path)

        # print(result)

        ### 结果汇总
        df_result = pd.DataFrame([asdict(result)])

        df_result = df_result[[
            "annual_return", "annual_excess_return",
            "max_drawdown", "excess_max_drawdown",
            "sharpe", "information_ratio"
        ]]
        logger.info(f"Backtest_result:")
        print(df_result.to_string(max_cols=None))


        df_result['Seed'] = seed
        df_result['Step'] = step
        df_result = df_result[[
            'Seed', 'Step',
            'annual_return', 'annual_excess_return', 'max_drawdown', 'excess_max_drawdown', 'sharpe',
            'information_ratio']]
        bt_result = pd.concat([bt_result, df_result],ignore_index=True)
    bt_result.to_csv(f'{output_path}/backtest_result.csv', index=False, date_format='%Y-%m-%d')

    ### 汇总其他信息
    # 读取train info
    train_info_file = f"{data_path}/Master_results/train_metrics_results.csv"
    train_info = pd.read_csv(train_info_file)

    # 读取test info
    test_info_file = f"{data_path}/Backtest_Results/predictions/test_metrics_results.csv"
    test_info = pd.read_csv(test_info_file)

    # 结果信息汇总
    backtest_result_info = pd.merge(train_info, test_info, on=['Seed', 'Step'], how='outer')
    backtest_result_info = pd.merge(backtest_result_info, bt_result, on=['Seed', 'Step'], how='outer')
    backtest_result_info.to_csv(f'{data_path}/Backtest_Results/info_result_{market_name}.csv', index=False, date_format='%Y-%m-%d')

    print(backtest_result_info.to_string(max_cols=None))

if __name__ == "__main__":
    fire.Fire(main)