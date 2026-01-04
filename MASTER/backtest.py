import os
import sys
import pickle
import warnings
from pathlib import Path
from loguru import logger
import pandas as pd
from typing import Union, TypeVar, Callable, Optional, Tuple

from dataclasses import dataclass, asdict
from dataclasses_json import DataClassJsonMixin

# QLIB_DIRNAME = '/home/idc2/notebook/qlib'
# sys.path.insert(0, QLIB_DIRNAME)

from qlib import init
from qlib.constant import REG_CN
from qlib.backtest import backtest, executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from qlib.contrib.strategy import TopkDropoutStrategy


_T = TypeVar("_T")


def _create_parents(path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def write_all_text(path: str, text: str) -> None:
    _create_parents(path)
    with open(path, "w") as f:
        f.write(text)


def dump_pickle(path: str,
                factory: Callable[[], _T],
                invalidate_cache: bool = False) -> Optional[_T]:
    if invalidate_cache or not os.path.exists(path):
        _create_parents(path)
        obj = factory()
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj


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
    ):
        self._benchmark = benchmark
        self._top_k = top_k
        self._n_drop = n_drop if n_drop is not None else top_k
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost

    def run(
        self,
        prediction: Union[pd.Series, pd.DataFrame],
        seed: int,
        steps: int,
        output_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, BacktestResult]:
        # 确保datetime列存在
        if 'datetime' not in prediction.columns:
            # 尝试找到时间列
            time_cols = [col for col in prediction.columns if col.lower() in ['date', 'time', 'timestamp']]
            if time_cols:
                prediction = prediction.rename(columns={time_cols[0]: 'datetime'})
            else:
                # 如果索引是时间类型
                if isinstance(prediction.index, pd.DatetimeIndex):
                    prediction = prediction.reset_index()
                    if 'index' in prediction.columns:
                        prediction = prediction.rename(columns={'index': 'datetime'})
        
        prediction['datetime'] = pd.to_datetime(prediction['datetime'])
        prediction = prediction.sort_index()

        prediction.set_index(['datetime', 'instrument'], inplace=True)
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        dates = index.levels[0]


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
                    time_per_step="day",
                    generate_portfolio_metrics=True
                )


                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=100_000_000,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": 0.095,
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

        graph = report_graph(report, show_notebook=False)[0]
        if output_path is not None:
            dump_pickle(output_path + f"{seed}-{steps}-report.pkl", lambda: report, True)
            dump_pickle(output_path + f"{seed}-{steps}-graph.pkl", lambda: graph, True)
            write_all_text(output_path + f"{seed}-{steps}-result.json", result.to_json())
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
        seed: int,
        step: Union[int, str] = '-',  # 修改为支持int或str类型，默认值为'-'
        top_k: int=30,
        n_drop: int=3,
        # qlib_path: str="~/notebook/qlib_bin",
        qlib_path = f"~/notebook/qlib_bin/cn_data_backtest",
        predfile_path: str="",
        pred_data: pd.DataFrame=None,
        output_path: str="./backtest",
        benchmark: str="SH000906",
):
    # read pred
    if (pred_data is None) or (not isinstance(pred_data, pd.DataFrame)) or (isinstance(pred_data, pd.DataFrame) and pred_data.empty):
        # pred_data无效，读取预测文件
        try:
            pred_data = pd.read_csv(predfile_path)
            logger.info(f"Read_pred_data from file: {predfile_path}")
        except Exception as e:
            logger.error(f"Pred_data error: {str(e)}")
            sys.exit(1)

    # qlib初始化
    init(provider_uri=qlib_path, region=REG_CN)
    # 回测
    qlib_backtest = QlibBacktest(top_k=top_k, n_drop=n_drop, benchmark=benchmark)
    
    # 将step转换为int（如果是数字字符串）或保持原样
    steps_value = int(step) if isinstance(step, str) and step.isdigit() else 0
    
    report, result = qlib_backtest.run(pred_data, output_path=output_path, seed=seed, steps=steps_value)

    ### 结果汇总
    df_result = pd.DataFrame([asdict(result)])
    df_result = df_result[[
        "annual_return", "annual_excess_return",
        "max_drawdown", "excess_max_drawdown",
        "sharpe", "information_ratio"
    ]]
    
    # 添加seed和step字段
    df_result['seed'] = seed
    df_result['step'] = step  # 使用原始的step值（可能是数字或'-'）
    
    print(df_result.to_string(max_cols=None))

    return df_result


def parse_filename_for_seed_step(filename):
    """
    从文件名解析seed和step
    支持多种文件名格式：
    - master_predictions_backday_8_csi800_0_100.csv (seed=0, step=100)
    - master_predictions_backday_8_csi800_0.csv (seed=0, step='-')
    - master_predictions_backday_8_csi800.csv (seed='-', step='-')
    """
    parts = filename.replace('.csv', '').split('_')
    
    # 查找包含数字的部分
    numeric_parts = []
    for part in parts:
        if part.isdigit():
            numeric_parts.append(part)
    
    if len(numeric_parts) >= 2:
        # 有至少两个数字，最后两个分别是seed和step
        seed = numeric_parts[-2]
        step = numeric_parts[-1]
    elif len(numeric_parts) == 1:
        # 只有一个数字，认为是seed，step设为缺省值
        seed = numeric_parts[0]
        step = '-'
    else:
        # 没有数字，都设为缺省值
        seed = '-'
        step = '-'
    
    return seed, step


if __name__ == "__main__":
    ### 1. 参数
    # qlib数据目录
    # qlib_path = f"~/notebook/qlib_bin"
    qlib_path = f"~/notebook/qlib_bin/cn_data_backtest"
    # 回测结果存放目录
    output_path = "./Backtest_results/predictions"

    # 参数
    benchmark = "SH000906"
    deal = "close"

    # 读取预测值文件列表
    folder_path = Path(f'./Backtest_results/predictions')
    filename_list = [file.name for file in folder_path.iterdir() if file.is_file() and file.name.endswith('.csv')]

    result = pd.DataFrame()
    for filename in filename_list:
        # 解析文件名获取seed和step
        seed, step = parse_filename_for_seed_step(filename)
        
        # 尝试将seed转换为int，如果不能转换则保持字符串
        try:
            seed_int = int(seed)
        except ValueError:
            seed_int = 0  # 如果seed不是数字，使用默认值0
            logger.warning(f"无法从文件名 {filename} 解析seed，使用默认值0")

        predfile_path = Path(folder_path / filename)
        logger.info(f"Backtest： {filename}, Seed: {seed}, Step: {step}")

        df_result = main(
            qlib_path=qlib_path,
            predfile_path=predfile_path,
            output_path=output_path,
            benchmark=benchmark,
            seed=seed_int,  # 传入转换后的seed
            step=step       # 传入step（可能是数字或'-'）
        )
        result = pd.concat([result, df_result], ignore_index=True)

    print(result)
    
    # 保存结果，包含seed和step字段
    result.to_csv(f'{folder_path}/backtest_result.csv', index=False)
    logger.info(f"回测结果已保存到: {folder_path}/backtest_result.csv")
    
    # 输出统计信息
    logger.info(f"共处理了 {len(result)} 个文件")
    logger.info(f"包含step信息的文件: {len(result[result['step'] != '-'])}")
    logger.info(f"缺少step信息的文件: {len(result[result['step'] == '-'])}")