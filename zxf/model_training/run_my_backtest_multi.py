import os
import sys
import fire
import pandas as pd
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from My_backtest import Backtest


# ============================================================
# 单个回测任务（多进程 worker，必须是顶层函数）
# ============================================================
def run_single_backtest(args):
    (
        seed, d_model, lr, dropout, step,
        top_k, n_drop, hold_p,
        pred_filename, output_path
    ) = args

    try:
        backtest = Backtest(
            top_k=top_k,
            n_drop=n_drop,
            hold_p=hold_p,
            pred_filename=pred_filename,
            backtest_start_date="2025-09-01",
        )

        account_detail = backtest.run()

        # 保存完整回测明细
        out_file = (
            f"{output_path}/backtest_my_"
            f"{seed}_{d_model}_{lr}_{dropout}_{step}_"
            f"{top_k}_{n_drop}_{hold_p}.csv"
        )
        account_detail.to_csv(out_file, index=False)

        # 汇总最后一行
        s = account_detail.iloc[-1:].copy()
        s['Seed'] = seed
        s['D_model'] = d_model
        s['LR'] = lr
        s['Dropout'] = dropout
        s['Step'] = step
        s['top_k'] = top_k
        s['n_drop'] = n_drop
        s['hold_p'] = hold_p

        return s[
            ['Seed','D_model','LR','Dropout','Step',
             'top_k','n_drop','hold_p',
             'total_return','annual_return','max_drawdown',
             'annual_turnover','volatility','sharpe',
             'information_ratio','alpha','beta',
             'annual_benchmark_return','winning_rate_cumulant']
        ]

    except Exception as e:
        logger.error(
            f"[FAILED] seed={seed}, top_k={top_k}, "
            f"n_drop={n_drop}, hold_p={hold_p}, err={e}"
        )
        return None


# ============================================================
# 主函数
# ============================================================
def main(
    market_name: str = "csi800",
    folder_name: str = "csi800_20251229_20150101_20251226",
    data_path: str = "home/idc2/notebook/zxf/data/master_results",
    bt_n: int = 5,
):

        # pred_filename="/home/idc2/notebook/zxf/data/master_results/csi800_20251229_20150101_20251226/Backtest_Results/predictions/master_predictions_backday_8_csi800_80_33.csv",
    
    expt_path = f"{data_path}/{folder_name}"
    output_path = f"{expt_path}/Backtest_Results/my"
    os.makedirs(output_path, exist_ok=True)

    backday = 8

    # --------------------------------------------------------
    # 1. 读取 qlib 回测结果
    # --------------------------------------------------------
    try:
        # qlib_file = f"{expt_path}/Backtest_Results/qlib/backtest_qlib_results.csv"
        qlib_file = f"{expt_path}/Backtest_Results/info_result.csv"
        qlib_bt_r = pd.read_csv(qlib_file)
    except Exception as e:
        logger.error(f"Qlib backtest results get failed: {e}")
        sys.exit(1)

    # 每组模型参数，取 annual_return 最大的 bt_n 条
    top_per_group = (
        qlib_bt_r
        .groupby(['Seed', 'D_model', 'LR', 'Dropout'])
        .apply(lambda x: x.nlargest(bt_n, 'annual_return'))
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 2. 构造并发任务列表
    # --------------------------------------------------------
    tasks = []

    top_k_list = [80, 50, 30, 20, 10]
    n_drop_ratios = [1, 0.1]
    hold_p_list = [10, 5, 1]

    for _, row in top_per_group.iterrows():
        seed = int(row['Seed'])
        d_model = int(row['D_model'])
        lr = row['LR']
        dropout = row['Dropout']
        step = int(row['Step'])

        pred_filename = (
            f"{expt_path}/Predictions/"
            f"master_predictions_backday_{backday}_"
            f"{market_name}_{seed}_{d_model}_{lr}_{dropout}_{step}.csv"
        )

        for top_k in top_k_list:
            for ratio in n_drop_ratios:
                for hold_p in hold_p_list:
                    n_drop = max(1, int(top_k * ratio))
                    if n_drop > top_k:
                        continue

                    tasks.append((
                        seed, d_model, lr, dropout, step,
                        top_k, n_drop, hold_p,
                        pred_filename, output_path
                    ))

    logger.info(f"Total backtest tasks: {len(tasks)}")

    # --------------------------------------------------------
    # 3. 并发执行（带进度条）
    # --------------------------------------------------------
    max_workers = os.cpu_count() - 1
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_backtest, t) for t in tasks]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Running Backtests",
            ncols=100
        ):
            res = future.result()
            if res is not None:
                results.append(res)

    if not results:
        logger.error("No backtest results generated.")
        sys.exit(1)

    backtest_my_results = pd.concat(results, ignore_index=True)

    backtest_my_results_file = f"{output_path}/backtest_my_results_new5.csv"
    backtest_my_results.to_csv(backtest_my_results_file, index=False)
    logger.info(f"Saved my backtest results to {backtest_my_results_file}")

    # --------------------------------------------------------
    # 4. merge qlib 结果
    # --------------------------------------------------------
    df_qlib = qlib_bt_r.rename(columns={
        'annual_return': 'qlib_annual_return',
        'annual_excess_return': 'qlib_annual_excess_return',
        'max_drawdown': 'qlib_max_drawdown',
        'excess_max_drawdown': 'qlib_excess_max_drawdown',
        'sharpe': 'qlib_sharpe',
        'information_ratio': 'qlib_information_ratio',
    })

    df_results = pd.merge(
        backtest_my_results,
        df_qlib,
        on=['Seed', 'D_model', 'LR', 'Dropout', 'Step'],
        how='left'
    )

    final_file = f"{expt_path}/Backtest_Results/Backtest_Results_new5.csv"
    df_results.to_csv(final_file, index=False)
    logger.info(f"Final merged results saved to {final_file}")


if __name__ == "__main__":
    fire.Fire(main)
