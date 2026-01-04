import pandas as pd
import fire

from My_backtest import Backtest

def main(
    top_k_list=[30,20,10,5],
    n_drop_list=[1,0.7,0.5,0.3,0.1],
    hold_p_list=[5,4,3,2,1],
    # data_path=f"/home/idc2/notebook/zxf/data/master_results/master_20251207_csi800_test_data/Backtest_Results",
    data_path=f"/home/idc2/notebook/zxf/data/beta_data/csi800c_20251209_20200101_20251208",
):
    backtest_results = pd.DataFrame()

    for top_k in top_k_list:
        for n_drop in n_drop_list:
            n_drop = int(top_k * n_drop)
            if n_drop == 0:
                continue
            for hold_p in hold_p_list:
                print(f"top_k:{top_k} / n_drop:{n_drop} / hold_p:{hold_p}")

                backtest = Backtest(
                    top_k=top_k,
                    n_drop=n_drop,
                    hold_p=hold_p,        
                    # pred_filename=f"{data_path}/predictions/master_predictions_backday_8_csi800_867_44.csv"    
                    pred_filename=f"{data_path}/master_predictions_backday_8_csi800_1199_56.csv"    

                )
                account_detail = backtest.run()
                account_detail_s = account_detail[-1:]
                account_detail_s['top_k'] = top_k
                account_detail_s['n_drop'] = n_drop
                account_detail_s['hold_p'] = hold_p
                
                backtest_results = pd.concat([backtest_results, account_detail_s],ignore_index=True)

    
    backtest_results.to_csv(f'./results/backtest_results_result.csv', index=False, date_format='%Y-%m-%d',float_format='%.3f')



if __name__ == "__main__":
    fire.Fire(main)