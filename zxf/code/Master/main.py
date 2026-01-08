from master import MASTERModel
import pickle
import random
import numpy as np
import time
import os
import sys
import pandas as pd
import yaml
import fire


def main(
        # project_dir: str = "/c/Quant",
        market_name: str="csi800",
        folder_name: str="csi800_20251105_20150101_20251103",
        seed_num: int= None,
        data_path: str=f"/home/idc2/notebook/zxf/data",
):
    experimental_data_path = f"{data_path}/master_results/{folder_name}"

    ### 1.读取配置文件
    with open(f"{experimental_data_path}/workflow_config_master_Alpha158_{market_name}.yaml", 'r') as f:
        config = yaml.safe_load(f)
    universe = config["market"] # 优化，直接从配置文件取值

    ### 2.读取实验数据
    # data_dir = f'../../Data/Results/{folder_name}'

    with open(f'{experimental_data_path}/{universe}_self_dl_train.pkl', 'rb') as f:
        dl_train = pickle.load(f)
    with open(f'{experimental_data_path}/{universe}_self_dl_valid.pkl', 'rb') as f:
        dl_valid = pickle.load(f)
    with open(f'{experimental_data_path}/{universe}_self_dl_test.pkl', 'rb') as f:
        dl_test = pickle.load(f)
    print("Data Loaded.")

    ### 3.实验参数【修改到config中】
    if seed_num is None:
        seed_num = config["task"]["model"]["kwargs"]["seed_num"]
    d_feat = config["task"]["model"]["kwargs"]["d_feat"]                                        # d_feat = 174    # 158
    d_model = config["task"]["model"]["kwargs"]["d_model"]                                      # d_model = 256
    t_nhead = config["task"]["model"]["kwargs"]["t_nhead"]                                      # t_nhead = 4
    s_nhead = config["task"]["model"]["kwargs"]["s_nhead"]                                      # s_nhead = 2
    dropout = config["task"]["model"]["kwargs"]["dropout"]                                      # dropout = 0.2  # 0.5
    gate_input_start_index = config["task"]["model"]["kwargs"]["gate_input_start_index"]        # gate_input_start_index = 174    #158
    gate_input_end_index = config["task"]["model"]["kwargs"]["gate_input_end_index"]            # gate_input_end_index = 237      # 221

    n_epoch = config["task"]["model"]["kwargs"]["n_epochs"]                                      # n_epoch = 100
    lr = config["task"]["model"]["kwargs"]["lr"]                                                # lr = 1e-5
    GPU = config["task"]["model"]["kwargs"]["GPU"]                                              # GPU = 0
    train_stop_loss_thred = config["task"]["model"]["kwargs"]["train_stop_loss_thred"]          # train_stop_loss_thred = 0.92

    beta = config["task"]["model"]["kwargs"]["beta"]                                            # csi300:5;csi500:3;csi800:2;else:2

    benchmark =  config["benchmark"]
    backday = config['task']['dataset']['kwargs']['step_len']


    # added by xhy
    if '--enable_rank_loss' in sys.argv:
        print('Rank loss enabled!')
        enable_rank_loss = True
        universe = universe + '_rank'
    else:
        enable_rank_loss = False


    ### 4. 实验
    # Training
    ######################################################################################
    save_path = f'{experimental_data_path}/Master_results'
    os.makedirs(save_path, exist_ok=True)

    ic = []
    icir = []
    ric = []
    ricir = []
    seed_list = []

    # 随机生成种子
    rng = random.Random(int(time.time()))
    seed_number_list = rng.sample(range(0, 100), seed_num)
    seed_number_list = [17, 27, 58, 84, 91]
    
    print(f"Seed List:{seed_number_list}")
    
    train_process_info = pd.DataFrame([])
    # for seed in range(seed_num):
    for seed in seed_number_list:
        model = MASTERModel(
            d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
            save_path=save_path, save_prefix=f'{universe}_backday_{backday}_self_exp_{seed}',
            enable_rank_loss=enable_rank_loss
        )
        start = time.time()

        # Train
        train_process_info_df = model.fit(dl_train, dl_valid)
        train_process_info_df['Seed'] = seed
        train_process_info_df = train_process_info_df[['Seed', 'Step', 'Train_loss', 'Valid_IC', 'Valid_ICIR', 'Valid_RIC', 'Valid_RICIR']]
        train_process_info = pd.concat([train_process_info,train_process_info_df],ignore_index=True)
        print(f"{seed} Model Trained.")

        # Test
        predictions, metrics = model.predict(dl_test)

        pred_frame = predictions.to_frame()
        pred_frame.columns = ['score']
        pred_frame.reset_index(inplace=True)
        pred_frame.to_csv(f'{save_path}/master_predictions_backday_{backday}_{universe}_{seed}.csv', index=False, date_format='%Y-%m-%d')

        running_time = time.time()-start

        print('Seed: {:d} time cost : {:.2f} sec'.format(seed, running_time))
        # print(metrics)
        # print(predictions)

        seed_list.append(seed)
        ic.append(metrics['IC'])
        icir.append(metrics['ICIR'])
        ric.append(metrics['RIC'])
        ricir.append(metrics['RICIR'])
    ######################################################################################
    train_process_info.to_csv(f'{save_path}/train_metrics_results.csv', index=False)
    ic_data = {
        'SEED': seed_list,
        'IC': ic,
        'ICIR': icir,
        'RIC': ric,
        'RICIR': ricir
    }
    df = pd.DataFrame(ic_data)
    df.to_csv(f'{save_path}/test_metrics_results.csv', index=False)

    print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
    print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
    print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
    print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))

if __name__ == "__main__":
    fire.Fire(main)