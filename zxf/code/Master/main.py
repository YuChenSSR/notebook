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
from pathlib import Path
import re


def _resolve_init_ckpt_path(
        init_param_path: str = None,
        init_dir: str = None,
        init_seed: int = None,
        init_step: int = None,
):
    """
    Resolve warm-start checkpoint path.

    Priority:
    1) init_param_path (explicit ckpt file)
    2) init_dir + init_seed (+ optional init_step) -> pick step max if init_step is None
    """
    if init_param_path:
        ckpt_path = Path(init_param_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"init_param_path not found: {ckpt_path}")
        return str(ckpt_path)

    if not init_dir:
        raise ValueError("rolling=True requires init_dir or init_param_path")
    if init_seed is None:
        raise ValueError("rolling=True requires init_seed")

    init_dir_path = Path(init_dir)
    if not init_dir_path.exists():
        raise FileNotFoundError(f"init_dir not found: {init_dir_path}")

    # filenames like: {universe}_backday_{backday}_self_exp_{seed}_{step}.pkl
    pattern = re.compile(rf".*_self_exp_{init_seed}_(\d+)\.pkl$")
    candidates = []
    for f in init_dir_path.glob(f"*self_exp_{init_seed}_*.pkl"):
        m = pattern.match(f.name)
        if not m:
            continue
        step = int(m.group(1))
        if init_step is not None and step != int(init_step):
            continue
        candidates.append((step, f))

    if not candidates:
        extra = f" with init_step={init_step}" if init_step is not None else ""
        raise FileNotFoundError(
            f"No checkpoint matched in init_dir={init_dir_path} for init_seed={init_seed}{extra}"
        )

    # choose max step by default
    step, ckpt = max(candidates, key=lambda x: x[0])
    print(f"[rolling] resolved init ckpt: seed={init_seed}, step={step}, path={ckpt}")
    return str(ckpt)


def main(
        # project_dir: str = "/c/Quant",
        market_name: str="csi800",
        folder_name: str="csi800_20251105_20150101_20251103",
        seed_num: int= None,
        seed: int = None,
        use_fixed_seeds: bool = True,
        fixed_seed_list: str = "67,80",
        rolling: bool = False,
        init_param_path: str = None,
        init_dir: str = None,
        init_seed: int = None,
        init_step: int = None,
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
    if rolling:
        if init_seed is None:
            raise ValueError("rolling=True requires init_seed (manually selected best seed).")
        seed_number_list = [int(init_seed)]
        ckpt_path = _resolve_init_ckpt_path(
            init_param_path=init_param_path,
            init_dir=init_dir,
            init_seed=int(init_seed),
            init_step=init_step,
        )
        # Warn if output dir equals init dir (risk of overwrite)
        if init_dir and os.path.abspath(init_dir) == os.path.abspath(save_path):
            print("[rolling][warn] init_dir is the same as current save_path; checkpoints may be overwritten.")
    else:
        if seed is not None:
            seed_number_list = [int(seed)]
        elif use_fixed_seeds:
            # Backward-compatible default behavior: use fixed seeds (previously hard-coded)
            fixed_seeds = [int(x.strip()) for x in str(fixed_seed_list).split(",") if x.strip() != ""]
            if not fixed_seeds:
                raise ValueError("use_fixed_seeds=True but fixed_seed_list is empty.")
            if seed_num is not None and seed_num > 0:
                # keep behavior stable while allowing smaller seed_num
                seed_number_list = fixed_seeds[:int(seed_num)]
            else:
                seed_number_list = fixed_seeds
        else:
            rng = random.Random(int(time.time()))
            seed_number_list = rng.sample(range(0, 100), seed_num)
    
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
        if rolling:
            print(f"[rolling] loading init params from: {ckpt_path}")
            model.load_param(ckpt_path)
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