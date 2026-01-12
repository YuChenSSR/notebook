from master import MASTERModel
import pickle
import random
import numpy as np
import time
import os
import sys
import json
import pandas as pd
import yaml
import fire


# 允许在任意工作目录运行：把上层 `master_futures/code/` 加入 sys.path，以便复用 roll_config / data_generator
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)  # .../master_futures/code
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

try:
    from roll_config import roll_config  # type: ignore
except Exception:
    roll_config = None  # 在 incremental=False 的常规训练中不会用到

try:
    from data_generator import data_generator as generate_data  # type: ignore
except Exception:
    generate_data = None  # 在 incremental=False 的常规训练中不会用到

try:
    from select_best_ckpt import select_best_ckpt  # 同目录
except Exception:
    select_best_ckpt = None  # 在 incremental=False 的常规训练中不会用到


def main(
        # project_dir: str = "/c/Quant",
        market_name: str = "f88",
        folder_name: str = "f88_20260101_20100104_20251212",
        seed_num: int= None,
        data_path: str = "/home/idc2/notebook/master_futures/data",
        enable_rank_loss: bool = False,
        # ===== 增量训练实验开关 =====
        incremental: bool = False,
        prev_folder_name: str | None = None,
        qlib_path: str = "/home/idc2/notebook/futures/data/qlib_bin/cn_data_train",
        roll_to_latest: bool = True,
        resume_from: str = "best",
        best_metric: str = "valid_IC",
        force_eval: bool = False,
        n_epochs_override: int | None = None,
):
    experimental_data_path = f"{data_path}/master_results/{folder_name}"

    # -----------------------------
    # 增量训练：滚动窗口 + 选择 best ckpt + warm-start
    # -----------------------------
    resume_ckpt_path = None
    resume_seed = None
    resume_epoch = None
    if bool(incremental):
        if prev_folder_name is None or str(prev_folder_name).strip() == "":
            raise ValueError("incremental=True 时必须指定 --prev_folder_name")
        if roll_config is None or generate_data is None or select_best_ckpt is None:
            raise ImportError("增量训练依赖 roll_config/data_generator/select_best_ckpt，请检查代码与环境")

        # 1) 生成滚动后的 workflow_config（写入本次实验目录）
        cfg_path = None
        if bool(roll_to_latest):
            cfg_path = roll_config(
                market_name=market_name,
                data_path=data_path,
                provider_uri=qlib_path,
                prev_folder_name=prev_folder_name,
                out_folder_name=folder_name,
                dry_run=False,
            )
        else:
            cfg_path = f"{experimental_data_path}/workflow_config_master_Alpha158_{market_name}.yaml"

        # 2) 生成本次实验数据（dl_train/dl_valid/dl_test + handler.pkl）
        #    注意：这里会把实际使用的 config（handler 替换为 file://）写入实验目录，保证可复现
        generate_data(
            market_name=market_name,
            qlib_path=qlib_path,
            data_path=data_path,
            folder_name=folder_name,
            config_path=cfg_path,
            overwrite_exp_config=True,
        )

        # 3) 从上一轮实验选择 best ckpt（默认 valid_IC 最大；评估对象=每 seed 最后 ckpt）
        best_json = os.path.join(experimental_data_path, "best_ckpt_valid_ic.json")
        best_json = select_best_ckpt(
            market_name=market_name,
            prev_folder_name=prev_folder_name,
            data_path=data_path,
            split="valid",
            scope="last",
            metric=best_metric,
            force_eval=force_eval,
            out_json=best_json,
        )
        meta = json.loads(open(best_json, "r", encoding="utf-8").read())
        sel = meta["selected"]
        resume_ckpt_path = sel["ckpt_path"]
        resume_seed = int(sel["seed"])
        resume_epoch = int(sel["epoch"])
        print(f"[INCREMENTAL] resume_from={resume_from} best_metric={best_metric} seed={resume_seed} epoch={resume_epoch}")
        print(f"[INCREMENTAL] ckpt_path={resume_ckpt_path}")

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
    if n_epochs_override is not None:
        n_epoch = int(n_epochs_override)
    lr = config["task"]["model"]["kwargs"]["lr"]                                                # lr = 1e-5
    GPU = config["task"]["model"]["kwargs"]["GPU"]                                              # GPU = 0
    train_stop_loss_thred = config["task"]["model"]["kwargs"]["train_stop_loss_thred"]          # train_stop_loss_thred = 0.92

    beta = config["task"]["model"]["kwargs"]["beta"]                                            # csi300:5;csi500:3;csi800:2;else:2

    benchmark = config.get("benchmark", None)
    backday = config['task']['dataset']['kwargs']['step_len']


    # enable ranking-based auxiliary loss (fire 参数：--enable_rank_loss=True)
    universe_tag = universe
    if enable_rank_loss:
        print('Rank loss enabled!')
        universe_tag = universe + '_rank'


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
    if bool(incremental):
        seed_number_list = [int(resume_seed)]
    else:
        rng = random.Random(int(time.time()))
        seed_number_list = rng.sample(range(0, 100), seed_num)
    # seed_number_list = [17, 27, 58, 84, 91]
    
    print(f"Seed List:{seed_number_list}")
    
    train_process_info = pd.DataFrame([])
    # for seed in range(seed_num):
    for seed in seed_number_list:
        model = MASTERModel(
            d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
            save_path=save_path, save_prefix=f'{universe_tag}_backday_{backday}_self_exp_{seed}',
            enable_rank_loss=enable_rank_loss
        )
        start = time.time()

        # Warm-start：仅在 incremental=True 时加载历史 ckpt
        if bool(incremental):
            if resume_ckpt_path is None:
                raise RuntimeError("incremental=True 但 resume_ckpt_path 为空（best ckpt 选择失败？）")
            model.load_param(resume_ckpt_path)
            model.fitted = resume_epoch

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
        pred_frame.to_csv(f'{save_path}/master_predictions_backday_{backday}_{universe_tag}_{seed}.csv', index=False, date_format='%Y-%m-%d')

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