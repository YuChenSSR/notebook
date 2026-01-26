from master import MASTERModel
import pickle
import random
import numpy as np
import time
import os
import re
import sys
import json
import pandas as pd
import yaml
import fire

import shutil


# 允许在任意工作目录运行：把上层 `zxf/code/` 加入 sys.path，以便复用 roll_config / data_generator
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)  # .../zxf/code
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


def _parse_ckpt_filename(fname: str):
    """
    从 checkpoint 文件名解析 (seed, epoch)。

    期望格式：..._self_exp_{seed}_{epoch}.pkl
    - seed / epoch 都必须是整数
    - 返回: (seed:int, epoch:int)；解析失败返回 None
    """
    if not isinstance(fname, str) or not fname.endswith(".pkl"):
        return None
    stem = fname[:-4]
    if "_" not in stem:
        return None
    prefix, epoch_str = stem.rsplit("_", 1)
    try:
        epoch = int(epoch_str)
    except Exception:
        return None
    m = re.search(r"_self_exp_(\d+)$", prefix)
    if m is None:
        return None
    seed = int(m.group(1))
    return seed, epoch


def main(
        # project_dir: str = "/c/Quant",
        market_name: str="csi800",
        folder_name: str="csi800_20251105_20150101_20251103",
        seed_num: int= None,
        data_path: str=f"/home/idc2/notebook/zxf/data",
        enable_rank_loss: bool = False,
        # ===== 增量训练实验开关 =====
        incremental: bool = False,
        prev_folder_name: str | None = None,
        qlib_path: str = "/home/idc2/notebook/qlib_bin/cn_data_train",
        roll_to_latest: bool = True,
        resume_from: str = "best",
        best_metric: str = "valid_IC",
        ckpt_path: str | None = None,
        resume_seed_override: int | None = None,
        resume_epoch_override: int | None = None,
        force_eval: bool = False,
        n_epochs_override: int | None = None,
        # ===== warm-start 输入目录（直接复用既有 YAML + dl_*.pkl）=====
        warmstart_input_dir: str | None = None,
):
    experimental_data_path = f"{data_path}/master_results/{folder_name}"

    def _safe_copy(src: str, dst: str) -> None:
        """
        将 src 拷贝到 dst（仅 copy；不使用 hardlink/软链）。
        """
        src_p = os.path.abspath(os.path.expanduser(str(src)))
        dst_p = os.path.abspath(os.path.expanduser(str(dst)))
        os.makedirs(os.path.dirname(dst_p), exist_ok=True)
        if os.path.exists(dst_p):
            return
        shutil.copy2(src_p, dst_p)

    def _stage_warmstart_inputs(warm_dir: str) -> tuple[str, str]:
        """
        将 warm-start 输入目录里的 YAML + dl_*.pkl（以及可选 ckpt）放到实验目录，返回：
        - cfg_path（实验目录下的 workflow_config_master_Alpha158_{market}.yaml）
        - universe（config["market"]）
        """
        warm_dir = os.path.abspath(os.path.expanduser(str(warm_dir)))
        if not os.path.isdir(warm_dir):
            raise FileNotFoundError(f"warmstart_input_dir 不存在: {warm_dir}")

        # 1) config：优先找标准命名；找不到则允许目录内唯一 yaml
        cand_cfg = os.path.join(warm_dir, f"workflow_config_master_Alpha158_{market_name}.yaml")
        if not os.path.isfile(cand_cfg):
            ymls = [p for p in os.listdir(warm_dir) if p.endswith(".yaml") or p.endswith(".yml")]
            # 排除 template
            ymls = [p for p in ymls if "template" not in p.lower()]
            if len(ymls) == 1:
                cand_cfg = os.path.join(warm_dir, ymls[0])
            else:
                raise FileNotFoundError(
                    "warmstart_input_dir 下找不到标准 config："
                    f"{os.path.basename(cand_cfg)}；且无法从多个 yaml 中自动选择：{ymls}"
                )

        # 读取 config 以获取 universe
        with open(cand_cfg, "r") as f:
            warm_cfg = yaml.safe_load(f)
        if not isinstance(warm_cfg, dict) or "market" not in warm_cfg:
            raise ValueError(f"warm-start config 非法（缺少 market）：{cand_cfg}")
        universe = str(warm_cfg["market"])

        # 2) stage 到实验目录，便于后续离线评估/复现实验
        os.makedirs(experimental_data_path, exist_ok=True)
        exp_cfg = os.path.join(experimental_data_path, f"workflow_config_master_Alpha158_{market_name}.yaml")
        _safe_copy(cand_cfg, exp_cfg)

        for split in ("train", "valid", "test"):
            src_pkl = os.path.join(warm_dir, f"{universe}_self_dl_{split}.pkl")
            if not os.path.isfile(src_pkl):
                raise FileNotFoundError(f"warm-start 缺少数据文件: {src_pkl}")
            dst_pkl = os.path.join(experimental_data_path, f"{universe}_self_dl_{split}.pkl")
            _safe_copy(src_pkl, dst_pkl)

        return exp_cfg, universe

    # -----------------------------
    # 增量训练：滚动窗口 + 选择 best ckpt + warm-start
    # -----------------------------
    resume_ckpt_path = None
    resume_seed = None
    resume_epoch = None
    if bool(incremental):
        # warmstart_input_dir 模式：直接复用既有 YAML + dl_*.pkl，不再要求 prev_folder_name / roll_config / data_generator
        if warmstart_input_dir is not None and str(warmstart_input_dir).strip() != "":
            warm_dir_abs = os.path.abspath(os.path.expanduser(str(warmstart_input_dir)))
            cfg_path, _ = _stage_warmstart_inputs(warm_dir_abs)
            print(f"[WARMSTART] Using warm-start inputs from: {os.path.abspath(os.path.expanduser(str(warmstart_input_dir)))}")
            print(f"[WARMSTART] Staged config: {cfg_path}")

            # 如果用户没显式指定 ckpt_path，则尝试从 warm-start 目录自动选择
            if ckpt_path is None or str(ckpt_path).strip() == "":
                cand = []
                for fn in os.listdir(warm_dir_abs):
                    if not fn.endswith(".pkl"):
                        continue
                    # 排除数据 pkl（dl）
                    if "_self_dl_" in fn:
                        continue
                    # 经验规则：checkpoint 文件名包含 _self_exp_
                    if "_self_exp_" in fn:
                        cand.append(os.path.join(warm_dir_abs, fn))
                if len(cand) == 1:
                    ckpt_path = cand[0]
                    print(f"[WARMSTART] Auto-selected ckpt_path: {ckpt_path}")
                elif len(cand) > 1:
                    raise ValueError(f"[WARMSTART] warm-start 目录下发现多个 ckpt，请显式指定 --ckpt_path：{cand}")
        else:
            if prev_folder_name is None or str(prev_folder_name).strip() == "":
                raise ValueError("incremental=True 时必须指定 --prev_folder_name（或提供 --warmstart_input_dir）")
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

        # 3) 选择 warm-start 起点：
        #    - 优先使用用户指定的 ckpt_path
        #    - 否则从上一轮实验自动选择 best ckpt（默认 valid_IC 最大；评估对象=每 seed 最后 ckpt）
        if ckpt_path is not None and str(ckpt_path).strip() != "":
            resume_ckpt_path = os.path.abspath(os.path.expanduser(str(ckpt_path)))
            if not os.path.isfile(resume_ckpt_path):
                raise FileNotFoundError(f"指定 ckpt_path 不存在: {resume_ckpt_path}")

            parsed = _parse_ckpt_filename(os.path.basename(resume_ckpt_path))
            if parsed is not None:
                resume_seed, resume_epoch = parsed
            else:
                # 文件名不符合预期时：seed 必须由用户显式给出；epoch 可选
                if resume_seed_override is None:
                    raise ValueError(
                        "无法从 ckpt 文件名解析 seed（期望 ..._self_exp_{seed}_{epoch}.pkl）。"
                        "请额外传 --resume_seed_override=xxx（必要）与 --resume_epoch_override=yyy（可选）。"
                    )
                resume_seed = int(resume_seed_override)
                resume_epoch = int(resume_epoch_override) if resume_epoch_override is not None else 0

            print(f"[INCREMENTAL] resume_from=path seed={int(resume_seed)} epoch={int(resume_epoch)}")
            print(f"[INCREMENTAL] ckpt_path={resume_ckpt_path}")
        else:
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
    lr_scheduler = config["task"]["model"]["kwargs"].get("lr_scheduler", None)
    lr_scheduler_kwargs = config["task"]["model"]["kwargs"].get("lr_scheduler_kwargs", None)
    lr_scheduler_monitor = config["task"]["model"]["kwargs"].get("lr_scheduler_monitor", None)
    GPU = config["task"]["model"]["kwargs"]["GPU"]                                              # GPU = 0
    train_stop_loss_thred = config["task"]["model"]["kwargs"]["train_stop_loss_thred"]          # train_stop_loss_thred = 0.92

    beta = config["task"]["model"]["kwargs"]["beta"]                                            # csi300:5;csi500:3;csi800:2;else:2

    benchmark =  config["benchmark"]
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
    # seed_number_list = [91]
    # seed_number_list = [5]
    # seed_number_list = [27]
    # seed_number_list = [16,95]
    
    print(f"Seed List:{seed_number_list}")
    
    train_process_info = pd.DataFrame([])
    # for seed in range(seed_num):
    for seed in seed_number_list:
        model = MASTERModel(
            d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
            lr_scheduler=lr_scheduler, lr_scheduler_kwargs=lr_scheduler_kwargs, lr_scheduler_monitor=lr_scheduler_monitor,
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
        train_process_info_df = train_process_info_df[['Seed', 'Step', 'LR', 'Train_loss', 'Valid_IC', 'Valid_ICIR', 'Valid_RIC', 'Valid_RICIR']]
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