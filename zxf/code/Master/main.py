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


def _fmt_bytes(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return str(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}TB"


def _load_pickle_with_log(path: str, name: str = ""):
    p = Path(path)
    size = p.stat().st_size if p.exists() else -1
    tag = f"{name} " if name else ""
    print(f"[load] {tag}start: {p} (size={_fmt_bytes(size)})", flush=True)
    t0 = time.perf_counter()
    with open(p, "rb") as f:
        obj = pickle.load(f)
    t1 = time.perf_counter()
    print(f"[load] {tag}done:  {p} (elapsed={t1 - t0:.2f}s)", flush=True)
    return obj


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
        # 增量训练常用覆盖：rolling 时一般只跑少量 epoch / 更小 lr
        n_epochs_override: int = None,
        lr_override: float = None,
        train_stop_loss_thred_override: float = None,
        strict_load: bool = True,
        # GPU/数据加载性能参数（默认保持旧行为，脚本可按需打开）
        amp: bool = None,
        amp_dtype: str = None,          # "bf16" / "fp16"
        tf32: bool = None,
        deterministic: bool = None,
        num_workers: int = None,
        pin_memory: bool = None,
        persistent_workers: bool = None,
        prefetch_factor: int = None,
        data_path: str=f"/home/idc2/notebook/zxf/data",
):
    experimental_data_path = f"{data_path}/master_results/{folder_name}"
    print(f"[main] pid={os.getpid()} market_name={market_name} folder_name={folder_name}", flush=True)
    print(f"[main] experimental_data_path={experimental_data_path}", flush=True)

    ### 1.读取配置文件
    cfg_file = f"{experimental_data_path}/workflow_config_master_Alpha158_{market_name}.yaml"
    print(f"[load] reading config: {cfg_file}", flush=True)
    with open(cfg_file, 'r') as f:
        config = yaml.safe_load(f)
    universe = config["market"] # 优化，直接从配置文件取值

    ### 2.读取实验数据
    # data_dir = f'../../Data/Results/{folder_name}'

    dl_train = _load_pickle_with_log(f'{experimental_data_path}/{universe}_self_dl_train.pkl', name="train")
    dl_valid = _load_pickle_with_log(f'{experimental_data_path}/{universe}_self_dl_valid.pkl', name="valid")
    dl_test = _load_pickle_with_log(f'{experimental_data_path}/{universe}_self_dl_test.pkl', name="test")
    print("[load] Data Loaded.", flush=True)

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

    # 额外性能参数：优先 CLI，其次 YAML（若存在），最后默认值
    mk = config.get("task", {}).get("model", {}).get("kwargs", {}) if isinstance(config, dict) else {}
    if amp is None:
        amp = bool(mk.get("amp", False))
    if amp_dtype is None:
        amp_dtype = str(mk.get("amp_dtype", "bf16"))
    if tf32 is None:
        tf32 = bool(mk.get("tf32", False))
    if deterministic is None:
        deterministic = bool(mk.get("deterministic", True))
    if num_workers is None:
        num_workers = int(mk.get("num_workers", 0))
    if pin_memory is None:
        pin_memory = bool(mk.get("pin_memory", False))
    if persistent_workers is None:
        persistent_workers = bool(mk.get("persistent_workers", False))
    if prefetch_factor is None:
        prefetch_factor = int(mk.get("prefetch_factor", 2))

    # CLI 覆盖（便于滚动增量训练时不改 yaml）
    if n_epochs_override is not None:
        n_epoch = int(n_epochs_override)
    if lr_override is not None:
        lr = float(lr_override)
    if train_stop_loss_thred_override is not None:
        train_stop_loss_thred = float(train_stop_loss_thred_override)
    print(
        f"[config] n_epochs={n_epoch}, lr={lr}, train_stop_loss_thred={train_stop_loss_thred}, GPU={GPU}, "
        f"amp={amp}, amp_dtype={amp_dtype}, tf32={tf32}, deterministic={deterministic}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}, prefetch_factor={prefetch_factor}"
    )


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
            enable_rank_loss=enable_rank_loss,
            # 性能参数透传到 SequenceModel
            amp=amp,
            amp_dtype=amp_dtype,
            tf32=tf32,
            deterministic=deterministic,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        if rolling:
            print(f"[rolling] loading init params from: {ckpt_path}")
            model.load_param(ckpt_path, strict=bool(strict_load))
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