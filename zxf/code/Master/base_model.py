import numpy as np
import pandas as pd
import copy
import math
import os

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


def calc_ic(pred, label):
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


def zscore(x):
    # print(x.std())
    return (x - x.mean()).div(x.std())

# def zscore(x, eps=1e-8):
#     """标准化处理，添加极小值保护"""
#     # print(f"numel:{x.numel()}")
#     # print(f"x:{x}")
#
#     mean = x.mean()
#     std = x.std()
#     # 处理标准差为0或极小的情形
#     std = torch.where(std < eps, torch.ones_like(std) * eps, std)
#     return (x - mean) / std


def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025 * N)
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]


def drop_na(x):
    N = x.shape[0]
    mask = ~x.isnan()
    return mask, x[mask]


class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index(), dtype='int64').groupby(
            "datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path='model/',
                 save_prefix='', enable_rank_loss=False,
                 num_workers=None, pin_memory=None, prefetch_factor=None, persistent_workers=None, non_blocking=None,
                 eval_freq=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None, lr_scheduler_monitor=None):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix

        self.enable_rank_loss = enable_rank_loss

        # -------------------------
        # LR Scheduler（可选；默认关闭保持原行为：固定 lr）
        # -------------------------
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs if isinstance(lr_scheduler_kwargs, dict) else {}
        self.lr_scheduler_monitor = lr_scheduler_monitor
        self.lr_scheduler = None

        # DataLoader / H2D pipeline options (不会改变数值结果；仅影响吞吐)
        def _env_int(name, default):
            val = os.getenv(name, None)
            if val is None or val == "":
                return default
            try:
                return int(val)
            except Exception:
                return default

        def _env_bool(name, default: bool):
            val = os.getenv(name, None)
            if val is None or val == "":
                return default
            return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

        if num_workers is None:
            num_workers = _env_int("MASTER_NUM_WORKERS", 0)
        if pin_memory is None:
            pin_memory = _env_bool("MASTER_PIN_MEMORY", torch.cuda.is_available())
        if prefetch_factor is None:
            prefetch_factor = _env_int("MASTER_PREFETCH_FACTOR", 2)
        if persistent_workers is None:
            persistent_workers = _env_bool("MASTER_PERSISTENT_WORKERS", True)
        if non_blocking is None:
            non_blocking = _env_bool("MASTER_NON_BLOCKING", True)
        # 训练过程中验证频率：0=不验证；N>0=每 N 个 epoch 验证一次（默认=1，保持原行为）
        if eval_freq is None:
            eval_freq = _env_int("MASTER_EVAL_FREQ", 1)

        num_workers = max(int(num_workers), 0)
        self.num_workers = num_workers
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()
        self.prefetch_factor = max(int(prefetch_factor), 1)
        self.persistent_workers = bool(persistent_workers) if self.num_workers > 0 else False
        self.non_blocking = bool(non_blocking) and self.pin_memory
        self.eval_freq = max(int(eval_freq), 0)

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self._init_lr_scheduler()
        self.model.to(self.device)

    def _init_lr_scheduler(self):
        """
        初始化学习率调度器（可选）。

        - 默认关闭：与原实现完全一致（固定 lr）
        - 支持：
          - cosine  : CosineAnnealingLR
          - step    : StepLR
          - plateau : ReduceLROnPlateau
        """
        name = self.lr_scheduler_name
        if name is None:
            self.lr_scheduler = None
            return
        name = str(name).strip().lower()
        if name in {"", "none", "null", "false", "0"}:
            self.lr_scheduler = None
            return

        kw = dict(self.lr_scheduler_kwargs) if isinstance(self.lr_scheduler_kwargs, dict) else {}

        if name in {"cosine", "cosineannealing", "cosineannealinglr"}:
            T_max = int(kw.pop("T_max", self.n_epochs))
            eta_min = float(kw.pop("eta_min", 0.0))
            self.lr_scheduler = CosineAnnealingLR(self.train_optimizer, T_max=T_max, eta_min=eta_min, **kw)
            return

        if name in {"step", "steplr"}:
            step_size = int(kw.pop("step_size", max(1, int(self.n_epochs) // 3)))
            gamma = float(kw.pop("gamma", 0.5))
            self.lr_scheduler = StepLR(self.train_optimizer, step_size=step_size, gamma=gamma, **kw)
            return

        if name in {"plateau", "reducelronplateau", "reduce"}:
            # 默认用 train_loss 做监控（mode=min）；这样即使 eval_freq=0 也能工作
            mode = str(kw.pop("mode", "min"))
            factor = float(kw.pop("factor", 0.5))
            patience = int(kw.pop("patience", 5))
            min_lr = float(kw.pop("min_lr", 0.0))
            self.lr_scheduler = ReduceLROnPlateau(
                self.train_optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                **kw,
            )
            return

        raise ValueError(f"Unknown lr_scheduler: {name}. Supported: none|cosine|step|plateau")

    def _step_lr_scheduler(self, train_loss, metrics, do_eval: bool):
        if self.lr_scheduler is None:
            return

        # ReduceLROnPlateau: 需要一个“被监控的标量”
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            monitor = self.lr_scheduler_monitor
            if monitor is None or str(monitor).strip() == "":
                monitor = "train_loss"
            m = str(monitor).strip().lower()

            # 如果监控的是 valid_*，但本轮未做验证，则直接跳过（避免把 NaN 喂给 scheduler）
            if m.startswith("valid_") and (not bool(do_eval)):
                return

            if m in {"train_loss", "loss", "train"}:
                value = train_loss
            elif m in {"valid_ic", "ic"}:
                value = metrics.get("IC")
            elif m in {"valid_icir", "icir"}:
                value = metrics.get("ICIR")
            elif m in {"valid_ric", "ric"}:
                value = metrics.get("RIC")
            elif m in {"valid_ricir", "ricir"}:
                value = metrics.get("RICIR")
            else:
                raise ValueError(
                    f"Unknown lr_scheduler_monitor: {monitor}. "
                    "Supported: train_loss | valid_IC | valid_ICIR | valid_RIC | valid_RICIR"
                )

            try:
                v = float(value)
            except Exception:
                return
            if not math.isfinite(v):
                return
            self.lr_scheduler.step(v)
            return

        # 其它 scheduler：按 epoch step
        self.lr_scheduler.step()

    def loss_fn(self, pred, label, rank_loss_ratio=0.01, topk_loss_ratio=0.01, topk_ratio=0.15):
        mask = ~torch.isnan(label)
        loss = torch.mean((pred[mask] - label[mask]) ** 2)

        if self.enable_rank_loss:
            diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)  # [N, N]
            diff_label = label.unsqueeze(1) - label.unsqueeze(0)
            S = torch.sign(diff_label)  # [N, N], +1/-1/0

            mask_upper = torch.triu(torch.ones_like(S), diagonal=1).bool()  # 仅考虑 i<j
            S_pair = S[mask_upper]
            D_pair = diff_pred[mask_upper]

            rank_loss = - torch.log(torch.sigmoid(S_pair * D_pair) + 1e-12).mean()

            loss += rank_loss_ratio * rank_loss

            N = pred.shape[0]
            k = max(1, int(topk_ratio * N))  # 比如 topk_ratio = 0.2

            _, idx = torch.sort(label, descending=True)
            G = idx[:k]  # 正例
            B = idx[k:]  # 负例

            if len(G) > 0 and len(B) > 0:
                pG = pred[G].unsqueeze(1)  # [k, 1]
                pB = pred[B].unsqueeze(0)  # [1, N-k]
                topk_loss = -torch.log(torch.sigmoid(pG - pB) + 1e-12).mean()

            loss += topk_loss_ratio * topk_loss

        return loss

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            # DataLoader 输出在 CPU；优先一次性搬到 GPU，再在 GPU 上切片，减少 H2D 拷贝次数与 CPU 侧切片开销
            data = torch.squeeze(data, dim=0).to(self.device, non_blocking=self.non_blocking)
            # print(data.shape)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158/360 factors + 63 market information + 1 label           
            '''
            feature = data[:, :, 0:-1]
            label = data[:, -1, -1]

            # print("\n" + "feature" + "*" * 100)
            # print(feature.shape)
            # print("-" * 50)
            # print(label.shape)

            # Additional process on labels
            # If you use original data to train, you won't need the following lines because we already drop extreme when we dumped the data.
            # If you use the opensource data to train, use the following lines to drop extreme labels.
            #########################
            mask, label = drop_extreme(label)
            feature = feature[mask, :, :]
            label = zscore(label)  # CSZscoreNorm
            #########################

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.detach())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        if len(losses) == 0:
            return float("nan")
        losses_t = torch.stack(losses)
        losses_t = losses_t[~torch.isnan(losses_t)]
        if losses_t.numel() == 0:
            return float("nan")
        # 为了严格复现原实现：原来是逐 batch loss.item()（float64）后用 numpy.mean 做均值。
        # 这里保持“只在 epoch 末同步一次”，但均值仍用 numpy 的 float64 计算路径。
        losses_np = losses_t.detach().cpu().numpy().astype(np.float64, copy=False)
        return float(np.mean(losses_np))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0).to(self.device, non_blocking=self.non_blocking)
            feature = data[:, :, 0:-1]
            label = data[:, -1, -1]

            # You cannot drop extreme labels for test.
            label = zscore(label)

            with torch.inference_mode():
                pred = self.model(feature.float())
                loss = self.loss_fn(pred, label)
            losses.append(loss.detach())

        if len(losses) == 0:
            return float("nan")
        losses_t = torch.stack(losses)
        losses_t = losses_t[~torch.isnan(losses_t)]
        if losses_t.numel() == 0:
            return float("nan")
        losses_np = losses_t.detach().cpu().numpy().astype(np.float64, copy=False)
        return float(np.mean(losses_np))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        # prefetch_factor 仅在 num_workers>0 时生效；否则传入会报错
        kwargs = dict(
            sampler=sampler,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        data_loader = DataLoader(data, **kwargs)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)

        process_info = pd.DataFrame([])     # 过程信息
        for step in range(self.n_epochs):
            curr_lr = float(self.train_optimizer.param_groups[0].get("lr", float("nan"))) if self.train_optimizer else float("nan")
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            metrics = {
                'IC': float("nan"),
                'ICIR': float("nan"),
                'RIC': float("nan"),
                'RICIR': float("nan"),
            }

            # 是否在当前 epoch 做验证：
            # - eval_freq=0: 完全不验证（训练最快，避免 GPU 等待 CPU 指标计算）
            # - eval_freq>0: 每 eval_freq 个 epoch 验证一次
            do_eval = (dl_valid is not None) and (self.eval_freq > 0) and (step % self.eval_freq == 0)
            if do_eval:
                _, metrics = self.predict(dl_valid)
                if self.lr_scheduler is not None:
                    print(
                        "Seed %d, Epoch %d, lr %.2e, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f."
                        % (self.seed, step, curr_lr, train_loss, metrics['IC'], metrics['ICIR'], metrics['RIC'], metrics['RICIR'])
                    )
                else:
                    print(
                        "Seed %d, Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f."
                        % (self.seed, step, train_loss, metrics['IC'], metrics['ICIR'], metrics['RIC'], metrics['RICIR'])
                    )
            else:
                # 跳过验证时，只打印训练损失
                if self.lr_scheduler is not None:
                    print("Seed %d, Epoch %d, lr %.2e, train_loss %.6f" % (self.seed, step, curr_lr, train_loss))
                else:
                    print("Seed %d, Epoch %d, train_loss %.6f" % (self.seed, step, train_loss))

            # 输出信息
            df = {
                'Step': step,
                'LR': curr_lr,
                'Train_loss': train_loss,
                'Valid_IC': metrics['IC'],
                'Valid_ICIR': metrics['ICIR'],
                'Valid_RIC': metrics['RIC'],
                'Valid_RICIR': metrics['RICIR']
            }
            process_info = pd.concat([process_info, pd.DataFrame([df])], ignore_index=True)


            # Add stop train condition.
            # if (train_loss <= self.train_stop_loss_thred) and (last_valid_ic - metrics['IC'] <= 0.005):
            # Do not use valid data performance to judge the train stop contidion.
            # 每个种子均输出
            # 直接保存当前 state_dict 即可；避免 deepcopy 带来的额外拷贝/同步开销
            torch.save(self.model.state_dict(), f'{self.save_path}/{self.save_prefix}_{step}.pkl')

            # epoch 结束后更新学习率（影响下一 epoch）
            self._step_lr_scheduler(train_loss=train_loss, metrics=metrics, do_eval=do_eval)

            if train_loss <= self.train_stop_loss_thred:
                break

        return process_info

    # 256特征值生成输出
    def encode(self, dl_test):
        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)
        encs = []
        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0).to(self.device, non_blocking=self.non_blocking)
            feature = data[:, :, 0:-1]
            with torch.inference_mode():
                enc = self.model.encode(feature.float()).detach().cpu().numpy()
            encs.append(enc)
        out = pd.DataFrame(
            np.concatenate(encs),
            index=dl_test.get_index(),
            columns=[f'master_{i}' for i in range(128)]         # 输出master值
        )
        return out

    def predict(self, dl_test):
        print('self.fitted:', self.fitted)
        # if self.fitted<0:
        #     raise ValueError("model is not fitted yet!")
        # else:
        #     print('Epoch:', self.fitted)

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds = []
        ic = []
        ric = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            # label 仅用于 CPU 上的 IC/RIC 统计，保持在 CPU 以避免额外 D2H
            label = data[:, -1, -1]
            data = data.to(self.device, non_blocking=self.non_blocking)
            feature = data[:, :, 0:-1]

            # nan label will be automatically ignored when compute metrics.
            # zscorenorm will not affect the results of ranking-based metrics.

            with torch.inference_mode():
                pred = self.model(feature.float()).detach().cpu().numpy()
            preds.append(pred.ravel())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())
        # print(predictions)

        # ic 与 ric 列表中去除空值（最后5天没有label，值为nan）
        ic = [x for x in ic if not math.isnan(x)]
        ric = [x for x in ric if not math.isnan(x)]

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic) / np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric) / np.std(ric)
        }

        return predictions, metrics
