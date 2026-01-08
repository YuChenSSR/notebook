import numpy as np
import pandas as pd
import copy
import math

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim
from contextlib import nullcontext
from functools import partial


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

def _seed_worker(worker_id: int, base_seed: int = None):
    """
    DataLoader worker 初始化随机种子，保证 num_workers>0 时也可复现。
    """
    if base_seed is None:
        return
    s = int(base_seed) + int(worker_id)
    np.random.seed(s)
    torch.manual_seed(s)


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
    def __init__(
            self,
            n_epochs,
            lr,
            GPU=None,
            seed=None,
            train_stop_loss_thred=None,
            save_path='model/',
            save_prefix='',
            enable_rank_loss=False,
            # DataLoader 性能参数（默认保持旧行为）
            num_workers: int = 0,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            prefetch_factor: int = 2,
            # CUDA 计算性能参数（默认保持旧行为）
            amp: bool = False,
            amp_dtype: str = "bf16",   # "bf16" / "fp16"
            tf32: bool = False,
            deterministic: bool = True,
    ):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        # DataLoader 参数
        self.num_workers = max(0, int(num_workers)) if num_workers is not None else 0
        # pin_memory 只有在 CUDA 下才有意义
        self.pin_memory = bool(pin_memory) and (self.device.type == "cuda")
        self.persistent_workers = bool(persistent_workers) and (self.num_workers > 0)
        self.prefetch_factor = int(prefetch_factor) if (prefetch_factor is not None) else None

        # CUDA 性能参数
        self.deterministic = bool(deterministic)
        self.tf32 = bool(tf32) and (self.device.type == "cuda")
        self.amp_enabled = bool(amp) and (self.device.type == "cuda")
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        # 训练可复现 vs 性能：默认保持旧行为（seed 存在且 deterministic=True 时，走确定性）
        if self.device.type == "cuda":
            if self.deterministic and (self.seed is not None):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                # 非确定性允许 benchmark，通常更快
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True

            # TF32：对 Ampere+ 的 GEMM/conv 通常能显著提速，精度略有折中
            if self.tf32:
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    # PyTorch 2.x 推荐接口
                    if hasattr(torch, "set_float32_matmul_precision"):
                        torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

        # AMP：优先 bf16（避免 fp16 下 log/sigmoid 的数值问题）
        if self.amp_enabled and (self.amp_dtype == torch.bfloat16) and self.device.type == "cuda":
            # 旧版 torch 可能没有 is_bf16_supported；不影响运行，失败则继续尝试
            try:
                if hasattr(torch.cuda, "is_bf16_supported") and (not torch.cuda.is_bf16_supported()):
                    # 回退 fp16
                    self.amp_dtype = torch.float16
            except Exception:
                pass

        self.scaler = None
        if self.amp_enabled and (self.amp_dtype == torch.float16) and self.device.type == "cuda":
            # GradScaler 仅对 fp16 必要
            try:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            except Exception:
                self.scaler = None

        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix

        self.enable_rank_loss = enable_rank_loss

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str):
        s = str(amp_dtype).strip().lower() if amp_dtype is not None else ""
        if s in ("bf16", "bfloat16"):
            return torch.bfloat16
        if s in ("fp16", "float16", "half", "16"):
            return torch.float16
        # 其它情况：不启用 autocast dtype 选择，仍可通过 amp=False 关闭
        return torch.bfloat16

    def _autocast_ctx(self):
        if self.amp_enabled and (self.device.type == "cuda"):
            return torch.cuda.amp.autocast(enabled=True, dtype=self.amp_dtype)
        return nullcontext()

    def loss_fn(self, pred, label, rank_loss_ratio=0.01, topk_loss_ratio=0.01, topk_ratio=0.15):
        # AMP 下 pred 可能是 fp16/bf16；loss/排序相关计算建议用 fp32 保证数值稳定
        pred = pred.float()
        label = label.float()

        mask = ~torch.isnan(label)
        loss = torch.mean((pred[mask] - label[mask]) ** 2)

        if self.enable_rank_loss:
            diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)  # [N, N]
            diff_label = label.unsqueeze(1) - label.unsqueeze(0)
            S = torch.sign(diff_label)  # [N, N], +1/-1/0

            mask_upper = torch.triu(torch.ones_like(S), diagonal=1).bool()  # 仅考虑 i<j
            S_pair = S[mask_upper]
            D_pair = diff_pred[mask_upper]

            # 注意：fp16 下 1e-12 会下溢为 0，导致 -inf；这里已确保 fp32
            rank_loss = - torch.log(torch.sigmoid(S_pair * D_pair) + 1e-12).mean()

            loss += rank_loss_ratio * rank_loss

            N = pred.shape[0]
            k = max(1, int(topk_ratio * N))  # 比如 topk_ratio = 0.2

            _, idx = torch.sort(label, descending=True)
            G = idx[:k]  # 正例
            B = idx[k:]  # 负例

            topk_loss = torch.tensor(0.0, device=pred.device)
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
            data = torch.squeeze(data, dim=0)
            # print(data.shape)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158/360 factors + 63 market information + 1 label           
            '''
            # 传输优化：pin_memory + non_blocking 可以提升 H2D 吞吐
            if self.device.type == "cuda":
                feature = data[:, :, 0:-1].to(self.device, non_blocking=self.pin_memory)
                label = data[:, -1, -1].to(self.device, non_blocking=self.pin_memory)
            else:
                feature = data[:, :, 0:-1].to(self.device)
                label = data[:, -1, -1].to(self.device)

            # Additional process on labels
            # If you use original data to train, you won't need the following lines because we already drop extreme when we dumped the data.
            # If you use the opensource data to train, use the following lines to drop extreme labels.
            #########################
            mask, label = drop_extreme(label)
            feature = feature[mask, :, :]
            label = zscore(label)  # CSZscoreNorm
            #########################

            self.train_optimizer.zero_grad(set_to_none=True)
            with self._autocast_ctx():
                pred = self.model(feature.float())
                loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.train_optimizer)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.scaler.step(self.train_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.train_optimizer.step()

        losses = [x for x in losses if not math.isnan(x)]
        # print(losses)
        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            if self.device.type == "cuda":
                feature = data[:, :, 0:-1].to(self.device, non_blocking=self.pin_memory)
                label = data[:, -1, -1].to(self.device, non_blocking=self.pin_memory)
            else:
                feature = data[:, :, 0:-1].to(self.device)
                label = data[:, -1, -1].to(self.device)

            # You cannot drop extreme labels for test.
            label = zscore(label)

            with torch.no_grad():
                with self._autocast_ctx():
                    pred = self.model(feature.float())
                    loss = self.loss_fn(pred, label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        kwargs = dict(
            sampler=sampler,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        # 多进程可复现：固定 worker seed + generator
        if self.seed is not None:
            try:
                kwargs["worker_init_fn"] = partial(_seed_worker, base_seed=int(self.seed))
            except Exception:
                pass
            try:
                g = torch.Generator()
                g.manual_seed(int(self.seed))
                kwargs["generator"] = g
            except Exception:
                pass
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
        try:
            data_loader = DataLoader(data, **kwargs)
        except TypeError:
            # 兼容旧版 torch：某些参数可能不存在
            kwargs.pop("persistent_workers", None)
            kwargs.pop("prefetch_factor", None)
            kwargs.pop("generator", None)
            data_loader = DataLoader(data, **kwargs)
        return data_loader

    def load_param(self, param_path, strict: bool = True):
        """
        Load model parameters.

        Notes:
        - Supports some legacy checkpoints where keys are prefixed with `layers.*`
          (mapped to current `encoder.*` / `decoder.*`).
        """
        state = torch.load(param_path, map_location=self.device)

        # common checkpoint wrapper
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        if not isinstance(state, dict):
            raise ValueError(f"Invalid checkpoint format: {param_path}")

        # remove DataParallel prefix if exists
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}

        # legacy key mapping: layers.* -> encoder/decoder.*
        if any(k.startswith("layers.") for k in state.keys()):
            def _map_legacy_key(k: str) -> str:
                if not k.startswith("layers."):
                    return k
                if k.startswith("layers.0."):
                    return "encoder.0." + k[len("layers.0."):]
                if k.startswith("layers.1."):
                    return "encoder.1." + k[len("layers.1."):]
                if k.startswith("layers.2."):
                    return "encoder.2." + k[len("layers.2."):]
                if k.startswith("layers.3."):
                    return "encoder.3." + k[len("layers.3."):]
                if k.startswith("layers.4."):
                    return "encoder.4." + k[len("layers.4."):]
                if k.startswith("layers.5."):
                    return "decoder." + k[len("layers.5."):]
                return k

            state = {_map_legacy_key(k): v for k, v in state.items()}
            print("[load_param] legacy key mapping applied: layers.* -> encoder/decoder.*")

        try:
            incompatible = self.model.load_state_dict(state, strict=strict)
            if (getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None)):
                print(f"[load_param] missing_keys={incompatible.missing_keys}, unexpected_keys={incompatible.unexpected_keys}")
        except RuntimeError as e:
            raise RuntimeError(
                f"Load param failed: {param_path}\n"
                f"Common causes: d_feat/backday/model arch mismatch between ckpt and current config.\n"
                f"Original error: {str(e)}"
            ) from e

        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_param = None
        # 验证集IC辅助判断
        last_valid_ic = 0

        process_info = pd.DataFrame([])     # 过程信息
        for step in range(self.n_epochs):
            # 先打印“epoch 开始”，避免一个 epoch 很久导致看起来像没输出
            print(f"Seed {self.seed}, Epoch {step} start...", flush=True)
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            if dl_valid:
                predictions, metrics = self.predict(dl_valid)
                print("Seed %d, Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." % (self.seed, step, train_loss, metrics['IC'], metrics['ICIR'], metrics['RIC'], metrics['RICIR']))
            else:
                print("Seed %d, Epoch %d, train_loss %.6f" % (self.seed, step, train_loss))

            last_valid_ic = metrics['IC']

            # 输出信息
            df = {
                'Step': step,
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
            best_param = copy.deepcopy(self.model.state_dict())
            torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{step}.pkl')
            if train_loss <= self.train_stop_loss_thred:
                break

        return process_info

    # 256特征值生成输出
    def encode(self, dl_test):
        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)
        encs = []
        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            if self.device.type == "cuda":
                feature = data[:, :, 0:-1].to(self.device, non_blocking=self.pin_memory)
            else:
                feature = data[:, :, 0:-1].to(self.device)
            with torch.no_grad():
                with self._autocast_ctx():
                    enc = self.model.encode(feature.float()).float().detach().cpu().numpy()
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
            if self.device.type == "cuda":
                feature = data[:, :, 0:-1].to(self.device, non_blocking=self.pin_memory)
            else:
                feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]

            # nan label will be automatically ignored when compute metrics.
            # zscorenorm will not affect the results of ranking-based metrics.

            with torch.no_grad():
                with self._autocast_ctx():
                    pred = self.model(feature.float()).float().detach().cpu().numpy()
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
