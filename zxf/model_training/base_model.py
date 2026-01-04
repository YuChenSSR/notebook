import numpy as np
import pandas as pd
import copy
import math

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim


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
                 save_prefix='', enable_rank_loss=False):
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

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

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
            data = torch.squeeze(data, dim=0)
            # print(data.shape)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158/360 factors + 63 market information + 1 label           
            '''
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

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
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
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # You cannot drop extreme labels for test.
            label = zscore(label)

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_param = None
        # 验证集IC辅助判断
        last_valid_ic = 0

        process_info = pd.DataFrame([])     # 过程信息
        for step in range(self.n_epochs):
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
            feature = data[:, :, 0:-1].to(self.device)
            with torch.no_grad():
                enc = self.model.encode(feature.float()).detach().cpu().numpy()
            encs.append(enc)
        out = pd.DataFrame(
            np.concatenate(encs),
            index=dl_test.get_index(),
            columns=[f'master_{i}' for i in range(1160)]         # 输出master值
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
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]

            # nan label will be automatically ignored when compute metrics.
            # zscorenorm will not affect the results of ranking-based metrics.

            with torch.no_grad():
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
