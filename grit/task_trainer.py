import os
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from utils.utils import save_model, get_regresssion_metrics, get_metrics, LossAnomalyDetector
import torch.nn.functional as F
from sklearn.metrics import classification_report, matthews_corrcoef
import random


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.3, 0.6, 0.1], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha, device='cuda')
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1)  # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  # 对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class TaskTrainer:

    def __init__(self, model, output_dir, grad_norm_clip=1.0, device='cuda',
                 max_epochs=10, use_amp=True, task_type='regression',
                 learning_rate=1e-4, lr_patience=20, lr_decay=0.5, min_lr=1e-5, weight_decay=0.0):
        self.model = model
        self.output_dir = output_dir
        self.grad_norm_clip = grad_norm_clip
        self.writer = SummaryWriter(self.output_dir)
        self.learning_rate = learning_rate
        self.device = device
        self.n_epochs = max_epochs
        self.use_amp = use_amp
        self.task_type = task_type
        self.loss_anomaly_detector = LossAnomalyDetector()
        if task_type == 'regression':
            self.loss_fn = nn.MSELoss()
        elif task_type == 'classification':
            # self.loss_fn = nn.BCEWithLogitsLoss()
            # self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1.0], device='cuda'),
            #                                    label_smoothing=0.1, ignore_index=2)
            # self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1.0], device='cuda'),
            #                                    ignore_index=2)
            self.loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor([0.2, 1.0], device='cuda'))
            # self.loss_fn = MultiClassFocalLossWithAlpha()
        else:
            raise Exception(f'Unknown task type: {task_type}')

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, raw_model.parameters()), lr=learning_rate,
                                          weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=lr_decay,
                                                                    patience=lr_patience, verbose=True)
        self.min_lr = min_lr

    def fit(self, train_loader, val_loader=None, test_loader=None, save_ckpt=True):
        model = self.model.to(self.device)

        best_loss = np.float32('inf')
        best_appearance = np.float32(0)

        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(epoch, model, train_loader)
            if val_loader is not None:
                val_loss, mcc_metric = self.eval_epoch(epoch, model, val_loader, e_type='val')

            if test_loader is not None:
                test_loss, _ = self.eval_epoch(epoch, model, test_loader, e_type='test')

            curr_loss = val_loss if 'val_loss' in locals() else train_loss
            curr_appearance = mcc_metric

            if self.output_dir is not None and save_ckpt and curr_loss < best_loss:  # only save better loss
                best_loss = curr_loss
                self._save_model(self.output_dir, str(epoch + 1), curr_loss)

            if self.output_dir is not None and save_ckpt and curr_appearance > best_appearance:  # 表现更好的也可以保存
                best_appearance = curr_appearance
                self._save_model(self.output_dir, str(epoch + 1) + "_mcc_", curr_appearance)

            if self.optimizer.param_groups[0]['lr'] < float(self.min_lr):
                logger.info("Learning rate == min_lr, stop!")
                break
            self.scheduler.step(val_loss)

        if self.output_dir is not None and save_ckpt:  # save final model
            self._save_model(self.output_dir, 'final', curr_loss)

    def run_forward(self, model, batch):
        batch = batch.to(self.device)
        pred, true = model(batch)

        pred = pred.squeeze(-1) if pred.ndim > 1 else pred
        true = true.squeeze(-1) if true.ndim > 1 else true

        logger.debug(f'pred: {pred.shape}, true: {true.shape}')
        print(shit)
        # print(batch)
        # print(pred)
        # print(true)
        # pred = F.log_softmax(pred, dim=-1)
        # loss = F.nll_loss(pred.squeeze(), true.squeeze())

        loss = self.loss_fn(pred, true)
        pred = torch.sigmoid(pred)

        return loss, pred, true

    @torch.no_grad()
    def mix_up(self, real_batch, ratio=0.5, alpha=1.0):
        batch = real_batch.clone()
        lam = np.random.beta(alpha, alpha)
        batch_size = len(batch)
        for i in range(batch_size):
            data = batch[i]
            origin_feature = data.x.clone()
            origin_one_hot = data.binary_label.clone()
            origin_index = data.edge_index.clone()

            num_vectors = origin_index.size(1)
            num_select = int(num_vectors * ratio)
            fusion_list = []
            mode = ['top', 'bot']
            while len(fusion_list) <= num_select:
                index = random.randint(0, num_vectors - 1)
                source_dot = origin_index[0][index].item()
                target_dot = origin_index[1][index].item()
                if target_dot in fusion_list and source_dot in fusion_list:
                    continue

                act = random.choices(mode, weights=[1, 1], k=1)

                x_i = origin_feature[target_dot, :]
                x_j = origin_feature[source_dot, :]
                y_i = origin_one_hot[target_dot]
                y_j = origin_one_hot[source_dot]

                # 源点融合给汇点
                if act == 'top':
                    if target_dot not in fusion_list:
                        data.x[target_dot, :] = lam * x_i + (1 - lam) * x_j
                        data.binary_label[target_dot] = lam * y_i + (1 - lam) * y_j
                        fusion_list.append(target_dot)
                    else:
                        data.x[source_dot, :] = lam * x_j + (1 - lam) * x_i
                        data.binary_label[source_dot] = lam * y_j + (1 - lam) * y_i
                        fusion_list.append(source_dot)
                # 汇点融合给源点
                else:
                    if source_dot not in fusion_list:
                        data.x[source_dot, :] = lam * x_j + (1 - lam) * x_i
                        data.binary_label[source_dot] = lam * y_j + (1 - lam) * y_i
                        fusion_list.append(source_dot)
                    else:
                        data.x[target_dot, :] = lam * x_i + (1 - lam) * x_j
                        data.binary_label[target_dot] = lam * y_i + (1 - lam) * y_j
                        fusion_list.append(target_dot)

        return batch

    def train_epoch(self, epoch, model, train_loader):
        model.train()
        losses = []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for it, batch in pbar:
            if self.device == 'cuda':
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    batch = self.mix_up(batch)
                    print(shit)
                    loss, _, _ = self.run_forward(model, batch)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
            else:
                loss, _, _ = self.run_forward(model, batch)

            if self.loss_anomaly_detector(loss.item()):
                logger.info(f"Anomaly loss detected at epoch {epoch + 1} iter {it}: train loss {loss:.5f}.")
                del loss, batch
                continue
            else:
                losses.append(loss.item())
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss:.5f}.")
                # logger.debug(f"epoch {epoch + 1} iter {it}: train loss {loss:.5f}.")
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                # del loss, batch
                # torch.cuda.empty_cache()
                # print("now_allocated:{}".format(torch.cuda.memory_allocated(0)))

                # import pynvml
                # pynvml.nvmlInit()
                # handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号
                # # 在每一个要查看的地方都要重新定义一个meminfo
                # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # print(meminfo.total / 1024 ** 2)  # 总的显存大小
                # print(meminfo.used / 1024 ** 2)  # 已用显存大小
                # print(meminfo.free / 1024 ** 2)  # 剩余显存大小
                # # 单位是MB，如果想看G就再除以一个1024

        loss = float(np.mean(losses))
        logger.info(f'train epoch: {epoch + 1}/{self.n_epochs}, loss: {loss:.4f}')
        self.writer.add_scalar(f'train_loss', loss, epoch + 1)
        return loss

    @torch.no_grad()
    def eval_epoch(self, epoch, model, test_loader, e_type='test'):
        model.eval()
        losses = []
        y_test = []
        y_test_hat = []

        pbar = enumerate(test_loader)
        for it, batch in pbar:
            if self.device == 'cuda':
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    loss, y_hat, y = self.run_forward(model, batch)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
            else:
                loss, y_hat, y = self.run_forward(model, batch)
            losses.append(loss.item())
            y_test_hat.append(y_hat.cpu().numpy())
            y_test.append(y.cpu().numpy())

        loss = float(np.mean(losses))
        logger.info(f'{e_type} epoch: {epoch + 1}, loss: {loss:.4f}')
        self.writer.add_scalar(f'{e_type}_loss', loss, epoch + 1)

        y_test = np.concatenate(y_test, axis=0).squeeze()
        y_test_hat = np.concatenate(y_test_hat, axis=0).squeeze()
        y_test_hat = np.argmax(y_test_hat, axis=1)

        # logger.info(f'y_test: {y_test.shape}, y_test_hat: {y_test_hat.shape}')
        if self.task_type == 'regression':
            mae, mse, _, spearman, pearson = get_regresssion_metrics(y_test_hat, y_test, print_metrics=False)
            logger.info(
                f'{e_type} epoch: {epoch + 1}, spearman: {spearman:.3f}, pearson: {pearson:.3f}, mse: {mse:.3f}, mae: {mae:.3f}')
            self.writer.add_scalar('spearman', spearman, epoch + 1)
            metric = spearman
        elif self.task_type == 'classification':
            # 找到 y_test 中值不等于2的索引
            indices_to_keep = (y_test != 2)
            # 从 y_test 中移除值为2的部分
            y_test = y_test[indices_to_keep]
            y_test_hat = y_test_hat[indices_to_keep]

            mcc = matthews_corrcoef(y_test, y_test_hat)
            logger.info(
                f'\n{classification_report(y_test, y_test_hat, digits=4)}\n{e_type} epoch: {epoch + 1},  mcc: {mcc:.3f}')
            # acc, pr, sn, sp, mcc, auroc = get_metrics(y_test_hat > 0.5, y_test, print_metrics=False)
            # logger.info(
            #     f'{e_type} epoch: {epoch + 1}, acc: {acc * 100:.2f}, pr: {pr * 100:.3f}, sn: {sn * 100:.3f}, sp: {sp:.2f}, mcc: {mcc:.3f}, auroc: {auroc:.3f}')
            self.writer.add_scalar('mcc', mcc, epoch + 1)
            metric = mcc
        return loss, metric

    def _save_model(self, base_dir, info, valid_loss):
        """ Save model with format: model_{info}_{valid_loss} """
        base_name = f'model_{info}_{valid_loss:.3f}'
        # logger.info(f'Save model {base_name}')
        save_model(self.model, base_dir, base_name)
