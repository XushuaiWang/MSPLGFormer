import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
from tqdm import tqdm

from Models import analysis, utils
from Models.loss import l2_reg_loss

logger = logging.getLogger('__main__')
NEG_METRICS = {'loss'}


class BaseTrainer(object):
    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None,
                 print_conf_mat=False):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError


class SupervisedTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = analysis.Analyzer(print_conf_mat=kwargs.get('print_conf_mat', False))

    def train_epoch(self, epoch_num=None):
        self.model.train()
        epoch_loss = 0.0
        total_samples = 0

        for x, targets, _ in self.dataloader:
            x = x.to(self.device)
            targets = targets.to(self.device)
            predictions = self.model(x)
            losses = self.loss_module(predictions, targets)
            batch_loss = torch.sum(losses)
            mean_loss = batch_loss / len(losses)
            total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model) if self.l2_reg else mean_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            total_samples += len(losses)
            epoch_loss += batch_loss.item()

        self.epoch_metrics = OrderedDict(epoch=epoch_num, loss=epoch_loss / total_samples)
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):
        self.model.eval()
        epoch_loss = 0.0
        total_samples = 0
        all_targets, all_predictions = [], []

        with torch.no_grad():
            for x, targets, _ in self.dataloader:
                x = x.to(self.device)
                targets = targets.to(self.device)
                predictions = self.model(x)
                losses = self.loss_module(predictions, targets)
                batch_loss = torch.sum(losses).item()

                total_samples += len(losses)
                epoch_loss += batch_loss
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(predictions.detach().cpu().numpy())

        logits = torch.from_numpy(np.concatenate(all_predictions, axis=0))
        probs = torch.softmax(logits, dim=1).numpy()
        pred_labels = np.argmax(probs, axis=1)
        targets = np.concatenate(all_targets, axis=0).flatten()
        class_names = np.arange(probs.shape[1])
        metrics_dict = self.analyzer.analyze_classification(pred_labels, targets, class_names)

        self.epoch_metrics = OrderedDict(
            epoch=epoch_num,
            loss=epoch_loss / total_samples,
            accuracy=metrics_dict['total_accuracy'],
            precision=metrics_dict['prec_avg'],
        )
        return self.epoch_metrics, metrics_dict


def is_better(metric_name, value, best_value):
    return value < best_value if metric_name in NEG_METRICS else value > best_value


def initial_best_value(metric_name):
    return float('inf') if metric_name in NEG_METRICS else -float('inf')


def validate(val_evaluator, tensorboard_writer, config, epoch):
    aggr_metrics, _ = val_evaluator.evaluate(epoch, keep_all=True)
    print_str = 'Validation Summary: '
    for key, value in aggr_metrics.items():
        tensorboard_writer.add_scalar(f'{key}/val', value, epoch)
        print_str += f'{key}: {value:8f} | '
    logger.info(print_str)
    return aggr_metrics


def train_runner(config, model, trainer, val_evaluator, save_path):
    epochs = config['epochs']
    key_metric = config['key_metric']
    best_value = initial_best_value(key_metric)
    best_metrics = None
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    for epoch in tqdm(range(1, epochs + 1), desc='Training Epoch', leave=False):
        train_metrics = trainer.train_epoch(epoch)
        train_str = f'Epoch {epoch} Training Summary: '
        for key, value in train_metrics.items():
            tensorboard_writer.add_scalar(f'{key}/train', value, epoch)
            train_str += f'{key}: {value:8f} | '
        logger.info(train_str)

        should_validate = epoch == 1 or epoch % config['val_interval'] == 0 or epoch == epochs
        if should_validate:
            val_metrics = validate(val_evaluator, tensorboard_writer, config, epoch)
            current_value = val_metrics[key_metric]
            if is_better(key_metric, current_value, best_value):
                best_value = current_value
                best_metrics = val_metrics.copy()
                utils.save_model(save_path, epoch, model, config['optimizer'])
                logger.info('New best validation %s: %.6f at epoch %s. Saved: %s',
                            key_metric, best_value, epoch, save_path)

    tensorboard_writer.close()
    if best_metrics is None:
        # Should never happen because epoch 1 is always validated, but keep a safe fallback.
        utils.save_model(save_path, epochs, model, config['optimizer'])
        best_metrics = OrderedDict(epoch=epochs, loss=float('nan'), accuracy=float('nan'), precision=float('nan'))

    total_runtime = time.time() - total_start_time
    logger.info('Train Time: %s hours, %s minutes, %s seconds\n', *utils.readable_time(total_runtime))
    return best_metrics
