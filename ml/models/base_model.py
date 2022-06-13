import time
from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn

from ml.save_load import init_drive_and_folder, save_file, load_file
from ml.misc_utils import format_time
from ml.session import SessionOptions


class BaseModel(ABC):

    def __init__(self, opt: SessionOptions):
        self.opt = opt
        self.training_start_time = None
        self.last_batch_time = None
        self.epoch_eval_loss = None
        self.epoch_start_time = None
        self.training_start_time = None
        self.this_epoch_evaluated = False

    ###
    # Pre & Post Train
    ###
    def pre_train(self):
        init_drive_and_folder(self.opt)  # for saving and loading

        self.training_start_time = time.time()
        print(f'Using device {self.opt.device}')
        print(f'Number of training samples: {len(self.opt.train_loader)}')
        print(f'Number of training batches: {len(self.opt.train_dataset)}')
        print(f'Number of testing samples: {len(self.opt.test_loader)}')
        print(f'Number of testing batches: {len(self.opt.test_dataset)}')
        print(f'Training started at {format_time(self.training_start_time)}')

    def post_train(self):
        training_end_time = time.time()
        print(f'Training finished at {format_time(training_end_time)}')
        print(f'Time taken: {format_time(training_end_time - self.training_start_time)}')
        self.save_checkpoint(tag='final')

    ###
    # Pre & Post Epoch
    ###
    def pre_epoch(self):
        self.last_batch_time = time.time()
        self.epoch_start_time = time.time()

    def post_epoch(self, epoch):
        if self.opt.eval_freq is not None and (epoch % self.opt.eval_freq == 0 or epoch == self.opt.start_epoch):
            print('Evaluation in progress ... ', end='')
            self.evaluate(epoch)
            print('done')
            self.this_epoch_evaluated = True
        else:
            self.this_epoch_evaluated = False

        if self.opt.log_freq is not None and (epoch % self.opt.log_freq == 0 or epoch == self.opt.start_epoch):
            print(self.log_epoch(epoch))

        if self.opt.save_freq is not None and (epoch % self.opt.save_freq == 0 or epoch == self.opt.start_epoch):
            self.save_checkpoint(epoch)

    ###
    # Pre & Post Batch
    ###
    def pre_batch(self, epoch, batch):
        pass

    def post_batch(self, epoch, batch, batch_out):
        if self.opt.batch_log_freq is not None and (epoch % self.opt.batch_log_freq == 0 or batch == 1):
            print(self.log_batch(batch))

    ###
    # Train Batch
    ###
    @abstractmethod
    def train_batch(self, batch, batch_data):
        pass

    @abstractmethod
    def evaluate(self, epoch):
        pass

    ###
    # Log, Save & Loading
    ###
    def log_epoch(self, epoch):
        curr_time = time.time()
        return f'[epoch={epoch}] ' + \
               f'[train_time={format_time(curr_time - self.training_start_time)}] ' + \
               f'[epoch_time={format_time(curr_time - self.epoch_start_time)}] '

    def _get_last_batch(self, this_batch):
        return max(1, this_batch - self.opt.batch_log_freq)

    def log_batch(self, batch):
        curr_time = time.time()
        self.last_batch_time = curr_time
        from_batch = self._get_last_batch(batch)
        return f'[batch={from_batch}-{batch}] ' + \
               f'[batch_time={format_time(curr_time - self.last_batch_time)}] ' + \
               f'[train_time={format_time(curr_time - self.training_start_time)}] '

    @abstractmethod
    def _get_checkpoint(self) -> Dict:
        pass

    def save_checkpoint(self, tag):
        print('Saving checkpoint ... ', end='')
        file_name = f'{self.opt.run_id}_{tag}.ckpt'
        torch.save(self._get_checkpoint(), file_name)
        save_file(self.opt, file_name, local=False)
        print(f'done: {file_name}')

    @abstractmethod
    def load_checkpoint(self, tag):
        file_name = f'{self.opt.run_id}_{tag}.ckpt'
        load_file(self.opt, file_name)  # ensure exists locally, will raise error if not exists
        print(f'Checkpoint file loaded: {file_name}')
        return torch.load(file_name)

    ###
    # Miscellaneous
    ###
    def _gaussian_init_weight(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1
                                     or classname.find('BatchNorm2d') != -1):
            nn.init.normal_(m.weight.data, 0.0, self.opt.init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def _decay_rule(self, epoch):
        return 1.0 - max(0, epoch + self.opt.start_epoch - self.opt.end_epoch) / float(self.opt.epochs_decay + 1)

    @staticmethod
    def _get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    @staticmethod
    def _set_requires_grad(net, requires_grad):
        for param in net.parameters():
            param.requires_grad = requires_grad
