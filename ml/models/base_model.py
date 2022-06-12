import time
from abc import ABC, abstractmethod

from torch import nn

from ml.save_load import format_time, init_drive_and_folder


class BaseModel(ABC):

    def __init__(self, opt):
        self.opt = opt
        self.training_start_time = None
        self.last_batch_time = None
        self.epoch_eval_loss = None
        self.epoch_start_time = None
        self.training_start_time = None

    def pre_train(self):
        self.training_start_time = time.time()
        print(f'Training started at: {format_time(self.training_start_time)}')

        init_drive_and_folder(self.opt)

    def post_train(self):
        training_end_time = time.time()
        print(f'Training finished at {format_time(training_end_time)}')
        print(f'Time taken: {format_time(training_end_time - self.training_start_time)}')
        self.save_checkpoint(tag='final')

    def pre_epoch(self):
        self.last_batch_time = time.time()
        self.epoch_start_time = time.time()

    def post_epoch(self, epoch):
        if self.opt.eval_freq is not None and (epoch % self.opt.eval_freq == 0 or epoch == self.opt.start_epoch):
            self.evaluate()

        if self.opt.log_freq is not None and (epoch % self.opt.log_freq == 0 or epoch == self.opt.start_epoch):
            self.log_epoch(epoch)

        if self.opt.save_freq is not None and (epoch % self.opt.save_freq == 0 or epoch == self.opt.start_epoch):
            self.save_checkpoint(epoch)

    def pre_batch(self, epoch, batch):
        pass

    def post_batch(self, epoch, batch, batch_out):
        if self.opt.batch_log_freq is not None and (epoch % self.opt.batch_log_freq == 0 or batch == 1):
            self.log_batch(batch)

    @abstractmethod
    def train_batch(self, batch, batch_data):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def log_epoch(self, epoch):
        pass

    @abstractmethod
    def log_batch(self, batch):
        pass

    @abstractmethod
    def save_checkpoint(self, tag):
        pass

    @abstractmethod
    def load_checkpoint(self, tag):
        pass

    def _gaussian_init_weight(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1
                                     or classname.find('BatchNorm2d') != -1):
            nn.init.normal_(m.weight.data, 0.0, self.opt.init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def decay_rule(self, epoch):
        return 1.0 - max(0, epoch + self.opt.start_epoch - self.opt.end_epoch) / float(self.opt.epochs_decay + 1)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def set_requires_grad(self, net, requires_grad):
        for param in net.parameters():
            param.requires_grad = requires_grad