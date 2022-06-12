from abc import ABC, abstractmethod

from torch import nn


class BaseModel(ABC):

    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def pre_train(self):
        pass

    @abstractmethod
    def post_train(self):
        pass

    @abstractmethod
    def pre_epoch(self):
        pass

    @abstractmethod
    def post_epoch(self):
        pass

    @abstractmethod
    def pre_batch(self, batch):
        pass

    @abstractmethod
    def post_batch(self, batch_out):
        pass

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
    def save_checkpoint(self, epoch):
        pass

    @abstractmethod
    def load_checkpoint(self, epoch):
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