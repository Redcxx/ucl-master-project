from typing import Dict, Tuple

import numpy as np
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from ml.models import BaseTrainModel
from ml.models.base import BaseInferenceModel
from ml.options import BaseTrainOptions
from ml.options.base import BaseInferenceOptions
from ml.options.default import DefaultTrainOptions


class DefaultInferenceModel(BaseInferenceModel):

    def __init__(self, opt: BaseInferenceOptions, inference_loader: DataLoader):
        super().__init__(opt, inference_loader)

        self.network = None

        self.setup()

    def init_from_checkpoint(self, checkpoint):
        loaded_opt = DefaultTrainOptions()
        loaded_opt.load_saved_dict(checkpoint['option_saved_dict'])

        self.network = ...
        self.network.load_state_dict(checkpoint['network_state_dict'])

    def inference_batch(self, i, batch_data):
        inp, tar = batch_data
        inp, tar = inp.to(self.opt.device), tar.to(self.opt.device)

        out = self.network(inp)

        return inp, tar, out


class DefaultTrainModel(BaseTrainModel):

    def __init__(self, opt: BaseTrainOptions, train_loader, test_loader):
        super().__init__(opt, train_loader, test_loader)

        self.network = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.losses = []

    def setup_from_train_checkpoint(self, checkpoint):
        loaded_opt = DefaultTrainOptions()
        loaded_opt.load_saved_dict(checkpoint['option_saved_dict'])

        self.setup_from_opt(loaded_opt)

        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def setup_from_opt(self, opt):
        self.network = ...
        self.optimizer = optim.Adam(self.network.parameters(), lr=opt.lr)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.95,
            last_epoch=self.opt.start_epoch - 1
        )

    def pre_train(self):
        super().pre_train()

    def pre_epoch(self):
        super().pre_epoch()
        self.losses = []

    def pre_batch(self, epoch, batch):
        super().pre_batch(epoch, batch)

    def train_batch(self, batch, batch_data):
        self.network = self.network.to(self.opt.device).train()
        self._set_requires_grad(self.network, True)

        inp, tar = batch_data
        inp, tar = inp.to(self.opt.device), tar.to(self.opt.device)

        out = self.network(inp)
        loss = self.criterion(tar, out)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def post_batch(self, epoch, batch, batch_out):
        super().post_batch(epoch, batch, batch_out)
        self.losses.append(batch_out)

    def evaluate_batch(self, i, batch_data) -> Tuple[float, Tensor, Tensor, Tensor]:
        self.network = self.network.eval().to(self.opt.device)

        inp, tar = batch_data
        inp, tar = inp.to(self.opt.device), tar.to(self.opt.device)

        out = self.network(inp)
        loss = self.criterion(tar, out)

        return loss.item(), inp, tar, out

    def post_epoch(self, epoch):
        super().post_epoch(epoch)
        self.scheduler.step()

    def post_train(self):
        super().post_train()

    def get_checkpoint(self) -> Dict:
        return {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'option_saved_dict': self.opt.saved_dict
        }

    def log_epoch(self, epoch):
        return super().log_epoch(epoch) + \
               f'[lr={self._get_lr(self.optimizer):.6f}] ' + \
               f'[loss={np.mean(self.losses):.4f}] '

    def log_batch(self, batch):
        from_batch = self._get_last_batch(batch)
        return super().log_batch(batch) + f'[loss={np.mean(self.losses[from_batch - 1:batch]):.4f}] '
