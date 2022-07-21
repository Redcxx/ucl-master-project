import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchsummaryX import summary

from ml.logger import log
from ml.models import BaseTrainModel
from ml.models.base import BaseInferenceModel
from ml.models.waifu2x_partials import Net
from ml.models.warmup_lr import WarmupLRScheduler
from ml.options.base import BaseInferenceOptions
from ml.options.default import DefaultTrainOptions
from ml.options.waifu2x import Waifu2xTrainOptions
from ml.plot_utils import plt_input_target


class Waifu2xInferenceModel(BaseInferenceModel):

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


class Waifu2xTrainModel(BaseTrainModel):

    def __init__(self, opt: Waifu2xTrainOptions, train_loader, test_loader):
        super().__init__(opt, train_loader, test_loader)
        self.opt = opt

        self.network = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.losses = []
        self.setup()

    def setup_from_train_checkpoint(self, checkpoint):
        loaded_opt = Waifu2xTrainOptions()
        loaded_opt.load_saved_dict(checkpoint['option_saved_dict'])

        self.setup_from_opt(loaded_opt)

        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def setup_from_opt(self, opt):
        self.network = Net(scale=opt.scale, multi_scale=opt.multi_scale, group=1)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=opt.lr)

        self.criterion = nn.L1Loss()

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=WarmupLRScheduler(opt.end_epoch - opt.start_epoch + 1, opt.decay)
        )

    def pre_train(self):
        super().pre_train()
        self._sanity_check()

    def pre_epoch(self):
        super().pre_epoch()
        self.losses = []

    def pre_batch(self, epoch, batch):
        super().pre_batch(epoch, batch)

    def _sanity_check(self):
        log('Generating Sanity Checks')
        # see if model architecture is alright
        summary(
            self.network,
            torch.rand(self.opt.batch_size, 3, self.opt.patch_size, self.opt.patch_size).to(self.opt.device),
            scale=self.opt.scale
        )
        # get some data and see if it looks good
        i = 1
        for real_cim, _, real_sim in self.train_loader:
            for inp, tar in zip(real_sim, real_cim):
                plt_input_target(inp, tar, save_file=f'sanity-check-im-{i}.jpg')
                i += 1
                if i > 5:
                    break
            if i > 5:
                break
        log('Sanity Checks Generated')

    def train_batch(self, batch, batch_data):
        self.network = self.network.to(self.opt.device).train()
        self._set_requires_grad(self.network, True)

        inputs = batch_data
        if self.opt.scale > 0:
            scale = self.opt.scale
            hr, lr = inputs[-1][0], inputs[-1][1]
        else:
            # only use one of multi-scale data
            # i know this is stupid but just temporary
            scale = random.randint(2, 4)
            hr, lr = inputs[scale - 2][0], inputs[scale - 2][1]

        hr, lr = hr.to(self.opt.device), lr.to(self.opt.device)

        out = self.network(lr, scale)
        loss = self.criterion(hr, out)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.network.parameters(), self.opt.clip)
        self.optimizer.step()

        return loss.item()

    def post_batch(self, epoch, batch, batch_out):
        super().post_batch(epoch, batch, batch_out)
        self.losses.append(batch_out)
        self.scheduler.step(epoch)

    def evaluate_batch(self, i, batch_data) -> Tuple[float, Tensor, Tensor, Tensor]:
        self.network = self.network.eval().to(self.opt.device)

        inputs = batch_data
        if self.opt.scale > 0:
            scale = self.opt.scale
            hr, lr = inputs[-1][0], inputs[-1][1]
        else:
            # only use one of multi-scale data
            # i know this is stupid but just temporary
            scale = random.randint(2, 4)
            hr, lr = inputs[scale - 2][0], inputs[scale - 2][1]

        hr, lr = hr.to(self.opt.device), lr.to(self.opt.device)

        out = self.network(lr, scale)
        loss = self.criterion(hr, out)

        return loss.item(), lr, hr, out

    def post_epoch(self, epoch):
        super().post_epoch(epoch)

    def post_train(self):
        super().post_train()

    def get_checkpoint(self) -> Dict:
        return {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'option_saved_dict': self.opt.saved_dict
        }

    def log_epoch(self, epoch):
        return super().log_epoch(epoch) + \
               f'[lr={self._get_lr(self.optimizer):.6f}] ' + \
               f'[loss={np.mean(self.losses):.4f}] '

    def log_batch(self, batch):
        from_batch = self._get_last_batch(batch)
        return super().log_batch(batch) + f'[loss={np.mean(self.losses[from_batch - 1:batch]):.4f}] '
