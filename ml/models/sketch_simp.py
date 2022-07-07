from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from ml.logger import log
from ml.models import BaseTrainModel
from ml.models.alac_gan_partials import NetF
from ml.models.base import BaseInferenceModel
from ml.models.sketch_simp_partials import SketchSimpModel
from ml.options.default import DefaultTrainOptions
from ml.options.sketch_simp import SketchSimpInferenceOptions, SketchSimpTrainOptions
from ml.plot_utils import plt_input_target


class SketchSimpInferenceModel(BaseInferenceModel):

    def __init__(self, opt: SketchSimpInferenceOptions, inference_loader: DataLoader):
        super().__init__(opt, inference_loader)

        self.network = None

        self.setup()

    def init_from_checkpoint(self, checkpoint):
        loaded_opt = DefaultTrainOptions()
        loaded_opt.load_saved_dict(checkpoint['option_saved_dict'])

        self.network = SketchSimpModel()
        self.network.load_state_dict(checkpoint['network_state_dict'])

    def inference_batch(self, i, batch_data):
        inp, tar = batch_data
        inp, tar = inp.to(self.opt.device), tar.to(self.opt.device)

        out = self.network(inp)

        return inp, tar, out


class SketchSimpTrainModel(BaseTrainModel):

    def __init__(self, opt: SketchSimpTrainOptions, train_loader, test_loader):
        super().__init__(opt, train_loader, test_loader)

        self.network = None
        self.net_F = None
        self.optimizer = None
        self.scheduler = None
        self.crt_mse = None
        self.crt_l1 = None

        self.l1_losses = []
        self.content_losses = []

        self.setup()

    def setup_from_train_checkpoint(self, checkpoint):
        loaded_opt = SketchSimpTrainOptions()
        loaded_opt.load_saved_dict(checkpoint['option_saved_dict'])

        self.setup_from_opt(loaded_opt)

        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def setup_from_opt(self, opt):
        self.network = SketchSimpModel().to(opt.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=opt.lr)
        self.crt_mse = nn.MSELoss()
        self.crt_l1 = nn.L1Loss()
        self.net_F = NetF(self.opt).to(self.opt.device)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=opt.opt_step_size,
            gamma=opt.opt_gamma
        )

    def _sanity_check(self):
        log('Generating Sanity Checks')
        i = 1
        n = 10
        for inp_batch, tar_batch in self.train_loader:
            for inp, tar in zip(inp_batch, tar_batch):
                plt_input_target(inp, tar, save_file=f'sanity-check-im-{i}.jpg')
                i += 1
                if i > n:
                    break
            if i > n:
                break
        log('Sanity Checks Generated')

    def pre_train(self):
        super().pre_train()
        self._sanity_check()

    def pre_epoch(self):
        super().pre_epoch()
        self.l1_losses = []
        self.content_losses = []

    def pre_batch(self, epoch, batch):
        super().pre_batch(epoch, batch)

    def train_batch(self, batch, batch_data):
        self.network = self.network.to(self.opt.device).train()
        self._set_requires_grad(self.network, True)

        inp, real = batch_data
        inp, real = inp.to(self.opt.device), real.to(self.opt.device)

        self.optimizer.zero_grad()

        fake = self.network(inp)
        # l1 loss
        l1_loss = self.crt_l1(fake, real)

        # content loss
        fake_feat = self.net_F(fake)
        with torch.no_grad():
            real_feat = self.net_F(real)
        content_loss = self.crt_mse(fake_feat, real_feat)

        loss = content_loss
        loss.backward()

        self.optimizer.step()

        return -1, content_loss.item()

    def post_batch(self, epoch, batch, batch_out):
        super().post_batch(epoch, batch, batch_out)
        self.l1_losses.append(batch_out[0])
        self.content_losses.append(batch_out[1])

    def evaluate_batch(self, i, batch_data) -> Tuple[float, Tensor, Tensor, Tensor]:
        self.network = self.network.eval().to(self.opt.device)

        inp, tar = batch_data
        inp, tar = inp.to(self.opt.device), tar.to(self.opt.device)

        out = self.network(inp)
        loss = self.crt_l1(tar, out)

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
               f'[l1_loss={np.mean(self.l1_losses):.4f}] ' + \
               f'[content_loss={np.mean(self.content_losses):.4f}] '

    def log_batch(self, batch):
        from_batch = self._get_last_batch(batch)
        return super().log_batch(batch) + \
               f'[loss={np.mean(self.l1_losses[from_batch - 1:batch]):.4f}] ' + \
               f'[loss={np.mean(self.content_losses[from_batch - 1:batch]):.4f}] '
