from typing import Tuple

import numpy as np
import scipy.stats as stats
import torch
from torch import optim, nn, Tensor
from torchsummaryX import summary

from ml.models.base import BaseTrainModel, BaseInferenceModel
from .alac_gan_partials import NetG, NetD, NetF, NetI
from ..criterion.GANBCELoss import GANBCELoss
from ..options.alac_gan import AlacGANTrainOptions, AlacGANInferenceOptions
from ..plot_utils import plot_inp_tar


def _mask_gen(opt, X):
    maskS = opt.image_size // 4

    mask1 = torch.cat(
        [torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(opt.batch_size // 2)], 0)
    mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(opt.batch_size // 2)], 0)
    mask = torch.cat([mask1, mask2], 0)

    return mask.to(opt.device)


def gradient_penalty(opt, critic, real, fake, gp_weight=10):
    bs, c, h, w = real.shape
    epsilon = torch.rand((bs, 1, 1, 1)).to(opt.device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    interpolated_images.requires_grad = True
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores).to(opt.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    return torch.mean((gradient.norm(2, dim=1) - 1) ** 2) * gp_weight


class AlacGANInferenceModel(BaseInferenceModel):
    def __init__(self, opt: AlacGANInferenceOptions, inference_loader):
        super().__init__(opt, inference_loader)

        self.net_I = None
        self.net_D = None
        self.net_G = None
        self.net_F = None
        self.opt = opt

        mu, sigma = 1, 0.005
        self.X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)

        self.setup()

    def init_from_checkpoint(self, checkpoint):
        loaded_opt = AlacGANTrainOptions()
        loaded_opt.load_saved_dict(checkpoint['opt'])

        self.net_G = NetG(loaded_opt).to(self.opt.device).eval()
        self.net_D = NetD(loaded_opt).to(self.opt.device).eval()

        self.net_F = NetF(loaded_opt).to(self.opt.device).eval()
        self.net_I = NetI(loaded_opt).to(self.opt.device).eval()

        self.net_G.load_state_dict(checkpoint['net_G_state_dict'])
        self.net_D.load_state_dict(checkpoint['net_D_state_dict'])

    def inference_batch(self, i, batch_data) -> Tuple[Tensor, Tensor, Tensor]:
        real_cim, real_vim, real_sim = batch_data

        real_cim = real_cim.to(self.opt.device)
        real_vim = real_vim.to(self.opt.device)
        real_sim = real_sim.to(self.opt.device)

        mask = _mask_gen(self.opt, self.X)
        hint = torch.cat((real_vim * mask, mask), 1)
        with torch.no_grad():
            # get sketch feature
            feat_sim = self.net_I(real_sim).detach()

        fake_cim = self.net_G(real_sim, hint, feat_sim)

        return real_sim, real_cim, fake_cim


class AlacGANTrainModel(BaseTrainModel):

    def __init__(self, opt: AlacGANTrainOptions, train_loader, test_loader):
        super().__init__(opt, train_loader, test_loader)

        # network
        self.net_G = None
        self.net_D = None
        self.net_F = None
        self.net_I = None

        # optimizer
        self.opt_G = None
        self.opt_D = None

        # scheduler
        self.sch_G = None
        self.sch_D = None

        # loss
        self.crt_mse = None
        self.crt_l1 = None
        self.crt_bce = None
        self.fixed_sketch = None
        self.fixed_hint = None
        self.fixed_sketch_feat = None

        # housekeeping
        self.net_G_losses = []
        self.net_D_losses = []
        # self.grad_penalties = []
        self.epoch_eval_loss = None

        # for generating mask
        mu, sigma = 1, 0.005
        self.X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)

        self.setup()

    def get_checkpoint(self):
        return {
            'net_G_state_dict': self.net_G.state_dict(),
            'net_D_state_dict': self.net_D.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
            'opt': self.opt.saved_dict,
        }

    def setup_from_train_checkpoint(self, checkpoint):
        loaded_opt = AlacGANTrainOptions()
        loaded_opt.load_saved_dict(checkpoint['opt'])

        self.setup_from_opt(loaded_opt)

        self.net_G.load_state_dict(checkpoint['net_G_state_dict'])
        self.net_D.load_state_dict(checkpoint['net_D_state_dict'])

        self.opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        self.opt_D.load_state_dict(checkpoint['opt_D_state_dict'])

    def setup_from_opt(self, opt):
        # network
        self.net_G = NetG(opt).to(self.opt.device)
        self.net_D = NetD(opt).to(self.opt.device)

        self.opt_G = optim.Adam(self.net_G.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        # backbones
        self.net_F = NetF(self.opt).to(self.opt.device)
        self.net_I = NetI(self.opt).to(self.opt.device).eval()

        # configs
        self._set_requires_grad(self.net_F, False)
        self._set_requires_grad(self.net_I, False)

        # criterion
        self.crt_mse = nn.MSELoss()
        self.crt_l1 = nn.L1Loss()
        self.crt_bce = GANBCELoss()

        self.fixed_sketch = torch.tensor(0, device=self.opt.device).float()
        self.fixed_hint = torch.tensor(0, device=self.opt.device).float()
        self.fixed_sketch_feat = torch.tensor(0, device=self.opt.device).float()

    def evaluate_batch(self, i, batch_data) -> Tuple[float, Tensor, Tensor, Tensor]:
        real_cim, real_vim, real_sim = batch_data

        real_cim = real_cim.to(self.opt.device)
        real_vim = real_vim.to(self.opt.device)
        real_sim = real_sim.to(self.opt.device)

        mask = _mask_gen(self.opt, self.X)
        hint = torch.cat((real_vim * mask, mask), 1)
        with torch.no_grad():
            # get sketch feature
            feat_sim = self.net_I(real_sim).detach()

        fake_cim = self.net_G(real_sim, hint, feat_sim)
        loss = self.crt_l1(fake_cim, real_cim)

        return loss.item(), real_sim, real_cim, fake_cim

    def _sanity_check(self):
        print('Generating Sanity Checks')
        # see if model architecture is alright
        summary(
            self.net_G,
            torch.rand(self.opt.batch_size, 1, self.opt.image_size, self.opt.image_size).to(self.opt.device),
            torch.rand(self.opt.batch_size, 4, self.opt.image_size // 4, self.opt.image_size // 4).to(self.opt.device),
            torch.rand(self.opt.batch_size, 512, 32, 32).to(self.opt.device),
        )
        summary(
            self.net_D,
            torch.rand(self.opt.batch_size, 3, self.opt.image_size, self.opt.image_size).to(self.opt.device),
            torch.rand(self.opt.batch_size, 512, 32, 32).to(self.opt.device),
        )
        # get some data and see if it looks good
        i = 1
        for real_cim, _, real_sim in self.train_loader:
            for inp, tar in zip(real_sim, real_cim):
                plot_inp_tar(inp, tar, save_file=f'sanity-check-im-{i}.jpg')
                i += 1
                if i > 5:
                    break
            if i > 5:
                break
        print('Sanity Checks Generated')

    def pre_train(self):
        super().pre_train()
        self._sanity_check()

    def pre_epoch(self):
        super().pre_epoch()
        self.net_G_losses = []
        self.net_D_losses = []
        # self.grad_penalties = []

    def pre_batch(self, epoch, batch):
        super().pre_batch(epoch, batch)

    def train_batch(self, batch, batch_data):
        self.net_G.train()
        self.net_D.train()

        real_cim, real_vim, real_sim = batch_data

        real_cim = real_cim.to(self.opt.device)
        real_vim = real_vim.to(self.opt.device)
        real_sim = real_sim.to(self.opt.device)

        ############################
        # (1) Update D network
        ###########################
        self._set_requires_grad(self.net_D, True)
        self._set_requires_grad(self.net_G, False)
        self.net_D.zero_grad()

        mask = _mask_gen(self.opt, self.X)
        hint = torch.cat((real_vim * mask, mask), 1)

        with torch.no_grad():
            # train with fake
            # get sketch feature
            feat_sim = self.net_I(real_sim).detach()
            # generate fake color image
            fake_cim = self.net_G(real_sim, hint, feat_sim).detach()

        # ask discriminator to calculate loss
        errD_fake = self.net_D(fake_cim, feat_sim)
        errD_fake = self.crt_bce(errD_fake, False)

        # train with real
        errD_real = self.net_D(real_cim, feat_sim)
        errD_real = self.crt_bce(errD_real, True)

        errD = (errD_fake + errD_real) * 0.5
        errD.backward()

        # grad_pen = gradient_penalty(self.opt, self.net_D, real_cim, fake_cim)
        # grad_pen.backward()

        self.opt_D.step()

        ############################
        # (2) Update G network
        ############################

        self._set_requires_grad(self.net_D, False)
        self._set_requires_grad(self.net_G, True)
        self.net_G.zero_grad()

        # generate a fake image
        fake = self.net_G(real_sim, hint, feat_sim)

        # discriminator loss
        errG = self.net_D(fake, feat_sim)
        self.crt_mse(errG, real_cim)
        errG.backward(retain_graph=True)

        # content loss
        feat1 = self.net_F(fake)
        with torch.no_grad():
            feat2 = self.net_F(real_cim)

        content_loss = self.crt_mse(feat1, feat2)
        content_loss.backward()

        self.opt_G.step()

        return errG.item(), errD.item()

    def post_batch(self, epoch, batch, batch_out):
        super().post_batch(epoch, batch, batch_out)

        self.net_G_losses.append(batch_out[0])
        self.net_D_losses.append(batch_out[1])
        # self.grad_penalties.append(batch_out[2])

    def post_epoch(self, epoch):
        super().post_epoch(epoch)

        self.sch_G.step(epoch)
        self.sch_D.step(epoch)

    def post_train(self):
        super().post_train()

    def log_epoch(self, epoch):
        return super().log_epoch(epoch) + \
               f'[lr={self._get_lr(self.opt_G):.6f}] ' + \
               f'[G_loss={np.mean(self.net_G_losses):.4f}] ' + \
               f'[D_loss={np.mean(self.net_D_losses):.4f}] ' + \
               (f'[eval_loss={self.epoch_eval_loss:.4f}]' if self.this_epoch_evaluated else '')
        # f'[grad_pen={np.mean(self.grad_penalties):.4f}] ' + \

    def log_batch(self, batch):
        from_batch = self._get_last_batch(batch)
        return super().log_batch(batch) + \
               f'[G_loss={np.mean(self.net_G_losses[from_batch - 1:batch]):.4f}] ' + \
               f'[D_loss={np.mean(self.net_D_losses[from_batch - 1:batch]):.4f}] '
        # f'[grad_pen={np.mean(self.grad_penalties[from_batch - 1:batch]):.4f}] '
