from typing import Tuple

import numpy as np
import torch
import scipy.stats as stats
from torch import optim, nn, Tensor
from torch.autograd import grad

from ml.models.base import BaseTrainModel
from .alac_gan_partials import NetG, NetD, NetF, NetI, WarmUpLRScheduler
from ..options.alac_gan import AlacGANTrainOptions


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
        self.fixed_sketch = None
        self.fixed_hint = None
        self.fixed_sketch_feat = None

        # housekeeping
        self.net_G_losses = []
        self.net_D_losses = []
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
        # generator
        self.net_G = NetG().to(self.opt.device)
        self.net_D = NetD().to(self.opt.device)

        self.opt_G = optim.Adam(self.net_G.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        self._init_fixed()

    def evaluate_batch(self, i, batch_data) -> Tuple[float, Tensor, Tensor, Tensor]:
        real_cim, real_vim, real_sim = batch_data

        real_cim = real_cim.to(self.opt.device)
        real_vim = real_vim.to(self.opt.device)
        real_sim = real_sim.to(self.opt.device)

        mask = self._mask_gen()
        hint = torch.cat((real_vim * mask, mask), 1)
        with torch.no_grad():
            # get sketch feature
            feat_sim = self.net_I(real_sim).detach()

        fake_cim = self.net_G(real_sim, hint, feat_sim)
        loss = self.crt_mse(fake_cim, real_cim)

        return loss, real_sim, real_cim, fake_cim

    def _init_fixed(self):
        self.net_F = NetF(self.opt).to(self.opt.device)
        self.net_I = NetI(self.opt).to(self.opt.device).eval()

        self.sch_D = WarmUpLRScheduler(self.opt_D, base_lr=self.opt.lr, warmup_steps=0,
                                       warmup_lr=0, last_iter=self.opt.start_epoch - 1)
        self.sch_G = WarmUpLRScheduler(self.opt_G, base_lr=self.opt.lr, warmup_steps=0,
                                       warmup_lr=0, last_iter=self.opt.start_epoch - 1)

        self._set_requires_grad(self.net_F, False)
        self._set_requires_grad(self.net_I, False)

        self.crt_mse = nn.MSELoss()

        self.fixed_sketch = torch.tensor(0, device=self.opt.device).float()
        self.fixed_hint = torch.tensor(0, device=self.opt.device).float()
        self.fixed_sketch_feat = torch.tensor(0, device=self.opt.device).float()

    def pre_train(self):
        super().pre_train()

    def pre_epoch(self):
        super().pre_epoch()
        self.net_G_losses = []
        self.net_D_losses = []

    def pre_batch(self, epoch, batch):
        super().pre_batch(epoch, batch)

    def _mask_gen(self):
        image_size = 512
        maskS = image_size // 4

        mask1 = torch.cat(
            [torch.rand(1, 1, maskS, maskS).ge(self.X.rvs(1)[0]).float() for _ in range(self.opt.batch_size // 2)], 0)
        mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(self.opt.batch_size // 2)], 0)
        mask = torch.cat([mask1, mask2], 0)

        return mask.to(self.opt.device)

    def calc_gradient_penalty(self, real_data, fake_data, sketch_feat):
        alpha = torch.rand(self.opt.batch_size, 1, 1, 1, device=self.opt.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates.requires_grad = True

        disc_interpolates = self.net_D(interpolates, sketch_feat)

        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size(), device=self.opt.device), create_graph=True,
                         retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # config.gpW  gradient penalty weight
        return gradient_penalty

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

        # for p in self.net_G_losses.parameters():
        #     p.requires_grad = False  # to avoid computation ft_params

        self.net_D.zero_grad()

        mask = self._mask_gen()
        hint = torch.cat((real_vim * mask, mask), 1)

        with torch.no_grad():
            # train with fake
            # get sketch feature
            feat_sim = self.net_I(real_sim).detach()
            # generate fake color image
            fake_cim = self.net_G(real_sim, hint, feat_sim).detach()

        # ask discriminator to calculate loss
        errD_fake = self.net_D(fake_cim, feat_sim)
        errD_fake = errD_fake.mean(0).view(1)

        errD_fake.backward(retain_graph=True)  # backward on score on real

        # train with real
        errD_real = self.net_D(real_cim, feat_sim)
        errD_real = errD_real.mean(0).view(1)
        errD = errD_real - errD_fake

        errD_realer = -1 * errD_real + errD_real.pow(2) * 0.001  # config.drift

        errD_realer.backward(retain_graph=True)  # backward on score on real

        gradient_penalty = self.calc_gradient_penalty(real_cim, fake_cim, feat_sim)
        gradient_penalty.backward()

        self.opt_D.step()

        ############################
        # (2) Update G network
        ############################

        self._set_requires_grad(self.net_D, False)
        self._set_requires_grad(self.net_G, True)
        self.net_G.zero_grad()

        real_cim, real_vim, real_sim = batch_data
        real_cim = real_cim.to(self.opt.device)
        real_vim = real_vim.to(self.opt.device)
        real_sim = real_sim.to(self.opt.device)

        # if flag:  # fix samples
        #     mask = mask_gen()
        #     hint = torch.cat((real_vim * mask, mask), 1)
        #     with torch.no_grad():
        #         feat_sim = netI(real_sim).detach()
        #
        #     tb_logger.add_image('target imgs', vutils.make_grid(real_cim.mul(0.5).add(0.5), nrow=4))
        #     tb_logger.add_image('sketch imgs', vutils.make_grid(real_sim.mul(0.5).add(0.5), nrow=4))
        #     tb_logger.add_image('hint', vutils.make_grid((real_vim * mask).mul(0.5).add(0.5), nrow=4))
        #
        #     fixed_sketch.resize_as_(real_sim).copy_(real_sim)
        #     fixed_hint.resize_as_(hint).copy_(hint)
        #     fixed_sketch_feat.resize_as_(feat_sim).copy_(feat_sim)
        #
        #     flag -= 1

        # discriminator loss
        mask = self._mask_gen()
        hint = torch.cat((real_vim * mask, mask), 1)

        with torch.no_grad():
            feat_sim = self.net_I(real_sim).detach()

        fake = self.net_G(real_sim, hint, feat_sim)

        errd = self.net_D(fake, feat_sim)
        errG = errd.mean() * 0.0001 * -1  # config.advW 0.0001
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

    def log_batch(self, batch):
        from_batch = self._get_last_batch(batch)
        return super().log_batch(batch) + \
               f'[G_loss={np.mean(self.net_G_losses[from_batch - 1:batch]):.4f}] ' + \
               f'[D_loss={np.mean(self.net_D_losses[from_batch - 1:batch]):.4f}] '