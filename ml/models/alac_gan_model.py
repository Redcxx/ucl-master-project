import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from scipy.stats import stats
from torch import optim, nn
from torch.autograd import grad
from tqdm import tqdm

from .alac_gan_partials import NetG, NetD, NetF, NetI, WarmUpLRScheduler
from ml.base_model import BaseModel
from ..base_options import BaseTrainOptions, BaseInferenceOptions
from ..plot_utils import plot_inp_tar_out


class AlacGANModel(BaseModel):

    def __init__(self, opt: Union[BaseTrainOptions, BaseInferenceOptions]):
        super().__init__(opt)

        # set later in pre-train, cannot initialize here because its init calls step()
        # which may requires attribute from TrainOptions that does not exist in InferenceOptions

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

    def _init_fixed(self):
        self.net_F = NetF(self.opt).to(self.opt.device)
        self.net_I = NetI(self.opt).to(self.opt.device).eval()

        self.sch_D = WarmUpLRScheduler(self.opt_D, base_lr=0.0001, warmup_steps=0,
                                       warmup_lr=0, last_iter=self.opt.start_epoch - 1)
        self.sch_G = WarmUpLRScheduler(self.opt_G, base_lr=0.0001, warmup_steps=0,
                                       warmup_lr=0, last_iter=self.opt.start_epoch - 1)

        self._set_requires_grad(self.net_F, False)
        self.crt_mse = nn.MSELoss()

        self.fixed_sketch = torch.tensor(0, device=self.opt.device).float()
        self.fixed_hint = torch.tensor(0, device=self.opt.device).float()
        self.fixed_sketch_feat = torch.tensor(0, device=self.opt.device).float()

    def _init_from_inference_checkpoint(self, checkpoint):
        self.opt, self.net_G, self.net_D, self.opt_G, self.opt_D = checkpoint
        self._init_fixed()

    def _init_from_train_checkpoint(self, checkpoint):
        _prev_opt, self.net_G, self.net_D, self.opt_G, self.opt_D = checkpoint
        self._init_fixed()

    def _init_from_opt(self):
        # generator
        self.net_G = NetG(self.opt).to(self.opt.device)
        self.net_D = NetD(self.opt).to(self.opt.device)

        self.opt_G = optim.Adam(self.net_G.parameters(), lr=self.opt.lr, betas=(0.5, 0.9))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=self.opt.lr, betas=(0.5, 0.9))

        self._init_fixed()

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
        # batch_time.update(time.time() - end)

        return errG.item(), errD.item()

    def post_batch(self, epoch, batch, batch_out):
        super().post_batch(epoch, batch, batch_out)

        self.net_G_losses.append(batch_out[0])
        self.net_D_losses.append(batch_out[1])

    def post_epoch(self, epoch):
        super().post_epoch(epoch)

        self.sch_G.step(epoch)
        self.sch_D.step(epoch)

    def evaluate(self, epoch, progress=False):
        self.net_G.eval()
        if self.opt.evaluate_save_images:
            Path(self.opt.evaluate_images_save_folder).mkdir(exist_ok=True, parents=True)

        eval_losses = []
        iterator = enumerate(self.opt.test_loader)
        if progress:
            iterator = tqdm(iterator, total=len(self.opt.test_loader))
        for i, (real_cim, real_vim, real_sim) in iterator:
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
            eval_losses.append(loss.item())

            if i < self.opt.evaluate_n_display_samples:
                plot_inp_tar_out(real_sim, real_cim, fake_cim, save_file=None)

            if self.opt.evaluate_save_images:
                save_filename = os.path.join(self.opt.evaluate_images_save_folder, f'epoch-{epoch}-eval-{i}.png')
                plot_inp_tar_out(real_sim, real_cim, fake_cim, save_file=save_filename)

        self.epoch_eval_loss = np.mean(eval_losses)

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

    def get_checkpoint(self):
        return {
            'net_G_state_dict': self.net_G.state_dict(),
            'net_D_state_dict': self.net_D.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
            'opt': self.opt.__dict__,
            'opt_name': self.opt.__class__.__name__
        }

    def load_checkpoint(self, tag):
        checkpoint = super().load_checkpoint(tag)

        opt_dict = checkpoint['opt']
        opt_name = checkpoint['opt_name']
        loaded_opt = BaseInferenceOptions(**opt_dict) if opt_name == 'InferenceOptions' else BaseTrainOptions(**opt_dict)

        net_G = NetG(loaded_opt).to(self.opt.device)
        net_D = NetD(loaded_opt).to(self.opt.device)
        net_G.load_state_dict(checkpoint['net_G_state_dict'])
        net_D.load_state_dict(checkpoint['net_D_state_dict'])

        opt_G = optim.Adam(net_G.parameters(), lr=loaded_opt.lr, betas=(0.5, 0.9))
        opt_D = optim.Adam(net_D.parameters(), lr=loaded_opt.lr, betas=(0.5, 0.9))
        opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        opt_D.load_state_dict(checkpoint['opt_D_state_dict'])

        print('Successfully created network with loaded checkpoint')
        return loaded_opt, net_G, net_D, opt_G, opt_D
