import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm

from .base_model import BaseModel
from .partials import Generator, Discriminator
from ..criterion.GANBCELoss import GANBCELoss
from ..plot_utils import plot_inp_tar_out
from ..options import TrainOptions


class Pix2pixModel(BaseModel):

    def __init__(self, opt: TrainOptions):
        super().__init__(opt)

        if opt.start_epoch > 1:
            # try resume training
            _prev_opt, self.net_G, self.net_D, self.opt_G, self.opt_D = self.load_checkpoint(
                tag=f'{opt.start_epoch - 1}')
            print('Checkpoint Loaded')
        else:
            # generator
            self.net_G = Generator(self.opt)
            self.net_D = Discriminator(self.opt)
            self.net_G.apply(self._gaussian_init_weight)
            self.net_D.apply(self._gaussian_init_weight)
            self.net_G.to(opt.device)
            self.net_D.to(opt.device)
            self.opt_G = optim.Adam(self.net_G.parameters(), lr=opt.lr,
                                    betas=(opt.optimizer_beta1, opt.optimizer_beta2),
                                    weight_decay=opt.weight_decay)

            # discriminator
            self.opt_D = optim.Adam(self.net_D.parameters(), lr=opt.lr,
                                    betas=(opt.optimizer_beta1, opt.optimizer_beta2),
                                    weight_decay=opt.weight_decay)

        self.sch_G = optim.lr_scheduler.LambdaLR(self.opt_G, lr_lambda=self._decay_rule)
        self.sch_D = optim.lr_scheduler.LambdaLR(self.opt_D, lr_lambda=self._decay_rule)

        # loss
        self.criterion_gan = GANBCELoss()
        self.criterion_l1 = nn.L1Loss()

        # housekeeping
        self.net_G_gan_losses = []
        self.net_G_l1_losses = []
        self.net_D_losses = []
        self.epoch_eval_loss = None

    def pre_train(self):
        super().pre_train()
        self.net_G = self.net_G.train().to(self.opt.device)
        self.net_D = self.net_D.train().to(self.opt.device)
        self.criterion_gan = self.criterion_gan.to(self.opt.device)

    def pre_epoch(self):
        super().pre_epoch()
        self.net_G_gan_losses = []
        self.net_G_l1_losses = []
        self.net_D_losses = []

    def pre_batch(self, epoch, batch):
        super().pre_batch(epoch, batch)

    def train_batch(self, batch, batch_data):

        self.net_G.train()
        self.net_D.train()

        real_A, real_B = batch_data
        real_A, real_B = real_A.to(self.opt.device), real_B.to(self.opt.device)

        # forward pass
        # generate fake image using generator
        fake_B = self.net_G(real_A)

        ###
        # DISCRIMINATOR
        ###
        self._set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()

        # discriminate fake image
        fake_AB = torch.cat((real_A, fake_B), dim=1)  # conditionalGAN takes both real and fake image
        pred_fake = self.net_D(fake_AB.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)

        # discriminate real image
        real_AB = torch.cat((real_A, real_B), dim=1)
        pred_real = self.net_D(real_AB)
        loss_D_real = self.criterion_gan(pred_real, True)

        # backward & optimize
        loss_D = (loss_D_fake + loss_D_real) * self.opt.d_loss_factor
        loss_D.backward()
        self.opt_D.step()

        ###
        # GENERATOR
        ###
        self._set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()

        # generator should fool the discriminator
        fake_AB = torch.cat((real_A, fake_B), dim=1)
        pred_fake = self.net_D(fake_AB)
        loss_G_fake = self.criterion_gan(pred_fake, True)

        # l1 loss between generated and real image for more accurate output
        loss_G_l1 = self.criterion_l1(fake_B, real_B) * self.opt.l1_lambda

        # backward & optimize
        loss_G = loss_G_fake + loss_G_l1
        loss_G.backward()
        self.opt_G.step()

        return loss_G_fake.item(), loss_G_l1.item(), loss_D.item()

    def post_batch(self, epoch, batch, batch_out):
        super().post_batch(epoch, batch, batch_out)

        self.net_G_gan_losses.append(batch_out[0])
        self.net_G_l1_losses.append(batch_out[1])
        self.net_D_losses.append(batch_out[2])

    def post_epoch(self, epoch):
        super().post_epoch(epoch)

        self.sch_G.step()
        self.sch_D.step()

    def evaluate(self, epoch, progress=False):
        self.net_G.eval()
        self.net_D.eval()
        if self.opt.save_inferred_images:
            Path(self.opt.inference_save_folder).mkdir(exist_ok=True, parents=True)

        eval_losses = []
        iterator = enumerate(self.opt.test_loader)
        if progress:
            iterator = tqdm(iterator, total=len(self.opt.test_loader))
        for i, (inp, tar) in iterator:
            inp, tar = inp.to(self.opt.device), tar.to(self.opt.device)

            out = self.net_G(inp)
            loss = self.criterion_l1(out, tar)
            eval_losses.append(loss.item())

            if i < self.opt.n_infer_display_samples:
                plot_inp_tar_out(inp, tar, out, save_file=None)

            if self.opt.save_inferred_images:
                save_filename = os.path.join(self.opt.inference_save_folder, f'epoch-{epoch}-eval-{i}.png')
                plot_inp_tar_out(inp, tar, out, save_file=save_filename)

        self.epoch_eval_loss = np.mean(eval_losses)

    def log_epoch(self, epoch):
        return super().log_epoch(epoch) + \
               f'[lr={self._get_lr(self.opt_G):.6f}] ' + \
               f'[G_l1_loss={np.mean(self.net_G_l1_losses):.4f}] ' + \
               f'[G_GAN_loss={np.mean(self.net_G_gan_losses):.4f}] ' + \
               f'[D_loss={np.mean(self.net_D_losses):.4f}] ' + \
               (f'[eval_loss={self.epoch_eval_loss:.4f}]' if self.this_epoch_evaluated else '')

    def log_batch(self, batch):
        from_batch = self._get_last_batch(batch)
        return super().log_batch(batch) + \
               f'[G_l1_loss={np.mean(self.net_G_l1_losses[from_batch - 1:batch]):.4f}] ' + \
               f'[G_GAN_loss={np.mean(self.net_G_gan_losses[from_batch - 1:batch]):.4f}] ' + \
               f'[D_loss={np.mean(self.net_D_losses[from_batch - 1:batch]):.4f}] '

    def _get_checkpoint(self):
        return {
            'net_G_state_dict': self.net_G.state_dict(),
            'net_D_state_dict': self.net_D.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
            'opt': self.opt
        }

    def load_checkpoint(self, tag):
        checkpoint = super().load_checkpoint(tag)

        loaded_opt = checkpoint['opt']

        net_G = Generator(loaded_opt).to(self.opt.device)
        net_D = Discriminator(loaded_opt).to(self.opt.device)
        net_G.load_state_dict(checkpoint['net_G_state_dict'])
        net_D.load_state_dict(checkpoint['net_D_state_dict'])

        optimizer_G = optim.Adam(net_G.parameters(), lr=loaded_opt.lr,
                                 betas=(loaded_opt.optimizer_beta1, loaded_opt.optimizer_beta2))
        optimizer_D = optim.Adam(net_D.parameters(), lr=loaded_opt.lr,
                                 betas=(loaded_opt.optimizer_beta1, loaded_opt.optimizer_beta2))
        optimizer_G.load_state_dict(checkpoint['opt_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['opt_D_state_dict'])

        print('Successfully created network with loaded checkpoint')
        return loaded_opt, net_G, net_D, optimizer_G, optimizer_D
