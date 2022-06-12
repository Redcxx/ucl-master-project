import time

import numpy as np
import torch
from torch import optim, nn

from .base_model import BaseModel
from .partials import Generator, Discriminator
from ..criterion.GANBCELoss import GANBCELoss
from ..save_load import format_time


class Pix2pixModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)

        # generator
        self.net_G = Generator(self.opt)
        self.net_G.apply(self._gaussian_init_weight)
        self.net_G.to(opt.device)
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=opt.lr,
                                betas=(opt.optimizer_beta1, opt.optimizer_beta2),
                                weight_decay=opt.weight_decay)
        self.sch_G = optim.lr_scheduler.LambdaLR(self.opt_G, lr_lambda=self.decay_rule)

        # discriminator
        self.net_D = Discriminator(self.opt)
        self.net_D.apply(self._gaussian_init_weight)
        self.net_D.to(opt.device)
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=opt.lr,
                                betas=(opt.optimizer_beta1, opt.optimizer_beta2),
                                weight_decay=opt.weight_decay)
        self.sch_D = optim.lr_scheduler.LambdaLR(self.opt_D, lr_lambda=self.decay_rule)

        # loss
        self.criterion_gan = GANBCELoss().to(opt.device)
        self.criterion_l1 = nn.L1Loss()

        # housekeeping
        self.net_G_gan_losses = []
        self.net_G_l1_losses = []
        self.net_D_losses = []
        self.last_batch_time = None
        self.epoch_eval_loss = None
        self.epoch_start_time = None
        self.training_start_time = None

    def pre_train(self):
        self.training_start_time = time.time()

    def post_train(self):
        pass

    def pre_epoch(self):
        self.net_G_gan_losses = []
        self.net_G_l1_losses = []
        self.net_D_losses = []
        self.last_batch_time = time.time()
        self.epoch_start_time = time.time()

        self.net_G = self.net_G.train().to(self.opt.device)
        self.net_D = self.net_D.train().to(self.opt.device)

    def post_epoch(self):
        self.sch_G.step()
        self.sch_D.step()

    def pre_batch(self, batch):
        pass

    def post_batch(self, batch_out):
        self.net_G_gan_losses.append(batch_out[0])
        self.net_G_l1_losses.append(batch_out[1])
        self.net_D_losses.append(batch_out[2])

    def train_batch(self, batch, batch_data):

        real_A, real_B = batch_data
        real_A, real_B = real_A.to(self.opt.device), real_B.to(self.opt.device)

        # forward pass
        # generate fake image using generator
        fake_B = self.net_G(real_A)

        ###
        # DISCRIMINATOR
        ###
        self.set_requires_grad(self.net_D, True)
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
        self.set_requires_grad(self.net_D, False)
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

    def evaluate(self):
        self.epoch_eval_loss = None

    def log_epoch(self, epoch):
        curr_time = time.time()
        print(
            f'[epoch={epoch}] ' +
            f'[lr={self.get_lr(self.opt_G):.6f}] ' +
            f'[G_l1_loss={np.mean(self.net_G_l1_losses):.4f}] ' +
            f'[G_GAN_loss={np.mean(self.net_G_gan_losses):.4f}] ' +
            f'[D_loss={np.mean(self.net_D_losses):.4f}] ' +
            f'[epoch_time={format_time(curr_time - self.epoch_start_time)}] ' +
            f'[train_time={format_time(curr_time - self.training_start_time)}] ' +
            (f'[eval_loss={self.epoch_eval_loss:.4f}]' if self.epoch_eval_loss is not None else '')
        )

    def log_batch(self, batch):
        from_batch = max(1, batch - self.opt.batch_log_freq)
        curr_time = time.time()
        print(
            f'[batch={from_batch}-{batch}] ' +
            f'[G_l1_loss={np.mean(self.net_G_l1_losses[from_batch - 1:batch]):.4f}] ' +
            f'[G_GAN_loss={np.mean(self.net_G_gan_losses[from_batch - 1:batch]):.4f}] ' +
            f'[D_loss={np.mean(self.net_D_losses[from_batch - 1:batch]):.4f}] ' +
            f'[batch_time={format_time(curr_time - self.last_batch_time)}] ' +
            f'[train_time={format_time(curr_time - self.training_start_time)}]'
        )
        self.last_batch_time = curr_time

    def save_checkpoint(self, epoch):
        pass

    def load_checkpoint(self, epoch):
        pass
