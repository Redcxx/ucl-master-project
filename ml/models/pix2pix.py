import os
from pathlib import Path

import numpy as np
import torch
from torch import optim, nn
from torchsummaryX import summary
from tqdm import tqdm

from ml.models.base import BaseTrainModel
from .alac_gan_partials import NetF
from .pix2pix_partials import Generator, Discriminator
from ml.models.criterion.GANBCELoss import GANBCELoss
from ..logger import log
from ..options.pix2pix import Pix2pixTrainOptions
from ..plot_utils import plt_input_target, plt_model_sample, plt_horizontals


class Pix2pixTrainModel(BaseTrainModel):

    def __init__(self, opt: Pix2pixTrainOptions, train_loader, test_loader):
        super().__init__(opt, train_loader, test_loader)

        self.opt = opt

        # network
        self.net_G = None
        self.net_D = None
        self.net_F = None

        # optimizer
        self.opt_G = None
        self.opt_D = None

        # scheduler
        self.sch_G = None
        self.sch_D = None

        # loss
        self.crt_gan = None
        self.crt_l1 = None

        # housekeeping
        self.net_G_gan_losses = []
        self.net_G_l1_losses = []
        self.net_D_losses = []

        self.setup()

    def _init_fixed(self):
        self.crt_gan = GANBCELoss().to(self.opt.device)
        self.crt_l1 = nn.L1Loss(reduction='none')
        self.sch_G = optim.lr_scheduler.LambdaLR(self.opt_G, lr_lambda=self._decay_rule)
        self.sch_D = optim.lr_scheduler.LambdaLR(self.opt_D, lr_lambda=self._decay_rule)
        self.net_F = NetF(self.opt).to(self.opt.device)

    def setup_from_train_checkpoint(self, checkpoint):
        _prev_opt, self.net_G, self.net_D, self.opt_G, self.opt_D = checkpoint
        self._init_fixed()

    def setup_from_opt(self, opt):
        # generator
        self.net_G = Generator(opt).to(opt.device).apply(self._gaussian_init_weight)
        self.net_D = Discriminator(opt).to(opt.device).apply(self._gaussian_init_weight)
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=opt.lr,
                                betas=(opt.optimizer_beta1, opt.optimizer_beta2),
                                weight_decay=opt.weight_decay)
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=opt.lr,
                                betas=(opt.optimizer_beta1, opt.optimizer_beta2),
                                weight_decay=opt.weight_decay)
        self._init_fixed()

    def _sanity_check(self):
        log('Generating Sanity Checks')
        self.net_G.to(self.opt.device).eval()
        self.net_D.to(self.opt.device).eval()
        # see if model architecture is alright
        summary(self.net_G, torch.rand(self.opt.batch_size, self.opt.generator_config['in_channels'],
                                       self.opt.image_size, self.opt.image_size)
                .to(self.opt.device))
        summary(self.net_D, torch.rand(self.opt.batch_size, self.opt.generator_config['in_channels'] * 2,
                                       self.opt.image_size, self.opt.image_size)
                .to(self.opt.device))
        # get some data and see if it looks good
        i = 0
        for inp_batch, tar_batch, _weight_map in self.train_loader:
            for inp, tar in zip(inp_batch, tar_batch):
                plt_horizontals(
                    [inp, tar],
                    titles=['input', 'target'],
                    un_normalize=True,
                    grayscale=True,
                    figsize=(2, 1),
                    save_file=f'sanity-check-im-{i}.jpg'
                )
                i += 1
                if i > 10:
                    break
            if i > 10:
                break
        log('Sanity Checks Generated')

    def pre_train(self):
        super().pre_train()
        self._sanity_check()

    def pre_epoch(self):
        super().pre_epoch()
        self.net_G_gan_losses = []
        self.net_G_l1_losses = []
        self.net_D_losses = []

    def pre_batch(self, epoch, batch):
        super().pre_batch(epoch, batch)

    def train_batch(self, batch, batch_data):
        self.net_G = self.net_G.train().to(self.opt.device)
        self.net_D = self.net_D.train().to(self.opt.device)

        real_A, real_B, weight_map = batch_data
        real_A, real_B = real_A.to(self.opt.device), real_B.to(self.opt.device)

        weight_map = weight_map.to(self.opt.device) * 10 + 1

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
        loss_D_fake = self.crt_gan(pred_fake, False)

        # discriminate real image
        real_AB = torch.cat((real_A, real_B), dim=1)
        pred_real = self.net_D(real_AB)
        loss_D_real = self.crt_gan(pred_real, True)

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
        loss_G_fake = self.crt_gan(pred_fake, True)

        # l1 loss between generated and real image for more accurate output
        pixel_wise_loss = self.crt_l1(fake_B, real_B) * self.opt.l1_lambda
        loss_G_l1 = torch.mean(pixel_wise_loss * (weight_map * 10 + 1) * self.opt.l1_lambda)

        # content loss
        # fake_feat = self.net_F(fake_AB)
        # with torch.no_grad():
        #     real_feat = self.net_F(real_AB)
        # content_loss = self.crt_l1(fake_feat, real_feat)

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

    def post_train(self):
        super().post_train()

    def evaluate_batch(self, i, batch_data):
        self.net_G = self.net_G.eval().to(self.opt.device)
        self.net_D = self.net_D.eval().to(self.opt.device)

        inp, tar, _weight_map = batch_data
        inp, tar = inp.to(self.opt.device), tar.to(self.opt.device)

        out = self.net_G(inp)
        loss = self.crt_l1(out, tar).mean()

        return loss.item(), inp, tar, out, (out > 0.5) * 1.0

    def evaluate(self, epoch):
        if self.opt.eval_n_save_samples > 0:
            # create directory, delete existing one
            Path(self.opt.eval_images_save_folder).mkdir(exist_ok=True, parents=True)

        eval_losses = []
        displayed_images = 0
        saved_images = 0

        iterator = enumerate(self.test_loader)
        if self.opt.eval_show_progress:
            iterator = tqdm(iterator, total=len(self.test_loader), desc='Evaluate')
        for i, batch_data in iterator:

            eval_loss, inp_batch, tar_batch, out_batch, thresholds = self.evaluate_batch(i, batch_data)
            eval_losses.append(eval_loss)

            for inp_im, tar_im, out_im, threshold in zip(inp_batch, tar_batch, out_batch, thresholds):
                if displayed_images < self.opt.eval_n_display_samples:
                    plt_model_sample(inp_im, tar_im, out_im, save_file=None)

                    plt_horizontals(
                        [inp_im, tar_im, out_im, threshold],
                        titles=['input', 'target', 'output', 'threshold'],
                        figsize=(4, 1),
                        dpi=512,
                        un_normalize=True,
                        grayscale=True,
                        save_file=None
                    )
                    displayed_images += 1

                if saved_images < self.opt.eval_n_save_samples:
                    save_filename = os.path.join(
                        self.opt.eval_images_save_folder,
                        f'epoch-{epoch}-eval-{saved_images}.png'
                    )
                    plt_horizontals(
                        [inp_im, tar_im, out_im, threshold],
                        titles=['input', 'target', 'output', 'threshold'],
                        figsize=(4, 1),
                        dpi=512,
                        un_normalize=True,
                        grayscale=True,
                        save_file=save_filename
                    )
                    saved_images += 1

        return np.mean(eval_losses)

    def log_epoch(self, epoch):
        return super().log_epoch(epoch) + \
               f'[lr={self._get_lr(self.opt_G):.6f}] ' + \
               f'[G_l1_loss={np.mean(self.net_G_l1_losses):.4f}] ' + \
               f'[G_GAN_loss={np.mean(self.net_G_gan_losses):.4f}] ' + \
               f'[D_loss={np.mean(self.net_D_losses):.4f}] '

    def log_batch(self, batch):
        from_batch = self._get_last_batch(batch)
        return super().log_batch(batch) + \
               f'[G_l1_loss={np.mean(self.net_G_l1_losses[from_batch - 1:batch]):.4f}] ' + \
               f'[G_GAN_loss={np.mean(self.net_G_gan_losses[from_batch - 1:batch]):.4f}] ' + \
               f'[D_loss={np.mean(self.net_D_losses[from_batch - 1:batch]):.4f}] '

    def get_checkpoint(self):
        return {
            'net_G_state_dict': self.net_G.state_dict(),
            'net_D_state_dict': self.net_D.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
            'opt': self.opt.saved_dict,
        }

    def load_checkpoint(self, tag, file_name=None):
        checkpoint = super().load_checkpoint(tag, file_name)

        opt_dict = checkpoint['opt']
        loaded_opt = Pix2pixTrainOptions()
        loaded_opt.load_saved_dict(opt_dict)

        net_G = Generator(loaded_opt).to(self.opt.device)
        net_D = Discriminator(loaded_opt).to(self.opt.device)
        net_G.load_state_dict(checkpoint['net_G_state_dict'])
        net_D.load_state_dict(checkpoint['net_D_state_dict'])

        opt_G = optim.Adam(net_G.parameters(), lr=loaded_opt.lr,
                           betas=(loaded_opt.optimizer_beta1, loaded_opt.optimizer_beta2))
        opt_D = optim.Adam(net_D.parameters(), lr=loaded_opt.lr,
                           betas=(loaded_opt.optimizer_beta1, loaded_opt.optimizer_beta2))
        opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        opt_D.load_state_dict(checkpoint['opt_D_state_dict'])

        log('Successfully created network with loaded checkpoint')
        return loaded_opt, net_G, net_D, opt_G, opt_D

    def _gaussian_init_weight(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1
                                     or classname.find('BatchNorm2d') != -1):
            nn.init.normal_(m.weight.data, 0.0, self.opt.init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def _decay_rule(self, epoch):
        return 1.0 - max(0, epoch + self.opt.start_epoch - (self.opt.end_epoch - self.opt.decay_epochs)) / float(
            self.opt.decay_epochs + 1)
