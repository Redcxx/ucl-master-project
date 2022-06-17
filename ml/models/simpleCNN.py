import os
from pathlib import Path

import numpy as np
from torch import optim, nn

from .base_model import BaseModel
from ..plot_utils import plot_inp_tar_out
from ..session import SessionOptions


class SimpleCNN(BaseModel):

    def __init__(self, opt: SessionOptions):
        super().__init__(opt)

        if opt.start_epoch > 1:
            # try resume training
            _prev_opt, self.network, self.optimizer = self.load_checkpoint(tag=f'{opt.start_epoch - 1}')
        else:
            self.network = ...
            self.optimizer = optim.Adam(self.network.parameters(), lr=opt.lr,
                                        betas=(opt.optimizer_beta1, opt.optimizer_beta2))
            self.network.apply(self._gaussian_init_weight)
            self.network.to(opt.device)

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self._decay_rule)

        # loss
        self.criterion = nn.L1Loss()

        # housekeeping
        self.losses = []
        self.epoch_eval_loss = None

    def pre_train(self):
        super().pre_train()
        self.network = self.network.train().to(self.opt.device)

    def pre_epoch(self):
        super().pre_epoch()
        self.losses = []

    def pre_batch(self, epoch, batch):
        super().pre_batch(epoch, batch)

    def train_batch(self, batch, batch_data):

        self.network.train()

        inp, tar = batch_data
        inp, tar = inp.to(self.opt.device), tar.to(self.opt.device)

        out = self.network(inp)
        loss = self.criterion(out, tar)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def post_batch(self, epoch, batch, batch_out):
        super().post_batch(epoch, batch, batch_out)

        self.losses.append(batch_out)

    def post_epoch(self, epoch):
        super().post_epoch(epoch)

        self.scheduler.step()

    def evaluate(self, epoch):
        self.network.eval()
        if self.opt.save_eval_images:
            Path(self.opt.eval_sample_folder).mkdir(exist_ok=True, parents=True)

        eval_losses = []
        for i, (inp, tar) in enumerate(self.opt.test_loader):
            inp, tar = inp.to(self.opt.device), tar.to(self.opt.device)

            out = self.network(inp)
            loss = self.criterion(out, tar)
            eval_losses.append(loss.item())

            if i < self.opt.n_eval_display_samples:
                if not self.opt.save_eval_images:
                    plot_inp_tar_out(inp, tar, out, save_file=None)
                else:
                    save_filename = os.path.join(self.opt.eval_sample_folder, f'epoch-{epoch}-eval-{i}.png')
                    plot_inp_tar_out(inp, tar, out, save_file=save_filename)

        self.epoch_eval_loss = np.mean(eval_losses)

    def log_epoch(self, epoch):
        return super().log_epoch(epoch) + \
               f'[lr={self._get_lr(self.optimizer):.6f}] ' + \
               f'[loss={np.mean(self.losses):.4f}] ' + \
               (f'[eval_loss={self.epoch_eval_loss:.4f}]' if self.this_epoch_evaluated else '')

    def log_batch(self, batch):
        from_batch = self._get_last_batch(batch)
        return super().log_batch(batch) + f'[loss={np.mean(self.losses[from_batch - 1:batch]):.4f}] '

    def _get_checkpoint(self):
        return {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'opt': self.opt
        }

    def load_checkpoint(self, tag):
        checkpoint = super().load_checkpoint(tag)

        loaded_opt = checkpoint['opt']

        network = ...  # (loaded_opt).to(self.opt.device)
        network.load_state_dict(checkpoint['network'])

        optimizer = optim.Adam(network.parameters(), lr=loaded_opt.lr,
                               betas=(loaded_opt.optimizer_beta1, loaded_opt.optimizer_beta2))
        optimizer.load_state_dict(checkpoint['optimizer'])

        print('Successfully created network with loaded checkpoint')
        return loaded_opt, network, optimizer
