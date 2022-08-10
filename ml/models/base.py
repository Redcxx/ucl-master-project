import os
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.logger import log
from ml.misc_utils import format_time, get_center_text
from ml.options.base import BaseOptions, BaseTrainOptions, BaseInferenceOptions
from ml.plot_utils import plt_model_sample
from ml.save_load import init_drive_and_folder, save_file, load_file


class BaseModel(ABC):

    def __init__(self, opt: BaseOptions):
        self.opt = opt
        init_drive_and_folder(self.opt)  # for saving and loading

    def load_checkpoint(self, tag=None, file_name=None):
        if file_name is None:
            assert tag is not None, "either file_name or tag must be supplied"
            file_name = f'{self.opt.run_id}_{tag}.ckpt'

        log(f'Loading checkpoint ... ', end='')
        load_file(self.opt, file_name)  # ensure exists locally, will raise error if not exists
        log(f'done: {file_name}')
        return torch.load(file_name)

    # abstract subclass should implement this method
    # concrete subclass should call this method in __init__
    @abstractmethod
    def setup(self):
        pass


class BaseInferenceModel(BaseModel, ABC):
    def __init__(self, opt: BaseInferenceOptions, inference_loader: DataLoader):
        super().__init__(opt)
        self.opt = opt

        self.inference_loader = inference_loader

    def setup(self):
        self.init_from_checkpoint(self.load_checkpoint(self.opt.inference_run_tag))

    @abstractmethod
    def init_from_checkpoint(self, checkpoint):
        pass

    @abstractmethod
    def inference_batch(self, i, batch_data) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    def inference(self):
        # create output directory, delete existing one
        save_path = Path(self.opt.output_images_path)
        save_path.mkdir(exist_ok=True, parents=True)

        iterator = enumerate(self.inference_loader)
        if self.opt.show_progress:
            iterator = tqdm(iterator, total=len(self.inference_loader), desc='Inference')
        for i, batch_data in iterator:

            inp_batch, tar_batch, out_batch = self.inference_batch(i, batch_data)

            for inp_im, tar_im, out_im in zip(inp_batch, tar_batch, out_batch):
                save_filename = os.path.join(self.opt.output_images_path, f'inference-{i}.png')
                plt_model_sample(inp_im, tar_im, out_im, save_file=save_filename)


class BaseTrainModel(BaseModel, ABC):

    def __init__(self, opt: BaseTrainOptions, train_loader: DataLoader, test_loader: DataLoader):
        super().__init__(opt)
        self.opt = opt

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.training_start_time = None
        self.last_batch_time = None
        self.last_epoch_time = None

        self.this_epoch_evaluated = False
        self.this_epoch_logged = False
        self.this_epoch_saved = False
        self.this_batch_logged = False

        self.epoch_eval_loss = None

    def setup(self):
        if self.opt.resume_ckpt_file is not None:
            log(f'Loading training checkpoint (resume id \"{self.opt.resume_ckpt_file}\" is not none) ...', end='')
            self.setup_from_train_checkpoint(self.load_checkpoint(file_name=self.opt.resume_ckpt_file))
            log('done')

        elif self.opt.start_epoch > 1:
            # try resume training
            log(f'Loading training checkpoint (start epoch {self.opt.start_epoch} > 1) ...', end='')
            self.setup_from_train_checkpoint(self.load_checkpoint(tag=f'{self.opt.start_epoch - 1}'))
            log('done')

        else:
            log('Initializing new model ...', end='')
            self.setup_from_opt(self.opt)
            log('done')

    @abstractmethod
    def get_checkpoint(self) -> Dict:
        pass

    def save_checkpoint(self, tag) -> None:
        log('Saving checkpoint ... ', end='')
        file_name = f'{self.opt.run_id}_{tag}.ckpt'
        torch.save(self.get_checkpoint(), file_name)
        save_file(self.opt, file_name, local=False)
        log(f'done: {file_name}')

    @abstractmethod
    def setup_from_train_checkpoint(self, checkpoint):
        pass

    @abstractmethod
    def setup_from_opt(self, opt):
        pass

    def _print_title(self):
        width = 100
        fill_char = '='
        option_text = 'OPTIONS'
        run_start_text = f'RUN ID: {self.opt.run_id}'

        log(fill_char * width)
        log(get_center_text(option_text, width, fill_char))
        log(fill_char * width)
        log(self.opt)
        log(fill_char * width)
        log(get_center_text(run_start_text, width, fill_char))
        log(fill_char * width)
        log(f'Train loader size: {len(self.train_loader)}')
        log(f'Test loader size: {len(self.test_loader)}')
        log(f'Using device {self.opt.device}')
        log(f'Run started at {format_time(self.training_start_time, datetime=True)}')
        log(fill_char * width)

    ###
    # Pre & Post method, subclass override to extend its functionalities
    ###

    @abstractmethod
    def pre_train(self):
        self.training_start_time = time.time()
        self.last_epoch_time = time.time()

        self._print_title()

        eval_path = Path(self.opt.eval_images_save_folder)
        if eval_path.is_dir():
            shutil.rmtree(str(eval_path))

    @abstractmethod
    def pre_epoch(self):
        self.last_batch_time = time.time()

    @abstractmethod
    def pre_batch(self, epoch, batch):
        pass

    @abstractmethod
    def train_batch(self, batch, batch_data):
        pass

    @abstractmethod
    def post_batch(self, epoch, batch, batch_out):
        if self.opt.batch_log_freq > 0 and (batch % self.opt.batch_log_freq == 0 or batch == 2):
            log(self.log_batch(batch))
            self.this_batch_logged = True
        else:
            self.this_batch_logged = False

    @abstractmethod
    def evaluate_batch(self, i, batch_data) -> Tuple[float, Tensor, Tensor, Tensor]:
        pass

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

            eval_loss, inp_batch, tar_batch, out_batch = self.evaluate_batch(i, batch_data)
            eval_losses.append(eval_loss)

            for inp_im, tar_im, out_im in zip(inp_batch, tar_batch, out_batch):
                if displayed_images < self.opt.eval_n_display_samples:
                    plt_model_sample(inp_im, tar_im, out_im, save_file=None)
                    displayed_images += 1

                if saved_images < self.opt.eval_n_save_samples:
                    save_filename = os.path.join(
                        self.opt.eval_images_save_folder,
                        f'epoch-{epoch}-eval-{saved_images}.png'
                    )
                    plt_model_sample(inp_im, tar_im, out_im, save_file=save_filename)
                    saved_images += 1

        return np.mean(eval_losses)

    @abstractmethod
    def post_epoch(self, epoch):

        if self.opt.save_freq > 0 and (epoch % self.opt.save_freq == 0 or epoch == self.opt.start_epoch):
            self.save_checkpoint(epoch)
            self.save_checkpoint('latest')

        if self.opt.log_freq > 0 and (epoch % self.opt.log_freq == 0 or epoch == self.opt.start_epoch):
            log(self.log_epoch(epoch))

        if self.opt.eval_freq > 0 and (epoch % self.opt.eval_freq == 0 or epoch == self.opt.start_epoch):
            self.epoch_eval_loss = self.evaluate(epoch)
        else:
            self.epoch_eval_loss = None

    @abstractmethod
    def post_train(self):
        training_end_time = time.time()
        log(f'Training finished at {format_time(training_end_time, datetime=True)}')
        log(f'Time taken: {format_time(training_end_time - self.training_start_time)}')
        self.save_checkpoint(tag='final')

    ##
    # Main training loop
    ##

    def train(self):
        self.pre_train()

        for epoch in range(self.opt.start_epoch, self.opt.end_epoch + 1):

            self.pre_epoch()

            for batch, batch_data in enumerate(self.train_loader, 1):
                self.pre_batch(epoch, batch)

                batch_out = self.train_batch(batch, batch_data)

                self.post_batch(epoch, batch, batch_out)

            self.post_epoch(epoch)

        self.post_train()

    ###
    # Logging
    ###

    def log_epoch(self, epoch):
        curr_time = time.time()
        time_remain = (curr_time - self.training_start_time) / epoch * (self.opt.end_epoch - epoch)
        text = f'[epoch={epoch}] ' + \
               f'[curr_time={format_time(curr_time, datetime=True)}] ' + \
               f'[train_time={format_time(curr_time - self.training_start_time)}] ' + \
               f'[epoch_time={format_time(curr_time - self.last_epoch_time)}] ' + \
               f'[time_remain={format_time(time_remain)}] ' + \
               (f'[eval_loss={self.epoch_eval_loss:.4f}]' if self.epoch_eval_loss is not None else '')

        self.last_epoch_time = curr_time
        return text

    def _get_last_batch(self, this_batch):
        return max(1, this_batch - self.opt.batch_log_freq)

    def log_batch(self, batch):
        curr_time = time.time()
        from_batch = self._get_last_batch(batch)

        text = f'[batch={from_batch}-{batch}] ' + \
               f'[train_time={format_time(curr_time - self.training_start_time)}] ' + \
               f'[batch_time={format_time(curr_time - self.last_batch_time)}] '

        self.last_batch_time = curr_time
        return text

    ###
    # Miscellaneous
    ###

    @staticmethod
    def _get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    @staticmethod
    def _set_requires_grad(net, requires_grad):
        for param in net.parameters():
            param.requires_grad = requires_grad
        return net
