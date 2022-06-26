import time
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Dict

import torch
from torch.utils.data import DataLoader

from ml.base_options import BaseOptions, BaseTrainOptions, BaseInferenceOptions
from ml.misc_utils import format_time, get_center_text
from ml.save_load import init_drive_and_folder, save_file, load_file


class BaseModel(ABC):

    def __init__(self, opt: BaseOptions):
        self.opt = opt
        init_drive_and_folder(self.opt)  # for saving and loading

    # def pre_train(self):
    #     shutil.rmtree(self.opt.inference_save_folder, ignore_errors=True)

    @abstractmethod
    def get_checkpoint(self) -> Dict:
        pass

    def save_checkpoint(self, tag) -> None:
        print('Saving checkpoint ... ', end='')
        file_name = f'{self.opt.run_id}_{tag}.ckpt'
        torch.save(self.get_checkpoint(), file_name)
        save_file(self.opt, file_name, local=False)
        print(f'done: {file_name}')

    def load_checkpoint(self, tag):
        print('Loading checkpoint ... ', end='')
        file_name = f'{self.opt.run_id}_{tag}.ckpt'
        load_file(self.opt, file_name)  # ensure exists locally, will raise error if not exists
        print(f'done: {file_name}')
        return torch.load(file_name)


class BaseInferenceModel(BaseModel, ABC):
    def __init__(self, opt: BaseInferenceOptions):
        super().__init__(opt)
        self.opt = opt

        self.init_from_checkpoint(self.load_checkpoint('latest'))
        self.inference_loader = self.create_inference_loader()

    @abstractmethod
    def create_inference_loader(self):
        pass

    @abstractmethod
    def init_from_checkpoint(self, checkpoint):
        pass

    @abstractmethod
    def inference(self):
        pass


class BaseTrainModel(BaseModel, ABC):

    def __init__(self, opt: BaseTrainOptions, train_loader: DataLoader, test_loader: DataLoader):
        super().__init__(opt)
        self.opt = opt

        if self.opt.start_epoch > 1:
            # try resume training
            print('Loading training checkpoint ...', end='')
            self.init_from_train_checkpoint(self.load_checkpoint(tag=f'{opt.start_epoch - 1}'))
            print('done')

        else:
            print('Initializing new model ...', end='')
            self.init_from_opt()
            print('done')

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.training_start_time = None
        self.last_batch_time = None
        self.last_epoch_time = None

        self.this_epoch_evaluated = False
        self.this_epoch_logged = False
        self.this_epoch_saved = False
        self.this_batch_logged = False

    @abstractmethod
    def init_from_train_checkpoint(self, checkpoint):
        pass

    @abstractmethod
    def init_from_opt(self):
        pass

    def _print_title(self):
        width = 100
        fill_char = '='
        option_text = 'OPTIONS'
        run_start_text = f'RUN ID: {self.opt.run_id}'

        print(fill_char * width)
        print(get_center_text(option_text, width, fill_char))
        print(fill_char * width)
        print(self.opt)
        print(fill_char * width)
        print(get_center_text(run_start_text, width, fill_char))
        print(fill_char * width)
        print(f'Train loader size: {len(self.train_loader)}')
        print(f'Test loader size: {len(self.test_loader)}')
        print(f'Using device {self.opt.device}')
        print(f'Run started at {format_time(self.training_start_time)}')
        print(fill_char * width)

    ###
    # Pre & Post method, subclass override to extend its functionalities
    ###

    @abstractmethod
    def pre_train(self):
        self.training_start_time = time.time()
        self.last_epoch_time = time.time()

        self._print_title()

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
        if self.opt.batch_log_freq > 0 and batch > 2: # (batch % self.opt.batch_log_freq == 0 or batch == 2):
            print(self.log_batch(batch))
            self.this_batch_logged = True
        else:
            self.this_batch_logged = False

    @abstractmethod
    def evaluate(self, epoch):
        pass

    @abstractmethod
    def post_epoch(self, epoch):
        if self.opt.eval_freq > 0 and (epoch % self.opt.eval_freq == 0 or epoch == self.opt.start_epoch):
            print('Evaluating ... ')
            self.evaluate(epoch)
            print('done')
            self.this_epoch_evaluated = True
        else:
            self.this_epoch_evaluated = False

        if self.opt.log_freq > 0 and (epoch % self.opt.log_freq == 0 or epoch == self.opt.start_epoch):
            print(self.log_epoch(epoch))
            self.this_epoch_logged = True
        else:
            self.this_epoch_logged = False

        if self.opt.save_freq > 0 and (epoch % self.opt.save_freq == 0 or epoch == self.opt.start_epoch):
            self.save_checkpoint(epoch)
            self.save_checkpoint('latest')
            self.this_epoch_saved = True
        else:
            self.this_epoch_saved = False

    @abstractmethod
    def post_train(self):
        training_end_time = time.time()
        print(f'Training finished at {format_time(training_end_time)}')
        print(f'Time taken: {format_time(training_end_time - self.training_start_time)}')
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
        text = f'[epoch={epoch}] ' + \
               f'[curr_time={format_time(curr_time)}] ' + \
               f'[train_time={format_time(curr_time - self.training_start_time)}] ' + \
               f'[epoch_time={format_time(curr_time - self.last_epoch_time)}] '

        self.last_epoch_time = curr_time
        return text

    def _get_last_batch(self, this_batch):
        return max(1, this_batch - self.opt.batch_log_freq)

    def log_batch(self, batch):
        curr_time = time.time()
        from_batch = self._get_last_batch(batch)

        text = f'[batch={from_batch}-{batch}] ' + \
               f'[batch_time={format_time(curr_time - self.last_batch_time)}] ' + \
               f'[train_time={format_time(curr_time - self.training_start_time)}] '

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
