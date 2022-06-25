import pprint
import random
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import torch.cuda


class BaseOptions(ABC):

    def __init__(self):

        # for saving and loading
        # this line must be on top here
        self.saved_dict = dict()

        # Housekeeping
        self.run_id = f'{self.tag}-' + datetime.now().strftime('%Y-%m-%d-%A-%Hh-%Mm-%Ss')

        # self.run_id = f'{self.tag}-2022-06-13-Monday-13h-06m-22s'
        self.random_seed = 42
        self.working_folder = 'WORK'  # shared on Google Drive
        self.pydrive2_settings_file = 'ucl-master-project/misc/settings.yaml'  # for saving and loading

        # Model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

    def __setattr__(self, key, value):
        if key != 'saved_dict':
            self.saved_dict[key] = value
        super().__setattr__(key, value)

    def __str__(self):
        return pprint.pformat(self.saved_dict)

    @property
    @abstractmethod
    def tag(self): pass

    @property
    @abstractmethod
    def a_to_b(self): pass


class BaseInferenceOptions(BaseOptions, ABC):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def images_folder_path(self): pass


class BaseTrainOptions(BaseOptions, ABC):
    def __init__(self):
        super().__init__()

    # training

    @property
    @abstractmethod
    def batch_size(self): pass

    @property
    @abstractmethod
    def start_epoch(self): pass

    @property
    @abstractmethod
    def end_epoch(self): pass

    @property
    @abstractmethod
    def eval_freq(self): pass

    @property
    @abstractmethod
    def log_freq(self): pass

    @property
    @abstractmethod
    def save_freq(self): pass

    @property
    @abstractmethod
    def batch_log_freq(self): pass

    # datasets

    @property
    @abstractmethod
    def num_workers(self): pass

    @property
    @abstractmethod
    def pin_memory(self): pass

    @property
    @abstractmethod
    def shuffle(self): pass
