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

        # Dataset
        self.a_to_b = True
        self.batch_size = 16
        self.num_workers = 4
        self.pin_memory = True
        self.shuffle = True
        self.image_size = 256

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

    def load_saved_dict(self, saved_dict):
        for k, v in saved_dict:
            setattr(self, k, v)

    def __str__(self):
        return pprint.pformat(self.saved_dict)

    @property
    @abstractmethod
    def tag(self): pass


class BaseInferenceOptions(BaseOptions, ABC):

    def __init__(self):
        super().__init__()
        self.input_images_path = 'BaseInferenceOptionsInputPath'
        self.shuffle = False
        self.output_images_path = 'BaseInferenceOptionsOutputPath'
        self.show_progress = True


class BaseTrainOptions(BaseOptions, ABC):
    def __init__(self):
        super().__init__()

        # Training
        self.lr = 0.0001
        self.batch_size = 16
        self.start_epoch = 1
        self.end_epoch = 100
        self.eval_freq = 10
        self.log_freq = 1
        self.save_freq = 10
        self.batch_log_freq = 100
        self.resume_training = False

        # Dataset
        self.dataset_root = 'BaseTrainOptionDatasetRoot'
        self.dataset_train_folder = 'train'
        self.dataset_test_folder = 'test'

        # Evaluate
        self.eval_n_display_samples = 0
        self.eval_n_save_samples = 10
        self.eval_images_save_folder = f'eval-images'
        self.eval_show_progress = True
