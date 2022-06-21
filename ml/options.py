import random
from datetime import datetime

import numpy as np
import torch.cuda


class BaseOptions(dict):
    def __init__(self):
        super().__init__()
        # dangerous, merging namespace,
        # but now you can access key using .key instead of ['key'] and pretty print it
        self.__dict__ = self

        # Housekeeping
        self.tag = 'raw-data-5-shots'
        # self.run_id = f'{self.tag}-' + datetime.now().strftime('%Y-%m-%d-%A-%Hh-%Mm-%Ss')
        self.run_id = f'{self.tag}-2022-06-01-Wednesday-14h-21m-44s'
        self.random_seed = 42
        self.working_folder = 'MasterProject'  # shared on Google Drive
        self.pydrive2_settings_file = 'ucl-master-project/misc/settings.yaml'

        # Model
        self.model_name = 'pix2pix_model'
        self.network_config = _pix2pix_network_config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Dataset
        self.num_workers = 4
        self.pin_memory = True
        self.A_to_B = False
        # will be set later in train or inference py
        self.test_loader = None
        self.test_dataset = None

        # Evaluate
        self.n_infer_display_samples = 0
        self.save_inferred_images = True
        self.inference_save_folder = f'eval-images'

        # reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)


class InferenceOptions(BaseOptions):

    def __init__(self):
        super().__init__()

        self.inference_images_folder = 'line_fill_inference'

        # overwrite
        self.n_infer_display_samples = 0
        self.save_inferred_images = True
        self.inference_save_folder = f'inference-images'


class TrainOptions(BaseOptions):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # Dataset
        self.dataset_dir = './line_tied'
        self.dataset_train_folder = 'train'
        self.dataset_test_folder = 'test'
        self.shuffle = True
        # transforms
        self.random_jitter = False
        self.random_mirror = True
        # dataset & loaders, will be set later in train.py
        self.train_loader = None
        self.train_dataset = None

        # Evaluate
        self.n_infer_display_samples = 5
        self.save_inferred_images = True
        self.inference_save_folder = f'eval-images'

        # Training
        self.batch_size = 1
        self.start_epoch = 0
        self.end_epoch = 500  # default 200
        self.decay_epochs = 100  # default 100
        self.eval_freq = 50  # eval frequency, unit epoch
        self.log_freq = 10  # log frequency, unit epoch
        self.save_freq = 50  # save checkpoint, unit epoch
        self.batch_log_freq = None

        # Optimizer
        self.lr = 0.0002
        self.optimizer_beta1 = 0.5  # default 0.5
        self.optimizer_beta2 = 0.999
        self.init_gain = 0.02  # default 0.02
        self.weight_decay = 0  # default 0

        # Loss
        self.l1_lambda = 100.0  # encourage l1 distance to actual output
        self.d_loss_factor = 0.5  # slow down discriminator learning

        # update above according to argument
        self.update(dict(*args, **kwargs))


def _discriminator_config():
    return {
        'in_channels': 3 * 2,  # conditionalGAN takes both real and fake image
        'blocks': [
            {
                'filters': 8,
            },
            {
                'filters': 16,
            },
            # {
            #     'filters': 32,
            # },
            # {
            #     'filters': 256,
            # },
            # {
            #     'filters': 512,
            # }
        ]
    }


def _generator_config():
    return {
        'in_channels': 3,
        'out_channels': 3,
        'blocks': [
            {
                'filters': 16,
                'dropout': False,
                'skip_connection': False
            },
            {
                'filters': 32,
                'dropout': False,
                'skip_connection': True
            },
            {
                'filters': 64,
                'dropout': False,
                'skip_connection': True
            },
            # {
            #     'filters': 256,
            #     'dropout': False,
            #     'skip_connection': True
            # },
            # {
            #     'filters': 256,
            #     'dropout': True,
            #     'skip_connection': True
            # },
            # {
            #     'filters': 512,
            #     'dropout': True,
            #     'skip_connection': True
            # },
            # {
            #     'filters': 512,
            #     'dropout': True,
            #     'skip_connection': True
            # },
            # {
            #     'filters': 512,
            #     'dropout': False,
            #     'skip_connection': True
            # }
        ]
    }


def _pix2pix_network_config():
    return {
        'generator_config': _generator_config(),
        'discriminator_config': _discriminator_config()
    }
