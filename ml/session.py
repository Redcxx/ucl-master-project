from datetime import datetime

import torch.cuda


class SessionOptions(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self  # dangerous, merging namespace, but now you can access key using .key instead of ['key']

        # Housekeeping
        self.tag = 'simple-GAN'
        self.run_id = f'{self.tag}-' + datetime.now().strftime('%Y-%m-%d-%A-%Hh-%Mm-%Ss')
        self.random_seed = 42
        self.working_folder = 'WORK'  # shared on Google Drive
        self.pydrive2_setting_file = 'settings.yaml'

        # Dataset
        self.dataset_dir = './line_tied'
        self.dataset_train_folder = 'train'
        self.dataset_test_folder = 'test'
        self.batch_size = 1  # default 1
        self.shuffle = False
        self.num_workers = 1
        self.pin_memory = True
        self.A_to_B = False
        # transforms
        self.random_jitter = True
        self.random_mirror = True
        # loaders, will be set later in train.py
        self.train_loader = None
        self.test_loader = None

        # Training
        self.start_epoch = 1
        self.end_epoch = 500  # default 200
        self.decay_epochs = 50
        self.eval_freq = 50  # eval frequency, unit epoch
        self.log_freq = 1  # log frequency, unit epoch
        self.save_freq = 100  # save checkpoint, unit epoch
        self.batch_log_freq = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Evaluate
        self.n_eval_display_samples = 5

        # Model
        self.model_name = 'pix2pixModel'
        self.generator_config = _generator_config()
        self.discriminator_config = _discriminator_config()

        # Optimizer
        self.lr = 0.0002
        self.optimizer_beta1 = 0.5  # default 0.5
        self.optimizer_beta2 = 0.999
        self.init_gain = 0.02  # default 0.02
        self.weight_decay = 0  # default 0

        # Scheduler
        self.epochs_decay = 0  # default 100

        # Loss
        self.l1_lambda = 100.0  # encourage l1 distance to actual output
        self.d_loss_factor = 0.5  # slow down discriminator learning

        self.update(dict(*args, **kwargs))


def _discriminator_config():
    return {
        'in_channels': 3 * 2,  # conditionalGAN takes both real and fake image
        'blocks': [
            {
                'filters': 32,
            },
            {
                'filters': 64,
            },
            {
                'filters': 64,
            },
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
                'filters': 64,
                'dropout': False,
                'skip_connection': False
            },
            {
                'filters': 128,
                'dropout': False,
                'skip_connection': True
            },
            {
                'filters': 128,
                'dropout': False,
                'skip_connection': True
            },
            {
                'filters': 128,
                'dropout': False,
                'skip_connection': True
            },
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
