from abc import ABC

from ml.options.base import BaseTrainOptions, BaseInferenceOptions


class Pix2pixOptions(ABC):
    def __init__(self):
        super().__init__()

    @property
    def tag(self):
        return 'pix2pix-sketch-simplification-MSE-DILATE-3000'


class Pix2pixInferenceOptions(BaseInferenceOptions):

    def __init__(self):
        super().__init__()
        self.input_images_path = r'sketch_simplification/test'
        self.output_images_path = 'noghost_sketch_simplification_MSE_DILATE'
        self.image_size = 512
        self.a_to_b = True
        self.batch_size = 8
        self.num_workers = 4

        self.generator_config = _generator_config()
        self.discriminator_config = _discriminator_config()

    @property
    def tag(self):
        return 'pix2pix-noghost-inference-sketch-simplification-NO-WEIGHT_MAP-MSE-DILATE'

    @property
    def inference_run_id(self):
        return 'pix2pix-sketch-simplification-MSE-DILATE-2022-09-11-Sunday-16h-40m-52s'

    @property
    def inference_run_tag(self):
        return 'final'


class Pix2pixTrainOptions(Pix2pixOptions, BaseTrainOptions):

    def __init__(self):
        super().__init__()

        # Training
        self.batch_size = 8
        # self.run_id = 'pix2pix-sketch-simplification-NO_WEIGHT_MAP-MSE-2022-09-09-Friday-10h-01m-56s'
        self.start_epoch = 1
        self.end_epoch = 3000
        self.eval_freq = 100
        self.log_freq = 5
        self.save_freq = 100
        self.batch_log_freq = 0

        # Dataset
        # pix2pix-sketch-simplification-MSE-DILATE-2022-09-11-Sunday-16h-40m-52s

        # pix2pix-sketch-simplification-CONTENT_LOSS-DILATE-2022-09-07-Wednesday-23h-16m-58s
        # pix2pix-sketch-simplification-DILATE-2022-09-07-Wednesday-13h-35m-26s
        # pix2pix-sketch-simplification-CONTENT_LOSS-2022-09-07-Wednesday-19h-44m-34s
        # pix2pix-sketch-simplification-NORMAL-2022-09-07-Wednesday-16h-09m-49s
        # pix2pix-sketch-simplification-NO-WEIGHT-MAP-2022-09-08-Thursday-10h-16m-21s
        # pix2pix-sketch-simplification-NO_WEIGHT_MAP-CONTENT_LOSS-2022-09-08-Thursday-18h-11m-45s
        # pix2pix-sketch-simplification-NO_WEIGHT_MAP-MSE-CONTENT_LOSS-2022-09-09-Friday-00h-15m-15s
        # pix2pix-sketch-simplification-NO_WEIGHT_MAP-MSE-2022-09-09-Friday-10h-01m-56s


        self.dataset_root = './sketch_simplification'
        self.a_to_b = True
        self.random_jitter = True
        self.random_mirror = True
        self.random_rotate = True

        self.weight_map = True
        self.dilate = True
        self.content_loss = False
        self.mse_loss = True
        # self.resume_ckpt_file = 'pix2pix-sketch-simplification-DILATE-2022-09-07-Wednesday-13h-35m-26s'

        # Model
        self.generator_config = _generator_config()
        self.discriminator_config = _discriminator_config()

        # Optimizer
        self.lr = 0.0002
        self.optimizer_beta1 = 0.5
        self.optimizer_beta2 = 0.999
        self.init_gain = 0.02
        self.weight_decay = 0
        self.decay_epochs = 500

        # Loss
        self.l1_lambda = 100.0  # encourage l1 distance to actual output
        self.d_loss_factor = 0.5  # slow down discriminator learning

        # transforms
        self.VGG16_PATH = 'vgg16-397923af.pth'
        self.a_to_b = True
        self.image_size = 512


def _discriminator_config():
    return {
        'in_channels': 1 * 2,  # conditionalGAN takes both real and fake image
        'blocks': [
            # {
            #     'filters': 512,
            # },
            {
                'filters': 256,
            },
            {
                'filters': 128,
            },
            {
                'filters': 64,
            },
            {
                'filters': 32,
            },
            {
                'filters': 16,
            },
            # {
            #     'filters': 4,
            # },
        ]
    }


def _generator_config():
    return {
        'in_channels': 1,
        'out_channels': 1,
        'blocks': [
            {
                'filters': 32,
                'dropout': False,
                'skip_connection': False
            },
            {
                'filters': 64,
                'dropout': False,
                'skip_connection': True
            },
            {
                'filters': 128,
                'dropout': False,
                'skip_connection': True
            },
            # {
            #     'filters': 128,
            #     'dropout': False,
            #     'skip_connection': True
            # },
            {
                'filters': 256,
                'dropout': False,
                'skip_connection': True
            },
            # {
            #     'filters': 256,
            #     'dropout': False,
            #     'skip_connection': True
            # },
            {
                'filters': 512,
                'dropout': True,
                'skip_connection': True
            },
            {
                'filters': 1024,
                'dropout': True,
                'skip_connection': True
            },
        ]
    }
