from ml.options import BaseTrainOptions
from ml.options.base import BaseInferenceOptions


class Waifu2xTrainOptions(BaseTrainOptions):

    def __init__(self):
        super().__init__()

        self.batch_size = 16
        self.start_epoch = 1
        self.end_epoch = 4000
        self.decay = 1500
        self.log_freq = 10
        self.eval_freq = 100
        self.save_freq = 100
        self.batch_log_freq = 0

        self.scale = 2
        self.patch_size = 64

        self.lr = 0.0001
        self.clip = 10.0

        self.multi_scale = self.scale == 0
        self.train_dataset_root = './DIV2K_train.h5'
        self.test_dataset_root = './DIV2K_valid.h5'
        self.eval_images_save_folder = self.tag + '-eval-images'

    @property
    def tag(self):
        return 'waifu2x-train-tag'


class Waifu2xInferenceOptions(BaseInferenceOptions):
    def __init__(self):
        super().__init__()

    @property
    def tag(self):
        return 'waifu2x-inference-tag'

    @property
    def inference_run_id(self):
        return 'waifu2x-inference-run-id'
