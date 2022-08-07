from ml.options import BaseTrainOptions
from ml.options.base import BaseInferenceOptions


class Waifu2x2TrainOptions(BaseTrainOptions):

    def __init__(self):
        super().__init__()

        self.batch_size = 16
        self.start_epoch = 1
        self.end_epoch = 3000
        self.decay = 1000
        self.log_freq = 10
        self.eval_freq = 100
        self.save_freq = 100
        self.batch_log_freq = 0

        self.scale = 2
        self.patch_size = 64

        self.lr = 0.0001
        self.clip = 10.0

        self.multi_scale = self.scale == 0
        self.dataset_root = 'upsample2x'
        self.train_dataset_root = 'train'
        self.test_dataset_root = 'test'
        self.high_res_root = 'hr'
        self.low_res_root = 'lr'
        self.eval_images_save_folder = self.tag + '-eval-images'

    @property
    def tag(self):
        return 'waifu2x2-train-tag'


class Waifu2x2InferenceOptions(BaseInferenceOptions):
    def __init__(self):
        super().__init__()

    @property
    def tag(self):
        return 'waifu2x2-inference-tag'

    @property
    def inference_run_id(self):
        return 'waifu2x2-inference-run-id'
