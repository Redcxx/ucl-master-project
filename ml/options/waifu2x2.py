from ml.options import BaseTrainOptions
from ml.options.base import BaseInferenceOptions


class Waifu2x2TrainOptions(BaseTrainOptions):

    def __init__(self):
        super().__init__()

        self.resume_ckpt_file = 'waifu2x2-train-tag-2022-08-08-Monday-08h-04m-03s_1000.ckpt'

        self.batch_size = 16
        self.start_epoch = 1001
        self.end_epoch = 2000
        self.decay = 250
        self.log_freq = 10
        self.eval_freq = 100
        self.save_freq = 100
        self.batch_log_freq = 0

        self.scale = 2
        self.patch_size = 64

        self.lr = 0.0001
        self.clip = 10.0

        self.multi_scale = self.scale == 0
        self.dataset_root = 'noghost_upsample/2/'
        self.train_dataset_root = 'train'
        self.test_dataset_root = 'test'
        self.high_res_root = 'hr'
        self.low_res_root = 'lr'
        self.eval_images_save_folder = self.tag + '-eval-images'

    @property
    def tag(self):
        return 'waifu2x2-noghost-train-tag'


class Waifu2x2InferenceOptions(BaseInferenceOptions):
    def __init__(self):
        super().__init__()

    @property
    def tag(self):
        return 'waifu2x2-inference-tag'

    @property
    def inference_run_id(self):
        return 'waifu2x2-inference-run-id'
