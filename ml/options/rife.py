from ml.options import BaseTrainOptions, BaseInferenceOptions


class RIFEInferenceOptions(BaseInferenceOptions):

    @property
    def inference_run_id(self):
        return 'RIFEInferenceRunID'

    @property
    def tag(self):
        return 'rife-inference-tag'


class RIFETrainOptions(BaseTrainOptions):

    def __init__(self):
        super().__init__()

        self.resume_ckpt_file = None

        self.batch_size = 32
        self.start_epoch = 0
        self.end_epoch = 1000
        self.decay = 250
        self.log_freq = 10
        self.eval_freq = 100
        self.save_freq = 100
        self.batch_log_freq = 0

        self.lr = 0.0001

        self.dataset_root = 'RIFETrainOptionsDatasetRoot'
        self.train_dataset_root = 'train'
        self.test_dataset_root = 'test'
        self.eval_images_save_folder = self.tag + '-eval-images'

    @property
    def tag(self):
        return 'rife-train-tag'
