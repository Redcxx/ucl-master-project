from ml.options import BaseTrainOptions
from ml.options.base import BaseInferenceOptions


class SketchSimpTrainOptions(BaseTrainOptions):

    def __init__(self):
        super().__init__()
        # Training
        self.batch_size = 8
        self.start_epoch = 1
        self.end_epoch = 100
        self.eval_freq = 5
        self.log_freq = 1
        self.save_freq = 1
        self.batch_log_freq = 100
        # Optimizer
        self.opt_step_size = 50
        self.opt_gamma = 0.5

        # Dataset
        self.dataset_dir = 'sketch-simp-train-dataset-root'
        self.a_to_b = True

        # Optimizer
        self.lr = 0.0002

    @property
    def tag(self):
        return 'sketch-simp-train'


class SketchSimpInferenceOptions(BaseInferenceOptions):
    @property
    def inference_run_id(self):
        return 'SketchSimpInferenceOptionsRunID'

    def __init__(self):
        super().__init__()

    @property
    def tag(self):
        return 'sketch-simp-inference'
