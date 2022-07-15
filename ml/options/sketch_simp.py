from ml.options import BaseTrainOptions
from ml.options.base import BaseInferenceOptions


class SketchSimpTrainOptions(BaseTrainOptions):

    def __init__(self):
        super().__init__()
        # Training
        self.batch_size = 16
        self.start_epoch = 1
        self.end_epoch = 3000
        self.eval_freq = 100
        self.log_freq = 10
        self.save_freq = 100
        self.batch_log_freq = 0
        # Optimizer
        self.lr = 0.0002
        self.opt_step_size = 1500
        self.opt_gamma = 0.1
        self.VGG16_PATH = 'vgg16-397923af.pth'

        # Dataset
        self.dataset_root = 'sketch_simplification'
        self.a_to_b = True
        self.image_size = 512
        self.make_fake_hint = False

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
