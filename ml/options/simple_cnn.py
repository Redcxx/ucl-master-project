from ml.options import BaseTrainOptions
from ml.options.base import BaseInferenceOptions


class SimpleCNNTrainOptions(BaseTrainOptions):

    def __init__(self):
        super().__init__()

    @property
    def tag(self):
        return 'simple-cnn-train'


class SimpleCNNInferenceOptions(BaseInferenceOptions):
    @property
    def inference_run_id(self):
        return 'SimpleCNNInferenceOptionsRunID'

    def __init__(self):
        super().__init__()

    @property
    def tag(self):
        return 'simple-cnn-inference'
