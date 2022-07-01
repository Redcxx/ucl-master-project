from ml.options import BaseTrainOptions
from ml.options.base import BaseInferenceOptions


class DefaultTrainOptions(BaseTrainOptions):

    def __init__(self):
        super().__init__()

    @property
    def tag(self):
        return 'default-train-tag'


class DefaultInferenceOptions(BaseInferenceOptions):
    def __init__(self):
        super().__init__()

    @property
    def tag(self):
        return 'default-inference-tag'

    @property
    def inference_run_id(self):
        return 'default-inference-run-id'
