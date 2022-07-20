from ml.options import BaseTrainOptions
from ml.options.base import BaseInferenceOptions


class Waifu2xTrainOptions(BaseTrainOptions):

    def __init__(self):
        super().__init__()

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
