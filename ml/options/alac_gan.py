from ml.options.base_options import BaseTrainOptions
from ml.options.pix2pix import Pix2pixOptions


class Pix2pixTrainOptions(Pix2pixOptions, BaseTrainOptions):

    def __init__(self):
        super().__init__()

    @property
    def batch_size(self): return 4

    @property
    def start_epoch(self): return 1

    @property
    def end_epoch(self): return 100

    @property
    def eval_freq(self): return 10

    @property
    def log_freq(self): return 1

    @property
    def save_freq(self): return 10

    @property
    def batch_log_freq(self): return 100

    @property
    def num_workers(self): return 4

    @property
    def pin_memory(self): return True

    @property
    def shuffle(self): return True

    @property
    def a_to_b(self): return True
