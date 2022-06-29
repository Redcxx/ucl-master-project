from abc import abstractmethod, ABC

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

import cv2 as cv

from ml.logger import log


class BaseDataset(Dataset, ABC):

    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, i):
        pass

    @staticmethod
    def _read_im_pil(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def _read_im_cv(path):
        return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)

    @staticmethod
    def _split_image_pil(AB):
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        return A, B

    @staticmethod
    def _split_image_cv(AB: np.array):
        assert len(AB.shape) == 3
        h, w = AB.shape[-2:]
        w2 = int(w / 2)
        A = AB[:, :, 0:w2]
        B = AB[:, :, w2:w]

        return A, B

    @staticmethod
    def _cv2pil_im(im: np.array):
        return Image.fromarray(im)

    @staticmethod
    def _pil2cv_im(im: Image):
        return np.asarray(im)
