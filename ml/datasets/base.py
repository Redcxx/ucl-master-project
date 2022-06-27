from abc import abstractmethod, ABC

from PIL import Image
from torch.utils.data.dataset import Dataset


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
    def _read_im(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def _split_image(AB):
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        return A, B
