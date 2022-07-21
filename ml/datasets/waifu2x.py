import random

import h5py
import numpy as np
from torchvision.transforms import transforms

from ml.datasets import BaseDataset
from ml.options.base import BaseInferenceOptions
from ml.options.waifu2x import Waifu2xTrainOptions


class Waifu2xDataset(BaseDataset):
    def __init__(self, opt: Waifu2xTrainOptions, root):
        super().__init__(opt)

        self.size = opt.patch_size
        scale = opt.scale

        h5f = h5py.File(root, "r")

        self.hr = [v[:] for v in h5f["X2"].values()]

        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v[:] for v in h5f["X{}".format(i)].values()] for i in self.scale]
        else:
            self.scale = [scale]
            self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]

        h5f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, i):
        size = self.size

        item = [(self.hr[i], self.lr[i][i]) for i, _ in enumerate(self.lr)]
        item = [random_crop(hr, lr, size, self.scale[i]) for i, (hr, lr) in enumerate(item)]
        item = [random_flip_and_rotate(hr, lr) for hr, lr in item]

        return [(self.transform(hr), self.transform(lr)) for hr, lr in item]


class Waifu2xInferenceDataset(Waifu2xDataset):

    def __init__(self, opt: BaseInferenceOptions):
        super().__init__(opt, opt.input_images_path)


class Waifu2xTestDataset(Waifu2xDataset):
    def __init__(self, opt: Waifu2xTrainOptions):
        super().__init__(opt, opt.test_dataset_root)


class Waifu2xTrainDataset(Waifu2xDataset):
    def __init__(self, opt: Waifu2xTrainOptions):
        super().__init__(opt, opt.train_dataset_root)


def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)

    hsize = size * scale
    hx, hy = x * scale, y * scale

    crop_lr = lr[y:y + size, x:x + size].copy()
    crop_hr = hr[hy:hy + hsize, hx:hx + hsize].copy()

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()
