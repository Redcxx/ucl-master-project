import os
import random

import numpy as np
from torchvision.transforms import transforms

from ml.datasets import BaseDataset
from ml.file_utils import get_all_image_paths
from ml.options.waifu2x2 import Waifu2x2TrainOptions


class Waifu2x2Dataset(BaseDataset):
    def __init__(self, opt: Waifu2x2TrainOptions, hr_root, lr_root):
        super().__init__(opt)

        self.hr_paths = sorted(get_all_image_paths(hr_root))
        self.lr_paths = sorted(get_all_image_paths(lr_root))
        self.a_to_b = opt.a_to_b
        self.opt = opt

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, i):
        hr, lr = self.hr_paths[i], self.lr_paths[i]
        hr, lr = self._read_im_cv(hr), self._read_im_cv(lr)
        hr, lr = random_crop(hr, lr, self.opt.patch_size, self.opt.scale)
        hr, lr = random_flip_and_rotate(hr, lr)
        hr, lr = self.transform(hr), self.transform(lr)

        return hr, lr


class Waifu2x2TrainDataset(Waifu2x2Dataset):

    def __init__(self, opt: Waifu2x2TrainOptions):
        hr_root = os.path.join(opt.dataset_root, opt.dataset_train_folder, opt.high_res_root)
        lr_root = os.path.join(opt.dataset_root, opt.dataset_train_folder, opt.low_res_root)
        super().__init__(opt, hr_root, lr_root)


class Waifu2x2TestDataset(Waifu2x2Dataset):

    def __init__(self, opt: Waifu2x2TrainOptions):
        hr_root = os.path.join(opt.dataset_root, opt.dataset_test_folder, opt.high_res_root)
        lr_root = os.path.join(opt.dataset_root, opt.dataset_test_folder, opt.low_res_root)
        super().__init__(opt, hr_root, lr_root)


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
