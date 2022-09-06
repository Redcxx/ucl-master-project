import os
import random

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ml.datasets.augmentation import pil_rotate_crop_max, FixedRandomResizedCrop
from ml.datasets.base import BaseDataset
from ml.file_utils import get_all_image_paths
from ml.options.pix2pix import Pix2pixTrainOptions

import cv2 as cv


class Pix2pixDataset(BaseDataset):

    def __init__(self, opt: Pix2pixTrainOptions):
        super().__init__(opt)
        self.opt = opt
        root = os.path.join(opt.dataset_root, opt.dataset_train_folder)
        self.paths = sorted(get_all_image_paths(root))
        self.a_to_b = opt.a_to_b
        self.random_jitter = opt.random_jitter
        self.random_mirror = opt.random_mirror
        self.random_rotate = opt.random_rotate

        self.weight_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.dilate_kernel = np.ones((3, 3), np.uint8)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        A, B = self._split_image_cv(self._read_im_cv(self.paths[i]))

        transform = self._generate_transform(A.shape[1], A.shape[0])

        # weight_map = B.point(lambda p: 255 if p < 200 else 0)  # threshold
        # weight_map = self._pil2cv_im(weight_map)
        weight_map = ((B < 200) * 255).astype(np.uint8)
        weight_map = cv.dilate(weight_map, kernel=self.dilate_kernel, iterations=1)
        weight_map = transform(weight_map)

        A, B = transform(A), transform(B)

        A, B = (A, B) if self.a_to_b else (B, A)

        return A, B, weight_map

    def _generate_transform(self, w, h):
        additional_transforms = []

        if self.random_jitter and random.random() > 0.1:
            # old_size = self.opt.image_size
            # new_size = int(old_size * 1.2)

            # rand_x = random.randint(0, new_size - old_size)
            # rand_y = random.randint(0, new_size - old_size)

            additional_transforms += [
                # transforms.Resize((new_size, new_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
                # transforms.Lambda(lambda im: self._crop(im, (rand_x, rand_y), (old_size, old_size)))
                FixedRandomResizedCrop(w, h, self.opt.image_size, scale=(0.6, 1.0), ratio=(1, 1)),
            ]

        if self.random_rotate and random.random() > 0.2:
            rotate_deg = random.randint(0, 180)
            additional_transforms += [
                transforms.Lambda(lambda im: pil_rotate_crop_max(im, rotate_deg))
            ]

        if self.random_mirror and random.random() > 0.5:
            additional_transforms += [
                transforms.Lambda(lambda im: self._flip(im)),
            ]

        in_channels = self.opt.generator_config['in_channels']
        if in_channels == 1:
            additional_transforms += [
                transforms.Grayscale()
            ]

        return transforms.Compose([
            transforms.Lambda(lambda im: self._cv2pil_im(im)),
            *additional_transforms,
            transforms.Resize(
                (self.opt.image_size, self.opt.image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * in_channels, [0.5] * in_channels)
        ])

    @staticmethod
    def _flip(im):
        return im.transpose(Image.FLIP_LEFT_RIGHT)

    @staticmethod
    def _crop(im, pos, size):
        return im.crop((pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]))


class Pix2pixTestDataset(Pix2pixDataset):
    def __init__(self, opt: Pix2pixTrainOptions):
        super().__init__(opt)


class Pix2pixTrainDataset(Pix2pixDataset):
    def __init__(self, opt: Pix2pixTrainOptions):
        super().__init__(opt)
