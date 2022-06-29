import os
import random

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ml.datasets.base import BaseDataset
from ml.datasets.default import DefaultTestDataset
from ml.file_utils import get_all_image_paths
from ml.options.pix2pix import Pix2pixTrainOptions


class Pix2pixTestDataset(DefaultTestDataset):
    def __init__(self, opt: Pix2pixTrainOptions):
        super().__init__(opt)


class Pix2pixTrainDataset(BaseDataset):

    def __init__(self, opt: Pix2pixTrainOptions):
        super().__init__(opt)
        self.opt = opt
        root = os.path.join(opt.dataset_dir, opt.dataset_train_folder)
        self.paths = sorted(get_all_image_paths(root))
        self.a_to_b = opt.a_to_b
        self.random_jitter = opt.random_jitter
        self.random_mirror = opt.random_mirror

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        A, B = self._split_image_pil(self._read_im_pil(self.paths[i]))

        transform = self._generate_transform()

        A, B = transform(A), transform(B)  # apply same transform to both A and B

        return (A, B) if self.a_to_b else (B, A)

    def _generate_transform(self):
        additional_transforms = []

        if self.random_jitter:
            old_size = self.opt.image_size
            new_size = int(old_size * 1.2)

            rand_x = random.randint(0, new_size - old_size)
            rand_y = random.randint(0, new_size - old_size)

            additional_transforms += [
                transforms.Resize((new_size, new_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
                transforms.Lambda(lambda im: self._crop(im, (rand_x, rand_y), (old_size, old_size)))
            ]

        if self.random_mirror and random.random() > 0.5:
            additional_transforms += [
                transforms.Lambda(lambda im: self._flip(im)),
            ]

        return transforms.Compose([
            transforms.Resize(self.opt.image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            *additional_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @staticmethod
    def _flip(im):
        return im.transpose(Image.FLIP_LEFT_RIGHT)

    @staticmethod
    def _crop(im, pos, size):
        return im.crop((pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]))
