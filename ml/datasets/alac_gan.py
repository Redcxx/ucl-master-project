import math
import numbers
import os
import random

from PIL import Image
from torchvision.transforms import transforms, InterpolationMode

from ml.algorithms.xdog import extract_edges_cv
from ml.datasets import BaseDataset
from ml.file_utils import get_all_image_paths
from ml.options.alac_gan import AlacGANTrainOptions, AlacGANInferenceOptions


class AlacGANTrainDataset(BaseDataset):

    @staticmethod
    def jitter(x):
        ran = random.uniform(0.7, 1)
        return x * ran + 1 - ran

    def __init__(self, opt: AlacGANTrainOptions):
        super().__init__(opt)
        root = os.path.join(opt.dataset_root, opt.dataset_train_folder)
        self.paths = sorted(get_all_image_paths(root))
        self.a_to_b = opt.a_to_b

        self.c_trans = transforms.Compose([
            transforms.Resize(opt.image_size, InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.v_trans = transforms.Compose([
            RandomSizedCrop(opt.image_size // 4, InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.s_trans = transforms.Compose([
            transforms.Resize(opt.image_size, InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Lambda(self.jitter),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        r = random.random()
        # if r < 0.25:
        #     # use sketch from extract sketch model
        #     A, B = self._split_image_pil(self._read_im_pil(self.paths[i]))
        # else:
        # create own sketch using xdog
        sigma = 0.3 if r < 0.333 else (0.4 if r < 0.666 else 0.5)

        _, B = self._split_image_cv(self._read_im_cv(self.paths[i]))
        A = extract_edges_cv(B, sigma=sigma)
        A, B = self._cv2pil_im(A), self._cv2pil_im(B)
        # A, B = self._split_image_pil(self._read_im_pil(self.paths[i]))

        s_im, c_im = (A, B) if self.a_to_b else (B, A)

        s_im = s_im.convert('L')
        if random.random() < 0.5:
            c_im, s_im = c_im.transpose(Image.FLIP_LEFT_RIGHT), s_im.transpose(Image.FLIP_LEFT_RIGHT)

        c_im, v_im, s_im = self.c_trans(c_im), self.v_trans(c_im), self.s_trans(s_im)

        return c_im, v_im, s_im


class AlacGANTestDataset(BaseDataset):

    def __init__(self, opt: AlacGANTrainOptions):
        super().__init__(opt)
        root = os.path.join(opt.dataset_root, opt.dataset_test_folder)
        self.paths = sorted(get_all_image_paths(root))
        self.a_to_b = opt.a_to_b

        self.c_trans = transforms.Compose([
            transforms.Resize(opt.image_size, InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.v_trans = transforms.Compose([
            RandomSizedCrop(opt.image_size // 4, InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.s_trans = transforms.Compose([
            transforms.Resize(opt.image_size, InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        _, B = self._split_image_cv(self._read_im_cv(self.paths[i]))
        A = extract_edges_cv(B, sigma=0.4)
        A, B = self._cv2pil_im(A), self._cv2pil_im(B)

        s_im, c_im = (A, B) if self.a_to_b else (B, A)

        s_im = s_im.convert('L')
        c_im, v_im, s_im = self.c_trans(c_im), self.v_trans(c_im), self.s_trans(s_im)

        return c_im, v_im, s_im


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1, img2):
        w, h = img1.size
        th, tw = self.size
        if w == tw and h == th:  # ValueError: empty range for randrange() (0,0, 0)
            return img1, img2

        if w == tw:
            x1 = 0
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))

        elif h == th:
            x1 = random.randint(0, w - tw)
            y1 = 0
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))

        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=InterpolationMode.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.9, 1.) * area
            aspect_ratio = random.uniform(7. / 8, 8. / 7)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BICUBIC)

        # Fallback
        scale = transforms.Resize(self.size, self.interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(scale(img))


class AlacGANInferenceDataset(AlacGANTestDataset):

    def __init__(self, opt):
        super().__init__(opt)
