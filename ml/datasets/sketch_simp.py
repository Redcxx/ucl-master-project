import math
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from ml.datasets import BaseDataset
from ml.datasets.augmentation import rotate_cv, flip_horizontal_cv
from ml.file_utils import get_all_image_paths
from ml.options.sketch_simp import SketchSimpInferenceOptions, SketchSimpTrainOptions


class SketchSimpDataset(BaseDataset):
    def __init__(self, opt, root):
        super().__init__(opt)
        self.paths = sorted(get_all_image_paths(root))
        self.a_to_b = opt.a_to_b
        self.transform = transforms.Compose([
            transforms.Resize(size=opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        A, B = self._split_image_pil(self._read_im_pil(self.paths[i]))
        A, B = self.transform(A), self.transform(B)
        return (A, B) if self.a_to_b else (B, A)


class SketchSimpInferenceDataset(SketchSimpDataset):

    def __init__(self, opt: SketchSimpInferenceOptions):
        super().__init__(opt, opt.input_images_path)


class SketchSimpTestDataset(SketchSimpDataset):
    def __init__(self, opt: SketchSimpTrainOptions):
        root = os.path.join(opt.dataset_root, opt.dataset_test_folder)
        super().__init__(opt, root)


class SketchSimpTrainDataset(BaseDataset):
    def __init__(self, opt: SketchSimpTrainOptions):
        super().__init__(opt)
        root = os.path.join(opt.dataset_root, opt.dataset_train_folder)
        self.paths = sorted(get_all_image_paths(root))
        self.a_to_b = opt.a_to_b
        self.transform = transforms.Compose([
            transforms.Resize(size=opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        A, B = self._split_image_pil(self._read_im_pil(self.paths[i]))
        A, B = A.convert('L'), B.convert('L')
        A, B = self.transform(A), self.transform(B)
        return (A, B) if self.a_to_b else (B, A)


def compute_weight_map(im: Image.Image, n_bins=10, dist=4, device='cuda'):
    lin = torch.linspace(0, 1, n_bins + 1)
    out = torch.zeros(im.size, device=device)

    w, h = im.size
    pixels = np.asarray(im)
    for x in range(w):
        x_min = max(0, x - dist)
        x_max = min(w, x + dist)
        for y in range(h):
            y_min = max(0, y - dist)
            y_max = min(h, y + dist)
            local = pixels[x_min:x_max, y_min:y_max]
            bins = np.empty(n_bins)
            for i in range(n_bins):
                bins[i] = np.sum((local > lin[i]) * (local < lin[i + 1])) / local.size
            p = im.getpixel((x, y))
            n = 0
            for i in range(n_bins):
                if lin[i] <= p <= lin[i + 1]:
                    n = i
                    break
            out[x, y] = torch.exp(-bins[n]) + 0.5

    return out


# maybe use this: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def extract_patch(patch_size, A: Image.Image, B: Image.Image, W: Image.Image):
    kaiten = ...
    A_out, B_out, W_out = None, None, None
    A, B, W = np.asarray(A), np.asarray(B), np.asarray(W) if W is not None else None
    while A_out is None:
        ur, vr = None, None
        kaitensuru = kaiten > 0
        if kaitensuru:
            kaiten = min(kaiten, 45) * math.pi / 180
            k_scale = abs(math.cos(kaiten)) + abs(math.cos(kaiten - 0.5 * torch.pi))

            if k_scale * patch_size + 2 < min(A.size):
                border = torch.ceil(((k_scale - 1) / 2) * patch_size)
                ur = random.randint(border + 1, A.size[0] - patch_size - border - 1)
                vr = random.randint(border + 1, A.size[1] - patch_size - border - 1)
            else:
                kaitensuru = False

        if ur is None or vr is None:
            ur = random.randint(1, A.size[0] - patch_size)
            vr = random.randint(1, A.size[1] - patch_size)

        ur_end = ur + patch_size + 1
        vr_end = vr + patch_size + 1
        B_out = B[ur:ur_end, vr:vr_end]
        if np.mean(B_out) < 0.99:
            if kaitensuru:
                kaiten = random.uniform(-kaiten, kaiten) * math.pi / 180
                k_scale = abs(math.cos(kaiten)) + abs(math.cos((kaiten - 0.5 * math.pi)))
                border = torch.ceil(((k_scale - 1) / 2) * patch_size)

                B = B[ur - border:ur_end + border, vr - border:vr_end + border]
                B = rotate_cv(B, kaiten)
                B_out = B[border:border + patch_size - 1, border:border + patch_size - 1]

                A = A[ur - border:ur_end + border, vr - border:vr_end + border]
                A = rotate_cv(A, kaiten)
                A_out = A[border:border + patch_size - 1, border:border + patch_size - 1]

                if W is not None:
                    W = W[ur - border:ur_end + border, vr - border:vr_end + border]
                    W = rotate_cv(W, kaiten)
                    W_out = W[border:border + patch_size - 1, border:border + patch_size - 1]

            else:
                A_out = A[ur:ur_end, vr:vr_end]
                if W is not None:
                    W_out = W[ur:ur_end, vr:vr_end]

    if random.random() < 0.5:
        A_out = flip_horizontal_cv(A_out)
        B_out = flip_horizontal_cv(B_out)
        if W is not None:
            W_out = flip_horizontal_cv(W_out)

    B_out[B_out < 0.6] = 0

    return Image.fromarray(A_out), Image.fromarray(B_out), Image.fromarray(W_out) if W is not None else None
