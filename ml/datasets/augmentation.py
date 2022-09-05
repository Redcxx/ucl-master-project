import math
import warnings
from typing import List, Tuple, Sequence

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode
from torchvision.transforms.transforms import _setup_size


def rotate_pil(im: Image.Image, rotate_deg):
    return im.rotate(rotate_deg, resample=Image.BILINEAR)


def rotate_cv(im: np.array, rotate_deg):
    image_center = tuple(np.array(im.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, rotate_deg, 1.0)
    im = cv.warpAffine(im, rot_mat, im.shape[1::-1], flags=cv.INTER_LINEAR)
    return im


def cv_flip_horizontal(im: np.array, flip):
    if flip:
        return cv.flip(im, 1)  # 1 = horizontal, 0 = vertical, -1 = both

    return im


def flip_horizontal_pil(im: Image.Image):
    return im.transpose(Image.FLIP_LEFT_RIGHT)


def cv_rotate_crop_max(im: np.array, rotate_deg):
    # rotate
    rotated = rotate_cv(im, rotate_deg)

    # crop to remove black background
    w, h = im.shape[1], im.shape[0]
    new_width, new_height = rotated_crop_dims(w, h, rotate_deg)
    left, top, right, bottom = center_crop_coord(w, h, new_width, new_height)
    left, top, right, bottom = round(left), round(top), round(right), round(bottom)
    cropped = rotated[top:bottom, left:right]

    return cropped


def pil_rotate_crop_max(im: Image, rotate_deg):
    rotated = im.rotate(rotate_deg)

    w, h = im.size
    new_width, new_height = rotated_crop_dims(w, h, rotate_deg)
    left, top, right, bottom = center_crop_coord(w, h, new_width, new_height)
    left, top, right, bottom = round(left), round(top), round(right), round(bottom)
    cropped = rotated.crop((left, top, right, bottom))

    return cropped


def center_crop_coord(width, height, new_width, new_height):
    if width < new_width:
        new_width = width
    if height < new_height:
        new_height = height

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return left, top, right, bottom


# https://stackoverflow.com/a/16778797/6880256
def rotated_crop_dims(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    angle = math.radians(angle)

    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


# a copy of torch vision's random resize crop, except that it is deterministic for every instance of it
# I did this by moving get_params to __init__ instead of forward
class FixedRandomResizedCrop(nn.Module):

    def __init__(self, height, width, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument 'interpolation' of type int is deprecated since 0.13 and will be removed in 0.15. "
                "Please use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

        self.i, self.j, self.h, self.w = self.get_params(height, width, self.scale, self.ratio)

    @staticmethod
    def get_params(height, width, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:

        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        return F.resized_crop(img, self.i, self.j, self.h, self.w, self.size, self.interpolation)
