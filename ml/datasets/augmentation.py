import cv2 as cv
import numpy as np
from PIL import Image


def rotate_pil(im: Image.Image, rotate_deg):
    return im.rotate(rotate_deg, resample=Image.BILINEAR)


def rotate_cv(im: np.array, rotate_deg):
    image_center = tuple(np.array(im.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, rotate_deg, 1.0)
    im = cv.warpAffine(im, rot_mat, im.shape[1::-1], flags=cv.INTER_LINEAR)
    return im


def flip_horizontal_cv(im: np.array):
    return cv.flip(im, 1)  # 1 = horizontal, 0 = vertical, -1 = both


def flip_horizontal_pil(im: Image.Image):
    return im.transpose(Image.FLIP_LEFT_RIGHT)
