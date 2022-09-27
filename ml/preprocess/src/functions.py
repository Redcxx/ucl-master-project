import os

import cv2 as cv
import imagehash
import numpy as np
from PIL import Image, ImageOps

from ml.preprocess.src.utils import find_all_image_paths


def make_input_output_paths(A_dir, B_dir, output_root, a_to_b=True):
    A_paths = sorted(find_all_image_paths(A_dir))
    B_paths = sorted(find_all_image_paths(B_dir))

    AB_paths = []
    out_paths = []
    for i, (a_path, b_path) in enumerate(zip(A_paths, B_paths), 1):
        out_paths.append(os.path.join(output_root, f'{i}.png'))
        AB_paths.append((a_path, b_path) if a_to_b else (b_path, a_path))

    return AB_paths, out_paths


def cv_read_images(prev_out, args):
    path_A, path_B = args

    # read image
    im_A = cv.imread(path_A, cv.IMREAD_UNCHANGED)
    im_B = cv.imread(path_B, cv.IMREAD_UNCHANGED)

    return im_A, im_B


def cv_read_images_unchanged(prev_out, args):
    path_A, path_B = args

    # read image
    im_A = cv.imread(path_A, cv.IMREAD_UNCHANGED)
    im_B = cv.imread(path_B, cv.IMREAD_UNCHANGED)

    return im_A, im_B


def cv_rgba_to_rgb_(im, bg_color=(1, 1, 1)):
    if im.shape[2] <= 3:
        return im
    im = im.astype(float) / 255
    alpha = im[:, :, -1][..., np.newaxis]
    rgb = im[:, :, :-1]
    bg = np.stack((
        np.full(im.shape[:-1], bg_color[0], dtype=float),
        np.full(im.shape[:-1], bg_color[1], dtype=float),
        np.full(im.shape[:-1], bg_color[2], dtype=float)
    ), axis=-1)
    bg = (1 - alpha) * bg
    fg = alpha * rgb
    return ((bg + fg) * 255).astype(np.uint8)


def cv_rgba_to_rgb(prev_out, args):
    im_A, im_B = prev_out
    bg_color = args

    if bg_color is None:
        bg_color = (1, 1, 1)

    return cv_rgba_to_rgb_(im_A, bg_color), cv_rgba_to_rgb_(im_B, bg_color)


def cv_convert_rgb(prev_out, args):
    im_A, im_B = prev_out

    im_A = cv.cvtColor(im_A, cv.COLOR_BGR2RGB)
    im_B = cv.cvtColor(im_B, cv.COLOR_BGR2RGB)

    return im_A, im_B


def cv_convert_gray(prev_out, args):
    im_A, im_B = prev_out

    im_A = cv.cvtColor(im_A, cv.COLOR_BGR2GRAY)
    im_B = cv.cvtColor(im_B, cv.COLOR_BGR2GRAY)

    return im_A, im_B


def pil_rgb_to_gray(prev_out, args):
    im_A, im_B = prev_out

    im_A = im_A.convert('L')
    im_B = im_B.convert('L')

    return im_A, im_B


def cv_resize_to_same(prev_out, args):
    im_A, im_B = prev_out

    if im_A.shape != im_B.shape:
        im_B = cv.resize(im_B, (im_A.shape[1], im_A.shape[0]))

    return im_A, im_B


def _cv_empty_surrounding_coord(im):
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

    # we want to remove white background,
    # so we convert the white background to black and use find non-zero later and crop it
    gray = 255 * (gray < 128).astype(np.uint8)
    coords = cv.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv.boundingRect(coords)  # Find minimum spanning bounding box
    return x, y, w, h


def cv_crop_empty_surrounding(prev_out, args):
    im_A, im_B = prev_out

    x, y, w, h = _cv_empty_surrounding_coord(im_A)

    im_A = im_A[y:y + h, x:x + w]
    im_B = im_B[y:y + h, x:x + w]

    return im_A, im_B


def cv_horizontal_concat_im(prev_out, args):
    im_A, im_B = prev_out
    return np.concatenate((im_A, im_B), axis=1)


def cv_save_im(prev_out, args):
    im = prev_out
    save_file = args

    if im.size == 0:
        return
    cv.imwrite(save_file, im)


def pil_read_images(prev_out, args):
    A, B, S = args

    im_A = Image.open(A)
    im_B = Image.open(B)

    return im_A, im_B, S


def pil_resize_to_same(prev_out, args):
    im_A, im_B, S = prev_out

    if im_A.size != im_B.size:
        im_B = im_B.resize(im_A.size, resample=Image.BICUBIC)

    return im_A, im_B, S


def pil_crop_empty_surrounding(prev_out, args):
    im_A, im_B, S = prev_out

    # crop to non empty region
    non_empty_coord = im_A.getbbox()

    if non_empty_coord is None:
        # empty image
        return None

    im_A = im_A.crop(non_empty_coord)
    im_B = im_B.crop(non_empty_coord)

    return im_A, im_B, S


def pil_pad_resize(prev_out, args):
    im_A, im_B = prev_out
    resize_size = args

    im_A = ImageOps.pad(im_A, (resize_size, resize_size), method=Image.BICUBIC, color='rgba(0,0,0,0)')
    im_B = ImageOps.pad(im_B, (resize_size, resize_size), method=Image.BICUBIC, color='rgba(0,0,0,0)')

    return im_A, im_B


def crop_resize(prev_out, args):
    im_A, im_B = prev_out
    resize_size = args

    crop_size = min(im_A.size[0], im_A.size[1])
    crop_coord = center_crop_coord(im_A, (crop_size, crop_size))

    im_A = im_A.crop(crop_coord)
    im_B = im_B.crop(crop_coord)

    im_A = im_A.resize((resize_size, resize_size), resample=Image.BICUBIC)
    im_B = im_B.resize((resize_size, resize_size), resample=Image.BICUBIC)

    return im_A, im_B


def combine_image_horizontal(prev_out, args):
    im_A, im_B = prev_out[:2]

    assert im_A.size[0] == im_B.size[0] and im_A.size[1] == im_B.size[1]
    width, height = im_A.size

    im = Image.new('RGBA', (width * 2, height))
    im.paste(im_A, (0, 0))
    im.paste(im_B, (width, 0))

    # convert to RGB, do not use .convert('RGB') as it gives black background
    rgb = Image.new("RGB", im.size, (255, 255, 255))
    rgb.paste(im, mask=im.split()[3])

    return rgb, *prev_out[2:]


def ab_to_ba(prev_out, args):
    A, B, S = prev_out

    return B, A, S


def pil_horizontal_combine_images(prev_out, args):
    im_A, im_B, S = prev_out

    assert im_A.size == im_B.size

    w, h = im_A.size

    im = Image.new(im_A.mode, (w * 2, h))
    im.paste(im_A, (0, 0))
    im.paste(im_B, (w, 0))

    return im, S


def pil_rgba_to_rgb(prev_out, args):
    im, S = prev_out

    rgb = Image.new("RGB", im.size, (255, 255, 255))
    rgb.paste(im, mask=im.split()[3])

    return rgb, S


def pil_save_image(prev_out, args):
    im, save_path = prev_out

    im.save(save_path)
    return im, save_path


def pil_compute_hash(prev_out, args):
    im, save_path = prev_out
    hash_size = args

    return imagehash.dhash(im, hash_size=hash_size), save_path


def save_image(prev_out, args):
    im = prev_out
    save_path = args

    im.save(save_path)


def compute_hash(prev_out, args):
    im = prev_out

    return imagehash.phash(im)


def center_crop_coord(im, new_size):
    width, height = im.size
    new_width, new_height = new_size

    assert width >= new_width
    assert height >= new_height

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return left, top, right, bottom


def compute_weight_map(prev_out, args):
    im = prev_out
    n_bins, dist = args

    lin = np.linspace(0, 1, n_bins + 1)
    out = np.zeros(im.shape)

    w, h = im.shape[:2]
    for x in range(w):
        x_min = max(0, x - dist)
        x_max = min(w, x + dist)
        for y in range(h):
            y_min = max(0, y - dist)
            y_max = min(h, y + dist)
            local = im[x_min:x_max, y_min:y_max]
            bins = np.empty(n_bins)
            for i in range(n_bins):
                bins[i] = np.sum((local > lin[i]) * (local < lin[i + 1])) / local.size
            p = im[x, y]
            n = 0
            for i in range(n_bins):
                if lin[i] <= p <= lin[i + 1]:
                    n = i
                    break
            out[x, y] = np.exp(-bins[n]) + 0.5

    return out
