import os
import shutil
from pathlib import Path

import cv2 as cv
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocess_delete_duplicates import delete_duplicates
from src.functions import cv_rgba_to_rgb_
from src.pipeline import Pipeline
from src.utils import find_all_image_paths


def read_image(prev_out, args):
    return cv.imread(args, cv.IMREAD_UNCHANGED)


def down_scale(prev_out, args):
    im = prev_out
    scale = 1 / args
    h, w, c = im.shape
    new_h, new_w = int(h * scale), int(w * scale)
    return cv.resize(im, (new_w, new_h), cv.INTER_CUBIC), im


def save_image(prev_out, args):
    lr, hr = prev_out
    lr_path, hr_path = args

    cv.imwrite(lr_path, lr)
    cv.imwrite(hr_path, hr)


def rgba_to_rgb(prev_out, _):
    lr, hr = prev_out
    lr = cv_rgba_to_rgb_(lr, bg_color=(1, 1, 1))
    hr = cv_rgba_to_rgb_(hr, bg_color=(1, 1, 1))
    return lr, hr


def drop_small_images(prev_out, min_size):
    lr, hr = prev_out
    if lr.shape[0] < min_size or lr.shape[1] < min_size \
            or hr.shape[0] < min_size or hr.shape[1] < min_size:
        return None
    else:
        return lr, hr


def pil_read_im(prev_out, args):
    return Image.open(args)


def pil_rgba_to_rgb(im, _):
    rgb = Image.new("RGB", im.size, (255, 255, 255))
    rgb.paste(im, mask=im.split()[3])
    return rgb


def pil_save_image(im, path):
    im.save(path)


def pil_crop_empty(im, _):
    non_empty_coord = im.getbbox()
    if non_empty_coord is None:
        return None  # return None means early termination
    im = im.crop(non_empty_coord)
    return im


def delete_im(args):
    im_path = args[0]
    Path(im_path).unlink(missing_ok=False)


def make_input(im_paths, out_root, scale):
    out_paths = []
    scales = []
    Path(os.path.join(out_root, str(scale))).mkdir(parents=True, exist_ok=True)
    train_paths, test_paths = train_test_split(im_paths,
                                               train_size=0.8,
                                               random_state=42,
                                               shuffle=True)
    train_size = len(train_paths)
    paths = train_paths + test_paths

    for i, path in enumerate(paths):
        split = 'train' if i < train_size else 'test'
        lr_root = os.path.join(out_root, str(scale), split, 'lr')
        hr_root = os.path.join(out_root, str(scale), split, 'hr')

        Path(lr_root).mkdir(parents=True, exist_ok=True)
        Path(hr_root).mkdir(parents=True, exist_ok=True)

        lr_path = os.path.join(lr_root, f'{i}.jpg')
        hr_path = os.path.join(hr_root, f'{i}.jpg')
        out_paths.append((lr_path, hr_path))
        scales.append(scale)

    return im_paths, out_paths, scales


def main():
    HIGH_RES_IMAGES_FOLDER = r'D:\UCL\labs\comp0122\datasets\RAW_DATA-20220624T112436Z-001'
    OUTPUT_ROOT = r'D:\UCL\labs\comp0122\datasets\upsample/noghost_upsample'  # should be an empty dir

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    # delete duplicates
    # first copy all images to a folder, i.e. output root
    all_images = find_all_image_paths(HIGH_RES_IMAGES_FOLDER)
    cp_im_paths = []
    for i, im_path in tqdm(enumerate(all_images), total=len(all_images), desc='copy'):
        basename = os.path.basename(im_path)
        new_im_path = os.path.join(OUTPUT_ROOT, f'{i}_{basename}')
        cp_im_paths.append(new_im_path)
        shutil.copy2(im_path, new_im_path)

    Pipeline(workers=4, multi_process=True) \
        .add(pil_read_im, args=cp_im_paths) \
        .add(pil_crop_empty) \
        .add(pil_save_image, args=cp_im_paths) \
        .on_early_terminate(delete_im) \
        .run(desc='crop empty')

    # sys.exit(0)

    # call delete duplicate function
    delete_duplicates(OUTPUT_ROOT)

    # proceed, but find images from the output root
    im_paths = find_all_image_paths(OUTPUT_ROOT)
    down_scales = [2, 4]

    in_paths = []
    out_paths = []
    scales = []
    for scale in down_scales:
        in_paths_, out_paths_, scales_ = make_input(im_paths, OUTPUT_ROOT, scale)
        in_paths += in_paths_
        out_paths += out_paths_
        scales += scales_

    Pipeline(workers=4, multi_process=True) \
        .add(read_image, args=in_paths) \
        .add(down_scale, args=scales) \
        .add(rgba_to_rgb) \
        .add(drop_small_images, args=64) \
        .add(save_image, args=out_paths) \
        .run(desc='process')

    # remove copied over original images
    for p in cp_im_paths:
        Path(p).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
