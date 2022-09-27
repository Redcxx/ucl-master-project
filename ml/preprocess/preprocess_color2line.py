import os
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import cv2 as cv

from src.process import center_crop_coord
from src.utils import find_all_image_paths
from src.xdog import extract_edges_cv

RESIZE_SIZE = 512


def get_last_path_comp(path):
    return os.path.basename(os.path.normpath(path))


def sketch_process(item):
    save_path, im_path, sigma = item

    im = cv.imread(im_path)
    im = extract_edges_cv(im, sigma=sigma)

    cv.imwrite(save_path, im*255)


# crop and resize to shape=(RESIZE_SIZE, RESIZE_SIZE)
def color_process(item):
    save_path, path = item

    im = Image.open(path)
    crop_size = min(im.size[0], im.size[1])
    crop_coord = center_crop_coord(im, (crop_size, crop_size))

    im = im.crop(crop_coord)

    im = im.resize((RESIZE_SIZE, RESIZE_SIZE), resample=Image.BICUBIC)

    im.save(save_path)


def main():
    # settings
    IMAGE_FOLDER_ROOT = r'D:\UCL\labs\comp0122\datasets\user_guided_processed\fill'
    OUTPUT_ROOT = r'D:\UCL\labs\comp0122\datasets\user_guided_processed\diff_xdog'

    COLOR_ROOT = os.path.join(OUTPUT_ROOT, 'color')
    SIGMA_03_ROOT = os.path.join(OUTPUT_ROOT, 'sigma_03')
    SIGMA_04_ROOT = os.path.join(OUTPUT_ROOT, 'sigma_04')
    SIGMA_05_ROOT = os.path.join(OUTPUT_ROOT, 'sigma_05')
    # create output dir
    Path(COLOR_ROOT).mkdir(parents=True, exist_ok=True)
    Path(SIGMA_03_ROOT).mkdir(parents=True, exist_ok=True)
    Path(SIGMA_04_ROOT).mkdir(parents=True, exist_ok=True)
    Path(SIGMA_05_ROOT).mkdir(parents=True, exist_ok=True)

    # find all image paths
    im_paths = find_all_image_paths(IMAGE_FOLDER_ROOT)

    # process color
    color_worker_inputs = []
    sketch_worker_inputs = []
    for i, im_path in enumerate(im_paths):
        filename = f'{i}.jpg'
        color_save_path = os.path.join(COLOR_ROOT, filename)
        sigma_03_save_path = os.path.join(SIGMA_03_ROOT, filename)
        sigma_04_save_path = os.path.join(SIGMA_04_ROOT, filename)
        sigma_05_save_path = os.path.join(SIGMA_05_ROOT, filename)

        color_worker_inputs.append((color_save_path, im_path))
        sketch_worker_inputs.append((sigma_03_save_path, color_save_path, 0.3))
        sketch_worker_inputs.append((sigma_04_save_path, color_save_path, 0.4))
        sketch_worker_inputs.append((sigma_05_save_path, color_save_path, 0.5))

    with Pool(os.cpu_count() - 2) as pool:
        for _ in tqdm(pool.imap_unordered(color_process, color_worker_inputs),
                      total=len(color_worker_inputs),
                      desc='color'):
            pass

    # compute sketches
    with Pool(os.cpu_count() - 2) as pool:
        for _ in tqdm(pool.imap_unordered(sketch_process, sketch_worker_inputs),
                      total=len(sketch_worker_inputs),
                      desc='sketch'):
            pass


if __name__ == '__main__':
    main()
