import os
import shutil
from pathlib import Path

import paddlehub as hub
from tqdm import tqdm

from src.utils import find_all_image_paths

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_last_path_comp(path):
    return os.path.basename(os.path.normpath(path))


def main():
    # settings
    IMAGE_FOLDER_ROOT = r'D:/UCL/labs/comp0122/datasets/illustrations_resized'
    OUTPUT_ROOT = r'D:/UCL/labs/comp0122/datasets/processed'

    # create output dir
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, get_last_path_comp(IMAGE_FOLDER_ROOT))
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # find all image paths
    im_paths = find_all_image_paths(IMAGE_FOLDER_ROOT)

    # initialize model for extracting line draft
    model = hub.Module(name='Extract_Line_Draft')

    for path in tqdm(im_paths):
        model.ExtractLine(path, use_gpu=True)
        output_path = os.path.join(OUTPUT_DIR, get_last_path_comp(path))
        shutil.move('output/output.png', output_path)

    print(f'done, output directory: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
