import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ml.preprocess.src.utils import find_all_image_paths


def split_dataset(dataset_path, train_test_ratio, random_seed):
    print(f'Splitting: {dataset_path}')
    train_out_folder = os.path.join(dataset_path, 'train')
    test_out_folder = os.path.join(dataset_path, 'test')

    Path(train_out_folder).mkdir(parents=True, exist_ok=True)
    Path(test_out_folder).mkdir(parents=True, exist_ok=True)

    im_paths = find_all_image_paths(dataset_path, recursive=False)
    if len(im_paths) == 0:
        print(f'Dataset is empty: {dataset_path}')
        return

    train_paths, test_paths = train_test_split(im_paths,
                                               train_size=train_test_ratio,
                                               random_state=random_seed,
                                               shuffle=True)

    print(f'Number of training samples: {len(train_paths)}')
    print(f'Number of testing  samples: {len(test_paths)}')

    # do not specify the output filename in shutil.move so that it raise
    # warning when file already exists.

    for train_path in tqdm(train_paths, total=len(train_paths), desc='train'):
        shutil.move(train_path, train_out_folder)

    for test_path in tqdm(test_paths, total=len(test_paths), desc='test'):
        shutil.move(test_path, test_out_folder)
