from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

from src.utils import find_all_image_paths


def delete_duplicates(root):
    im_paths = sorted(find_all_image_paths(root))

    seen = set()
    to_delete = []
    uniques = 0
    for im_path in tqdm(im_paths, total=len(im_paths), desc='del dup'):
        hash_code = imagehash.dhash(Image.open(im_path), hash_size=16)
        if hash_code not in seen:
            seen.add(hash_code)
            uniques += 1
        else:
            to_delete.append(im_path)

    for path in to_delete:
        Path(path).unlink(missing_ok=False)

    print(f'root: {root}')
    print(f'all: {len(im_paths)}')
    print(f'duplicates: {len(to_delete)}')
    print(f'left over: {uniques}')


def main():
    ROOT = r'D:\UCL\labs\comp0122\datasets\processed\sketch_simplification'

    delete_duplicates(ROOT)


if __name__ == '__main__':
    main()
