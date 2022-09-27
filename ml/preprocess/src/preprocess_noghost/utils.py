import os
from collections import defaultdict
from pprint import pprint
from typing import Iterable, Tuple

from ml.preprocess.src.preprocess_noghost.image_path import ImagePath
from ml.preprocess.src.utils import find_all_image_paths


def find_innermost_folder_path(root, relative=True):
    paths = []
    assert os.path.isdir(root)

    for root, inner_folders, filenames in sorted(os.walk(root)):
        if len(inner_folders) == 0:
            paths.append(os.path.relpath(root) if relative else root)

    return paths


# returns a map from identifier to image path
def compute_unique_paths(paths):
    id2paths = defaultdict(set)
    for path in paths:
        im_path = ImagePath(path)
        id2paths[im_path.identifier].add(im_path)

    unique_paths = []
    for identifier, paths in id2paths.items():
        if len(paths) > 1:
            print(f'Duplicate Identifier for \"{identifier}\"')
            pprint(paths)
        else:
            unique_paths.append(next(iter(paths)))

    return unique_paths


def compute_matched_dirs(image_paths: Iterable[ImagePath]):
    line_paths = dict()
    fill_paths = dict()
    tied_paths = dict()

    # remove corresponding attributes in the path, e.g. COL, FILL, TD, ...

    for path in image_paths:

        if path.identifier.line_matched():
            line_paths[path.identifier.remove_line_attrs()] = path

        if path.identifier.fill_matched():
            fill_paths[path.identifier.remove_fill_attrs()] = path

        if path.identifier.tied_matched():
            tied_paths[path.identifier.remove_tied_attrs()] = path

    # print('=' * 50)
    # print('line paths')
    # print('=' * 50)
    # pprint(line_paths)
    # print('=' * 50)
    # print('fill paths')
    # print('=' * 50)
    # pprint(fill_paths)
    # print('=' * 50)
    # print('tied paths')
    # print('=' * 50)
    # pprint(tied_paths)

    # if after removing these attributes, we get the same text, then these two folders should match

    line_fill_matched_dir = []
    line_tied_matched_dir = []
    for line_id in line_paths.keys():
        if line_id in fill_paths and line_paths[line_id] != fill_paths[line_id]:
            line_fill_matched_dir.append((line_paths[line_id], fill_paths[line_id]))

        if line_id in tied_paths and line_paths[line_id] != tied_paths[line_id]:
            line_tied_matched_dir.append((line_paths[line_id], tied_paths[line_id]))

    return line_fill_matched_dir, line_tied_matched_dir


def matched_dir_to_im_paths(matched_dirs: Iterable[Tuple[ImagePath, ImagePath]]):
    im_paths = []
    for dir_A, dir_B in matched_dirs:
        dir_A_im_paths = sorted(find_all_image_paths(dir_A.norm_path))
        dir_B_im_paths = sorted(find_all_image_paths(dir_B.norm_path))

        for path_A, path_B in zip(dir_A_im_paths, dir_B_im_paths):
            im_paths.append((path_A, path_B))

    return im_paths
