import re
from pathlib import Path
from pprint import pprint

from src.functions import *
from src.pipeline import Pipeline
from src.preprocess_noghost.utils import find_innermost_folder_path

_line_match = re.compile(r'line|clean', re.IGNORECASE)
_line_sub = re.compile(r'^\d+|line|clean|_', re.IGNORECASE)

_fill_match = re.compile(r'colour|col|fill', re.IGNORECASE)
_fill_sub = re.compile(r'^\d+|colour|col|fill|_', re.IGNORECASE)

_tied_match = re.compile(r'td|tiedown', re.IGNORECASE)
_tied_sub = re.compile(r'^\d+|td|tiedown|_', re.IGNORECASE)


def find_matched_dirs(inner_paths):
    line_paths = {}
    fill_paths = {}
    tied_paths = {}

    for path in inner_paths:
        path = os.path.abspath(path)
        components = path.split(os.path.sep)
        inner_folder = components[-1]
        parents = components[:-1]

        if _line_match.search(inner_folder) is not None:
            cleaned_inner_folder = _line_sub.sub('', inner_folder)
            cleaned_path = os.path.join(*parents, cleaned_inner_folder)
            line_paths[cleaned_path] = path

        if _fill_match.search(inner_folder) is not None:
            cleaned_inner_folder = _fill_sub.sub('', inner_folder)
            cleaned_path = os.path.join(*parents, cleaned_inner_folder)
            fill_paths[cleaned_path] = path

        if _tied_match.search(inner_folder) is not None:
            cleaned_inner_folder = _tied_sub.sub('', inner_folder)
            cleaned_path = os.path.join(*parents, cleaned_inner_folder)
            tied_paths[cleaned_path] = path

    # pprint(line_paths)
    # pprint(fill_paths)
    # pprint(tied_paths)

    line_fill_matched_dir = []
    line_tied_matched_dir = []
    for line_id in line_paths.keys():
        if line_id in fill_paths and line_paths[line_id] != fill_paths[line_id]:
            line_fill_matched_dir.append((line_paths[line_id], fill_paths[line_id]))

        if line_id in tied_paths and line_paths[line_id] != tied_paths[line_id]:
            line_tied_matched_dir.append((line_paths[line_id], tied_paths[line_id]))

    return line_fill_matched_dir, line_tied_matched_dir


def matched_dir_to_im_paths(matched_dirs):
    im_paths = []
    for dir_A, dir_B in matched_dirs:
        dir_A_im_paths = sorted(find_all_image_paths(dir_A))
        dir_B_im_paths = sorted(find_all_image_paths(dir_B))

        for path_A, path_B in zip(dir_A_im_paths, dir_B_im_paths):
            im_paths.append((path_A, path_B))

    return im_paths


_non_letter_number = re.compile('[^a-zA-Z0-9]')


def add_save_path(matched_im_paths, input_dataset_root):
    # tag for this input dataset
    input_dataset_tag = os.path.basename(os.path.normpath(input_dataset_root))

    def path_tag(path: str):
        return _non_letter_number.sub('', path)[-52:]

    paths = [
        (
            matched_im_paths[i][0],
            matched_im_paths[i][1],
            os.path.join(
                input_dataset_root,
                '-'.join([
                    input_dataset_tag,
                    path_tag(matched_im_paths[i][0]),
                    path_tag(matched_im_paths[i][1]),
                    str(i)
                ]) + '.jpg'
            )
        ) for i in range(len(matched_im_paths))
    ]

    return paths


def filter_bad_tie_down(paths):
    good_paths = []
    BAD_TAGS = {'sh_0120', 'sh_0120', 'sh_0190', 'piton'}
    for line_path, tied_path, save_path in paths:
        is_good_path = True
        for path in [line_path, tied_path]:
            if any(bad_tag in os.path.abspath(path).lower() for bad_tag in BAD_TAGS):
                is_good_path = False
                break
        if is_good_path:
            good_paths.append((line_path, tied_path, save_path))

    print(f'Filtered {len(paths) - len(good_paths)} bad paths')

    return good_paths


def main():
    print('Preprocess Started')

    # show running configuration
    INPUT_DATASET_ROOT = r'D:/UCL/labs/comp0122/datasets/RAW_DATA-20220624T112436Z-001'
    OUTPUT_DATASET_ROOT = r'D:/UCL/labs/comp0122/datasets/processed'

    # find innermost folder, which should be the one containing images
    im_folder_paths = find_innermost_folder_path(INPUT_DATASET_ROOT)

    # find paths that match
    line_fill_dirs, line_tied_dirs = find_matched_dirs(im_folder_paths)

    # find matched image paths from matched image dirs
    line_fill_paths = matched_dir_to_im_paths(line_fill_dirs)
    line_tied_paths = matched_dir_to_im_paths(line_tied_dirs)
    print(f'line fill matched dirs={len(line_fill_dirs)}, images={len(line_fill_paths)}')
    pprint(line_fill_dirs)
    print(f'line tied matched dirs={len(line_tied_dirs)}, images={len(line_tied_paths)}')
    pprint(line_tied_dirs)

    # output directory
    line_fill_dataset_root = os.path.join(OUTPUT_DATASET_ROOT, 'colorization')
    line_tied_dataset_root = os.path.join(OUTPUT_DATASET_ROOT, 'sketch_simplification')
    Path(line_fill_dataset_root).mkdir(parents=True, exist_ok=True)
    Path(line_tied_dataset_root).mkdir(parents=True, exist_ok=True)

    line_fill_paths = add_save_path(line_fill_paths, line_fill_dataset_root)
    line_tied_paths = add_save_path(line_tied_paths, line_tied_dataset_root)

    # extra processing
    line_tied_paths = filter_bad_tie_down(line_tied_paths)

    # Pipeline(workers=4, multi_process=True) \
    #     .add(pil_read_images, args=line_fill_paths) \
    #     .add(pil_resize_to_same) \
    #     .add(pil_crop_empty_surrounding) \
    #     .add(pil_horizontal_combine_images) \
    #     .add(pil_rgba_to_rgb) \
    #     .add(pil_save_image) \
    #     .run()

    Pipeline(workers=4, multi_process=True) \
        .add(pil_read_images, args=line_tied_paths) \
        .add(ab_to_ba) \
        .add(pil_resize_to_same) \
        .add(pil_crop_empty_surrounding) \
        .add(pil_horizontal_combine_images) \
        .add(pil_rgba_to_rgb) \
        .add(pil_rgb_to_gray) \
        .add(pil_save_image) \
        .run()


if __name__ == '__main__':
    main()
