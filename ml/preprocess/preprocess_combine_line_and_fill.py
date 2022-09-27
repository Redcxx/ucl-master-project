import os.path
from pathlib import Path

from src.process import multi_process
from src.utils import find_all_image_paths


def main():
    LINE_DATASET_ROOT = r'../../datasets/user_guided_processed/line'
    FILL_DATASET_ROOT = r'../../datasets/user_guided_processed/fill'
    OUTPUT_ROOT = '../../datasets/user_guided_processed/combined'

    print(f'Output Directory: {os.path.abspath(OUTPUT_ROOT)}')
    Path(OUTPUT_ROOT).mkdir(exist_ok=True, parents=True)

    line_im_paths = sorted(find_all_image_paths(LINE_DATASET_ROOT))
    fill_im_paths = sorted(find_all_image_paths(FILL_DATASET_ROOT))
    print(f'line paths: {len(line_im_paths)}')
    print(f'fill paths: {len(fill_im_paths)}')
    print('Above image path should be corresponding and same amount')

    worker_inputs = []
    for i, (line_path, fill_path) in enumerate(zip(line_im_paths, fill_im_paths), 1):
        out_path = os.path.join(OUTPUT_ROOT, f'{i}.png')
        worker_inputs.append((line_path, fill_path, out_path, i))

    print('Combine Started')

    multi_process(worker_inputs)

    print('Combine Ended')


if __name__ == '__main__':
    main()
