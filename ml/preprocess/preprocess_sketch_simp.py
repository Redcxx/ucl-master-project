from pathlib import Path

from src.functions import make_input_output_paths, cv_read_images, cv_convert_gray, cv_resize_to_same, \
    cv_horizontal_concat_im, cv_save_im
from src.pipeline import Pipeline


def threshold_second(prev_out, args):
    A, B = prev_out
    B[B < 0.9] = 0

    return A, B


def main():
    INPUT_PATH = r'/datasets/sketch_simplication/sketch_simp_pretrain_data/Input'
    TARGET_PATH = r'/datasets/sketch_simplication/sketch_simp_pretrain_data/Target'
    OUTPUT_ROOT = r'D:\UCL\labs\comp0122\datasets\sketch_simplication\sketch_simp_pretrain_data/processed'

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    AB_paths, save_paths = make_input_output_paths(INPUT_PATH, TARGET_PATH, OUTPUT_ROOT)

    Pipeline(workers=4, multi_process=True) \
        .add(cv_read_images, args=AB_paths) \
        .add(cv_convert_gray) \
        .add(cv_resize_to_same) \
        .add(threshold_second) \
        .add(cv_horizontal_concat_im) \
        .add(cv_save_im, args=save_paths) \
        .run()


if __name__ == '__main__':
    main()
