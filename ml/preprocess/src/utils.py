import os

IMG_EXTENSIONS = [
    '.jpg', '.jpeg',
    '.png', '.ppm', '.bmp',
    '.tif', '.tiff',
]


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def find_all_image_paths(root, recursive=True):
    paths = []
    assert os.path.isdir(root)

    for root, _, filenames in sorted(os.walk(root)):
        for filename in filenames:
            if is_image_file(filename):
                paths.append(os.path.join(root, filename))
        if not recursive:
            break

    return paths


def get_last_comp(path):
    return os.path.basename(os.path.normpath(path))