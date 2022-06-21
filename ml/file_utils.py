import os

IMG_EXTENSIONS = [
    '.jpg', '.jpeg',
    '.png', '.ppm', '.bmp',
    '.tif', '.tiff',
]


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def get_all_image_paths(root):
    paths = []
    assert os.path.isdir(root), root

    for root, _, filenames in sorted(os.walk(root)):
        for filename in filenames:
            if is_image_file(filename):
                paths.append(os.path.join(root, filename))

    return paths
