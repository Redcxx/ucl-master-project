import importlib
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


def _find_cls_using_name(name: str, package: str, parent_class: type, cls_postfix: str) -> type:
    model_lib = importlib.import_module(f"ml.{package}." + name.lower())

    found_cls = None
    for cls_name, cls in model_lib.__dict__.items():
        if cls_name.replace('_', '').lower() == (name + cls_postfix).replace('_', '').lower() \
                and issubclass(cls, parent_class):
            found_cls = cls

    if found_cls is None:
        raise FileNotFoundError(f'Class not found: {name} in package: {package} with parent: {parent_class}')

    return found_cls
