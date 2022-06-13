import os
import random

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ml.file_utils import get_all_image_paths


class MyDataset(Dataset):

    def __init__(self, opt, train=True):
        dataset_folder = opt.dataset_train_folder if train else opt.dataset_test_folder
        root = os.path.join(opt.dataset_dir, dataset_folder)

        self.paths = sorted(get_all_image_paths(root))
        self.A_to_B = opt.A_to_B
        self.random_jitter = opt.random_jitter
        self.random_mirror = opt.random_mirror

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        A, B = self._split_input_output(self._read_im(self.paths[i]))

        transform = self._generate_transform()

        A, B = transform(A), transform(B)  # apply same transform to both A and B

        return (A, B) if self.A_to_B else (B, A)

    def _read_im(self, path):
        return Image.open(path).convert('RGB')

    def _split_input_output(self, AB):
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        return A, B

    def _generate_transform(self):
        additional_transforms = []

        if self.random_jitter:
            new_size = 286
            old_size = 256

            rand_x = random.randint(0, new_size - old_size)
            rand_y = random.randint(0, new_size - old_size)

            additional_transforms += [
                transforms.Resize((new_size, new_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
                transforms.Lambda(lambda im: self._crop(im, (rand_x, rand_y), (old_size, old_size)))
            ]

        if self.random_mirror:
            flip = random.random() > 0.5
            additional_transforms += [
                transforms.Lambda(lambda im: self._flip(im, flip)),
            ]

        return transforms.Compose([
            *additional_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _flip(self, im, flip):
        if flip:
            return im.transpose(Image.FLIP_LEFT_RIGHT)
        return im

    def _crop(self, im, pos, size):
        return im.crop((pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]))
