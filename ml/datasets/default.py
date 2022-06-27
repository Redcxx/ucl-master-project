import os

from torchvision.transforms import transforms

from ml.datasets import BaseDataset
from ml.file_utils import get_all_image_paths
from ml.options import BaseTrainOptions
from ml.options.base import BaseInferenceOptions


class DefaultDataset(BaseDataset):
    def __init__(self, opt, root):
        super().__init__(opt)
        self.paths = sorted(get_all_image_paths(root))
        self.a_to_b = opt.a_to_b
        self.transform = transforms.Compose([
            transforms.Resize(size=opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        A, B = self._split_image(self._read_im(self.paths[i]))
        A, B = self.transform(A), self.transform(B)
        return (A, B) if self.a_to_b else (B, A)


class DefaultInferenceDataset(DefaultDataset):

    def __init__(self, opt: BaseInferenceOptions):
        super().__init__(opt, opt.input_images_path)


class DefaultTestDataset(DefaultDataset):
    def __init__(self, opt: BaseTrainOptions):
        root = os.path.join(opt.dataset_root, opt.dataset_test_folder)
        super().__init__(opt, root)


class DefaultTrainDataset(DefaultDataset):
    def __init__(self, opt: BaseTrainOptions):
        root = os.path.join(opt.dataset_root, opt.dataset_train_folder)
        super().__init__(opt, root)
