from torch.utils.data import DataLoader

from ml.base_options import BaseTrainOptions
from ml.base_dataset import BaseDataset
from ml.file_utils import _find_cls_using_name


def create_train_dataloaders(opt: BaseTrainOptions, name):
    print(f'Finding train dataloaders with name: [{name}] ...')
    train_dataset = _find_cls_using_name(
        name,
        package='datasets',
        parent_class=BaseDataset,
        cls_postfix='TrainDataset'
    )()
    print(f'Dataset: [{train_dataset.__class__.__name__}] was created')

    test_dataset = _find_cls_using_name(
        name,
        package='datasets',
        parent_class=BaseDataset,
        cls_postfix='TestDataset'
    )()
    print(f'Dataset: [{test_dataset.__class__.__name__}] was created')

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle,
                                  num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers,
                                 pin_memory=opt.pin_memory)

    return train_dataloader, test_dataloader
