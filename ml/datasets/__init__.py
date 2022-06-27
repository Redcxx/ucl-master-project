from torch.utils.data import DataLoader

from ml.options.base import BaseTrainOptions
from ml.datasets.base import BaseDataset
from ml.file_utils import _find_cls_using_name


def create_train_dataloaders(opt: BaseTrainOptions, name):
    print(f'Finding train dataloaders with name: [{name}] ... ', end='')
    train_dataset = _find_cls_using_name(
        name,
        package='datasets',
        parent_class=BaseDataset,
        cls_postfix='TrainDataset'
    )(opt)
    print(f'done: [{train_dataset.__class__.__name__}] was created')

    print(f'Finding test dataloaders with name: [{name}] ... ', end='')
    test_dataset = _find_cls_using_name(
        name,
        package='datasets',
        parent_class=BaseDataset,
        cls_postfix='TestDataset'
    )(opt)
    print(f'done: [{test_dataset.__class__.__name__}] was created')

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle,
                                  num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                 pin_memory=opt.pin_memory)

    return train_dataloader, test_dataloader
