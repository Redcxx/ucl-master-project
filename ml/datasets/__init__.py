from torch.utils.data import DataLoader

from ml.datasets.base import BaseDataset
from ml.file_utils import _find_cls_using_name
from ml.options.base import BaseTrainOptions, BaseInferenceOptions


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
                                  num_workers=opt.num_workers, pin_memory=opt.pin_memory, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                 pin_memory=opt.pin_memory, drop_last=True)

    return train_dataloader, test_dataloader


def create_inference_dataloaders(opt: BaseInferenceOptions, name):
    print(f'Finding inference dataloaders with name: [{name}] ... ', end='')
    inference_dataset = _find_cls_using_name(
        name,
        package='datasets',
        parent_class=BaseDataset,
        cls_postfix='InferenceDataset'
    )(opt)
    print(f'done: [{inference_dataset.__class__.__name__}] was created')

    inference_dataloader = DataLoader(inference_dataset,
                                      batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                      pin_memory=opt.pin_memory, drop_last=True)

    return inference_dataloader
