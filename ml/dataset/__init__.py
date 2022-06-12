from torch.utils.data import DataLoader

from ml.dataset.dataset import MyDataset
from ml.session import SessionOptions


def create_dataloaders(opt: SessionOptions):
    train_dataset = MyDataset(opt, train=True)
    test_dataset = MyDataset(opt, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle,
                                  num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers,
                                 pin_memory=opt.pin_memory)

    return train_dataloader, test_dataloader
