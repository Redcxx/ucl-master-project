from torch.utils.data import DataLoader

from ml.dataset.dataset import TrainDataset, InferenceDataset
from ml.options import TrainOptions, InferenceOptions


def create_train_dataloaders(opt: TrainOptions):
    train_dataset = TrainDataset(opt, train=True)
    test_dataset = TrainDataset(opt, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle,
                                  num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers,
                                 pin_memory=opt.pin_memory)

    return train_dataloader, test_dataloader, train_dataset, train_dataloader


def create_inference_dataloaders(opt: InferenceOptions):
    inference_dataset = InferenceDataset(opt)

    inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False,
                                      num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    return inference_dataloader, inference_dataset
