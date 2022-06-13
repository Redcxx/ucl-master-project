from ml.dataset import create_dataloaders
from ml.models import create_model
from ml.session import SessionOptions


def main():
    opt = SessionOptions()
    opt.train_loader, opt.test_loader, opt.train_dataset, opt.test_dataset = create_dataloaders(opt)
    model = create_model(opt)

    model.pre_train()

    for epoch in range(opt.start_epoch, opt.end_epoch + 1 + opt.decay_epochs):

        model.pre_epoch()

        for batch, batch_data in enumerate(opt.train_loader, 1):
            model.pre_batch(epoch, batch)

            batch_out = model.train_batch(batch, batch_data)

            model.post_batch(epoch, batch, batch_out)

        model.post_epoch(epoch)

    model.post_train()


if __name__ == '__main__':
    main()
