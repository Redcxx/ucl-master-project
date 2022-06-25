from ml.datasets import create_train_dataloaders
from ml.models import create_train_model
from ml.options import create_train_options


def main():
    opt = create_train_options('pix2pix')
    train_loader, test_loader = create_train_dataloaders(opt, 'pix2pix')
    model = create_train_model(opt, train_loader, test_loader, 'pix2pix')
    model.train()


if __name__ == '__main__':
    main()
