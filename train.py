from ml.datasets import create_train_dataloaders
from ml.models import create_train_model
from ml.options import create_train_options


def main():
    opt = create_train_options('waifu2x')
    train_loader, test_loader = create_train_dataloaders(opt, 'waifu2x')
    model = create_train_model(opt, train_loader, test_loader, 'waifu2x')
    model.train()


if __name__ == '__main__':
    main()
