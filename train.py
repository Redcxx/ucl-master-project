from ml.datasets import create_train_dataloaders
from ml.models import create_train_model
from ml.options import create_train_options


def main():
    name = 'alac_gan'
    opt = create_train_options(name)
    train_loader, test_loader = create_train_dataloaders(opt, name)
    model = create_train_model(opt, train_loader, test_loader, name)
    model.train()


if __name__ == '__main__':
    main()
