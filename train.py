from ml.datasets import create_train_dataloaders
from ml.models import create_train_model
from ml.options import create_train_options


def main():
    opt = create_train_options('alacgan')
    train_loader, test_loader = create_train_dataloaders(opt, 'alacgan')
    model = create_train_model(opt, train_loader, test_loader, 'alacgan')
    model.train()


if __name__ == '__main__':
    main()
