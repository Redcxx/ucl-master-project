from ml.datasets import create_train_dataloaders
from ml.models import create_train_model
from ml.options import create_train_options


def main():
    opt = create_train_options('sketch_simp')
    train_loader, test_loader = create_train_dataloaders(opt, 'sketch_simp')
    model = create_train_model(opt, train_loader, test_loader, 'sketch_simp')
    model.train()


if __name__ == '__main__':
    main()
