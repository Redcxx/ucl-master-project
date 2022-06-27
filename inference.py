from ml.datasets import create_inference_dataloaders
from ml.models import create_inference_model
from ml.options import create_inference_options


def main():
    name = 'alac_gan'
    opt = create_inference_options(name)
    train_loader, test_loader = create_inference_dataloaders(opt, name)
    model = create_inference_model(opt, train_loader, test_loader)

    print('Inference started')
    model.inference()
    print('Inference finished')


if __name__ == '__main__':
    main()
