from ml.datasets import create_inference_dataloaders
from ml.models import create_train_model
from ml.options.base_options import BaseInferenceOptions


def main():
    opt = BaseInferenceOptions()
    opt.test_loader, opt.test_dataset = create_inference_dataloaders(opt)
    model = create_train_model(opt)

    print('Evaluate Started')
    model.evaluate(0, progress=True)
    print('Evaluate Finish')


if __name__ == '__main__':
    main()
