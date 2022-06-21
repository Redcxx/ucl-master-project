from ml.dataset import create_inference_dataloaders
from ml.models import create_model
from ml.options import InferenceOptions


def main():
    opt = InferenceOptions()
    opt.test_loader, opt.test_dataset = create_inference_dataloaders(opt)
    model = create_model(opt)

    print('Evaluate Started')
    model.evaluate(0, progress=True)
    print('Evaluate Finish')


if __name__ == '__main__':
    main()
