import importlib

from ml.models.base_model import BaseModel
from ml.models.pix2pix_model import Pix2pixModel


def _find_model_using_name(model_name):
    modellib = importlib.import_module("ml.models." + model_name)
    model = None
    for name, cls in modellib.__dict__.items():
        if name.replace('_', '').lower() == model_name.lower() \
                and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        raise FileNotFoundError(f'model not found: {model_name}')

    return model


def create_model(opt) -> BaseModel:
    instance = _find_model_using_name(opt.model_name)(opt)
    print(f'[{instance.__class__.__name__}] was created')
    return instance
