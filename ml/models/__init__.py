import importlib

from ml.models.base_model import BaseModel
from ml.models.pix2pix_model import Pix2pixModel


def _find_model_using_name(model_name):
    model_lib = importlib.import_module("ml.models." + model_name)
    model = None
    for name, cls in model_lib.__dict__.items():
        if name.lower() == model_name.replace('_', '').lower() \
                and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        raise FileNotFoundError(f'model not found: {model_name}')

    return model


def create_model(opt) -> BaseModel:
    instance = _find_model_using_name(opt.model_name)(opt)
    print(f'Model: [{instance.__class__.__name__}] was created')
    return instance
