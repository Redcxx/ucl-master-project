import importlib

from ml.models.base_model import BaseModel
from ml.models.pix2pixModel import Pix2pixModel


def _find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".
    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    modellib = importlib.import_module("ml.models." + model_name)
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == model_name.lower() \
                and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        raise FileNotFoundError(f'model not found: {model_name}')

    return model


def create_model(opt) -> BaseModel:
    instance = _find_model_using_name(opt.model_name)(opt)
    print(f'[{instance.__class__.__name__}] was created')
    return instance
