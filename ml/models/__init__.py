from ml.logger import log
from ml.models.base import BaseTrainModel, BaseInferenceModel
from ml.file_utils import _find_cls_using_name


def create_train_model(opt, train_loader, test_loader, name):

    log(f'Finding train model with name: [{name}] ... ')

    cls = _find_cls_using_name(
        name,
        package='models',
        parent_class=BaseTrainModel,
        cls_postfix='TrainModel'
    )

    instance = cls(opt, train_loader, test_loader)

    log(f'done: [{instance.__class__.__name__}] was created')
    return instance


def create_inference_model(opt, inference_loader, name):

    log(f'Finding inference model with name: [{name}] ... ')

    cls = _find_cls_using_name(
        name,
        package='models',
        parent_class=BaseInferenceModel,
        cls_postfix='InferenceModel'
    )

    instance = cls(opt, inference_loader)

    log(f'done: [{instance.__class__.__name__}] was created')
    return instance
