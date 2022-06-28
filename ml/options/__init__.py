from ml.logger import log
from ml.options.base import BaseTrainOptions, BaseInferenceOptions
from ml.file_utils import _find_cls_using_name


def create_train_options(name):
    log(f'Finding train option with name: [{name}] ... ', end='')
    instance = _find_cls_using_name(
        name,
        package='options',
        parent_class=BaseTrainOptions,
        cls_postfix='TrainOptions'
    )()
    log(f'done, [{instance.__class__.__name__}] was created')
    return instance


def create_inference_options(name):
    log(f'Finding inference option with name: [{name}] ... ', end='')
    instance = _find_cls_using_name(
        name,
        package='options',
        parent_class=BaseInferenceOptions,
        cls_postfix='InferenceOptions'
    )()
    log(f'done, [{instance.__class__.__name__}] was created')
    return instance
