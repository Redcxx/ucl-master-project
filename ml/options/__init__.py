from ml.base_options import BaseTrainOptions
from ml.file_utils import _find_cls_using_name


def create_train_options(name):
    print(f'Finding train option with name: [{name}] ...')
    instance = _find_cls_using_name(
        name,
        package='options',
        parent_class=BaseTrainOptions,
        cls_postfix='TrainOptions'
    )()
    print(f'Option: [{instance.__class__.__name__}] was created')
    return instance
