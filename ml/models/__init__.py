from ml.base_model import BaseTrainModel
from ml.file_utils import _find_cls_using_name


def create_train_model(opt, train_loader, test_loader, name):

    print(f'Finding train model with name: [{name}] ...')

    cls = _find_cls_using_name(
        name,
        package='models',
        parent_class=BaseTrainModel,
        cls_postfix='TrainModel'
    )

    instance = cls(opt, train_loader, test_loader)

    print(f'Model: [{instance.__class__.__name__}] was created')
    return instance
