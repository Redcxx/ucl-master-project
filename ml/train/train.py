import time

from ml.dataset import create_dataloaders
from ml.models import create_model
from ml.models.base_model import BaseModel
from ml.session import SessionOptions

if __name__ == '__main__':
    opt = SessionOptions()
    train_loader, test_loader = create_dataloaders(opt)
    model: BaseModel = create_model(opt)

    model.pre_train()

    for epoch in range(opt.start_epoch, opt.end_epoch + 1 + opt.decay_epochs):

        model.pre_epoch()

        for batch, batch_data in enumerate(train_loader, 1):

            model.pre_batch(batch)

            batch_out = model.train_batch(batch, batch_data)

            model.post_batch(batch_out)

            if opt.batch_log_freq is not None and (epoch % opt.batch_log_freq == 0 or batch == 1):
                model.log_batch(batch)

        model.post_epoch()

        if opt.eval_freq is not None and (epoch % opt.eval_freq == 0 or epoch == opt.start_epoch):
            model.evaluate()

        if opt.log_freq is not None and (epoch % opt.log_freq == 0 or epoch == opt.start_epoch):
            model.log_epoch(epoch)

        if opt.save_freq is not None and (epoch % opt.save_freq == 0 or epoch == opt.start_epoch):
            model.save_checkpoint(epoch)

    model.post_train()
