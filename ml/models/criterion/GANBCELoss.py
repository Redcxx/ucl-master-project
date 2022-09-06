import torch
from torch import nn


class GANBCELoss(nn.Module):

    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()

        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))

        self.loss = nn.BCEWithLogitsLoss()  # nn.MSELoss()

    def __call__(self, model_output, target_is_real):
        label = self.real_label if target_is_real else self.fake_label
        label = label.expand_as(model_output)

        return self.loss(model_output, label)
