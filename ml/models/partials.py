import functools

import torch
from torch import nn

from ml.session import SessionOptions


class UnetBlock(nn.Module):
    def __init__(
            self,
            in_filters, out_filters,

            submodule=None,
            sub_in_filters=None,
            sub_out_filters=None,
            sub_skip_connection=False,

            skip_connection=True,
            dropout=nn.Dropout,
            in_norm=nn.BatchNorm2d, out_norm=nn.BatchNorm2d,
            in_act=nn.LeakyReLU, out_act=nn.ReLU,
    ):
        super().__init__()

        if submodule is None:
            sub_in_filters = in_filters
            sub_out_filters = in_filters
            sub_skip_connection = False

        conv_common_args = {
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'bias': in_norm.func != nn.BatchNorm2d if type(in_norm) == functools.partial else in_norm != nn.BatchNorm2d
            # batch norm has bias
        }

        layers = []

        # encoder
        layers.append(nn.Conv2d(in_channels=in_filters, out_channels=sub_in_filters, **conv_common_args))

        if in_norm:
            layers.append(in_norm(sub_in_filters))

        if in_act:
            layers.append(in_act())

        # submodule
        if submodule:
            layers.append(submodule)

        # decoder
        if sub_skip_connection:
            layers.append(
                nn.ConvTranspose2d(in_channels=sub_out_filters * 2, out_channels=out_filters, **conv_common_args))
        else:
            layers.append(nn.ConvTranspose2d(in_channels=sub_out_filters, out_channels=out_filters, **conv_common_args))

        if out_norm:
            layers.append(out_norm(out_filters))

        if dropout:
            layers.append(dropout())

        if out_act:
            layers.append(out_act())

        self.model = nn.Sequential(*layers)

        self.skip_connection = skip_connection

    def forward(self, x):
        if self.skip_connection:
            return torch.cat([x, self.model(x)], dim=1)
        else:
            return self.model(x)


class Generator(nn.Module):

    def __init__(self, opt: SessionOptions):
        super().__init__()

        config = opt.network_config['generator_config']

        # dependency injection
        batch_norm = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        relu = functools.partial(nn.ReLU, inplace=True)
        leaky_relu = functools.partial(nn.LeakyReLU, inplace=True, negative_slope=0.2)
        dropout = functools.partial(nn.Dropout, p=0.5)
        tahn = nn.Tanh

        # build model recursively inside-out
        blocks = config['blocks'][::-1]

        self.model = None

        # build innermost block
        self.model = UnetBlock(
            in_filters=blocks[0]['filters'],
            out_filters=blocks[0]['filters'],

            submodule=None,
            sub_in_filters=None,
            sub_out_filters=None,
            sub_skip_connection=False,

            skip_connection=blocks[0]['skip_connection'],
            dropout=dropout if blocks[0]['dropout'] else None,
            in_norm=None, out_norm=False,
            in_act=relu, out_act=relu
        )

        # build between blocks
        for i, layer in enumerate(blocks[1:], 1):
            self.model = UnetBlock(
                in_filters=layer['filters'],
                out_filters=layer['filters'],

                submodule=self.model,
                sub_in_filters=blocks[i - 1]['filters'],
                sub_out_filters=blocks[i - 1]['filters'],
                sub_skip_connection=blocks[i - 1]['skip_connection'],

                skip_connection=blocks[i]['skip_connection'],
                dropout=dropout if layer['dropout'] else None,
                in_norm=batch_norm, out_norm=batch_norm,
                in_act=leaky_relu, out_act=relu
            )

        # build outermost block
        self.model = UnetBlock(
            in_filters=config['in_channels'],
            out_filters=config['out_channels'],

            submodule=self.model,
            sub_in_filters=blocks[-1]['filters'],
            sub_out_filters=blocks[-1]['filters'],
            sub_skip_connection=blocks[-1]['skip_connection'],

            skip_connection=blocks[-1]['skip_connection'],
            dropout=None,
            in_norm=None, out_norm=None,
            in_act=leaky_relu, out_act=tahn
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, opt: SessionOptions):
        super().__init__()

        config = opt.network_config['discriminator_config']

        # we do not use bias in conv2d layer if batch norm is used, because batch norm already has bias
        batch_norm = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        leaky_relu = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)

        conv_common_args = {
            'kernel_size': 4,
            'padding': 1,
        }

        blocks = config['blocks']
        layers = []

        # build first block
        layers += [
            nn.Conv2d(config['in_channels'], blocks[0]['filters'], stride=2, **conv_common_args),
            leaky_relu()
        ]

        # build between block
        prev_filters = blocks[0]['filters']
        for i, layer in enumerate(blocks[1:-1], 1):
            curr_filters = min(blocks[i]['filters'], blocks[0]['filters'] * 8)
            layers += [
                nn.Conv2d(prev_filters, curr_filters, bias=False, stride=2, **conv_common_args),
                batch_norm(curr_filters),
                leaky_relu()
            ]
            prev_filters = curr_filters

        # build last block
        curr_filters = min(blocks[-1]['filters'], blocks[0]['filters'] * 8)
        layers += [
            # stride = 1 for last block
            nn.Conv2d(prev_filters, curr_filters, stride=1, bias=False, **conv_common_args),
            batch_norm(curr_filters),
            leaky_relu(),
            # convert to 1 dimensional output
            nn.Conv2d(curr_filters, 1, stride=1, **conv_common_args)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
