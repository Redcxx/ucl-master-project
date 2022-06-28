from bisect import bisect_right

import torch
import torch.nn.functional as F
import torchvision.models as M
from torch import nn

VGG16_PATH = 'resources/vgg16-397923af.pth'
I2V_PATH = 'resources/i2v.pth'


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super().__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        x = self.shortcut.forward(x)
        return x + bottleneck


class NetG(nn.Module):
    def __init__(self, opt, ngf=64):
        super(NetG, self).__init__()

        self.toH = nn.Sequential(nn.Conv2d(4, ngf, kernel_size=7, stride=1, padding=3), nn.LeakyReLU(0.2, True))

        self.to0 = nn.Sequential(nn.Conv2d(1, ngf // 2, kernel_size=3, stride=1, padding=1),  # 512
                                 nn.LeakyReLU(0.2, True))
        self.to1 = nn.Sequential(nn.Conv2d(ngf // 2, ngf, kernel_size=4, stride=2, padding=1),  # 256
                                 nn.LeakyReLU(0.2, True))
        self.to2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),  # 128
                                 nn.LeakyReLU(0.2, True))
        self.to3 = nn.Sequential(nn.Conv2d(ngf * 3, ngf * 4, kernel_size=4, stride=2, padding=1),  # 64
                                 nn.LeakyReLU(0.2, True))
        self.to4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),  # 32
                                 nn.LeakyReLU(0.2, True))

        tunnel4 = nn.Sequential(*[ResNeXtBottleneck(ngf * 8, ngf * 8, cardinality=opt.cardinality, dilate=1) for _ in range(20)])

        self.tunnel4 = nn.Sequential(nn.Conv2d(ngf * 8 + 512, ngf * 8, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel4,
                                     nn.Conv2d(ngf * 8, ngf * 4 * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 64

        depth = 2
        tunnel = [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=opt.cardinality, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=opt.cardinality, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=opt.cardinality, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=opt.cardinality, dilate=2),
                   ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=opt.cardinality, dilate=1)]
        tunnel3 = nn.Sequential(*tunnel)

        self.tunnel3 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel3,
                                     nn.Conv2d(ngf * 4, ngf * 2 * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 128

        tunnel = [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=opt.cardinality, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=opt.cardinality, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=opt.cardinality, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=opt.cardinality, dilate=2),
                   ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=opt.cardinality, dilate=1)]
        tunnel2 = nn.Sequential(*tunnel)

        self.tunnel2 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel2,
                                     nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        tunnel = [ResNeXtBottleneck(ngf, ngf, cardinality=opt.cardinality//2, dilate=1)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=opt.cardinality//2, dilate=2)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=opt.cardinality//2, dilate=4)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=opt.cardinality//2, dilate=2),
                   ResNeXtBottleneck(ngf, ngf, cardinality=opt.cardinality//2, dilate=1)]
        tunnel1 = nn.Sequential(*tunnel)

        self.tunnel1 = nn.Sequential(nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel1,
                                     nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        self.exit = nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, sketch, hint, sketch_feat):
        hint = self.toH(hint)

        x0 = self.to0(sketch)
        x1 = self.to1(x0)
        x2 = self.to2(x1)
        x3 = self.to3(torch.cat([x2, hint], 1))  # !
        x4 = self.to4(x3)

        x = self.tunnel4(torch.cat([x4, sketch_feat], 1))
        x = self.tunnel3(torch.cat([x, x3], 1))
        x = self.tunnel2(torch.cat([x, x2], 1))
        x = self.tunnel1(torch.cat([x, x1], 1))
        x = torch.tanh(self.exit(torch.cat([x, x0], 1)))

        return x


class NetD(nn.Module):
    def __init__(self, ndf=64):
        super(NetD, self).__init__()

        self.feed = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=7, stride=1, padding=3, bias=False),  # 512
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # 256
            nn.LeakyReLU(0.2, True),

            ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1, stride=2),  # 128
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, True),

            ResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1, stride=2),  # 64
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, True),

            ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1, stride=2)  # 32
        )

        self.feed2 = nn.Sequential(
            nn.Conv2d(ndf * 12, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),  # 32
            nn.LeakyReLU(0.2, True),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 16
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 8
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 4
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=0, bias=False),  # 1
            nn.LeakyReLU(0.2, True)
        )

        self.out = nn.Linear(512, 1)

    def forward(self, color, sketch_feat):
        x = self.feed(color)

        x = self.feed2(torch.cat([x, sketch_feat], 1))

        out = self.out(x.view(color.size(0), -1))
        return out


class NetF(nn.Module):
    def __init__(self, opt):
        super(NetF, self).__init__()

        vgg16 = M.vgg16()
        vgg16.load_state_dict(torch.load(opt.VGG16_PATH))
        vgg16.features = nn.Sequential(
            *list(vgg16.features.children())[:9]
        )
        self.model = vgg16.features
        self.register_buffer('mean', torch.FloatTensor([0.485 - 0.5, 0.456 - 0.5, 0.406 - 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        return self.model((images.mul(0.5) - self.mean) / self.std)


class NetI(nn.Module):
    def __init__(self, opt):
        super(NetI, self).__init__()
        i2v_model = nn.Sequential(  # Sequential,
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 1539, (3, 3), (1, 1), (1, 1)),
            nn.AvgPool2d((7, 7), (1, 1), (0, 0), ceil_mode=True),  # AvgPool2d,
        )
        i2v_model.load_state_dict(torch.load(opt.I2V_PATH))
        i2v_model = nn.Sequential(
            *list(i2v_model.children())[:15]
        )
        self.model = i2v_model
        self.register_buffer('mean', torch.FloatTensor([164.76139251, 167.47864617, 181.13838569]).view(1, 3, 1, 1))

    def forward(self, images):
        images = F.avg_pool2d(images, 2, 2)
        images = images.mul(0.5).add(0.5).mul(255)
        return self.model(images.expand(-1, 3, 256, 256) - self.mean)


class _LRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_iter = last_iter

    def _get_new_lr(self):
        raise NotImplementedError

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            param_group['lr'] = lr


class WarmUpLRScheduler(_LRScheduler):

    def __init__(self, optimizer, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        if warmup_steps == 0:
            self.warmup_lr = base_lr
        else:
            self.warmup_lr = warmup_lr
        super(WarmUpLRScheduler, self).__init__(optimizer, last_iter)

    def _get_warmup_lr(self):
        if self.warmup_steps > 0 and self.last_iter < self.warmup_steps:
            # first compute relative scale for self.base_lr, then multiply to base_lr
            scale = ((self.last_iter / self.warmup_steps) * (
                    self.warmup_lr - self.base_lr) + self.base_lr) / self.base_lr
            # print('last_iter: {}, warmup_lr: {}, base_lr: {}, scale: {}'.format(self.last_iter, self.warmup_lr, self.base_lr, scale))
            return [scale * base_lr for base_lr in self.base_lrs]
        else:
            return None


class StepLRScheduler(WarmUpLRScheduler):
    def __init__(self, optimizer, milestones, lr_mults, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        super(StepLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)

        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestones, lr_mults)
        for x in milestones:
            assert isinstance(x, int)
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.lr_mults = [1.0]
        for x in lr_mults:
            self.lr_mults.append(self.lr_mults[-1] * x)

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        pos = bisect_right(self.milestones, self.last_iter)
        scale = self.warmup_lr * self.lr_mults[pos] / self.base_lr
        return [base_lr * scale for base_lr in self.base_lrs]
