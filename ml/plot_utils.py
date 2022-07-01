import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms


def plt_tensor(im):
    im = unnormalize_im(im)
    im = im.cpu().detach().numpy()
    im = im.transpose(1, 2, 0).squeeze()
    plt.imshow(im)


def unnormalize_im(im):
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    return transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())(im)


def plt_horizontals(images, titles=None, figsize=(10, 10), save_file=None):
    fig = plt.figure(figsize=figsize)

    for i, im in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt_tensor(im)
        if titles:
            plt.title(titles[i])

    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plt_model_sample(inp, tar, out, save_file=None):
    return plt_horizontals(
        [inp, tar, out],
        ['input', 'target', 'output'],
        (15, 15),
        save_file=save_file
    )


def plt_input_target(inp, tar, save_file=None):
    return plt_horizontals(
        [inp, tar],
        ['input', 'target'],
        (15, 15),
        save_file=save_file
    )
