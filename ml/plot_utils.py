import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms


def plt_tensor(im, un_normalize=True):
    if un_normalize:
        im = unnormalize_im(im)
    im = im.cpu().detach().numpy()
    im = im.transpose(1, 2, 0).squeeze()
    plt.imshow(im)


def unnormalize_im(im):
    num_dims = im.shape[0]
    mean = torch.tensor([0.5 for _ in range(num_dims)], dtype=torch.float32)
    std = torch.tensor([0.5 for _ in range(num_dims)], dtype=torch.float32)
    return transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())(im)


def plt_horizontals(images, titles=None, figsize=(10, 10), dpi=512, un_normalize=True, save_file=None):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if type(un_normalize) == bool:
        un_normalize = [un_normalize for _ in range(len(images))]

    for i, im in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt_tensor(im, un_normalize=un_normalize[i])
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
        (15, 5),
        save_file=save_file
    )


def plt_input_target(inp, tar, save_file=None):
    return plt_horizontals(
        [inp, tar],
        ['input', 'target'],
        (10, 5),
        save_file=save_file
    )
