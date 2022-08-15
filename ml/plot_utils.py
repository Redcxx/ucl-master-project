import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

def preprocess_im(im, un_normalize=True):
    if un_normalize:
        im = unnormalize_im(im)
    im = im.cpu().detach().numpy()
    im = im.transpose(1, 2, 0).squeeze()
    return im

def plt_tensor(im, un_normalize=True, grayscale=False):
    im = preprocess_im(im, un_normalize=un_normalize)
    if grayscale:
        plt.imshow(im, cmap='gray')
    else:
        plt.imshow(im)


def unnormalize_im(im):
    num_dims = im.shape[0]
    mean = torch.tensor([0.5 for _ in range(num_dims)], dtype=torch.float32)
    std = torch.tensor([0.5 for _ in range(num_dims)], dtype=torch.float32)
    return transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())(im)


def save_raw_im(im: torch.tensor, filename, dpi=512, un_normalize=True):
    im = preprocess_im(im, un_normalize=un_normalize)
    w, h = im.shape[:2]
    fig = plt.figure(frameon=False, dpi=dpi)
    # fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im, aspect='auto')
    fig.savefig(filename)

def plt_horizontals(images, titles=None, figsize=(10, 10), dpi=512, un_normalize=True, save_file=None, grayscale=False):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if type(un_normalize) == bool:
        un_normalize = [un_normalize for _ in range(len(images))]

    if type(grayscale) == bool:
        grayscale = [grayscale for _ in range(len(images))]

    for i, im in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt_tensor(im, un_normalize=un_normalize[i], grayscale=grayscale[i])
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
