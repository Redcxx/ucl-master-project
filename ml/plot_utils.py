import numpy as np
from matplotlib import pyplot as plt


def plot_im(tensor):
    im = tensor.squeeze().cpu().detach().numpy()
    im = unnormalize_im(im)
    im = im.transpose(1, 2, 0)
    plt.imshow(im)


def unnormalize_im(im):
    return np.clip(im * 0.5 + 0.5, 0, 1)


def plot_h_images(images, titles=None, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)

    for i, im in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plot_im(im)
        if titles:
            plt.title(titles[i])

    plt.show()

    return fig


def plot_inp_tar_out(inp, tar, out):
    return plot_h_images(
        [inp, tar, out],
        ['input', 'target', 'output'],
        (10, 10)
    )
