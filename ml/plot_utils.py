import numpy as np
from matplotlib import pyplot as plt


def plot_im(tensor):
    im = tensor.cpu().detach().numpy()
    im = unnormalize_im(im)
    im = im.transpose(1, 2, 0)
    plt.imshow(im)


def unnormalize_im(im):
    return np.clip(im * 0.5 + 0.5, 0, 1)


def plot_h_images(images, titles=None, figsize=(10, 10), save_file=None):
    fig = plt.figure(figsize=figsize)

    for i, im in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plot_im(im)
        if titles:
            plt.title(titles[i])

    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_inp_tar_out(inp, tar, out, save_file=None):
    return plot_h_images(
        [inp, tar, out],
        ['input', 'target', 'output'],
        (5, 10),
        save_file=save_file
    )


def plot_inp_tar(inp, tar, save_file=None):
    return plot_h_images(
        [inp, tar],
        ['input', 'target'],
        (5, 10),
        save_file=save_file
    )
