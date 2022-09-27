import cv2 as cv
import numpy as np


def xdog_filter(
        image,
        kernel_size=0,   # open cv will choose this automatically if it is 0, and given sigma
        sigma=0.3,  # 0.3/0.4/0.5
        k_sigma=4.5,  # 4.5
        eps=0.0,
        phi=10e9,  # 10e9
        gamma=0.95
):
    """XDoG(Extended Difference of Gaussians)
        Args:
            image: opencv np image, 0-1 ranged
            kernel_size: Gaussian Blur Kernel Size
            sigma: sigma for small Gaussian filter
            k_sigma: large/small for sigma Gaussian filter
            eps: threshold value between dark and bright
            phi: soft threshold
            gamma: scale parameter for DoG signal to make sharp
        Returns:
            Image after applying the XDoG, 0-1 ranged.
        """
    dog = dog_filter(image, kernel_size, sigma, k_sigma, gamma)
    dog /= dog.max()
    e = 1 + np.tanh(phi * (dog - eps))
    e[e >= 1] = 1
    return e


def dog_filter(image, kernel_size=0, sigma=1.4, k_sigma=1.6, gamma=1.0):
    g1 = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    g2 = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma * k_sigma)
    return g1 - gamma * g2


def extract_edges_cv(
        im: np.array,
        kernel_size=0,
        sigma=0.3,  # 0.3/0.4/0.5
        k_sigma=4.5,
        eps=0.0,
        phi=10e9,  # 10e9
        gamma=0.95,  # tau 0.95
):
    im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)  # convert to grayscale
    im = cv.GaussianBlur(im, (5, 5), 0)  # blur to remove noise
    im = xdog_filter(im, kernel_size, sigma, k_sigma, eps, phi, gamma)  # xdog

    return im
