"""This module is for processing the data.

Serval processing methods are included in this module,  including 
gradient computation,  fast Fourier transform,  and spatial average.
These methods are used to add some needed features to original brain
image.
"""

from collections import namedtuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve

__all__ = [
    "compute_gradient",
    "fft",
    "spatial_average",
]


def _get_gaussian_kernel(ndim: int, size: int, sigma: float) -> np.ndarray:
    shape = (size,) * ndim
    center = (size - 1) // 2
    center_loc = (center,) * ndim
    kernel = np.zeros(shape=shape, dtype=np.float64)
    kernel[center_loc] = 1
    return gaussian_filter(input=kernel, sigma=sigma)


_FFTRes = namedtuple("_FFTRes", ["real", "imag"])


def fft(
    data: np.ndarray,
) -> _FFTRes:
    data = np.nan_to_num(data, copy=True, nan=0.0)
    data = np.fft.fftshift(np.fft.fftn(data))
    return _FFTRes(
        real=np.real(data),
        imag=np.imag(data),
    )


def compute_gradient(*args, **kwargs):
    return np.gradient(*args, **kwargs)


def spatial_average(
    data: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> np.ndarray:
    """Performs spatial average on the data.

    We use convolve function to perform spatial average on the data.
    The kernel for convolution is distributed according to Gaussian
    distribution with the given sigma.  We normalized the total weight of
    the kernel to 1.

    Args:
        data: The data need to compute spatial average.
        kernel_size: The edge size of the kernel for convolution.
          Defaults to 3.
        sigma: The standard deviation of the Gaussian kernel.  Defaults
          to 1.0.

    Returns:
        The spatial averaged result of the data.
    """
    kernel = _get_gaussian_kernel(
        ndim=data.ndim,
        size=kernel_size,
        sigma=sigma,
    )

    pad_width = (kernel_size - 1) // 2
    mask = np.isnan(data)
    processed_data = np.nan_to_num(data, copy=True, nan=0.0)
    processed_data = np.pad(
        processed_data,
        pad_width=pad_width,
        mode="constant",
        constant_values=0,
    )
    processed_data = convolve(processed_data, kernel, mode="valid")
    processed_data[mask] = np.nan

    return processed_data


def spatial_difference(data: np.ndarray) -> np.ndarray:
    data = data.flatten().copy()
    return np.array([np.full_like(data, num) - data for num in data])
