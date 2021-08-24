"""
filty
~~~~~

Filter functions for image data.
"""
import cv2
from PIL import Image, ImageOps
import numpy as np

from filty.utility import (grayscale_to_rgb, uses_uint8, COLOR,
                           processes_by_grayscale_frame)


# Useful constants.
X, Y, Z = -1, -2, -3


# Image filter functions.
@processes_by_grayscale_frame
def filter_box_blur(a: np.ndarray, size: int) -> np.ndarray:
    """Perform a box blur."""
    kernel = np.ones((size, size), float) / size ** 2
    return cv2.filter2D(a, -1, kernel)


@processes_by_grayscale_frame
@uses_uint8
def filter_colorize(a: np.ndarray, 
                    colorkey: str = '',
                    white: str = '#FFFFFF',
                    black: str = '#000000') -> np.array:
    """Colorize a grayscale image.
    
    :param a: The image data to colorize.
    :param colorkey: (Optional.) The key for the pre-defined 
        colors to use in the colorization. These are defined
        in utility.COLOR.
    :param white: (Optional.) The color name for the color
        to use to replace white in the image. Color names
        are defined by PIL.ImageColor.
    :param black: (Optional.) The color name for the color
        to use to replace black in the image. Color names
        are defined by PIL.ImageColor.
    :returns: The colorized image data.
    :rtype: A :class:numpy.ndarray object
    """
    src_space = 'L'
    dst_space = 'RGB'
    if colorkey:
        white, black = COLOR[colorkey]
    img = Image.fromarray(a, mode=src_space)
    img = ImageOps.colorize(**{
        'image': img,
        'black': black,
        'white': white,
        'blackpoint': 0x00,
        'midpoint': 0x7f,
        'whitepoint': 0xff,
    })
    img = img.convert(dst_space)
    out = np.array(img, dtype=a.dtype)
    return out


def filter_contrast(a: np.ndarray) -> np.ndarray:
    """Adjust the image to fill the full dynamic range."""
    a_min = np.min(a)
    a_max = np.max(a)
    scale = a_max - a_min
    if scale != 0:
        a = a - a_min
        a = a / scale
    return a


def filter_flip(a: np.ndarray, axis: int) -> np.ndarray:
    """Flip the image around an axis."""
    return np.flip(a, axis)


if __name__ == '__main__':
    from tests.common import A, VIDEO_2_5_5
    from filty.utility import print_array
    
    filter = filter_flip
    kwargs = {
        'a': VIDEO_2_5_5.copy(),
        'axis': Z,
    }
    out = filter(**kwargs)
    print_array(out, 2)
