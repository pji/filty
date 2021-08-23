"""
filty
~~~~~

Filter functions for image data.
"""
from PIL import Image, ImageOps
import numpy as np

from filty.utility import (grayscale_to_rgb, uses_uint8, COLOR,
                           processes_by_grayscale_frame)


# Image filter functions.
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


if __name__ == '__main__':
    from tests.common import VIDEO_2_3_3
    from filty.utility import print_array
    
    filter = filter_colorize
    kwargs = {
        'a': VIDEO_2_3_3.copy(),
        'white': 'hsv(350, 100%, 100%)',
        'black': 'hsv(10, 100%, 0%)',
    }
    out = filter(**kwargs)
    print_array(out, 2)
