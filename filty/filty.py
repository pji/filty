"""
filty
~~~~~

Filter functions for image data.
"""
from PIL import Image, ImageOps
import numpy as np

from filty.utility import grayscale_to_rgb


# Image filter functions.
def filter_colorize(a: np.ndarray, 
                    white: str,
                    black: str) -> np.array:
    """Colorize a grayscale image."""
    src_space = 'L'
    dst_space = 'RGB'
    a_L_mode = (a * 0xff).astype(np.uint8)
    img = Image.fromarray(a_L_mode, mode=src_space)
    img = ImageOps.colorize(**{
        'image': img,
        'black': black,
        'white': white,
        'blackpoint': 0x00,
        'midpoint': 0x7f,
        'whitepoint': 0xff,
    })
    img = img.convert(dst_space)
    out = np.array(img, dtype=a.dtype) / 0xff
    return out


if __name__ == '__main__':
    from tests.common import F
    from filty.utility import print_array
    
    filter = filter_colorize
    kwargs = {
        'a': F.copy(),
        'white': 'hsv(350, 100%, 100%)',
        'black': 'hsv(10, 100%, 0%)',
    }
    out = filter(**kwargs)
    print_array(out)
