"""
filty
~~~~~

Filter functions for image data.
"""
import cv2
from PIL import Image, ImageOps
import numpy as np

from filty.utility import (grayscale_to_rgb, uses_uint8, COLOR,
                           processes_by_grayscale_frame,
                           trilinear_interpolation, X, Y, Z,
                           bilinear_interpolation, will_square)


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


def filter_gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Perform a gaussian blur."""
    return cv2.GaussianBlur(a, (0, 0), sigma, sigma, 0)


def filter_flip(a: np.ndarray, axis: int) -> np.ndarray:
    """Flip the image around an axis."""
    return np.flip(a, axis)


def filter_grow(a: np.ndarray, factor: float) -> np.ndarray:
    """Increase the size of an image."""
    if len(a.shape) == 2:
        return bilinear_interpolation(a, factor)
    return trilinear_interpolation(a, factor)


def filter_inverse(a: np.ndarray) -> np.ndarray:
    """Inverse the colors of an image."""
    return 1 - a


@processes_by_grayscale_frame
@will_square
def filter_linear_to_polar(a: np.ndarray) -> np.ndarray:
    """Convert the linear coordinates of the image data to
    polar coordinates.
    """
    center = tuple(n / 2 for n in a.shape)
    max_radius = np.sqrt(sum(n ** 2 for n in center))
    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
    return cv2.warpPolar(a, a.shape, center, max_radius, flags)


@processes_by_grayscale_frame
def filter_motion_blur(a: np.ndarray,
                       amount: int,
                       axis: int) -> np.ndarray:
    """Perform a motion blur."""
    kernel = np.zeros((amount, amount), float)
    if axis == X:
        y = int(amount // 2)
        for x in range(amount):
            kernel[y][x] = 1 / amount
    if axis == Y:
        x = int(amount // 2)
        for y in range(amount):
            kernel[y][x] = 1 / amount
    return cv2.filter2D(a, -1, kernel)


@processes_by_grayscale_frame
def filter_pinch(a: np.ndarray,
                 amount: float,
                 radius: float,
                 scale: tuple[float],
                 offset: tuple[int]) -> np.ndarray:
    """Distort an image to make it appear as though it is being
    pinched or swelling.
    """
    # Set up for creating the maps.
    center = tuple((n) / 2 + o for n, o in zip(a.shape, offset))
    flex_x = np.zeros(a.shape, np.float32)
    flex_y = np.zeros(a.shape, np.float32)

    # Create a map of the distance from each pixel in the image to
    # the center of the image.
    indices = np.indices(a.shape)
    y = indices[Y]
    x = indices[X]
    delta_y = scale[Y] * (y - center[Y])
    delta_x = scale[X] * (x - center[X])
    distance = delta_x ** 2 + delta_y ** 2

    # Mask out the area covered by not within the radius of the effect.
    r_mask = np.zeros(x.shape, bool)
    r_mask[distance >= radius ** 2] = True
    flex_x[r_mask] = x[r_mask]
    flex_y[r_mask] = y[r_mask]

    # Create maps with the barrel/pincushion formula.
    pmask = np.zeros(x.shape, bool)
    pmask[distance > 0.0] = True
    pmask[r_mask] = False
    factor = np.sin(np.pi * np.sqrt(distance) / radius / 2)
    factor[factor > 0] = factor[factor > 0] ** -amount
    factor[factor < 0] = -((-factor[factor < 0]) ** -amount)
    flex_x[pmask] = factor[pmask] * delta_x[pmask] / scale[X] + center[X]
    flex_y[pmask] = factor[pmask] * delta_y[pmask] / scale[Y] + center[Y]

    flex_x[~pmask] = 1.0 * delta_x[~pmask] / scale[X] + center[X]
    flex_y[~pmask] = 1.0 * delta_y[~pmask] / scale[Y] + center[Y]
    
    # Perform the pinch using the maps and return.
    return cv2.remap(a, flex_x, flex_y, cv2.INTER_LINEAR)


@processes_by_grayscale_frame
@will_square
def filter_polar_to_linear(a: np.ndarray) -> np.ndarray:
    """Convert the polar coordinates of the image data to
    linear coordinates.
    """
    center = tuple(n / 2 for n in a.shape)
    max_radius = np.sqrt(sum(n ** 2 for n in center))
    return cv2.linearPolar(a, center, max_radius, cv2.WARP_FILL_OUTLIERS)


@processes_by_grayscale_frame
def filter_ripple(a: np.ndarray,
                  wave: tuple[float],
                  amp: tuple[float],
                  distaxis: tuple[int],
                  offset: tuple[float] = (0, 0, 0)) -> np.ndarray:
    """Perform a ripple distortion."""
    # Map out the volume of the given image and make sure everything is
    # in float32 to keep the cv2.remap function happy.
    flex = np.indices(a.shape, np.float32)
    flex_x = flex[X].copy()
    flex_y = flex[Y].copy()

    # Modify the mapping to apply the ripple to create the flex
    # maps for cv.remap. The flex map value for each pixel will
    # indicate how far that pixel moves in the remapped image.
    da_x, da_y = distaxis
    off_y, off_x = offset
    if wave[X]:
        flex_x = np.cos((off_x + flex[da_x]) / wave[X] * 2 * np.pi)
        flex_x = flex[X] + flex_x * amp[X]
    if wave[Y]:
        flex_y = np.cos((off_y + flex[da_y]) / wave[Y] * 2 * np.pi)
        flex_y = flex[Y] + flex_y * amp[Y]

    # Remap the color values in the original image using the
    # rippled flex map.
    return cv2.remap(a, flex_x, flex_y, cv2.INTER_LINEAR)


if __name__ == '__main__':
    from tests.common import A, F, VIDEO_2_5_5
    from filty.utility import print_array
    
    filter = filter_ripple
    kwargs = {
        'a': VIDEO_2_5_5.copy(),
        'wave': (2, 2),
        'amp': (2, 2),
        'distaxis': (Y, X),
        'offset': (0, 0),
    }
    out = filter(**kwargs)
    print_array(out, 2)
