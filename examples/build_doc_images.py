"""
build_doc_images
~~~~~~~~~~~~~~~~

Build the documentation images for :mod:`imgfilt`.
"""
from pathlib import Path
from typing import Callable

import imggen as ig
import imgwriter as iw

import imgfilt as ift


# Constants.
X, Y, Z = -1, -2, -3


# Make the example images.
def make_image(
    filter: Callable, kwargs: dict, size: ig.Size, color: bool = False
) -> ig.ImgAry:
    """Make an example image for a filter."""
    # Create image.
    source = ig.Spheres(radius=51, offset='x')
    a = source.fill(size)
    a /= 2
    a += 0.25
    
    # Filter the image data.
    filtered = filter(a, **kwargs)
    
    # Because we are putting the filtered data back into the bottom half
    # of the original data, work needs to be done to make sure the shape
    # of the filtered data matches the shape of the bottom half of the
    # original data.
    if color:
        a = ift.filter_colorize(a)
    if filtered.shape != a.shape and not color:
        diffs = [an - fn for an, fn in zip(filtered.shape, a.shape)]
        starts = [n // 2 for n in diffs]
        stops = [sn + an for an, sn in zip(a.shape, starts)]
        filtered = filtered[
            starts[Z]:stops[Z], starts[Y]:stops[Y], starts[X]:stops[X]
        ]
    
    # Replace the bottom half of the image data with the filtered data.
    midheight = size[Y] // 2
    a[:, midheight:, :] = filtered[:, midheight:, :]
    
    # Add the label.
    label, ystart, ystop = make_label(size)
    if color:
        label = ift.filter_colorize(label)
    a[:, ystart:ystop, :] = label
    return a


def make_images(path: Path, size: ig.Size, ext: str = 'jpg') -> None:
    """Make the example images."""
    filters = [
        (ift.filter_box_blur, {'size': size[X] // 32,}, False),
        (ift.filter_colorize, {'colorkey': 'g',}, True),
        (ift.filter_contrast, {}, False),
        (ift.filter_flip, {'axis': X,}, False),
        (ift.filter_gaussian_blur, {'sigma': 12.0,}, False),
        (ift.filter_glow, {'sigma': 3,}, False),
        (ift.filter_grow, {'factor': 2,}, False),
        (ift.filter_inverse, {}, False),
        (ift.filter_linear_to_polar, {}, False),
        (ift.filter_motion_blur, {'amount': size[X] // 32, 'axis': X}, False),
        (ift.filter_pinch, {
            'amount': 0.5,
            'radius': size[X] // 3,
            'scale': (0, 0.5, 0.5),
            'offset': (0, 0, 0)
        }, False),
        (ift.filter_polar_to_linear, {}, False),
        (ift.filter_ripple, {
            'wave': (0, size[Y] // 5, size[Y] // 5),
            'amp': (0, size[Y] // 80, size[Y] // 80),
            'distaxis': (Z, Y, X),
        }, False),
    ]
    for item in filters:
        filter, kwargs, color = item
        a = make_image(filter, kwargs, size, color)
        fname = f'{filter.__name__}.{ext}'
        iw.write(path / fname, a)


def make_label(size: ig.Size) -> tuple[ig.ImgAry, int, int]:
    """Make the label to insert between the halves of the example image."""
    # Original side.
    oheight = int(size[Y] * 1.1 // 12)
    origin = (size[X] // 10, oheight // 20)
    text_orig = ig.Text(
        '\u25b2 ORIGINAL \u25b2',
        font='Menlo',
        size=size[Y] // 30,
        origin=origin
    )
    height = int(size[Y] * 1.1 // 24)
    label = text_orig.fill((size[Z], height, size[X]))
    
    # Filtered side.
    origin = (size[X] * 8 // 10, oheight // 20)
    text_filt = ig.Text(
        '\u25bc FILTER \u25bc',
        font='Menlo',
        size=size[Y] // 30,
        origin=origin
    )
    height = int(size[Y] * 1.1 // 24)
    label += text_filt.fill((size[Z], height, size[X]))

    # Cap the color values.
    label[label > 1.0] = 1.0
    label[label < 0.0] = 0.0
    
    # Locate the label in the image.
    ystart = size[Y] // 2 - height // 2
    ystop = ystart + height
    return label, ystart, ystop
    

# Mainline.
if __name__ == '__main__':
    path = Path('docs/source/images')
    size = (1, 720, 1280)
    make_images(path, size)
    