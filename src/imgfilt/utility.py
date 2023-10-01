"""
utility
~~~~~~~

Utility functions for the imgfilt module.
"""
from functools import wraps
from inspect import getmembers, isfunction
from typing import Callable, NewType

import numpy as np


# Types.
Color = NewType('Color', tuple[str, str])
ColorDict = NewType('ColorDict', dict[str, Color])


# Useful constants.
X, Y, Z = -1, -2, -3
COLORS = ColorDict({
    # Grayscale
    'a': Color(('hsv(0, 0%, 100%)', 'hsv(0, 0%, 0%)')),
    'A': Color(('hsl(0, 0%, 75%)', 'hsl(0, 0%, 25%)')),

    # Electric blue.
    'b': Color(('hsv(200, 100%, 100%)', 'hsv(200, 100%, 0%)')),
    'B': Color(('hsl(200, 100%, 75%)', 'hsl(200, 100%, 25%)')),

    'bw': Color(('hsv(205, 100%, 100%)', 'hsv(200, 100%, 0%)')),
    'Bw': Color(('hsl(205, 100%, 75%)', 'hsl(200, 100%, 25%)')),

    'bk': Color(('hsv(200, 30%, 20%)', 'hsv(200, 30%, 0%)')),
    'BK': Color(('hsl(200, 30%, 30%)', 'hsl(200, 30%, 10%)')),

    # Cream
    'c': Color(('hsv(35, 100%, 100%)', 'hsv(35, 100%, 0%)')),
    'C': Color(('hsl(35, 100%, 80%)', 'hsl(35, 100%, 25%)')),

    'cw': Color(('hsv(30, 100%, 100%)', 'hsv(35, 100%, 0%)')),
    'Cw': Color(('hsl(30, 100%, 80%)', 'hsl(35, 100%, 25%)')),

    'cc': Color(('hsv(40, 100%, 100%)', 'hsv(35, 100%, 0%)')),
    'Cc': Color(('hsl(40, 100%, 80%)', 'hsl(35, 100%, 25%)')),

    'ck': Color(('hsv(35, 30%, 20%)', 'hsv(35, 30%, 0%)')),
    'CK': Color(('hsl(35, 30%, 30%)', 'hsl(35, 30%, 10%)')),

    # Dark.
    'k': Color(('hsv(220, 30%, 20%)', 'hsv(220, 30%, 0%)')),
    'K': Color(('hsl(220, 30%, 30%)', 'hsl(220, 30%, 10%)')),

    'kk': Color(('hsv(220, 30%, 10%)', 'hsv(220, 30%, 0%)')),
    'KK': Color(('hsl(220, 30%, 15%)', 'hsl(220, 30%, 5%)')),

    # Ectoplasmic teal.
    'e': Color(("hsv(190, 50%, 100%)", "hsv(190, 100%, 0%)")),
    'E': Color(("hsl(190, 50%, 100%)", "hsl(190, 100%, 30%)")),

    # Electric green.
    'g': Color(('hsv(90, 100%, 100%)', 'hsv(90, 100%, 0%)')),
    'G': Color(('hsl(90, 100%, 75%)', 'hsl(90, 100%, 25%)')),

    'gk': Color(('hsv(90, 30%, 20%)', 'hsv(90, 30%, 0%)')),
    'GK': Color(('hsl(90, 30%, 30%)', 'hsl(90, 30%, 10%)')),

    # Slate.
    'l': Color(('hsv(220, 30%, 50%)', 'hsv(220, 30%, 0%)')),
    'L': Color(('hsl(220, 30%, 75%)', 'hsl(220, 30%, 25%)')),

    # Electric pink.
    'p': Color(('hsv(320, 100%, 100%)', 'hsv(320, 100%, 0%)')),
    'P': Color(('hsl(320, 100%, 75%)', 'hsl(320, 100%, 25%)')),

    # Royal purple.
    'r': Color(('hsv(280, 100%, 100%)', 'hsv(280, 100%, 0%)')),
    'R': Color(('hsl(280, 100%, 75%)', 'hsl(280, 100%, 25%)')),

    'rw': Color(('hsv(285, 100%, 100%)', 'hsv(280, 100%, 0%)')),
    'Rw': Color(('hsl(285, 100%, 75%)', 'hsl(280, 100%, 25%)')),

    # Scarlet.
    's': Color(('hsv(350, 100%, 100%)', 'hsv(10, 100%, 0%)')),
    'S': Color(('hsl(350, 100%, 75%)', 'hsl(10, 100%, 25%)')),

    'sw': Color(('hsv(0, 100%, 100%)', 'hsv(10, 100%, 0%)')),
    'Sw': Color(('hsl(0, 100%, 75%)', 'hsl(10, 100%, 25%)')),

    'sk': Color(('hsv(350, 30%, 20%)', 'hsv(10, 30%, 0%)')),
    'SK': Color(('hsl(350, 30%, 30%)', 'hsl(10, 30%, 10%)')),

    # White.
    'w': Color(('hsv(0, 0%, 100%)', 'hsv(0, 0%, 0%)')),
    'W': Color(('hsl(0, 0%, 75%)', 'hsl(0, 0%, 25%)')),

    # Hue templates.
    't': Color(('hsv({}, 100%, 100%)', 'hsv({}, 100%, 0%)')),
    'T': Color(('hsl({}, 100%, 75%)', 'hsl({}, 100%, 25%)')),
})


# Color functions.
def get_color_for_key(colorkey: str) -> Color:
    return COLORS[colorkey]


def grayscale_to_rgb(a: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to RGB."""
    new_shape = (*a.shape, 3)
    new_a = np.zeros(new_shape, dtype=a.dtype)
    for channel in range(3):
        new_a[..., channel] = a
    return new_a


# Debugging functions.
def print_array(a: np.ndarray, depth: int = 0, color: bool = True) -> None:
    """Write the values of the given array to stdout."""
    if len(a.shape) > 1:
        print(' ' * (4 * depth) + '[')
        for i in range(a.shape[0]):
            print_array(a[i], depth + 1, color)
        print(' ' * (4 * depth) + '],')

    else:
        if a.dtype == np.float32 or a.dtype == np.float64:
            tmp = '{:>1.4f}'
        else:
            tmp = '{}'
        nums = [tmp.format(n) for n in a]
        print(' ' * (4 * depth) + '[' + ', '.join(nums) + '],')


# Decorators.
def processes_by_grayscale_frame(fn:Callable) -> Callable:
    """If the given array is more than two dimensions, iterate
    through each two dimensional slice. This is used when the
    filter can't handle more than two dimensions in an array.
    """
    @wraps(fn)
    def wrapper(a: np.ndarray, *args, **kwargs) -> np.ndarray:
        if len(a.shape) > 2:
            frames = [fn(frame, *args, **kwargs) for frame in a]
            out = np.array(frames)
        else:
            out = fn(a, *args, **kwargs)
        return out
    return wrapper


def uses_uint8(fn: Callable) -> Callable:
    """Converts the image data from floats to ints."""
    @wraps(fn)
    def wrapper(a: np.ndarray, *args, **kwargs) -> np.ndarray:
        # The wrapped function requires the image data be 8-bit
        # unsigned integers. If it's not, do the conversion.
        original_type = a.dtype
        if original_type != np.uint8:
            a = (a * 0xff).astype(np.uint8)

        # Pass the converted array to the wrapped function.
        a = fn(a, *args, **kwargs)

        # Ensure the image data is back to the type that was
        # originally passed to the function when it is returned.
        if original_type != a.dtype:
            a = a.astype(original_type) / 0xff
        return a
    return wrapper


def will_square(fn: Callable) -> Callable:
    """The array needs to have equal sized X and Y axes. The result
    will be sliced to the size of the original array.
    """
    @wraps(fn)
    def wrapper(a: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Determine if the Y and X axes aren't square.
        old_size = None
        if a.shape[X] != a.shape[Y]:
            old_size = a.shape
            largest = max(a.shape[Y:])
            new_size = (*a.shape[:Y], largest, largest)
            new_a = np.zeros(new_size, dtype=a.dtype)
            x_start = (largest - old_size[X]) // 2
            x_end = x_start + old_size[X]
            y_start = (largest - old_size[Y]) // 2
            y_end = y_start + old_size[Y]
            new_a[..., y_start:y_end, x_start:x_end] = a
            a = new_a
            del new_a

        # Send to the wrapped function.
        a = fn(a, *args, **kwargs)

        # Resize result back to the size of the original image if
        # needed before returning.
        if old_size:
            y_start = (a.shape[Y] - old_size[Y]) // 2
            y_end = y_start + old_size[Y]
            x_start = (a.shape[X] - old_size[X]) // 2
            x_end = x_start + old_size[X]
            a = a[..., y_start:y_end, x_start:x_end]
        return a
    return wrapper


# Discovery functions.
def get_prefixed_functions(prefix: str, obj: object) -> dict:
    """Return the functions within the given object that start with
    the prefix.
    """
    names = getmembers(obj, isfunction)
    p_len = len(prefix)
    fns = {name[p_len:]: fn for name, fn in names if name.startswith(prefix)}
    return fns


# Interpolation functions.
def bilinear_interpolation(a: np.ndarray, factor: float) -> np.ndarray:
    """Resize an two dimensional array using trilinear
    interpolation.

    :param a: The array to resize. The array is expected to have at
        least two dimensions.
    :param factor: The amount to resize the array. Given how the
        interpolation works, you probably don't get great results
        with factor less than or equal to .5. Consider multiple
        passes of interpolation with larger factors in those cases.
    :return: A :class:ndarray object.
    :rtype: numpy.ndarray
    """
    # Return the array unchanged if the array won't be magnified.
    if factor == 1:
        return a

    # Perform a defensive copy of the original array to avoid
    # unexpected side effects.
    a = a.copy()

    # Since we are magnifying the given array, the new array's shape
    # will increase by the magnification factor.
    mag_size = tuple(int(s * factor) for s in a.shape)

    # Map out the relationship between the old space and the
    # new space.
    indices = np.indices(mag_size)
    if factor > 1:
        whole = (indices // factor).astype(int)
        parts = (indices / factor - whole).astype(float)
    else:
        new_ends = [s - 1 for s in mag_size]
        old_ends = [s - 1 for s in a.shape]
        true_factors = [n / o for n, o in zip(new_ends, old_ends)]
        for i in range(len(true_factors)):
            if true_factors[i] == 0:
                true_factors[i] = .5
        whole = indices.copy()
        parts = indices.copy()
        for i in Y, X:
            whole[i] = (indices[i] // true_factors[i]).astype(int)
            parts[i] = (indices[i] / true_factors[i] - whole[i]).astype(float)
    del indices

    # Trilinear interpolation determines the value of a new pixel by
    # comparing the values of the eight old pixels that surround it.
    # The hashes are the keys to the dictionary that contains those
    # old pixel values. The key indicates the position of the pixel
    # on each axis, with one meaning the position is ahead of the
    # new pixel, and zero meaning the position is behind it.
    hashes = [f'{n:>02b}'[::-1] for n in range(2 ** 2)]
    hash_table = {}

    # The original array needs to be made one dimensional for the
    # numpy.take operation that will occur as we build the tables.
    raveled = np.ravel(a)

    # Build the table that contains the old pixel values to
    # interpolate.
    for hash in hashes:
        hash_whole = whole.copy()

        # Use the hash key to adjust the which old pixel we are
        # looking at.
        for axis in Y, X:
            if hash[axis] == '1':
                hash_whole[axis] += 1

                # Handle the pixels that were pushed off the far
                # edge of the original array by giving them the
                # value of the last pixel along that axis in the
                # original array.
                m = np.zeros(hash_whole[axis].shape, dtype=bool)
                m[hash_whole[axis] >= a.shape[axis]] = True
                hash_whole[axis][m] = a.shape[axis] - 1

        # Since numpy.take() only works in one dimension, we need to
        # map the three dimensional indices of the original array to
        # the one dimensional indices used by the raveled version of
        # that array.
        raveled_indices = hash_whole[Y] * a.shape[X]
        raveled_indices += hash_whole[X]

        # Get the value of the pixel in the original array.
        hash_table[hash] = np.take(raveled, raveled_indices.astype(int))

    # Once the hash table has been built, clean up the working arrays
    # in case we are running short on memory.
    else:
        del hash_whole, raveled_indices, whole

    # Everything before this was to set up the interpolation. Now that
    # it's set up, we perform the interpolation. Since we are doing
    # this across three dimensions, it's a three stage process. Stage
    # one is along the X axis.
    x1 = lerp(hash_table['00'], hash_table['01'], parts[X])
    x2 = lerp(hash_table['10'], hash_table['11'], parts[X])

    # And stage three is along the Z axis. Since this is the last step
    # we can just return the result.
    return lerp(x1, x2, parts[Y])


def lerp(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Perform a linear interpolation on the values of two arrays

    :param a: The "left" values.
    :param b: The "right" values.
    :param x: An array of how close the location of the final value
        should be to the "left" value.
    :return: A :class:ndarray object
    :rtype: numpy.ndarray

    Usage::

        >>> import numpy as np
        >>>
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([3, 4, 5])
        >>> x = np.array([.5, .5, .5])
        >>> lerp(a, b, x)
        array([2., 3., 4.])
    """
    return a.astype(float) * (1 - x.astype(float)) + b.astype(float) * x


def trilinear_interpolation(a: np.ndarray, factor: float) -> np.ndarray:
    """Resize an three dimensional array using trilinear
    interpolation.

    :param a: The array to resize. The array is expected to have at
        least three dimensions.
    :param factor: The amount to resize the array. Given how the
        interpolation works, you probably don't get great results
        with factor less than or equal to .5. Consider multiple
        passes of interpolation with larger factors in those cases.
    :return: A :class:ndarray object.
    :rtype: numpy.ndarray

    Usage::

        >>> import numpy as np
        >>>
        >>> a = np.array([
        ...     [
        ...             [0, 1],
        ...             [1, 0],
        ...     ],
        ...     [
        ...             [1, 0],
        ...             [0, 1],
        ...     ],
        ... ])
        >>> trilinear_interpolation(a, 2)
        array([[[0. , 0.5, 1. , 1. ],
                [0.5, 0.5, 0.5, 0.5],
                [1. , 0.5, 0. , 0. ],
                [1. , 0.5, 0. , 0. ]],
        <BLANKLINE>
               [[0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5]],
        <BLANKLINE>
               [[1. , 0.5, 0. , 0. ],
                [0.5, 0.5, 0.5, 0.5],
                [0. , 0.5, 1. , 1. ],
                [0. , 0.5, 1. , 1. ]],
        <BLANKLINE>
               [[1. , 0.5, 0. , 0. ],
                [0.5, 0.5, 0.5, 0.5],
                [0. , 0.5, 1. , 1. ],
                [0. , 0.5, 1. , 1. ]]])
    """
    # Return the array unchanged if the array won't be magnified.
    if factor == 1:
        return a

    # Perform a defensive copy of the original array to avoid
    # unexpected side effects.
    a = a.copy()

    # Since we are magnifying the given array, the new array's shape
    # will increase by the magnification factor.
    mag_size = tuple(int(s * factor) for s in a.shape)

    # Map out the relationship between the old space and the
    # new space.
    indices = np.indices(mag_size)
    if factor > 1:
        whole = (indices // factor).astype(int)
        parts = (indices / factor - whole).astype(float)
    else:
        new_ends = [s - 1 for s in mag_size]
        old_ends = [s - 1 for s in a.shape]
        true_factors = [n / o for n, o in zip(new_ends, old_ends)]
        for i in range(len(true_factors)):
            if true_factors[i] == 0:
                true_factors[i] = .5
        whole = indices.copy()
        parts = indices.copy()
        for i in Z, Y, X:
            whole[i] = (indices[i] // true_factors[i]).astype(int)
            parts[i] = (indices[i] / true_factors[i] - whole[i]).astype(float)
    del indices

    # Trilinear interpolation determines the value of a new pixel by
    # comparing the values of the eight old pixels that surround it.
    # The hashes are the keys to the dictionary that contains those
    # old pixel values. The key indicates the position of the pixel
    # on each axis, with one meaning the position is ahead of the
    # new pixel, and zero meaning the position is behind it.
    hashes = [f'{n:>03b}'[::-1] for n in range(2 ** 3)]
    hash_table = {}

    # The original array needs to be made one dimensional for the
    # numpy.take operation that will occur as we build the tables.
    raveled = np.ravel(a)

    # Build the table that contains the old pixel values to
    # interpolate.
    for hash in hashes:
        hash_whole = whole.copy()

        # Use the hash key to adjust the which old pixel we are
        # looking at.
        for axis in Z, Y, X:
            if hash[axis] == '1':
                hash_whole[axis] += 1

                # Handle the pixels that were pushed off the far
                # edge of the original array by giving them the
                # value of the last pixel along that axis in the
                # original array.
                m = np.zeros(hash_whole[axis].shape, dtype=bool)
                m[hash_whole[axis] >= a.shape[axis]] = True
                hash_whole[axis][m] = a.shape[axis] - 1

        # Since numpy.take() only works in one dimension, we need to
        # map the three dimensional indices of the original array to
        # the one dimensional indices used by the raveled version of
        # that array.
        raveled_indices = hash_whole[Z] * a.shape[Y] * a.shape[X]
        raveled_indices += hash_whole[Y] * a.shape[X]
        raveled_indices += hash_whole[X]

        # Get the value of the pixel in the original array.
        hash_table[hash] = np.take(raveled, raveled_indices.astype(int))

    # Once the hash table has been built, clean up the working arrays
    # in case we are running short on memory.
    else:
        del hash_whole, raveled_indices, whole

    # Everything before this was to set up the interpolation. Now that
    # it's set up, we perform the interpolation. Since we are doing
    # this across three dimensions, it's a three stage process. Stage
    # one is along the X axis.
    x1 = lerp(hash_table['000'], hash_table['001'], parts[X])
    x2 = lerp(hash_table['010'], hash_table['011'], parts[X])
    x3 = lerp(hash_table['100'], hash_table['101'], parts[X])
    x4 = lerp(hash_table['110'], hash_table['111'], parts[X])

    # Stage two is along the Y axis.
    y1 = lerp(x1, x2, parts[Y])
    y2 = lerp(x3, x4, parts[Y])
    del x1, x2, x3, x4

    # And stage three is along the Z axis. Since this is the last step
    # we can just return the result.
    return lerp(y1, y2, parts[Z])
