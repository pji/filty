"""
utility
~~~~~~~

Utility functions for the filty module.
"""
from functools import wraps
from typing import Callable

import numpy as np


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


# Convenience utilities.
def grayscale_to_rgb(a: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to RGB."""
    new_shape = (*a.shape, 3)
    new_a = np.zeros(new_shape, dtype=a.dtype)
    for channel in range(3):
        new_a[..., channel] = a
    return new_a


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


# Color constants.
COLOR = {
    # Don't colorize.
    '': [],

    # Grayscale
    'a': ['hsv(0, 0%, 100%)', 'hsv(0, 0%, 0%)'],
    'A': ['hsl(0, 0%, 75%)', 'hsl(0, 0%, 25%)'],

    # Electric blue.
    'b': ['hsv(200, 100%, 100%)', 'hsv(200, 100%, 0%)'],
    'B': ['hsl(200, 100%, 75%)', 'hsl(200, 100%, 25%)'],

    'bw': ['hsv(205, 100%, 100%)', 'hsv(200, 100%, 0%)'],
    'Bw': ['hsl(205, 100%, 75%)', 'hsl(200, 100%, 25%)'],

    'bk': ['hsv(200, 30%, 20%)', 'hsv(200, 30%, 0%)'],
    'BK': ['hsl(200, 30%, 30%)', 'hsl(200, 30%, 10%)'],

    # Cream
    'c': ['hsv(35, 100%, 100%)', 'hsv(35, 100%, 0%)'],
    'C': ['hsl(35, 100%, 80%)', 'hsl(35, 100%, 25%)'],

    'cw': ['hsv(30, 100%, 100%)', 'hsv(35, 100%, 0%)'],
    'Cw': ['hsl(30, 100%, 80%)', 'hsl(35, 100%, 25%)'],

    'cc': ['hsv(40, 100%, 100%)', 'hsv(35, 100%, 0%)'],
    'Cc': ['hsl(40, 100%, 80%)', 'hsl(35, 100%, 25%)'],

    'ck': ['hsv(35, 30%, 20%)', 'hsv(35, 30%, 0%)'],
    'CK': ['hsl(35, 30%, 30%)', 'hsl(35, 30%, 10%)'],

    # Dark.
    'k': ['hsv(220, 30%, 20%)', 'hsv(220, 30%, 0%)'],
    'K': ['hsl(220, 30%, 30%)', 'hsl(220, 30%, 10%)'],

    'kk': ['hsv(220, 30%, 10%)', 'hsv(220, 30%, 0%)'],
    'KK': ['hsl(220, 30%, 15%)', 'hsl(220, 30%, 5%)'],

    # Ectoplasmic teal.
    'e': ["hsv(190, 50%, 100%)", "hsv(190, 100%, 0%)"],
    'E': ["hsl(190, 50%, 100%)", "hsl(190, 100%, 30%)"],

    # Electric green.
    'g': ['hsv(90, 100%, 100%)', 'hsv(90, 100%, 0%)'],
    'G': ['hsl(90, 100%, 75%)', 'hsl(90, 100%, 25%)'],

    'gk': ['hsv(90, 30%, 20%)', 'hsv(90, 30%, 0%)'],
    'GK': ['hsl(90, 30%, 30%)', 'hsl(90, 30%, 10%)'],

    # Slate.
    'l': ['hsv(220, 30%, 50%)', 'hsv(220, 30%, 0%)'],
    'L': ['hsl(220, 30%, 75%)', 'hsl(220, 30%, 25%)'],

    # Electric pink.
    'p': ['hsv(320, 100%, 100%)', 'hsv(320, 100%, 0%)'],
    'P': ['hsl(320, 100%, 75%)', 'hsl(320, 100%, 25%)'],

    # Royal purple.
    'r': ['hsv(280, 100%, 100%)', 'hsv(280, 100%, 0%)'],
    'R': ['hsl(280, 100%, 75%)', 'hsl(280, 100%, 25%)'],

    'rw': ['hsv(285, 100%, 100%)', 'hsv(280, 100%, 0%)'],
    'Rw': ['hsl(285, 100%, 75%)', 'hsl(280, 100%, 25%)'],

    # Scarlet.
    's': ['hsv(350, 100%, 100%)', 'hsv(10, 100%, 0%)'],
    'S': ['hsl(350, 100%, 75%)', 'hsl(10, 100%, 25%)'],

    'sw': ['hsv(0, 100%, 100%)', 'hsv(10, 100%, 0%)'],
    'Sw': ['hsl(0, 100%, 75%)', 'hsl(10, 100%, 25%)'],

    'sk': ['hsv(350, 30%, 20%)', 'hsv(10, 30%, 0%)'],
    'SK': ['hsl(350, 30%, 30%)', 'hsl(10, 30%, 10%)'],

    # White.
    'w': ['hsv(0, 0%, 100%)', 'hsv(0, 0%, 0%)'],
    'W': ['hsl(0, 0%, 75%)', 'hsl(0, 0%, 25%)'],

    # Hue templates.
    't': ['hsv({}, 100%, 100%)', 'hsv({}, 100%, 0%)'],
    'T': ['hsl({}, 100%, 75%)', 'hsl({}, 100%, 25%)'],
}
