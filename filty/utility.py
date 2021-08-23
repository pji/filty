"""
utility
~~~~~~~

Utility functions for the filty module.
"""
import numpy as np


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
#         if a.dtype == np.float32 or a.dtype == np.float64:
#             a = np.around(a, 4)
        nums = [f'{n}' for n in a]
        print(' ' * (4 * depth) + '[' + ', '.join(nums) + '],')
