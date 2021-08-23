"""
utility
~~~~~~~

Utility functions for the filty module.
"""
import numpy as np


# Convenience utilities.
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
