"""
filty
~~~~~

Filter functions for image data.
"""
import numpy as np

from filty.utility import grayscale_to_rgb


# Image filter functions.
def filter_colorize(a: np.ndarray, color: tuple[float]) -> np.array:
    """Colorize a grayscale image."""
    # Grayscale data can't have color.
    if a.shape[-1] != 3 or len(a.shape) == 2:
        a = grayscale_to_rgb(a)

    # Multiply the solid color with the grayscale.
    a *= color
    return a
