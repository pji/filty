"""
filty
~~~~~

Filter functions for image data.
"""
import numpy as np


# Image filter functions.
def colorize(a: np.ndarray, color: tuple[float]) -> np.array:
    """Colorize a grayscale image."""
    # Convert grayscale image to RGB.
    new_shape = (*a.shape, 3)
    new_a = np.zeros(new_shape, dtype=a.dtype)
    for channel in range(3):
        new_a[..., channel] = a

    # Multiply the solid color with the grayscale.
    new_a *= color
    return new_a