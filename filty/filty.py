"""
filty
~~~~~

Filter functions for image data.
"""
import numpy as np

from filty.utility import grayscale_to_rgb


# Image filter functions.
def filter_colorize(a: np.ndarray, 
                    color: tuple[float],
                    blend: str = 'multiply') -> np.array:
    """Colorize a grayscale image."""
    # Grayscale data can't have color.
    if a.shape[-1] != 3 or len(a.shape) == 2:
        a = grayscale_to_rgb(a)

    # Blend the solid color with the grayscale.
    if blend == 'multiply':
        a *= color
    elif blend == 'screen':
        rev_a = 1 - a
        rev_color = tuple(1 - c for c in color)
        rev_a *= rev_color
        a = 1 - rev_a
    return a


if __name__ == '__main__':
    from tests.common import F
    from filty.utility import print_array
    
    filter = filter_colorize
    kwargs = {
        'a': F.copy(),
        'color': (1.0, 0.5, 0.5),
        'blend': 'screen',
    }
    out = filter(**kwargs)
    print_array(out)
