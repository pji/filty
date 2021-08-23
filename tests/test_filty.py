"""
test_filty
~~~~~~~~~~
"""
import unittest as ut

import numpy as np

from tests.common import ArrayTestCase, A, F
from filty import filty as f


# Base test case.
class FilterTestCase(ArrayTestCase):
    def run_test(self, filter, exp, a=None, *args, **kwargs):
        """Run a basic test case for a filter."""
        # Test data and state.
        if a is None:
            a = A

        # Run test.
        act = filter(a, *args, **kwargs)

        # Determine test result.
        self.assertArrayEqual(exp, act)


# Test cases.
class ColorizeTestCase(FilterTestCase):
    def test_filter(self):
        """Given an RGB color and grayscale image data, the color is
        applied to the image data.
        """
        filter = f.filter_colorize
        exp = np.array([
            [[1.00, 0.50, 0.50], [0.50, 0.25, 0.25], [0.00, 0.00, 0.00]],
            [[0.50, 0.25, 0.25], [0.00, 0.00, 0.00], [0.50, 0.25, 0.25]],
            [[0.00, 0.00, 0.00], [0.50, 0.25, 0.25], [1.00, 0.50, 0.50]],
        ], dtype=np.float32)
        a = F.copy()
        color = (1.0, 0.5, 0.5)
        self.run_test(filter, exp, a, color=color)

    def test_colorize_rgb(self):
        """Given an RGB color and RGB image data, the color is applied
        to the image data.
        """
        filter = f.filter_colorize
        exp = np.array([
            [
                [[1.00, 0.50, 0.50], [0.50, 0.25, 0.25], [0.00, 0.00, 0.00]],
                [[0.50, 0.25, 0.25], [0.00, 0.00, 0.00], [0.50, 0.25, 0.25]],
                [[0.00, 0.00, 0.00], [0.50, 0.25, 0.25], [1.00, 0.50, 0.50]],
            ],
        ], dtype=np.float32)
        a = np.array([
            [
                [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
                [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
            ],
        ], dtype=np.float32)
        color = (1.0, 0.5, 0.5)
        self.run_test(filter, exp, a, color=color)

    def test_colorize_screen(self):
        """Given the screen mode, the color is applied using the screen
        blend rather than multiply.
        """
        filter = f.filter_colorize
        exp = np.array([
            [[1.00, 1.00, 1.00], [1.00, 0.75, 0.75], [1.00, 0.50, 0.50]],
            [[1.00, 0.75, 0.75], [1.00, 0.50, 0.50], [1.00, 0.75, 0.75]],
            [[1.00, 0.50, 0.50], [1.00, 0.75, 0.75], [1.00, 1.00, 1.00]],
        ], dtype=np.float32)
        a = F.copy()
        color = (1.0, 0.5, 0.5)
        blend = 'screen'
        self.run_test(filter, exp, a, color=color, blend=blend)
