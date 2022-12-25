"""
test_imgfilt
~~~~~~~~~~
"""
import unittest as ut

import numpy as np

from tests.common import (ArrayTestCase, A, E, F, VIDEO_2_3_3, VIDEO_2_5_5,
                          IMAGE_5_5_LOW_CONTRAST)
from imgfilt import imgfilt as f


# Base test case.
class FilterTestCase(ArrayTestCase):
    def run_test(self, filter, exp, a=None, *args, **kwargs):
        """Run a basic test case for a filter."""
        # Test data and state.
        if a is None:
            a = A.copy()

        # Run test.
        act = filter(a, *args, **kwargs)

        # Determine test result.
        self.assertArrayEqual(exp, act, round_=True)


# Test cases.
class BoxBlurTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data and a box size, perform a box blur on the
        image data.
        """
        filter = f.filter_box_blur
        exp = np.array([
            [0.2500, 0.2500, 0.5000, 0.7500, 0.8750],
            [0.2500, 0.2500, 0.5000, 0.7500, 0.8750],
            [0.5000, 0.5000, 0.7500, 0.8750, 0.7500],
            [0.7500, 0.7500, 0.8750, 0.7500, 0.5000],
            [0.8750, 0.8750, 0.7500, 0.5000, 0.2500],
        ], dtype=np.float32)
        kwargs = {
            'size': 2,
        }
        self.run_test(filter, exp, **kwargs)

    def test_filter_on_video(self):
        """Given three dimensional image data, the blur should be
        performed on all frames of the image data.
        """
        filter = f.filter_box_blur
        exp = np.array([
            [
                [0.2500, 0.2500, 0.5000, 0.7500, 0.8750],
                [0.2500, 0.2500, 0.5000, 0.7500, 0.8750],
                [0.5000, 0.5000, 0.7500, 0.8750, 0.7500],
                [0.7500, 0.7500, 0.8750, 0.7500, 0.5000],
                [0.8750, 0.8750, 0.7500, 0.5000, 0.2500],
            ],
            [
                [0.8750, 0.8750, 0.7500, 0.5000, 0.2500],
                [0.8750, 0.8750, 0.7500, 0.5000, 0.2500],
                [0.7500, 0.7500, 0.8750, 0.7500, 0.5000],
                [0.5000, 0.5000, 0.7500, 0.8750, 0.7500],
                [0.2500, 0.2500, 0.5000, 0.7500, 0.8750],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_5_5.copy()
        kwargs = {
            'size': 2,
        }
        self.run_test(filter, exp, a, **kwargs)


class ColorizeTestCase(FilterTestCase):
    def test_filter(self):
        """Given an RGB color and grayscale image data, the color is
        applied to the image data.
        """
        filter = f.filter_colorize
        exp = np.array([
            [
                [1.0000, 0.0000, 0.1686],
                [0.4980, 0.0000, 0.0824],
                [0.0000, 0.0000, 0.0000],
            ],
            [
                [0.4980, 0.0000, 0.0824],
                [0.0000, 0.0000, 0.0000],
                [0.4980, 0.0000, 0.0824],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [0.4980, 0.0000, 0.0824],
                [1.0000, 0.0000, 0.1686],
            ],
        ], dtype=np.float32)
        a = F.copy()
        kwargs = {
            'white': 'hsv(350, 100%, 100%)',
            'black': 'hsv(10, 100%, 0%)',
        }
        self.run_test(filter, exp, a, **kwargs)

    def test_filter_by_colorkey(self):
        """Given an RGB color and grayscale image data, the color is
        applied to the image data.
        """
        filter = f.filter_colorize
        exp = np.array([
            [
                [1.0000, 0.0000, 0.1686],
                [0.4980, 0.0000, 0.0824],
                [0.0000, 0.0000, 0.0000],
            ],
            [
                [0.4980, 0.0000, 0.0824],
                [0.0000, 0.0000, 0.0000],
                [0.4980, 0.0000, 0.0824],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [0.4980, 0.0000, 0.0824],
                [1.0000, 0.0000, 0.1686],
            ],
        ], dtype=np.float32)
        a = F.copy()
        kwargs = {
            'colorkey': 's',
        }
        self.run_test(filter, exp, a, **kwargs)

    def test_filter_by_colorkey_with_mid(self):
        """Given an RGB color and grayscale image data, the color is
        applied to the image data. If the given colorkey has a mid, then
        it is used in the colorization.
        """
        filter = f.filter_colorize
        exp = np.array([
            [
                [1.0000, 1.0000, 1.0000],
                [0.0000, 0.8314, 1.0000],
                [0.0000, 0.0000, 0.0000],
            ],
            [
                [0.0000, 0.8314, 1.0000],
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.8314, 1.0000],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.8314, 1.0000],
                [1.0000, 1.0000, 1.0000],
            ],
        ], dtype=np.float32)
        a = F.copy()
        kwargs = {
            'colorkey': 'em',
        }
        self.run_test(filter, exp, a, **kwargs)

    def test_filter_on_video(self):
        """Given three-dimensional image data, the filter should
        colorize every frame of the data.
        """
        filter = f.filter_colorize
        exp = np.array([
            [
                [
                    [1.0000, 0.0000, 0.1686],
                    [0.4980, 0.0000, 0.0824],
                    [0.0000, 0.0000, 0.0000],
                ],
                [
                    [0.4980, 0.0000, 0.0824],
                    [0.0000, 0.0000, 0.0000],
                    [0.4980, 0.0000, 0.0824],
                ],
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.4980, 0.0000, 0.0824],
                    [1.0000, 0.0000, 0.1686],
                ],
            ],
            [
                [
                    [1.0000, 0.0000, 0.1686],
                    [0.4980, 0.0000, 0.0824],
                    [0.0000, 0.0000, 0.0000],
                ],
                [
                    [0.4980, 0.0000, 0.0824],
                    [0.0000, 0.0000, 0.0000],
                    [0.4980, 0.0000, 0.0824],
                ],
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.4980, 0.0000, 0.0824],
                    [1.0000, 0.0000, 0.1686],
                ],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_3_3.copy()
        kwargs = {
            'colorkey': 's',
        }
        self.run_test(filter, exp, a, **kwargs)


class ContrastTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data, adjust the range of the data to ensure
        the darkest color is black and the lightest is white.
        """
        filter = f.filter_contrast
        exp = np.array([
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
        ], dtype=np.float32)
        a = IMAGE_5_5_LOW_CONTRAST.copy()
        self.run_test(filter, exp, a)


class FlipTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data and an axis, flip the image around
        that axis.
        """
        filter = f.filter_flip
        exp = np.array([
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            [0.7500, 1.0000, 0.7500, 0.5000, 0.2500],
            [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.2500, 0.5000, 0.7500, 1.0000, 0.7500],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
        ], dtype=np.float32)
        kwarg = {
            'axis': f.X,
        }
        self.run_test(filter, exp, **kwarg)

    def test_flip_y_axis(self):
        """Given image data and an axis, flip the image around
        that axis.
        """
        filter = f.filter_flip
        exp = np.array([
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            [0.7500, 1.0000, 0.7500, 0.5000, 0.2500],
            [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.2500, 0.5000, 0.7500, 1.0000, 0.7500],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
        ], dtype=np.float32)
        kwarg = {
            'axis': f.X,
        }
        self.run_test(filter, exp, **kwarg)

    def test_flip_z_axis(self):
        """Given image data and an axis, flip the image around
        that axis.
        """
        filter = f.filter_flip
        exp = np.array([
            [
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
                [0.7500, 1.0000, 0.7500, 0.5000, 0.2500],
                [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
                [0.2500, 0.5000, 0.7500, 1.0000, 0.7500],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            ],
            [
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
                [0.2500, 0.5000, 0.7500, 1.0000, 0.7500],
                [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
                [0.7500, 1.0000, 0.7500, 0.5000, 0.2500],
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_5_5.copy()
        kwarg = {
            'axis': f.Z,
        }
        self.run_test(filter, exp, a, **kwarg)


class GaussianBlurTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data and a sigma, perform a gaussian blur
        on the image data.
        """
        filter = f.filter_gaussian_blur
        exp = np.array([
            [0.1070, 0.3036, 0.5534, 0.7918, 0.9158],
            [0.3036, 0.5002, 0.7442, 0.9046, 0.7918],
            [0.5534, 0.7442, 0.9044, 0.7442, 0.5534],
            [0.7918, 0.9046, 0.7442, 0.5002, 0.3036],
            [0.9158, 0.7918, 0.5534, 0.3036, 0.1070],
        ], dtype=np.float32)
        kwarg = {
            'sigma': 0.5,
        }
        self.run_test(filter, exp, **kwarg)

    def test_blur_video(self):
        """Given image data and a sigma, perform a gaussian blur
        on the image data.
        """
        filter = f.filter_gaussian_blur
        exp = np.array([
            [
                [0.2436, 0.4099, 0.5535, 0.6968, 0.7564],
                [0.3565, 0.5952, 0.7500, 0.8516, 0.6435],
                [0.5000, 0.7499, 0.9465, 0.7499, 0.5000],
                [0.6435, 0.8516, 0.7500, 0.5952, 0.3565],
                [0.7564, 0.6968, 0.5535, 0.4099, 0.2436],
            ],
            [
                [0.7564, 0.6968, 0.5535, 0.4099, 0.2436],
                [0.6435, 0.8516, 0.7500, 0.5952, 0.3565],
                [0.5000, 0.7499, 0.9465, 0.7499, 0.5000],
                [0.3565, 0.5952, 0.7500, 0.8516, 0.6435],
                [0.2436, 0.4099, 0.5535, 0.6968, 0.7564],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_5_5.copy()
        kwarg = {
            'sigma': 0.5,
        }
        self.run_test(filter, exp, a, **kwarg)


class GlowTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data and a size factor, zoom into the image
        by the size factor.
        """
        filter = f.filter_glow
        exp = np.array([
            [
                [0.7477, 0.9173, 0.9596, 0.9729, 1.0000],
                [0.8113, 0.9510, 0.9844, 1.0000, 0.9379],
                [0.8750, 0.9784, 1.0000, 0.9784, 0.8750],
                [0.9379, 1.0000, 0.9844, 0.9510, 0.8113],
                [1.0000, 0.9729, 0.9596, 0.9173, 0.7477],
            ],
            [
                [1.0000, 0.9729, 0.9596, 0.9173, 0.7477],
                [0.9379, 1.0000, 0.9844, 0.9510, 0.8113],
                [0.8750, 0.9784, 1.0000, 0.9784, 0.8750],
                [0.8113, 0.9510, 0.9844, 1.0000, 0.9379],
                [0.7477, 0.9173, 0.9596, 0.9729, 1.0000],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_5_5.copy()
        kwarg = {
            'sigma': 4,
        }
        self.run_test(filter, exp, a, **kwarg)


class GrowTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data and a size factor, zoom into the image
        by the size factor.
        """
        filter = f.filter_grow
        exp = np.array([
            [
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000, 0.0000],
                [0.7500, 0.5000, 0.2500, 0.2500, 0.2500, 0.2500],
                [0.5000, 0.2500, 0.0000, 0.2500, 0.5000, 0.5000],
                [0.2500, 0.2500, 0.2500, 0.5000, 0.7500, 0.7500],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
            ],
            [
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000, 0.0000],
                [0.7500, 0.5000, 0.2500, 0.2500, 0.2500, 0.2500],
                [0.5000, 0.2500, 0.0000, 0.2500, 0.5000, 0.5000],
                [0.2500, 0.2500, 0.2500, 0.5000, 0.7500, 0.7500],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
            ],
            [
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000, 0.0000],
                [0.7500, 0.5000, 0.2500, 0.2500, 0.2500, 0.2500],
                [0.5000, 0.2500, 0.0000, 0.2500, 0.5000, 0.5000],
                [0.2500, 0.2500, 0.2500, 0.5000, 0.7500, 0.7500],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
            ],
            [
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000, 0.0000],
                [0.7500, 0.5000, 0.2500, 0.2500, 0.2500, 0.2500],
                [0.5000, 0.2500, 0.0000, 0.2500, 0.5000, 0.5000],
                [0.2500, 0.2500, 0.2500, 0.5000, 0.7500, 0.7500],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_3_3.copy()
        kwarg = {
            'factor': 2,
        }
        self.run_test(filter, exp, a, **kwarg)

    def test_two_dimensional_grow(self):
        """Given image data and a size factor, zoom into the image
        by the size factor. This should work on still image data.
        """
        filter = f.filter_grow
        exp = np.array([
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000, 0.0000],
            [0.7500, 0.5000, 0.2500, 0.2500, 0.2500, 0.2500],
            [0.5000, 0.2500, 0.0000, 0.2500, 0.5000, 0.5000],
            [0.2500, 0.2500, 0.2500, 0.5000, 0.7500, 0.7500],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
        ], dtype=np.float32)
        a = F.copy()
        kwarg = {
            'factor': 2,
        }
        self.run_test(filter, exp, a, **kwarg)


class InverseTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data, invert the colors of the image data."""
        filter = f.filter_inverse
        exp = np.array([
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            [0.7500, 0.5000, 0.2500, 0.0000, 0.2500],
            [0.5000, 0.2500, 0.0000, 0.2500, 0.5000],
            [0.2500, 0.0000, 0.2500, 0.5000, 0.7500],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
        ], dtype=np.float32)
        self.run_test(filter, exp)


class LinearToPolarTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data, convert the linear coordinates to
        polar coordinates.
        """
        filter = f.filter_linear_to_polar
        exp = np.array([
            [0.0000, 0.2500, 0.0000, 0.0000, 0.0000],
            [0.2500, 0.5000, 0.7500, 0.5000, 0.2500],
            [0.2500, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.5000, 1.0000, 0.7500, 0.5000, 0.5000],
            [0.5000, 0.7500, 1.0000, 0.7500, 1.0000],
        ], dtype=np.float32)
        self.run_test(filter, exp)

    def test_filter_video(self):
        """Convert coordinates for video."""
        filter = f.filter_linear_to_polar
        exp = np.array([
            [
                [0.0000, 0.2500, 0.0000, 0.0000, 0.0000],
                [0.2500, 0.5000, 0.7500, 0.5000, 0.2500],
                [0.2500, 0.7500, 1.0000, 0.7500, 0.5000],
                [0.5000, 1.0000, 0.7500, 0.5000, 0.5000],
                [0.5000, 0.7500, 1.0000, 0.7500, 1.0000],
            ],
            [
                [0.0000, 0.7500, 1.0000, 1.0000, 1.0000],
                [0.7500, 1.0000, 0.7500, 0.5000, 0.7500],
                [0.7500, 0.7500, 0.5000, 0.2500, 0.5000],
                [0.5000, 1.0000, 0.7500, 1.0000, 0.5000],
                [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_5_5.copy()
        self.run_test(filter, exp, a)


class MotionBlurTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data, an amount, and a direction, perform a
        motion blur on the image data.
        """
        filter = f.filter_motion_blur
        exp = np.array([
            [0.1250, 0.1250, 0.3750, 0.6250, 0.8750],
            [0.3750, 0.3750, 0.6250, 0.8750, 0.8750],
            [0.6250, 0.6250, 0.8750, 0.8750, 0.6250],
            [0.8750, 0.8750, 0.8750, 0.6250, 0.3750],
            [0.8750, 0.8750, 0.6250, 0.3750, 0.1250],
        ], dtype=np.float32)
        kwargs = {
            'amount': 2,
            'axis': f.X,
        }
        self.run_test(filter, exp, **kwargs)

    def test_filter_vertical(self):
        """The motion blur should be vertical when done on the
        Y axis.
        """
        filter = f.filter_motion_blur
        exp = np.array([
            [0.1250, 0.3750, 0.6250, 0.8750, 0.8750],
            [0.1250, 0.3750, 0.6250, 0.8750, 0.8750],
            [0.3750, 0.6250, 0.8750, 0.8750, 0.6250],
            [0.6250, 0.8750, 0.8750, 0.6250, 0.3750],
            [0.8750, 0.8750, 0.6250, 0.3750, 0.1250],
        ], dtype=np.float32)
        kwargs = {
            'amount': 2,
            'axis': f.Y,
        }
        self.run_test(filter, exp, **kwargs)

    def test_filter_video(self):
        """The motion blur work on video.
        """
        filter = f.filter_motion_blur
        exp = np.array([
            [
                [0.1250, 0.1250, 0.3750, 0.6250, 0.8750],
                [0.3750, 0.3750, 0.6250, 0.8750, 0.8750],
                [0.6250, 0.6250, 0.8750, 0.8750, 0.6250],
                [0.8750, 0.8750, 0.8750, 0.6250, 0.3750],
                [0.8750, 0.8750, 0.6250, 0.3750, 0.1250],
            ],
            [
                [0.8750, 0.8750, 0.6250, 0.3750, 0.1250],
                [0.8750, 0.8750, 0.8750, 0.6250, 0.3750],
                [0.6250, 0.6250, 0.8750, 0.8750, 0.6250],
                [0.3750, 0.3750, 0.6250, 0.8750, 0.8750],
                [0.1250, 0.1250, 0.3750, 0.6250, 0.8750],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_5_5.copy()
        kwargs = {
            'amount': 2,
            'axis': f.X,
        }
        self.run_test(filter, exp, a, **kwargs)

    def test_filter_along_disallowed_axis(self):
        """The motion blur work on video.
        """
        # Expected value.
        exp_ex = ValueError
        exp_msg = 'motion_blur can only affect the X or Y axis.'

        # Test data and setup.
        filter = f.filter_motion_blur
        a = A.copy()
        kwargs = {
            'amount': 2,
            'axis': -3,
        }

        # Run test and determine result.
        with self.assertRaisesRegex(exp_ex, exp_msg):
            _ = filter(a, **kwargs)


class PinchTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data, an amount of the pinch, a radius, a
        scale, and an offset, perform a pinch on the image data.
        """
        filter = f.filter_pinch
        exp = np.array([
            [0.0000, 0.0859, 0.1465, 0.2441, 0.3438],
            [0.0859, 0.2188, 0.4609, 0.8340, 0.3896],
            [0.1465, 0.4609, 0.6719, 0.7500, 0.0713],
            [0.2441, 0.8340, 0.7500, 0.1719, 0.0225],
            [0.3438, 0.3896, 0.0713, 0.0225, 0.0000],
        ], dtype=np.float32)
        kwargs = {
            'amount': 0.5,
            'radius': 3.0,
            'scale': (0.5, 0.5),
            'offset': (0, 0, 0),
        }
        self.run_test(filter, exp, **kwargs)

    def test_filter_video(self):
        """Pinch should work on video.
        """
        filter = f.filter_pinch
        exp = np.array([
            [
                [0.0000, 0.0859, 0.1465, 0.2441, 0.3438],
                [0.0859, 0.2188, 0.4609, 0.8340, 0.3896],
                [0.1465, 0.4609, 0.6719, 0.7500, 0.0713],
                [0.2441, 0.8340, 0.7500, 0.1719, 0.0225],
                [0.3438, 0.3896, 0.0713, 0.0225, 0.0000],
            ],
            [
                [0.5166, 0.4141, 0.1660, 0.0684, 0.0000],
                [0.4141, 0.8770, 0.6016, 0.2109, 0.0479],
                [0.1660, 0.6016, 0.8872, 0.4219, 0.0537],
                [0.0684, 0.2109, 0.4219, 0.8872, 0.1025],
                [0.0000, 0.0479, 0.0537, 0.1025, 0.1914],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_5_5.copy()
        kwargs = {
            'amount': 0.5,
            'radius': 3.0,
            'scale': (0.5, 0.5),
            'offset': (0, 0, 0),
        }
        self.run_test(filter, exp, a, **kwargs)


class PolarToLinearTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data, convert the polar coordinates to
        linear coordinates.
        """
        filter = f.filter_polar_to_linear
        exp = np.array([
            [1.0000, 0.7500, 0.5000, 0.0000, 0.0000],
            [1.0000, 0.5000, 0.2500, 0.0000, 0.0000],
            [1.0000, 0.7500, 1.0000, 0.7500, 1.0000],
            [1.0000, 1.0000, 0.7500, 0.5000, 0.2500],
            [1.0000, 0.7500, 1.0000, 0.7500, 0.7500],
        ], dtype=np.float32)
        self.run_test(filter, exp)

    def test_filter_video(self):
        """Convert coordinates for video."""
        filter = f.filter_polar_to_linear
        exp = np.array([
            [
                [1.0000, 0.7500, 0.5000, 0.0000, 0.0000],
                [1.0000, 0.5000, 0.2500, 0.0000, 0.0000],
                [1.0000, 0.7500, 1.0000, 0.7500, 1.0000],
                [1.0000, 1.0000, 0.7500, 0.5000, 0.2500],
                [1.0000, 0.7500, 1.0000, 0.7500, 0.7500],
            ],
            [
                [1.0000, 0.7500, 0.5000, 0.0000, 0.0000],
                [1.0000, 1.0000, 0.7500, 0.0000, 0.0000],
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
                [1.0000, 1.0000, 0.7500, 1.0000, 0.7500],
                [1.0000, 0.7500, 0.5000, 0.2500, 0.2500],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_5_5.copy()
        self.run_test(filter, exp, a)


class RippleTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data, a wavelength of the ripples, an
        amplitude, a distortion axis, and an offset, perform a
        ripple distortion on the image data.
        """
        filter = f.filter_ripple
        exp = np.array([
            [1.0000, 0.0000, 0.5000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.7500, 0.0000, 0.7500],
            [0.5000, 0.7500, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.5000, 0.0000],
            [0.0000, 0.7500, 0.0000, 0.0000, 0.0000],
        ], dtype=np.float32)
        kwargs = {
            'wave': (2, 2),
            'amp': (2, 2),
            'distaxis': (f.Y, f.X),
            'offset': (0, 0),
        }
        self.run_test(filter, exp, **kwargs)

    def test_filter_video(self):
        """The filter should work on video.
        """
        filter = f.filter_ripple
        exp = np.array([
            [
                [1.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.7500, 0.0000, 0.7500],
                [0.5000, 0.7500, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.5000, 0.0000],
                [0.0000, 0.7500, 0.0000, 0.0000, 0.0000],
            ],
            [
                [1.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.2500, 0.0000, 0.7500],
                [0.5000, 0.2500, 1.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000, 0.7500, 0.0000, 0.0000, 0.0000],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_5_5.copy()
        kwargs = {
            'wave': (2, 2),
            'amp': (2, 2),
            'distaxis': (f.Y, f.X),
            'offset': (0, 0),
        }
        self.run_test(filter, exp, a, **kwargs)


class Rotate90TestCase(FilterTestCase):
    def test_filter(self):
        """Given image data and a direction, rotate the image data
        90° in that direction.
        """
        filter = f.filter_rotate_90
        exp = np.array([
            [
                [0.8000, 0.6000, 0.4000, 0.2000, 0.0000],
                [0.9000, 0.7000, 0.5000, 0.3000, 0.1000],
                [1.0000, 0.8000, 0.6000, 0.4000, 0.2000],
                [0.7000, 0.9000, 0.7000, 0.5000, 0.3000],
                [0.8000, 1.0000, 0.8000, 0.6000, 0.4000],
            ],
        ], dtype=np.float32)
        a = E.copy()
        self.run_test(filter, exp, a)

    def test_filter_counter_clockwise(self):
        """Given image data and a direction, rotate the image data
        90° in that direction.
        """
        filter = f.filter_rotate_90
        exp = np.array([
            [
                [0.4000, 0.6000, 0.8000, 1.0000, 0.8000],
                [0.3000, 0.5000, 0.7000, 0.9000, 0.7000],
                [0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.1000, 0.3000, 0.5000, 0.7000, 0.9000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000],
            ],
        ], dtype=np.float32)
        a = E.copy()
        kwargs = {
            'direction': 'ccw',
        }

        self.run_test(filter, exp, a, **kwargs)


class SkewTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data and a slope, skew the image data by an
        amount equal to the slope.
        """
        filter = f.filter_skew
        exp = np.array([
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [1.0000, 0.7500, 0.2500, 0.5000, 0.7500],
            [0.7500, 1.0000, 0.7500, 0.5000, 0.5000],
            [0.2500, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.5000, 0.2500, 0.0000, 1.0000, 0.7500],
        ], dtype=np.float32)
        kwargs = {
            'slope': 2.0,
        }
        self.run_test(filter, exp, **kwargs)

    def test_filter_video(self):
        """Video data should be processed one frame at a time."""
        filter = f.filter_skew
        exp = np.array([
            [
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
                [1.0000, 0.7500, 0.2500, 0.5000, 0.7500],
                [0.7500, 1.0000, 0.7500, 0.5000, 0.5000],
                [0.2500, 0.7500, 1.0000, 0.7500, 0.5000],
                [0.5000, 0.2500, 0.0000, 1.0000, 0.7500],
            ],
            [
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
                [0.5000, 0.2500, 0.7500, 1.0000, 0.7500],
                [0.7500, 1.0000, 0.7500, 0.5000, 0.5000],
                [0.7500, 0.2500, 0.5000, 0.7500, 1.0000],
                [0.5000, 0.7500, 1.0000, 0.0000, 0.2500],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_5_5.copy()
        kwargs = {
            'slope': 2.0,
        }
        self.run_test(filter, exp, a, **kwargs)


class TwirlTestCase(FilterTestCase):
    def test_filter(self):
        """Given image data, a radius, a strength, and an offset,
        perform a twirl distortion on the data.
        """
        filter = f.filter_twirl
        exp = np.array([
            [0.0019, 0.2537, 0.5047, 0.7547, 0.9963],
            [0.2491, 0.5001, 0.7565, 0.9871, 0.7499],
            [0.4969, 0.7438, 0.9785, 0.7275, 0.4935],
            [0.7468, 0.9873, 0.7715, 0.5010, 0.2438],
            [0.9963, 0.7586, 0.5129, 0.2627, 0.0088],
        ], dtype=np.float32)
        kwargs = {
            'radius': 5.0,
            'strength': 0.25,
        }
        self.run_test(filter, exp, **kwargs)

    def test_filter_video(self):
        """The filter should operate on each frame of a video."""
        filter = f.filter_twirl
        exp = np.array([
            [
                [0.0019, 0.2537, 0.5047, 0.7547, 0.9963],
                [0.2491, 0.5001, 0.7565, 0.9871, 0.7499],
                [0.4969, 0.7438, 0.9785, 0.7275, 0.4935],
                [0.7468, 0.9873, 0.7715, 0.5010, 0.2438],
                [0.9963, 0.7586, 0.5129, 0.2627, 0.0088],
            ],
            [
                [0.9981, 0.7491, 0.4968, 0.2469, 0.0037],
                [0.7537, 0.9912, 0.7373, 0.4938, 0.2588],
                [0.5047, 0.7626, 0.9775, 0.7510, 0.5127],
                [0.2547, 0.5065, 0.7510, 0.9775, 0.7626],
                [0.0037, 0.2501, 0.4938, 0.7435, 0.9914],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_5_5.copy()
        kwargs = {
            'radius': 5.0,
            'strength': 0.25,
        }
        self.run_test(filter, exp, a, **kwargs)

    def test_filter_with_offset(self):
        """Given an offset, the center of the effect should be moved
        by the amount of the offset.
        """
        filter = f.filter_twirl
        exp = np.array([
            [
                [0.0005, 0.2515, 0.5047, 0.7626, 0.9785],
                [0.2496, 0.4985, 0.7453, 0.9873, 0.7715],
                [0.4998, 0.7487, 0.9963, 0.7586, 0.5129],
                [0.7499, 0.9992, 0.7519, 0.5037, 0.2547],
                [0.9999, 0.7503, 0.5008, 0.2513, 0.0015],
            ],
            [
                [0.9995, 0.7511, 0.5031, 0.2562, 0.0225],
                [0.7505, 0.9985, 0.7468, 0.4935, 0.2490],
                [0.5004, 0.7505, 0.9963, 0.7499, 0.5062],
                [0.2503, 0.5001, 0.7500, 0.9963, 0.7531],
                [0.0001, 0.2500, 0.4999, 0.7495, 0.9985],
            ],
        ], dtype=np.float32)
        a = VIDEO_2_5_5.copy()
        kwargs = {
            'radius': 5.0,
            'strength': 0.25,
            'offset': (-2, 2)
        }
        self.run_test(filter, exp, a, **kwargs)
