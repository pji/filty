"""
test_utility
~~~~~~~~~~~~

Unit tests for the filty.utility module.
"""
import numpy as np

from filty import utility as u
from tests.common import ArrayTestCase


# Test cases.
class WillSquareTestCase(ArrayTestCase):
    def test_will_square(self):
        """Given an array with the X axis having a different size
        than the Y axis, make the size of those axes the same before
        passing the array to the function. Then slice the result to
        be the original size of the given array.
        """
        # Expected value.
        exp = np.array([
            [0.0, 1.0, 1.0, 1.0, 0.0, ],
            [0.0, 1.0, 1.0, 1.0, 0.0, ],
            [0.0, 1.0, 1.0, 1.0, 0.0, ],
        ], dtype=np.float32)

        # Test data and state.
        a = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0, ],
            [1.0, 1.0, 1.0, 1.0, 1.0, ],
            [1.0, 1.0, 1.0, 1.0, 1.0, ],
        ], dtype=np.float32)

        @u.will_square
        def spam(a):
            return np.rot90(a, 1, (0, 1))

        # Run test.
        act = spam(a)

        # Determine test result.
        self.assertArrayEqual(exp, act)
