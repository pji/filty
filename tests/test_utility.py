"""
test_utility
~~~~~~~~~~~~

Unit tests for the imgfilt.utility module.
"""
import numpy as np

from imgfilt import utility as u


# Test cases.
def test_get_prefixed_functions():
    """When given a prefix and an object, :func:`get_prefixed_functions`
    should return the dict of functions within that object's namespace
    that start with the prefix.
    """
    from tests import spam
    assert u.get_prefixed_functions('spam_', spam) == {
        'eggs': spam.spam_eggs,
        'bacon': spam.spam_bacon,
        'baked_beans': spam.spam_baked_beans,
    }


def test_will_square():
    """Given an array with the X axis having a different size
    than the Y axis, :func:`will_square` should make the size
    of those axes the same before passing the array to the
    function. Then slice the result to be the original size of
    the given array.
    """
    @u.will_square
    def spam(a):
        return np.rot90(a, 1, (0, 1))

    assert (spam(np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0,],
        [1.0, 1.0, 1.0, 1.0, 1.0,],
        [1.0, 1.0, 1.0, 1.0, 1.0,],
    ], dtype=float)) == np.array([
        [0.0, 1.0, 1.0, 1.0, 0.0,],
        [0.0, 1.0, 1.0, 1.0, 0.0,],
        [0.0, 1.0, 1.0, 1.0, 0.0,],
    ], dtype=float)).all()
