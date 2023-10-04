.. _api:

##########
Public API
##########

The following are the public functions that make up the public API
of :mod:`imgfilt`.


Filter Functions
================
The following is true of all filter functions:

*   They take a :class:`numpy.ndarray` of image data as the first parameter.
*   They may require other parameters.
*   They return a :class:`numpy.ndarray` of image data.

All filter functions are registered in the :class:`dict` `imgeaser.filters`
for convenience, but they can also be called directly.

.. autofunction:: imgfilt.filter_box_blur
.. autofunction:: imgfilt.filter_colorize
.. autofunction:: imgfilt.filter_contrast
.. autofunction:: imgfilt.filter_flip
.. autofunction:: imgfilt.filter_gaussian_blur
.. autofunction:: imgfilt.filter_glow
