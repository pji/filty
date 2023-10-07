####################
imgfilt Requirements
####################

The purpose of this document is to detail the requirements for
imgfilt, a Python module procedurally create image data. This is an
initial take for the purposes of planning. There may be additional
requirements or non-required features added in the future that are
not covered in this document.


*******
Purpose
*******
The purpose of imgfilt is to perform algorithmic changes to a set of
image data. It can be used on either still image data or video.


***********************
Functional Requirements
***********************
The following are the functional requirements for imgfilt:

*   imgfilt can alter image data.


**********************
Technical Requirements
**********************
The following are the technical requirements for imgblender:

*   imgfilt accepts "image data", which are three-dimensional array-
    like objects of floating-point numbers from 0 to 1, inclusive.
*   imgfilt outputs image data in array-like objects that can be
    saved by the imgwriter package.
