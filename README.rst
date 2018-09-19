`dodoimages`
############

This package contains a collection of functions for basic image processing. There are functions to

- remove monochromatic borders
- separate foreground using either contours or grabcut (requires OpenCV)
- remove duplicates based on structural similarity
- scale a list of images to the same size (adding black borders to normalize aspect ratio)
- delete corrupt images

Things I'd like to add include:

- rotate an image based on a person's position
- crop image around a person