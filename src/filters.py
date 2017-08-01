import cv2
import numpy as np
import scipy.interpolate


def stroke_edges(src, dst, blur_ksize=7, edge_size=5):
    if blur_ksize >= 3:
        # medianBlur() is expensive with a large ksize, such as 7.
        # blur to remove noise, espically in color images
        blurred_src = cv2.medianBlur(src, blur_ksize)
        gray_src = cv2.cvtColor(blurred_src, cv2.COLOR_BGR2GRAY)
    else:
        # to turn off blur in case performance issue
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray_src, cv2.CV_8U, gray_src, ksize=edge_size)
    normalized_inverse_alpha = (1.0 / 255) * (255 - gray_src)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalized_inverse_alpha
    cv2.merge(channels, dst)


class VConvolutionFilter(object):
    """A filter which applies a covolution to V (or all of BGR)"""

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, ddepth=-1, kernel=self._kernel, dst=dst)


class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with 1-pixel radius"""

    # the weights sum up to 1. This should be the case whenever we want to leave the
    # image’s overall brightness unchanged.
    def __init__(self):
        kernel = np.array(
            [[-1, -1, -1],
             [-1, 9, -1],
             [-1, -1, -1]]
        )
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    # If we modify a sharpening kernel slightly so that its weights sum up to 0 instead,
    # we have an edge detection kernel that turns edges white and non-edges black.
    def __init__(self):
        kernel = np.array(
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]]
        )
        VConvolutionFilter.__init__(self, kernel)


# for a blur effect, the weights should sum up to 1 and should be positive throughout the neighborhood.
class BlurFilter(VConvolutionFilter):
    """A blur filter with 2-pixel radius"""

    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


# Sometimes, though, kernels with less symmetry produce an interesting effect. Let’s
# consider a kernel that blurs on one side (with positive weights) and sharpens on the other
# (with negative weights).

class EmbossFilter(VConvolutionFilter):
    """ An emboss filter with a 1-pixel radius"""

    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)


# todo: fix it
class BGRPortraCurveFilter(EmbossFilter):
    pass
