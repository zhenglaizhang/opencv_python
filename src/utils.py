import cv2
import numpy
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
