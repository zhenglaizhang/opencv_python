import cv2
import numpy as np

# a very handy function called Canny (after the algorithm’s inventor, John F. Canny)
# The Canny edge detection algorithm is quite complex but also interesting: it’s a five-step
# process that denoises the image with a Gaussian filter, calculates gradients, applies non
# maximum suppression (NMS) on edges, a double threshold on all the detected edges to
# eliminate false positives, and, lastly, analyzes all the edges and their connection to each
# other to keep the real edges and discard the weak ones.
img = cv2.imread("../images/stature_small.jpg")
cv2.imwrite("canny.jpg", cv2.Canny(img, 200, 300))
cv2.imshow("canny", cv2.imread("canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()
