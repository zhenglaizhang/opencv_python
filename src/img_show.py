import numpy as np
import cv2

img = cv2.imread('../random_bgr.png')
cv2.imshow('random bgr', img)
cv2.waitKey()
cv2.destroyAllWindows()
