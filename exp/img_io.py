import numpy as np
import cv2
import os

img = np.zeros((3, 3), dtype=np.uint8)
print(img)
print(img.shape)

print()

bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(bgr_img)
print(bgr_img.shape)
print(bgr_img.size)
print(bgr_img.dtype)

# image = cv2.imread("myPic.png")
# cv2.imwrite("myPic2.jpg", image)

# By default, imread() returns an image in the BGR color format even if the file uses a
# grayscale format. BGR represents the same color space as red-green-blue (RGB), but the
# byte order is reversed


random_byte_array = bytearray(os.urandom(120000))
flat_numpy_array = np.array(random_byte_array)

# 400 x 300 gray img
gray_img = flat_numpy_array.reshape(300, 400)
cv2.imwrite('random_gray.png', gray_img)

# 400 x 100 colorful img
bgr_img = flat_numpy_array.reshape(100, 400, 3)
print(bgr_img.shape)
print(bgr_img.size)
print(bgr_img.dtype)

# set all `G` to 0
# indexing
bgr_img[:, :, 1] = 0
cv2.imwrite("random_bgr.png", bgr_img)



gray_img2 = np.random.randint(0, 256, 120000).reshape(300, 400)

