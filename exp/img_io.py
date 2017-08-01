import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

img = np.zeros((3, 3), dtype=np.uint8)
print(img)
print(img.shape)

print()

bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(bgr_img)
print(bgr_img.shape)
print(bgr_img.size)
print(bgr_img.dtype)

# IMREAD_ANYCOLOR = 4
# IMREAD_ANYDEPTH = 2
# IMREAD_COLOR = 1 读入一副彩色图像。图像的透明度会被忽略， 这是默认参数
# IMREAD_GRAYSCALE = 0 以灰度模式读入图像
# cv2.IMREAD_UNCHANGED：读入一幅图像，并且包括图像的 alpha 通道
image = cv2.imread("../images/stature_small.jpg", cv2.IMREAD_GRAYSCALE)
if image is not None:
    cv2.imshow("gray statue", image)    # cv2.WINDOW_AUTOSIZE
else:
    print("failed to read img")
cv2.waitKey(0)  # wait forever, in mili seconds
cv2.destroyAllWindows()
# cv2.imwrite("myPic2.jpg", image)

# create window first then load the image
cv2.namedWindow("pre-created", cv2.WINDOW_NORMAL)
cv2.imshow('pre-created', image)
k = cv2.waitKey(0)
if k == 27:   # `Esc`
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite("/tmp/saved.jpg", image)
    cv2.destroyAllWindows()

# By default, imread() returns an image in the BGR color format even if the file uses a
# grayscale format. BGR represents the same color space as red-green-blue (RGB), but the
# byte order is reversed


random_byte_array = bytearray(os.urandom(120000))
flat_numpy_array = np.array(random_byte_array)

# 400 x 300 gray img
gray_img = flat_numpy_array.reshape(300, 400)
cv2.imwrite('/tmp/random_gray.png', gray_img)

# 400 x 100 colorful img
bgr_img = flat_numpy_array.reshape(100, 400, 3)
print(bgr_img.shape)
print(bgr_img.size)
print(bgr_img.dtype)

# set all `G` to 0
# indexing
bgr_img[:, :, 1] = 0
cv2.imwrite("/tmp/random_bgr.png", bgr_img)

gray_img2 = np.random.randint(0, 256, 120000).reshape(300, 400)

# show with matplotlib
plt.imshow(gray_img2, cmap="gray", interpolation="bicubic")
plt.xticks([]), plt.yticks([])
plt.show()

# 彩色图像使用 OpenCV 加载时是 BGR 模式。但是 Matplotib 是 RGB
# 模式。所以彩色图像如果已经被 OpenCV 读取，那它将不会被 Matplotib 正
# 确显示。
