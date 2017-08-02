# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


# by default, optimization is turned on at compiler time
print(cv2.useOptimized())
print(cv2.useOpenVX())

# not enabled at compiler time
# print(cv2.setUseOpenVX(True))

print(cv2.setUseOptimized(True))


img1 = cv2.imread('../data/roi.jpg')
start = cv2.getTickCount()
for i in range(5, 49, 2):
    img1 = cv2.medianBlur(img1, i)
end = cv2.getTickCount()
t = (end - start) / cv2.getTickFrequency()
print(t, 's')

# 图像加法

# 两幅图像的大小，类型必须一致，或者第二个图像可以使一个简单的标量值
# OpenCV 中的加法与 Numpy 的加法是有所不同的。 OpenCV 的加法是一种饱和操作，而 Numpy 的加法是一种模操作

x = np.uint8([250])
y = np.uint8([10])

# 这种差别在你对两幅图像进行加法时会更加明显。 OpenCV 的结果会更好一点。所以我们尽量使用 OpenCV 中的函数
print(cv2.add(x, y))  # 260 => 255
print(x + y)  # 260 % 256 = 4

# 图像混合
# 这其实也是加法，但是不同的是两幅图像的权重不同，这就会给人一种混合或者透明的感觉。图像混合的计算公式如下：
# g (x) = (1 − α) f0 (x) + αf1 (x)
# 通过修改 α 的值（0 ! 1），可以实现非常酷的混合。
img1 = cv2.imread('../data/ml.png')[:380, :308]
print(img1.shape)
img2 = cv2.imread('../data/opencv-logo.png')[:380, :308, :]
print(img2.shape)

# dst = α · img1 + β · img2 + γ
dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
# dst2 = cv2.addWeighted(img1, 0.7, img2, 0.3, 50)
cv2.imshow('dst', dst)
# cv2.imshow('dst with gamma', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()


#  OpenCV 的函数要比 Numpy 函数快。所以对于相同的操 作最好使用 OpenCV 的函数。
# 当然也有例外，尤其是当使用 Numpy 对视图而非复制）进行操作时。
