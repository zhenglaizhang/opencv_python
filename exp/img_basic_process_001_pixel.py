import cv2
import numpy as np

img = cv2.imread('../images/stature_small.jpg')

px = img[100, 100]
print(px)
blue = img[100, 100, 0]
print(blue)

img[100, 100] = [255, 255, 255]
print(img[100, 100])
print(img[403, 699])

# better way to get/set pixel value
print(img.item(100, 100, 0))
img.itemset((100, 100, 0), 101)
print(img.item(100, 100, 0))

# 如果图像是灰度图，返回值仅有行数和列数。所以通过检查这个返回值就可以知道加载的是灰度图还是彩色图。
height, width, channels = img.shape
print(img.shape)
print(img.dtype)
print(img.size)

# ROI
ball = img[780:850, 40:200]
img[180:250, 100:260] = ball

# 这是你就需要把 BGR 拆分成单个通道。有时你需要把独立通道的图片合并成一个 BGR 图像
# cv2.split() 是一个比较耗时的操作。只有真正需要时才用它，能用 Numpy 索引就尽量用。
b, g, r = cv2.split(img)
img = cv2.merge(b, g, r)
b = img[:, :, 0]
img[:, :, 2] = 0  # set red channel value all as 0

# cv2.imshow('processed', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
