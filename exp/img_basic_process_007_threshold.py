import numpy as np
from matplotlib import pyplot as plt
import cv2

# 简单阀值
# 与名字一样，这种方法非常简单。但像素值高于阈值时，我们给这个像素
# 赋予一个新值（可能是白色），否则我们给它赋予另外一种颜色（也许是黑色）。
# 这个函数就是 cv2.threshhold()。这个函数的第一个参数就是原图像，原图
# 像应该是灰度图。第二个参数就是用来对像素值进行分类的阈值。
img = cv2.imread('../data/gradient.png', 0)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i], 'gray')
    plt.xticks([]), plt.yticks([])
plt.show()

# 自适应阈值
# 尤其是当同一幅图像上的不同部分的具有不
# 同亮度时。这种情况下我们需要采用自适应阈值。此时的阈值是根据图像上的
# 每一个小区域计算与其对应的阈值。因此在同一幅图像上的不同区域采用的是
# 不同的阈值，从而使我们能在亮度不同的情况下得到更好的结果。
# 种方法需要我们指定三个参数，返回值只有一个。
# • Adaptive Method- 指定计算阈值的方法。
#   – cv2.ADPTIVE_THRESH_MEAN_C：阈值取自相邻区域的平均值
#   – cv2.ADPTIVE_THRESH_GAUSSIAN_C：阈值取值相邻区域的加权和，权重为一个高斯窗口。
# • Block Size - 邻域大小（用来计算阈值的区域大小）。
# • C - 这就是是一个常数，阈值就等于的平均值或者加权平均值减去这个常数。

# 中值滤波
img = cv2.imread('../data/messi5.jpg', 0)
img = cv2.medianBlur(img, 5)
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# 11 为 Block size, 2 为 C 值
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# Otsu 二值化要做的。简单来说就是对
# 一副双峰图像自动根据其直方图计算出一个阈值。（对于非双峰图像，这种方法得到的结果可能会不理想）。
# 该方法不适合直方图中双峰差别很大或双峰间的谷比较宽广而平坦的图像，以及单峰直方图的情况。

img = cv2.imread('../images/noisy.jpg', 0)
# global thresholding
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# Otsu's thresholding
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
# （5,5）为高斯核的大小， 0 为标准差
blur = cv2.GaussianBlur(img, (5, 5), 0)
# 阈值一定要设为 0！
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
# 这里使用了 pyplot 中画直方图的方法， plt.hist, 要注意的是它的参数是一维数组
# 所以这里使用了（numpy）ravel 方法，将多维数组转换成一维，也可以使用 flatten 方法
# ndarray.flat 1-D iterator over an array.
# ndarray.flatten 1-D array copy of the elements of an array in row-major order.
for i in range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
plt.show()
