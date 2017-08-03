import numpy as np
import cv2

# 形态学操作是根据图像形状进行的简单操作。一般情况下对二值化图像进
# 行的操作。需要输入两个参数，一个是原始图像，第二个被称为结构化元素或
# 核，它是用来决定操作的性质的。两个基本的形态学操作是腐蚀和膨胀。他们
# 的变体构成了开运算，闭运算，梯度等。


# 腐蚀
# 就像土壤侵蚀一样，这个操作会把前景物体的边界腐蚀掉（但是前景仍然
# 是白色）。这是怎么做到的呢？卷积核沿着图像滑动，如果与卷积核对应的原图
# 像的所有像素值都是1，那么中心元素就保持原来的像素值，否则就变为零。
# 这回产生什么影响呢？根据卷积核的大小靠近前景的所有像素都会被腐蚀
# 掉（变为 0），所以前景物体会变小，整幅图像的白色区域会减少。这对于去除
# 白噪声很有用，也可以用来断开两个连在一块的物体等。
img = cv2.imread('../images/j.png', 0)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
cv2.imshow('erosion', erosion)
cv2.waitKey()
cv2.destroyAllWindows()

# 与腐蚀相反，与卷积核对应的原图像的像素值中只要有一个是 1，中心元
# 素的像素值就是 1。所以这个操作会增加图像中的白色区域（前景）。一般在去
# 噪声时先用腐蚀再用膨胀。因为腐蚀在去掉白噪声的同时，也会使前景对象变
# 小。所以我们再对他进行膨胀。这时噪声已经被去除了，不会再回来了，但是
# 前景还在并会增加。膨胀也可以用来连接两个分开的物体。
dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('dilation', dilation)
cv2.waitKey()
cv2.destroyAllWindows()

# 先进性腐蚀再进行膨胀就叫做开运算。就像我们上面介绍的那样，它被用
# 来去除噪声。这里我们用到的函数是 cv2.morphologyEx()。

img = cv2.imread('../images/j_bg_dotted.png', 0)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('dilation', opening)
cv2.waitKey()
cv2.destroyAllWindows()

# 先膨胀再腐蚀。它经常被用来填充前景物体中的小洞，或者前景物体上的小黑点。
img = cv2.imread('../images/j_fg_dotted.png', 0)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)
cv2.waitKey()
cv2.destroyAllWindows()


# 形态学梯度
# 其实就是一幅图像膨胀与腐蚀的差别。
# 结果看上去就像前景物体的轮廓。
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient', gradient)
cv2.waitKey()
cv2.destroyAllWindows()


# 原始图像与进行开运算之后得到的图像的差。下面的例子是用一个 9x9 的
# 核进行礼帽操作的结果。
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# 黑帽
# 进行闭运算之后得到的图像与原始图像的差。
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
