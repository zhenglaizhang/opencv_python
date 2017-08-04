# 有两类图像金字塔：高斯金字塔和拉普拉斯金字塔。
# 高斯金字塔的顶部是通过将底部图像中的连续的行和列去除得到的。顶
# 部图像中的每个像素值等于下一层图像中 5 个像素的高斯加权平均值。这样
# 操作一次一个 MxN 的图像就变成了一个 M/2xN/2 的图像。所以这幅图像
# 的面积就变为原来图像面积的四分之一。这被称为 Octave。连续进行这样
# 的操作我们就会得到一个分辨率不断下降的图像金字塔。

import cv2
import numpy as np

img = cv2.imread('../data/messi5.jpg')
lower_reso = cv2.pyrDown(img)
higher_reso2 = cv2.pyrUp(lower_reso)

cv2.imshow('img', img)
# cv2.imshow('lower', lower_reso)
# cv2.imshow('higher', higher_reso2)
cv2.waitKey()
cv2.destroyAllWindows()

# 因为一旦使用 cv2.pyrDown()，图像的分辨率就会降低，信息就会被丢失。

# 图像金字塔的一个应用是图像融合。例如，在图像缝合中，你需要将两幅
# 图叠在一起，但是由于连接区域图像像素的不连续性，整幅图的效果看起来会
# 很差。这时图像金字塔就可以排上用场了，他可以帮你实现无缝连接

A = cv2.imread('../data/apple.jpg')
B = cv2.imread('../data/orange.jpg')

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# Li = Gi − PyrUp (Gi+1)
# 拉普拉金字塔的图像看起来就像边界图，其中很多像素都是 0。他们经常
# 被用在图像压缩中。

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5, 0, -1):
    print(i)
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i - 1], GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5, 0, -1):
    print(i)
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i - 1], GE)
    lpB.append(L)

# Now add left and right halves of images in each level
# numpy.hstack(tup)
# Take a sequence of arrays and stack them horizontally
# to make a single array.
