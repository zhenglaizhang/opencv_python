import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../data/messi5.jpg')

moving_matrix = np.uint8([[1, 0, 100], [0, 1, 50]])
rows, cols = img.shape[:2]
print(img.shape)
print(rows, cols)
# res = cv2.warpAffine(img, moving_matrix, (rows, cols))

# 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
# 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1.0)
# 第三个参数是输出图像的尺寸中心
dst = cv2.warpAffine(img, M, (cols, rows))

# 在仿射变换中，原图中所有的平行线在结果图像中同样平行。
# 为了创建这个矩阵我们需要从原图像中找到三个点以及他们在输出图像中的位置。
rows, cols, channels = img.shape
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
dst2 = cv2.warpAffine(img, M, (cols, rows))

plt.subplot(221)
plt.title('Affine Input')
plt.imshow(img)
plt.subplot(222)
plt.title('Affine Output')
plt.imshow(dst2)

# 对于视角变换，我们需要一个 3x3 变换矩阵。在变换前后直线还是直线。
# 要构建这个变换矩阵，你需要在输入图像上找 4 个点，以及他们在输出图
# 像上对应的位置。这四个点中的任意三个都不能共线。
img = cv2.imread('../data/sudoku.png')
rows, cols, chs = img.shape
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (300, 300))
plt.subplot(223)
plt.title('Perspective Input')
plt.imshow(img)
plt.subplot(224)
plt.title('Perspective Output')
plt.imshow(dst)
plt.show()

while True:
    cv2.imshow('img', img)
    # cv2.imshow('res', res)
    cv2.imshow('dst', dst)
    cv2.imshow('dst2', dst2)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
