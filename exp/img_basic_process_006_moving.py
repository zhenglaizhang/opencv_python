import numpy as np
import cv2

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

while True:
    cv2.imshow('img', img)
    # cv2.imshow('res', res)
    cv2.imshow('dst', dst)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
