import cv2
import numpy as np
from matplotlib import pyplot as plt

# https://www.zhihu.com/question/20511799

# 人们把照片的亮度分为0到255共256个数值，数值越大，代表的亮度越高。
# 其中0代表纯黑色的最暗区域，255表示最亮的纯白色，而中间的数字就是不同亮度的灰色。
# 人们还进一步把这些亮度分为了5个区域，分别是黑色，阴影，中间调，高光和白色。

# 当我们用横轴代表0-255的亮度数值。竖轴代表照片中对应亮度的像素数量，这个函数图像就被称为直方图。
# 直方图中柱子的高度，代表了画面中有多少像素是那个亮度，其实就可以看出来画面中亮度的分布和比例。
# 比如下面一个直方图，波峰是在中间偏左的位置（阴影区域），说明画面中有很多深灰或者深色部分。
# 上面的这个直方图，准确来说应该叫RGB直方图，因为他是由红、绿、蓝三个通道的直方图叠加后除以3而成的。


# 我们拍摄照片的时候，相机会通过快门曝光，把现场环境的实际亮度映射到了0到255的照片记录区间上。
# 一张理想的曝光应该如下图，直方图堆积在中部，最左侧和最右侧都没有被切断（切断或者溢出，指的是直方图左右两个边缘，有很高的柱子堆积，可以参考再后面两张过曝/欠曝的直方图）。
# 一旦我们曝光参数设置不对，照片就会欠曝或者过曝。体现在直方图上，就是一侧边缘有大量像素堆积，看起来像被切断了一样。
# 一张欠曝的图片，最0值的纯黑区域也有大量的像素存在，直方图的左侧切断。需要我们在拍摄时增加曝光量。
# 如果直方图左侧和右侧都被切断，同时出现了上面说的过曝和欠曝现象，（或者一侧接触边缘，另一侧切断）则意味着环境里的亮度差别太大，相机已经难以记录下全部信息了。


# 直方图是根据灰度 图像绘制的，而不是彩色图像）。直方图的左边区域像是了暗一点的像素数量，右侧显示了亮一点的像素的数量。


# 你只需要把原来的 256 个值等分成 16 小组，取每组的
# 总和。而这里的每一个小组就被成为 BIN。第一个例子中有 256 个 BIN，第
# 二个例子中有 16 个 BIN。在 OpenCV 的文档中用 histSize 表示 BINS


img = cv2.imread('../data/home.jpg', cv2.IMREAD_GRAYSCALE)
color = ('b', 'g', 'r')

# channels: 如果输入图像是灰度图，它的值就是 [0]；如果是彩色图像的话，传入的参数可以是 [0]， [1]， [2] 它们分别对应着通道 B， G， R。
# cv2:calcHist(images; channels; mask; histSize; ranges[; hist[; accumulate]])
hist = cv2.calcHist(img, [0], None, [255], [0, 256])
# hist 是一个 256x1 的数组，每一个值代表了与次灰度值对应的像素点数目。

# img.ravel() 将图像转成一维数组，这里没有中括号。
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
print(len(img.ravel()))

# 对一个列表或数组既要遍历索引又要遍历元素时
# 使用内置 enumerrate 函数会有更加直接，优美的做法
# enumerate 会将数组或列表组成一个索引序列。
# 使我们再获取索引和索引内容的时候更加方便
img = cv2.imread('../data/messi5.jpg')
for i, col in enumerate(color):
    histr = cv2.calcHist(img, [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

img = cv2.imread('../data/home.jpg', 0)
# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(img, img, mask=mask)
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()
