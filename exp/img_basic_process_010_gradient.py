
# 梯度简单来说就是求导。
# OpenCV 提供了三种不同的梯度滤波器，或者说高通滤波器： Sobel，Scharr 和 Laplacian。

# Sobel， Scharr 其实就是求一阶或二阶导数。 Scharr 是对 Sobel（使用
# 小的卷积核求解求解梯度角度时）的优化。 Laplacian 是求二阶导数。

# Sobel 算子是高斯平滑与微分操作的结合体，所以它的抗噪声能力很好。
# 你可以设定求导的方向（xorder 或 yorder）。还可以设定使用的卷积核的大
# 小（ksize）。如果 ksize=-1，会使用 3x3 的 Scharr 滤波器，它的的效果要
# 比 3x3 的 Sobel 滤波器好（而且速度相同，所以在使用 3x3 滤波器时应该尽
# 量使用 Scharr 滤波器）。
