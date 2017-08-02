import numpy as np
import cv2

# 对于 BGR$Gray 的转换，我们要使用的 flag 就是 cv2.COLOR_BGR2GRAY。
# 同样对于 BGR$HSV 的转换，我们用的 flag 就是 cv2.COLOR_BGR2HSV。

# 在 OpenCV 的 HSV 格式中， H（色彩/色度）的取值范围是 [0， 179]，
# S（饱和度）的取值范围 [0， 255]， V（亮度）的取值范围 [0， 255]。但是不
# 同的软件使用的值可能不同。所以当你需要拿 OpenCV 的 HSV 值与其他软
# 件的 HSV 值进行对比时，一定要记得归一化。
flags = [f for f in dir(cv2) if f.startswith('COLOR_')]
print(flags)

# 在 HSV 颜色空间中要比在 BGR 空间中更容易表示一个特定颜色。

cap = cv2.VideoCapture(0)

while True:
    # get each frame
    ret, frame = cap.read()

    # to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # set blue threshold
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_green = np.array([50, 50, 50])
    upper_green = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()


# 这里的三层括号应该分别对应于 cvArray，cvMat，IplImage
green = np.uint8([[[0, 255, 0]]])

hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
print(hsv_green)
# 现在你可以分别用 [H-100，100，100] 和 [H+100，255，255] 做上 下阈值。
# or [H-10, H+10]?
blue = np.uint8([[[255, 0, 0]]])
hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
print(hsv_blue)
