import numpy as np
import cv2

img = cv2.imread('../data/messi5.jpg')

# set scale factor
res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

print(img.shape)
print(img.shape[:2])
height, width = img.shape[:2]

# set targeting size
res = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_LINEAR)
print(res.shape)

while True:
    cv2.imshow('img', img)
    cv2.imshow('res', res)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
