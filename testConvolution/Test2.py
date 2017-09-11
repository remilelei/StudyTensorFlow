# -*- coding: utf-8 -*-

import cv2
import numpy as np

# 读入原图
image = cv2.imread("testConvolution/test.jpg")
cv2.imshow('original', image)

# 边缘检测
kernal = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])
edges = cv2.filter2D(image, -1, kernal)
cv2.imwrite('temp/edges.jpg', edges)