'''
  -*- coding: utf-8 -*-
  @Time    : 2020/10/29 10:58
  @Author  : liuxu
  @File    : tophat.py
  @Software: PyCharm
  @Theme   : 
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# filepath = "./light.jpg"
filepath = "./closed_hole.jpg"
# 设置图像的宽和高
IMAGE_WIDTH = 650
IMAGE_HEIGHT = 650

# 显示图像
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

# 对图像resize处理
def row_image_resize(image, width, height):
    # image输入图像，width，height 期望得到的宽和高
    image = cv2.resize(image, (width, height))
    # print("resized image:", image.shape)
    return image


# sobel边缘检测
def sobel_edge(img):

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = img
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad


# Getting the kernel to be used in Top-Hat
filterSize = (3,3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                   filterSize)

# Reading the image named 'input.jpg'
input_image = cv2.imread(filepath)
input_image = row_image_resize(input_image,IMAGE_HEIGHT,IMAGE_WIDTH)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Applying the Top-Hat operation
tophat_img = cv2.morphologyEx(input_image,
                              cv2.MORPH_TOPHAT,
                              kernel)

open_img = cv2.morphologyEx(input_image,
                            cv2.MORPH_OPEN,
                            kernel)

close_img = cv2.morphologyEx(input_image,
                             cv2.MORPH_CLOSE,
                             kernel)

black_img = cv2.morphologyEx(input_image,
                             cv2.MORPH_BLACKHAT,
                             kernel)


grad_img = cv2.morphologyEx(input_image,
                            cv2.MORPH_GRADIENT,
                            kernel)

erode_img = cv2.morphologyEx(input_image,
                             cv2.MORPH_ERODE,
                             kernel)
dilate_img = cv2.morphologyEx(input_image,
                              cv2.MORPH_DILATE,
                              kernel)
external_grad = cv2.subtract(dilate_img,input_image)

inner_grad = cv2.subtract(input_image,erode_img)

sobel_img = sobel_edge(input_image)


cv_show("original", input_image)
cv_show("tophat", tophat_img)
cv_show("openimg",open_img)
cv_show("blackimg",black_img)
cv_show("gradimg",grad_img)
cv_show("erodeimg",erode_img)
cv_show("dilateimg",dilate_img)
cv_show("innergard",inner_grad)
cv_show("externalgrad",external_grad)
cv_show("sobelimg",sobel_img)





