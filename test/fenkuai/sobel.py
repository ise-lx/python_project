'''
  -*- coding: utf-8 -*-
  @Time    : 2020/10/21 20:50
  @Author  : liuxu
  @File    : sobel.py
  @Software: PyCharm
  @Theme   : 
'''

import sys
import cv2 as cv
import cv2 as cv2

import numpy as np

# file4 = "./1.jpg"
# file4 = "./test8.jpg"
# file4 = "./light_1.JPG"
# file4 = "./light_test12.jpg"
file4 = "./bianxing_test.jpg"


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
    image = cv.resize(image, (width, height))
    # print("resized image:", image.shape)
    return image



# #图像预处理函数,使用大津法
def pre_process_edge(rowimage):
    # 显示原图
    # plt_show_one_pic("original image", rowimage)
    # all_images.append(("orignial image",rowimage))      # 方便以后显示
    grayimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2GRAY)  # 灰度化
    # all_images.append(("gray image",grayimage))

    # plt_show_one_pic("gray image", grayimage)
    resizedimage = row_image_resize(grayimage, IMAGE_WIDTH, IMAGE_HEIGHT)  # resize

    # 运用大津算法进行图像分割
    ret, im_th = cv2.threshold(resizedimage, 0, 255, cv2.THRESH_OTSU)
    # all_images.append(("im_th",im_th))
    # plt_show_one_pic("OTSU", im_th)
    # plt.show()
    # #中值滤波去除椒盐噪声,卷积核为5x5时
    # blur_image = cv2.medianBlur(im_th, 3)
    # 显示中值滤波处理后图像
    # cv_show("image after medianBlur", blur_image)

    return im_th










# #图像预处理函数，使用边缘检测
def pre_process(original_img):
    cv.imshow("original img", original_img)
    cv.waitKey(0)

    # 边缘检测
    grad = sobel_edge(original_img)

    cv.imshow("window_name", grad)
    cv.waitKey(0)

    # 运用大津算法进行图像分割
    ret, im_th = cv2.threshold(grad, 0, 255, cv2.THRESH_OTSU)

    cv.imshow("im_th", im_th)
    cv.waitKey(0)
    return im_th



def extract_contour(img):
    # 闭运算填补细小裂缝

    # # 定义矩形结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv.imshow("image after close operation", img)
    cv.waitKey(0)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("contours type: ",type(contours))
    # print("contours: ",contours)
    # print("contours length: ",len(contours))

    # print("轮廓的类型：",type(contours))     # type is list
    areas = []  # 轮廓所围成的面积构成的数组
    for idx in range(len(contours)):
        areas.append(cv2.contourArea(contours[idx]))

    largest_idx = np.argmax(np.array(areas))  # 最大面积的下标



    black_img = np.ones_like(img)  # 创建一个同大小的(0,0,0)黑色图像
    black_img = black_img*255
    black_img_rgb = cv2.cvtColor(black_img, cv2.COLOR_GRAY2BGR)


    # 为显示彩色,把灰度图像转换为RGB三通道图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # draw_con_img = cv2.drawContours(img_rgb, contours, largest_idx, (255, 0, 0),thickness=2)
    draw_con_img = cv2.drawContours(black_img_rgb, contours, largest_idx, (0, 0,0),thickness=1)
    cv.imshow("draw con img",draw_con_img)
    cv.waitKey(0)


    original_img = file4
    src = cv.imread(original_img)
    src = row_image_resize(src, 650, 650)

    draw_con_img = cv2.drawContours(src, contours, largest_idx, (255, 0, 0),thickness=1)
    cv.imshow("draw con src img",draw_con_img)
    cv.waitKey(0)


    return contours,largest_idx


def sobel_edge(img):

    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad


def main():
    original_img = file4
    src = cv.imread(original_img)

    # #中值滤波去除椒盐噪声,卷积核为3x3时
    # src = cv2.medianBlur(src, 3)
    # 显示中值滤波处理后图像
    # cv_show("image after medianBlur", blur_image)

    src = row_image_resize(src, 650, 650)
    im_th = pre_process(src)        # 边缘检测方法
    im_th_otsu = pre_process_edge(src) # 大津法处理

    # im_th = cv2.medianBlur(im_th, 3)
    # im_th_otsu = cv2.medianBlur(im_th_otsu,3)


    cv_show("im_th",im_th)
    cv_show("im_th_otsu",im_th_otsu)
    contours,largest_idx = extract_contour(im_th)




    return 0


if __name__ == "__main__":
    main()






    #
    # # 最小外接矩形
    # rect = cv2.minAreaRect(contours[largest_idx])  # 返回  中心点坐标（x,y) width:rect[1][0] height: rect[1][1] 旋转角度θ
    # # print("rect type:", type(rect))
    # # print(rect)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)


    # # 画出样品的轮廓
    # img = src.copy()
    # # 为显示彩色,把灰度图像转换为RGB三通道图像
    # # img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #
    # img_with_rect = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    #
    # cv_show("image with rect", img_with_rect)



    #
    # # 为显示彩色,把灰度图像转换为RGB三通道图像
    # # img_rgb = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    #
    # draw_con_img = cv2.drawContours(src, contours, largest_idx, (255, 0, 0),cv2.FILLED)
    #
    # cv.imshow("draw con img",draw_con_img)
    # cv.waitKey(0)
    #
    # # 图像取差集 imagesub = imagesub1 - imagesub2
    # imgsub1 = draw_con_img.copy()
    # imgsub2 = src.copy()
    # # blur_image为单通道图像，需要转换为RGB图像
    # # imgsub2 = cv2.cvtColor(imgsub2, cv2.COLOR_GRAY2BGR)
    #
    # # 取差集
    # imgsub = cv2.subtract(imgsub1, imgsub2)
    #
    # cv.imshow("imgsub",imgsub)
    # cv.waitKey(0)
