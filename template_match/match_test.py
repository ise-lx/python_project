# -*- coding: utf-8 -*-
'''
   @Time    : 2020/10/1 8:42
   @Author  : liuxu
   @File    : match_test.py
   @Software: PyCharm
   @Content: 判断过滤网的形状；(圆形，长方形，正方形);模板匹配;占空比；细长度；
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

template_circle_path = "./images/template_circle.JPG"
template_rect_path = "./images/template_rect.JPG"
template_square_path = "./images/template_square.jpg"
test_circle_path = "./images/test_circle.JPG"
dirc_images_path = "./images"

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


# 读入一张图像,返回
def read_an_gray_image(filepath):
    return cv2.imread(filepath)


# 读入文件夹下的所有图像,返回一个路径的列表,(相对路径)
def read_images(dictoryname):
    dicnames = []
    for imagename in os.listdir(dictoryname):
        path = dictoryname + "/" + imagename
        image = cv2.imread(path)
        dicnames.append(image)

    # return dicnames
    return dicnames


# #图像预处理函数
def pre_process(rowimage):
    # 显示原图
    # cv_show("row image", rowimage)

    grayimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2GRAY)  # 灰度化
    resizedimage = row_image_resize(grayimage, IMAGE_WIDTH, IMAGE_HEIGHT)  # resize

    # 运用大津算法进行图像分割
    ret, im_th = cv2.threshold(resizedimage, 0, 255, cv2.THRESH_OTSU)
    # cv_show("image after OTSU", im_th)

    # #中值滤波去除椒盐噪声,卷积核为5x5
    blur_image = cv2.medianBlur(im_th, 3)
    # #显示中值滤波处理后图像
    # cv_show("image after medianBlur", blur_image)

    return blur_image


def find_Contours(binary_img):
    # 在分割后的图像找轮廓
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []  # 轮廓所围成的面积构成的数组
    for idx in range(len(contours)):
        areas.append(cv2.contourArea(contours[idx]))

    largest_idx = np.argmax(np.array(areas))  # 最大面积的下标
    largest_area = cv2.contourArea(contours[largest_idx])  # 最大面积的大小

    # print("最大面积为：", largest_area)
    # print("最大面积的下标为：", largest_idx)
    # 画出样品的轮廓
    img = binary_img.copy()
    # 为显示彩色,把灰度图像转换为RGB三通道图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_con_img = cv2.drawContours(img_rgb, contours, largest_idx, (255, 0, 0), cv2.FILLED)
    # 显示画出轮廓后的图像
    # cv_show("image with contours", draw_con_img)

    return contours[largest_idx]


def match_template(cont1, cont2):
    # 形状匹配
    similarity = cv2.matchShapes(cont1, cont2, 1, 0.0)
    # print("形状的相似度为：", similarity)
    return similarity


def match_template_refactor(orimg1, orimg2):
    pre_img1 = pre_process(orimg1)
    pre_img2 = pre_process(orimg2)

    cont1 = find_Contours(pre_img1)
    cont2 = find_Contours(pre_img2)

    # 形状匹配
    similarity = cv2.matchShapes(cont1, cont2, 1, 0.0)
    # print("形状的相似度为：", similarity)
    return similarity


def main():
    template_circle = read_an_gray_image(template_circle_path)
    template_rect = read_an_gray_image(template_rect_path)
    template_square = read_an_gray_image(template_square_path)
    test_circle = read_an_gray_image(test_circle_path)
    test_rect_1 = read_an_gray_image("./images/rect_test_1.jpg")
    test_squ_0 = read_an_gray_image("./images/squ_test_0.jpg")
    test_squ_1 = read_an_gray_image("./images/squ_test_1.jpg")


    # 重构后的模板匹配函数
    cir_obj_match_ans = match_template_refactor(template_circle, test_circle)
    squ_obj_match_ans = match_template_refactor(template_square, test_circle)
    rect_obj_match_ans = match_template_refactor(template_rect, test_circle)



    print("判断圆形：")
    if (cir_obj_match_ans < squ_obj_match_ans) & (cir_obj_match_ans < rect_obj_match_ans):
        print("圆形")
    elif (squ_obj_match_ans < cir_obj_match_ans) & (squ_obj_match_ans < rect_obj_match_ans):
        print("正方形")
    elif (rect_obj_match_ans < cir_obj_match_ans) & (rect_obj_match_ans < squ_obj_match_ans):
        print("长方形")


    print("判断长方形：")
    cir_obj_match_ans = match_template_refactor(template_circle, test_rect_1)
    squ_obj_match_ans = match_template_refactor(template_square, test_rect_1)
    rect_obj_match_ans = match_template_refactor(template_rect, test_rect_1)

    if (cir_obj_match_ans < squ_obj_match_ans) & (cir_obj_match_ans < rect_obj_match_ans):
        print("圆形")
    elif (squ_obj_match_ans < cir_obj_match_ans) & (squ_obj_match_ans < rect_obj_match_ans):
        print("正方形")
    elif (rect_obj_match_ans < cir_obj_match_ans) & (rect_obj_match_ans < squ_obj_match_ans):
        print("长方形")


    print("判断正方形1：")
    cir_obj_match_ans = match_template_refactor(template_circle, test_squ_0)
    squ_obj_match_ans = match_template_refactor(template_square, test_squ_0)
    rect_obj_match_ans = match_template_refactor(template_rect, test_squ_0)

    if (cir_obj_match_ans < squ_obj_match_ans) & (cir_obj_match_ans < rect_obj_match_ans):
        print("圆形")
    elif (squ_obj_match_ans < cir_obj_match_ans) & (squ_obj_match_ans < rect_obj_match_ans):
        print("正方形")
    elif (rect_obj_match_ans < cir_obj_match_ans) & (rect_obj_match_ans < squ_obj_match_ans):
        print("长方形")



    print("判断正方形2：")
    cir_obj_match_ans = match_template_refactor(template_circle, test_squ_1)
    squ_obj_match_ans = match_template_refactor(template_square, test_squ_1)
    rect_obj_match_ans = match_template_refactor(template_rect, test_squ_1)

    if (cir_obj_match_ans < squ_obj_match_ans) & (cir_obj_match_ans < rect_obj_match_ans):
        print("圆形")
    elif (squ_obj_match_ans < cir_obj_match_ans) & (squ_obj_match_ans < rect_obj_match_ans):
        print("正方形")
    elif (rect_obj_match_ans < cir_obj_match_ans) & (rect_obj_match_ans < squ_obj_match_ans):
        print("长方形")



    # print("cir_cir_match_similarity: ", cir_cir_match_ans)
    # print("squ_cir_match_similarity: ", squ_cir_match_ans)
    # print("rect_cir_match_similarity: ", rect_cir_match_ans)


if __name__ == '__main__':
    main()
