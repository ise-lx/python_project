# -*- coding: utf-8 -*-
'''
   @Time    : 2020/9/26 16:54
   @Author  : liuxu
   @File    : lost_corner_detect.py
   @Software: PyCharm
'''


import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import os


# 待检测图像路径
filepath = "./lost_corner_images/3.jpg"
lost_corner_img_dictoryname = "./lost_corner_images"
unlost_corner_img_dictoryname = "./unlost_corner_images"


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


# 定义预处理函数
def pre_process(rowimage):
    # 显示原图
    cv_show("row image",rowimage)

    # 灰度化处理
    grayimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2GRAY)
    resizedimage = row_image_resize(grayimage, IMAGE_WIDTH, IMAGE_HEIGHT)

    # 大津算法处理
    ret, im_th = cv2.threshold(resizedimage, 0, 255, cv2.THRESH_OTSU)
    cv_show("image after OTSU",im_th)

    # 中值滤波去除椒盐噪声，卷积核为3x3
    blur_image = cv2.medianBlur(im_th, 3)

    # 显示中值滤波处理后图像
    cv_show("image after mediaBlur",blur_image)

    return blur_image


# 计算最小外接矩形的四个角的坐标距离轮廓的大小
# 传入轮廓和外接矩形的四个坐标
def cal_distance(img,max_cont, coord_1, coord_2, coord_3, coord_4):
    # 初始化四个距离,取一个较大的数
    global point_1, point_2, point_3, point_4
    coord_1_min_distance = 100000.0
    coord_2_min_distance = 100000.0
    coord_3_min_distance = 100000.0
    coord_4_min_distance = 100000.0
    print("四个坐标：", coord_1, coord_2, coord_3, coord_4)

    for con_point in max_cont:  # 计算coor_1到max_cont的最短距离
        point_width = con_point[0][0]
        point_height = con_point[0][1]

        # 临时距离
        coord_1_temp_distance = float(np.sqrt((point_width - coord_1[0]) ** 2 + (point_height - coord_1[1]) ** 2))
        coord_2_temp_distance = float(np.sqrt((point_width - coord_2[0]) ** 2 + (point_height - coord_2[1]) ** 2))
        coord_3_temp_distance = float(np.sqrt((point_width - coord_3[0]) ** 2 + (point_height - coord_3[1]) ** 2))
        coord_4_temp_distance = float(np.sqrt((point_width - coord_4[0]) ** 2 + (point_height - coord_4[1]) ** 2))

        # 判断
        if coord_1_temp_distance < coord_1_min_distance:
            coord_1_min_distance = coord_1_temp_distance
            point_1 = con_point


        if coord_2_temp_distance < coord_2_min_distance:
            coord_2_min_distance = coord_2_temp_distance
            point_2 = con_point

        if coord_3_temp_distance < coord_3_min_distance:
            coord_3_min_distance = coord_3_temp_distance
            point_3 = con_point

        if coord_4_temp_distance < coord_4_min_distance:
            coord_4_min_distance = coord_4_temp_distance
            point_4 = con_point


    print("四个最小距离：", coord_1_min_distance, coord_2_min_distance, coord_3_min_distance, coord_4_min_distance)

    # 四个距离中最大的
    max_of_all_distance = max(coord_1_min_distance, coord_2_min_distance, coord_3_min_distance, coord_4_min_distance)
    print("max of all four:", max_of_all_distance)

    if max_of_all_distance > 25:
        print("存在掉角缺陷")
    else:
        print("不存在掉角缺陷")
    img = cv2.circle(img,(point_1[0][0],point_1[0][1]),3,(0,255,0),2)
    img = cv2.circle(img,(point_2[0][0],point_2[0][1]),3,(0,255,0),2)
    img = cv2.circle(img,(point_3[0][0],point_3[0][1]),3,(0,255,0),2)
    img = cv2.circle(img,(point_4[0][0],point_4[0][1]),3,(0,255,0),2)
    cv_show("img with circle:",img)

    # return coord_1_min_distance, coord_2_min_distance, coord_3_min_distance, coord_4_min_distance


# 找出掉角区域
def find_lost_corner_area(preprocessedimage):
    contours, hierarchy = cv2.findContours(preprocessedimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print("轮廓的类型：",type(contours))     # type is list
    areas = []  # 轮廓所围成的面积构成的数组
    for idx in range(len(contours)):
        areas.append(cv2.contourArea(contours[idx]))

    largest_idx = np.argmax(np.array(areas))  # 最大面积的下标
    largest_area = cv2.contourArea(contours[largest_idx])  # 最大面积的大小

    # 画出样品的轮廓
    img = preprocessedimage.copy()
    # 为显示彩色,把灰度图像转换为RGB三通道图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # draw_con_img = cv2.drawContours(img_rgb, contours, largest_idx, (0, 0, 255), cv2.FILLED) #填充
    draw_con_img = cv2.drawContours(img_rgb, contours, largest_idx, (0, 0, 255))  # 不填充
    # 显示画出轮廓后的图像
    cv_show("image with contours", draw_con_img)

    # 最大轮廓
    max_cnt = contours[largest_idx]

    # 最小外接矩形
    rect = cv2.minAreaRect(max_cnt)
    # print("rect type:", type(rect))
    # print(rect)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # print("box:", box)
    cord_1 = box[0]
    cord_2 = box[1]
    cord_3 = box[2]
    cord_4 = box[3]
    # print("cord_1:", cord_1)
    # print("type of box", type(box))

    img_with_rect = cv2.drawContours(img_rgb, [box], 0, (0, 0, 255), 2)
    cv_show("img_with_rect", img_with_rect)

    # 计算距离
    cal_distance(img_with_rect,max_cnt, cord_1, cord_2, cord_3, cord_4)


# 对单个图像检测
def single_detection(imagepath):
    rowimg = read_an_gray_image(imagepath)
    preproimg = pre_process(rowimg)
    find_lost_corner_area(preproimg)


# 对文件夹下的所有图像进行检测
def dictory_detection(dictorypath):
    imgslist = read_images(dictorypath)
    print("一共{}副图像".format(len(imgslist)))
    for i in range(len(imgslist)):
        rowimg = imgslist[i]
        preproimg = pre_process(rowimg)
        find_lost_corner_area(preproimg)


# main函数
def main():
    # 对一幅图像进行处理
    # single_detection(filepath)

    # 对文件夹下的所有掉角的图像进行处理
    dictory_detection(lost_corner_img_dictoryname)

    # 对文件夹下的所有正常的图像进行处理
    # dictory_detection(unlost_corner_img_dictoryname)


if __name__ == '__main__':
    main()
