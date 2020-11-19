'''
  -*- coding: utf-8 -*-
  @Time    : 2020/10/31 13:35
  @Author  : liuxu
  @File    : ellipse_test.py
  @Software: PyCharm
  @Theme   : 
'''

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math

filepath = "./images.png"

# 使用opencv显示图像
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)            # 控制显示



# #图像预处理函数
def pre_process(rowimage):
    # 显示原图
    cv_show("row image", rowimage)


    grayimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2GRAY)  # 灰度化
    # cv2.imwrite("output_images/grayimage.jpg",grayimage)

    grayimage = cv2.bitwise_not(grayimage)
    cv_show("gray",grayimage)


    # 运用大津算法进行图像分割
    ret, im_th = cv2.threshold(grayimage, 0, 255, cv2.THRESH_OTSU)
    # cv2.imwrite("output_images/imth.jpg",im_th)

    cv_show("image after OTSU", im_th)

    return im_th






# 根据等效椭圆、椭圆长轴与短轴之比
def judge_is_cracked_ellipse(img,contours):

    min_e = 1  # 初始化一个较大的细长度
    min_e_r1 = 0
    min_e_r2 = 0

    print("len contours:",len(contours))
    xichan_degree = []
    max_area = 0
    max_area_index = 0
    for i in range(len(contours)):
        temp_con = contours[i]
        temp_area = cv2.contourArea(temp_con)
        if temp_area > max_area:
            max_area = temp_area
            max_area_index = i

    for i in range(len(contours)):
        temp_cont = contours[max_area_index]
        tempimg = cv2.drawContours(img,temp_cont,-1,(0,255,0),1)
        cv_show("tempimg",tempimg)
        # cv2.waitKey(0)
        try:
            M = cv2.moments(temp_cont)
            print(M)
            area = M['m00']
            u20 = M['mu20']/area
            u02 = M['mu02']/area
            u11 = M['mu11']/area
            print(area)
            print(u20)
            print(u02)
            print(u11)


            if (area != 0) & (u02 != 0) & (u20 != 0) & (u11 != 0):
                # 求轮廓包围的面积
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # 求重心
                cilcel = cv2.circle(img,centroid,2,(0,0,255))
                cv_show("cil",cilcel)
                print(centroid)
                r1 = math.sqrt(1/2*(u20+u02+math.sqrt((u20-u02)*(u20-u02)+4*u11*u11)))
                print("r1: ",r1)
                r2 = math.sqrt(1/2*(u20+u02-math.sqrt((u20-u02)*(u20-u02)+4*u11*u11)))
                print("r2: ",r2)

                e = r2/r1                   #短轴与长轴比,越小说明越扁平

                radian = 1/2*math.atan(2*u11/(u20-u02))
                angle = radian/math.pi*180

                color = (0,0,255)

                # 绘制一个红色椭圆
                ptCenter = centroid  #
                axesSize = (int(104), int(12))  #
                rotateAngle = angle
                startAngle = 0
                endAngle = 360

                point_color = (0, 255, 255)  # BGR
                thickness = 2
                lineType = 4
                cv2.ellipse(img, ptCenter, axesSize, rotateAngle, startAngle, endAngle, point_color, thickness, lineType)
                cv2.imshow("ellipse",img)
                cv2.waitKey(0)


                if e < min_e:
                    min_e = e
                    min_e_r1 = r1
                    min_e_r2 = r2
        except:
            pass
        break
    print("min_e: ",min_e)
    print("min_e_r1: ",min_e_r1)
    print("min_e_r2: ",min_e_r2)






def main():
    src = cv2.imread(filepath)

    preimg = pre_process(src)
    contours, hierarchy = cv2.findContours(preimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frac_cont = judge_is_cracked_ellipse(preimg, contours)

    img = cv2.drawContours(preimg,frac_cont,-1,(0,0,255),2)
    cv_show("img",img)

    # cv_show("ellipse",frac_cont)



if __name__ == '__main__':
    main()
