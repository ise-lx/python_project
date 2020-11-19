# -*- coding: utf-8 -*-
'''
   @Time    : 2020/9/26 11:48
   @Author  : liuxu
   @File    : sunken_detect.py
   @Software: PyCharm
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os



filepath = "sunken_images/13.jpg"
# filepath = "./unsunken_images/bianxing_test.jpg"
# sunken_img_dictoryname = "./sunken_images"
# unsunken_img_dictoryname = "./unsunken_images"



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
    print("resized image:", image.shape)
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
    cv_show("row image", rowimage)

    grayimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2GRAY)  # 灰度化
    resizedimage = row_image_resize(grayimage, IMAGE_WIDTH, IMAGE_HEIGHT)  # resize

    # 运用大津算法进行图像分割
    ret, im_th = cv2.threshold(resizedimage, 0, 255, cv2.THRESH_OTSU)
    cv_show("image after OTSU", im_th)

    # #中值滤波去除椒盐噪声,卷积核为5x5
    # blur_image = cv2.medianBlur(im_th, 3)
    #显示中值滤波处理后图像
    # cv_show("image after medianBlur", blur_image)

    # return blur_image

    return im_th


# 计算所有孔洞轮廓外接圆半径的平均值
def cal_ave_radius(allcontours, max_id):                            # allcontours是一个存放轮廓的列表
    sum = 0.0
    for i in range(len(allcontours)):
        if (i is not max_id):                                       # 去除最大面积后再求平均
            temp_contour = allcontours[i]
            (x, y), radius = cv2.minEnclosingCircle(temp_contour)   # 最小外接圆
            sum = sum + radius

    return sum / (len(allcontours) - 1)





# 找出凹陷区域
def find_sunken_area(preprocessedimage):
    # 在分割后的图像找轮廓
    contours, hierarchy = cv2.findContours(preprocessedimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print("轮廓的类型：",type(contours))     # type is list
    areas = []  # 轮廓所围成的面积构成的数组
    for idx in range(len(contours)):
        areas.append(cv2.contourArea(contours[idx]))

    largest_idx = np.argmax(np.array(areas))  # 最大面积的下标
    largest_area = cv2.contourArea(contours[largest_idx])  # 最大面积的大小

    # #
    # # 第二大面积的下标初始化为0
    # second_largest_idx = 0
    # # 第二大面积的大小
    # second_largest_area = sorted(areas)[-2]
    #
    # for idx in range(len(areas)):
    #     temp_area = cv2.contourArea(contours[idx])
    #     if (temp_area == second_largest_area):
    #         second_largest_idx = idx
            # print("second largest index:",second_largest_idx)

    print("最大面积为：", largest_area)
    print("最大面积的下标为：", largest_idx)
    # print("第二大面积为：", second_largest_area)
    # print("第二大面积的下标为：", second_largest_idx)

    # 画出样品的轮廓
    img = preprocessedimage.copy()
    # 为显示彩色,把灰度图像转换为RGB三通道图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_con_img = cv2.drawContours(img_rgb, contours, largest_idx, (0, 0, 255), cv2.FILLED)
    # 显示画出轮廓后的图像
    cv_show("image with contours", draw_con_img)
    # 图像取差集 imagesub = imagesub1 - imagesub2
    imgsub1 = draw_con_img.copy()
    imgsub2 = preprocessedimage.copy()
    # blur_image为单通道图像，需要转换为RGB图像
    imgsub2 = cv2.cvtColor(imgsub2, cv2.COLOR_GRAY2BGR)

    # 取差集
    imgsub = cv2.subtract(imgsub1, imgsub2)

    # 显示取差集后图像
    cv_show("image substraction", imgsub)

    # 取差集后对图像进行闭运算操作,填补小孔洞

    # 定义矩形结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closedimg = cv2.morphologyEx(imgsub, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv_show("image after close operation", closedimg)

    # 在取差集后的图像找轮廓
    image_find_contours = closedimg.copy()
    # 二值化,要用到阈值函数cv2.threshold()
    # cv2.findContours()要输入二值图像

    # 灰度化
    image_find_contours_gray = cv2.cvtColor(image_find_contours, cv2.COLOR_BGR2GRAY)
    ret, image_find_contours_bin = cv2.threshold(image_find_contours_gray, 0, 255, cv2.THRESH_OTSU)

    # 在二值图像找出所有轮廓
    contours, hierarchy = cv2.findContours(image_find_contours_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for idx in range(len(contours)):
        areas.append(cv2.contourArea(contours[idx]))

    # 最大面积的下标
    max_idx = np.argmax(np.array(areas))
    # 最大面积的大小
    max_area = cv2.contourArea(contours[max_idx])

    # 计算所有轮廓外接圆的平均半径
    ave_radius = cal_ave_radius(contours, max_idx)
    print("平均半径为：", ave_radius)

    # 在差集图像中画出轮廓
    # image_with_con = cv2.drawContours(image_find_contours, contours, max_idx, (51, 153, 255),3)

    # 显示画出轮廓后的图像
    # cv_show("image with max contour", image_with_con)

    #画出所有轮廓
    # image_with_con_all = cv2.drawContours(image_find_contours,contours,-1,(0,0,255),1)        #注意(B,G,R)
    # cv_show("image with all contours",image_with_con_all)


    # 最大轮廓
    max_cnt = contours[max_idx]
    # 最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(max_cnt)
    # 圆心
    center = (int(x), int(y))
    radius = int(radius)
    # 打印半径
    print("取差集后图像中最大轮廓\n的最小外接圆的半径为: {}px".format(radius))

    # 在图像中标出最小外接圆
    # img = cv2.circle(image_with_con, center, radius, (255, 185, 15), 2)
    # cv_show("image with min enclosing circle", img)

    # 在图像中标出最小外接圆
    img_rgb = cv2.cvtColor(image_find_contours_bin,cv2.COLOR_GRAY2BGR)
    img = cv2.circle(img_rgb, center, radius, (0, 0, 255), 2)
    cv_show("image with min enclosing circle", img)

    #判断是否存在凹坑缺陷


    #设标志
    flag = False
    # 判断最大外接圆的半径是否大于平均半径（除最大面积外）的二倍, 参数可以调节
    if(radius > ave_radius*4):
        flag = True

    print('*'*30)
    print('\n')
    if(flag == True):
        print("该图像存在凹坑缺陷！")
        print("凹坑缺陷面积为：{}".format(max_area))
        print("凹坑外接圆的半径为: {}px".format(radius))
    else:
        print("该图像没有凹坑缺陷")
    print('\n')
    print('*'*30)




# 对单个图像检测
def single_detection(imagepath):
    rowimg = read_an_gray_image(imagepath)
    preproimg = pre_process(rowimg)
    find_sunken_area(preproimg)




# 对文件夹下的所有图像进行检测
def dictory_detection(dictorypath):

    imgslist = read_images(dictorypath)
    # imgslist = read_images(dictorypath)
    print("一共{}副图像".format(len(imgslist)))
    for i in range(len(imgslist)):
        rowimg = imgslist[i]
        preproimg = pre_process(rowimg)
        find_sunken_area(preproimg)


# main函数
def main():
    # 对一幅图像处理
    single_detection(filepath)
    # 对文件夹下所有存在凹陷的图像进行检测
    # dictory_detection(sunken_img_dictoryname)

    # 对文件下所有正常图像进行检测
    # dictory_detection(unsunken_img_dictoryname)



if __name__ == "__main__":
    main()
