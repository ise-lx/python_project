'''
  -*- coding: utf-8 -*-
  @Time    : 2020/10/21 17:47
  @Author  : liuxu
  @File    : fenkuaiimg.py
  @Software: PyCharm
  @Theme   : 
'''

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


file4 = "./test15.jpg"
# file4 = "./test10.jpg"
# file4 = "./test11.jpg"
# file4 = "./test12.jpg"
# file4 = "./test13.jpg"
# file4 = "./test14.jpg"
# file4 = "./histtest8.jpg"


# 设置图像的宽和高
IMAGE_WIDTH = 650
IMAGE_HEIGHT = 650

# 存放图像处理过程的列表
all_images = []


# 使用plt显示一张图片
def plt_show_one_pic(title,pic):
    temp_img = pic
    title = title
    plt.imshow(temp_img)
    plt.title(title,fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# 使用opencv显示图像
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)            # 控制显示


# 对图像resize处理
def row_image_resize(image, width, height):
    # image输入图像，width，height 期望得到的宽和高
    image = cv2.resize(image, (width, height))
    # print("resized image:", image.shape)
    return image




# 读入一张图像,返回
def read_an_image(image_path):
    return cv2.imread(image_path)

# #图像预处理函数
def pre_process(rowimage):
    # 显示原图
    # cv_show("row image", rowimage)
    grayimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2GRAY)  # 灰度化
    all_images.append((grayimage,"grayimage"))              # 填入列表
    resizedimage = row_image_resize(grayimage, IMAGE_WIDTH, IMAGE_HEIGHT)  # resize
    all_images.append((resizedimage,"resizedimage"))        # 填入列表

    edge = sobel_dect(resizedimage)

    cv_show("edge",edge)



    resizedimage = edge
    # 运用大津算法进行图像分割
    ret, im_th = cv2.threshold(resizedimage, 0, 255, cv2.THRESH_OTSU)
    cv_show("image after OTSU", im_th)
    all_images.append((im_th,"OTSU"))



    return im_th


def sobel_dect(img):
    # sobel边缘检测
    edges = cv2.Sobel(img, cv2.CV_16S, 1, 1)
    # 浮点型转成uint8型
    edges = cv2.convertScaleAbs(edges)
    # plt.figure()
    # plt.imshow(edges, plt.cm.gray)
    # plt.show()

    return edges


def fenkuai(img):
    # cv_show("img",img)
    plt_show_one_pic("img",img)

    h = img.shape[0]
    w = img.shape[1]
    print(h)
    print(w)

    n = 3
    m = 3
    dis_h = int(np.floor(h/n))
    dis_w = int(np.floor(w/m))
    num = 0
    for i in range(n):
        for j in range(m):
            num +=1
            print('i,j={}{}'.format(i,j))
            temp_img = img[dis_h*i:dis_h*(i+1),dis_w*j:dis_w*(j+1)]
            # cv_show(str(num),temp_img)
            plt_show_one_pic(str(num),temp_img)







    # new_wid = int(wid/2)
    # new_heig = int(height/2)
    # print(new_wid)
    # print(new_heig)
    #
    #
    # new_img_1 = img[0:new_heig,0:new_wid]
    # new_img_1_1 = img[int(new_heig/2):new_heig,int(new_wid/2):new_wid]
    # new_img_2 = img[0:new_heig,new_wid:wid]
    # new_img_3 = img[new_heig:height,0:new_wid]
    # new_img_4 = img[new_heig:height,new_wid:wid]
    #
    #
    #
    # print("new img1 shape:",new_img_1.shape)
    # print("new img1_1 shape:",new_img_1_1.shape)
    # print("new img2 shape:",new_img_2.shape)
    # print("new img3 shape:",new_img_3.shape)
    # print("new img4 shape:",new_img_4.shape)
    #
    # cv_show("new img1",new_img_1)
    # cv_show("new img1_1",new_img_1_1)
    # cv_show("new img2", new_img_2)
    # cv_show("new img3", new_img_3)
    # cv_show("new img4",new_img_4)


src = read_an_image(file4)
img_th = pre_process(src)
# fenkuai(img_th)

# cv2.waitKey(0)
