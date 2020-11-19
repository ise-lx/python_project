import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


# file1 = "./test_light3.jpg"
# file2 = "./test_light3_2.jpg"
# file3 = "./test_light3_3.jpg"
file4 = "./test8.jpg"
# file4_1 = "./test6_1.jpg"
# file4_2 = "./test6_2.jpg"
# file4_3 = "./test6_3.jpg"
# file4_4 = "./test6_4.jpg"


# 设置图像的宽和高
IMAGE_WIDTH = 750
IMAGE_HEIGHT = 750

# 存放图像处理过程的列表
all_images = []


# 使用plt显示一张图片
def plt_show_one_pic(pic):
    temp_img = pic[0]
    title = pic[1]
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



    # 运用大津算法进行图像分割
    ret, im_th = cv2.threshold(resizedimage, 0, 255, cv2.THRESH_OTSU)
    cv_show("image after OTSU", im_th)
    all_images.append((im_th,"OTSU"))

    # return im_th



# src = read_an_image(file1)
# pre_process(src)
# src = read_an_image(file2)
# pre_process(src)
# src = read_an_image(file3)
# pre_process(src)

src = read_an_image(file4)
pre_process(src)
#
# src = read_an_image(file4_1)
# pre_process(src)
#
# src = read_an_image(file4_2)
# pre_process(src)
#
# src = read_an_image(file4_3)
# pre_process(src)
#
# src = read_an_image(file4_4)
# pre_process(src)