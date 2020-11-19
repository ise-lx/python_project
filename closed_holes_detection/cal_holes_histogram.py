# 计算滑动窗口内孔径的直方图
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


# 待检测图像路径
# filepath = "./closed_holes_images/closed_hole.JPG"
filepath = "./closed_holes_images/4.jpg"
# filepath = "./closed_holes_images/closed_holes_circle.jpg"
# filepath = "./closed_holes_images/closed_holes_3.jpg"
# filepath = "./unclosed_holes_images/unclosed.JPG"
# filepath = "./unclosed_holes_images/unclosed_2.JPG"
# filepath = "./unclosed_holes_images/unclosed_3.jpg"
# filepath = "./unclosed_holes_images/homomorphic_filtered.png"

lost_corner_img_dictoryname = "./closed_holes_images"
unlost_corner_img_dictoryname = "./unclosed_holes_images"

# 设置图像的宽和高
IMAGE_WIDTH = 650
IMAGE_HEIGHT = 650

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



# 求滑动窗口内孔径的平均值
def cal_mean_hole_size(contours):
    radius_list = []
    for i in range(len(contours)):
        temp_con = contours[i]
        (x, y), radius = cv2.minEnclosingCircle(temp_con)
        if radius > 0:
            radius_list.append(radius)

    mean_radius = np.mean(radius_list)

    print("孔径的平均值为：",mean_radius)









