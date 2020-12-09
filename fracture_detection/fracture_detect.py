'''
  -*- coding: utf-8 -*-
  @Time    : 2020/10/16 13:35
  @Author  : liuxu
  @File    : fracture_detect.py
  @Software: PyCharm
  @Theme   : 检测裂缝
'''

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math

# 待检测图像路径
filepath = "./fracture_images/crack1.jpg"
# filepath = "./fracture_images/crack2.jpg"
# filepath = "./unfracture_images/paper_img1.jpg"

# lost_corner_img_dictoryname = "./closed_holes_images"
# unlost_corner_img_dictoryname = "./unclosed_holes_images"

# 设置图像的宽和高
IMAGE_WIDTH = 650
IMAGE_HEIGHT = 650

# 存放图像处理过程的列表
all_images = []

# 设置裂缝判断规则(e = r2/r1;裂缝的面积area)
# E ;AREA
E = 0.25
AREA = 200


# 使用plt显示一张图片
def plt_show_one_pic(pic):
    temp_img = pic[0]
    title = pic[1]
    plt.imshow(temp_img)
    plt.title(title, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# 使用plt显示多张图片
# pics为存放多张图片的列表
def plt_show_muti_pic(pics):
    num = len(pics)
    for i in range(num):
        temp_img = pics[i][0]
        temp_title = pics[i][1]
        title = temp_title
        # #行，列，索引
        col = 4  # 一行显示4张图像
        row = math.ceil(num / col)  # 行数
        plt.subplot(row, col, i + 1)
        plt.imshow(temp_img)
        plt.title(title, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        # plt.imshow(temp_img)
        # plt.title(title,fontsize=8)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        # break

    plt.show()


# 使用opencv显示图像
def cv_show(name, img):
    cv2.imshow(name, img)
    # cv2.waitKey(0)            # 控制显示


# 对图像resize处理
def row_image_resize(image, width, height):
    # image输入图像，width，height 期望得到的宽和高
    image = cv2.resize(image, (width, height))
    # print("resized image:", image.shape)
    return image


# 读入一张图像,返回
def read_an_gray_image(image_path):
    return cv2.imread(image_path)


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

    # cv2.imwrite("output_images/grayimage.jpg", grayimage)
    cv_show("gray", grayimage)

    all_images.append((grayimage, "grayimage"))  # 填入列表

    resizedimage = row_image_resize(grayimage, IMAGE_WIDTH, IMAGE_HEIGHT)  # resize
    all_images.append((resizedimage, "resizedimage"))  # 填入列表

    # 运用大津算法进行图像分割
    ret, im_th = cv2.threshold(resizedimage, 0, 255, cv2.THRESH_OTSU)
    # cv2.imwrite("output_images/imth.jpg", im_th)

    cv_show("image after OTSU", im_th)
    all_images.append((im_th, "OTSU"))

    return im_th


# Applies a Laplacian operator to the grayscale image,拉普拉斯变换
def laplacian_tranform(binary_image):
    ddepth = cv2.CV_16S
    kernel_size = 3
    lap_img = cv2.Laplacian(binary_image, ddepth=ddepth, ksize=kernel_size)
    abs_dst = cv2.convertScaleAbs(lap_img)

    cv_show("laplacian", abs_dst)


# 根据矩形度,细长度,周长和面积的比值
def judge_is_cracked(contours):
    print("len contours:", len(contours))
    xichan_degree = []

    cont_peri_area_rectradio_len_wid_ratio = []  # 用一个元组存放(轮廓,周长,面积,矩形度,细长度)
    # 遍历所有轮廓，根据规则判断, 设置temp_p1为矩形度,temp_p3为最小外接矩形的长宽比,值域(0,1)
    # temp_perimeter为周长,temp_area为面积
    for i in range(len(contours)):
        temp_cont = contours[i]
        temp_perimeter = cv2.arcLength(temp_cont, True)  # 周长
        # print("周长为:",temp_perimeter)
        temp_area = cv2.contourArea(temp_cont)  # 轮廓包围的面积
        # print("面积为：",temp_area)

        if (temp_area == 0) | (temp_perimeter == 0):  # 判空
            continue

        # 最小外接矩形
        rect = cv2.minAreaRect(temp_cont)  # 返回  中心点坐标（x,y) width:rect[1][0] height: rect[1][1] 旋转角度θ
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 外接矩形的的四个坐标
        coord1 = box[0]
        coord2 = box[1]
        coord3 = box[2]
        # coord4 = box[3]
        a = float(coord1[0] - coord2[0]) * float(coord1[0] - coord2[0]) + float(coord1[1] - coord2[1]) * float(
            coord1[1] - coord2[1])
        a = math.sqrt(a)  # 矩形的其中一条边长

        b = float(coord2[0] - coord3[0]) * float(coord2[0] - coord3[0]) + float(coord2[1] - coord3[1]) * float(
            coord2[1] - coord3[1])
        b = math.sqrt(b)  # 另一条边长

        rect_area = a * b  # 外接矩形的面积
        if rect_area == 0:  # 防止分母为0
            continue

        # 求矩形度
        rect_degree = temp_area / rect_area
        # print("矩形度为：",rect_degree)

        # 最小外接矩形长宽比,(0,1)
        small_v = min(a, b)
        big_v = max(a, b)
        if big_v == 0:
            continue
        xichan_ratio = big_v / small_v  # 细长度
        # print("细长度为：",xichan_ratio)
        xichan_degree.append(xichan_ratio)

        # 周长和面积的比值,对于裂缝来说周长/面积较大
        area_per_retio = temp_perimeter / temp_area
        # print("周长/面积=",area_per_retio)

        cont_peri_area_rectradio_len_wid_ratio.append(
            (contours[i], temp_perimeter, temp_area, rect_degree, xichan_ratio))

    max_xichang_degree_idx = 0  # 搜索细长度最大的轮廓的下标
    for i in range(len(cont_peri_area_rectradio_len_wid_ratio)):
        temp_xich = cont_peri_area_rectradio_len_wid_ratio[max_xichang_degree_idx][4]
        if cont_peri_area_rectradio_len_wid_ratio[i][4] > temp_xich:
            max_xichang_degree_idx = i

    max_xichang_degree = cont_peri_area_rectradio_len_wid_ratio[max_xichang_degree_idx][4]
    print("最大的细长度为：", max_xichang_degree)
    print("all xichang degree:", xichan_degree)
    frac_cont = cont_peri_area_rectradio_len_wid_ratio[max_xichang_degree_idx][0]
    print("裂缝的面积：", cv2.contourArea(frac_cont))
    xichan_degree.remove(max_xichang_degree)
    # 去掉最大的细长度，求其他的均值
    xichang_mean = np.mean(xichan_degree)
    print("均值为：", xichang_mean)

    return frac_cont


# 根据等效椭圆、椭圆长轴与短轴之比
def judge_is_cracked_ellipse(preprocessimg, img, contours):
    preprocessimg_rgb = cv2.cvtColor(preprocessimg, cv2.COLOR_GRAY2BGR)
    min_e = 1  # 初始化一个较大的细长度
    min_e_r1 = 0
    min_e_r2 = 0
    min_e_area = 0

    # print("len contours:",len(contours))

    for i in range(len(contours)):
        temp_cont = contours[i]
        # tempimg = cv2.drawContours(img,temp_cont,-1,(0,255,0),1)
        # cv_show("tempimg",tempimg)
        # cv2.waitKey(0)
        try:
            M = cv2.moments(temp_cont)
            # print(M)
            area = M['m00']
            u20 = M['mu20'] / area
            u02 = M['mu02'] / area
            u11 = M['mu11'] / area
            # print(area)
            # print(u20)
            # print(u02)
            # print(u11)

            if (area != 0) & (u02 != 0) & (u20 != 0) & (u11 != 0):
                # 求轮廓包围的面积
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # 求重心

                # cv_show("cil",cilcel)
                # print(centroid)
                r1 = math.sqrt(2 * (u20 + u02 + math.sqrt(pow((u20 - u02), 2) + 4 * pow(u11, 2))))
                # print("r1: ",r1)
                r2 = math.sqrt(2 * (u20 + u02 - math.sqrt(pow((u20 - u02), 2) + 4 * pow(u11, 2))))
                # print("r2: ",r2)

                e = r2 / r1  # 短轴与长轴比,越小说明越扁平

                if (e < E) & (int(area) > AREA):
                    # if int(area) > 20000:
                    cilcel = cv2.circle(img, centroid, 2, (0, 0, 255),thickness=2)
                    radian = 1 / 2 * math.atan(2 * u11 / (u20 - u02))
                    angle = radian / math.pi * 180

                    color = (0, 0, 255)

                    # 绘制一个红色椭圆
                    ptCenter = centroid  #
                    axesSize = (int(r1), int(r2))  #
                    rotateAngle = angle
                    startAngle = 0
                    endAngle = 360
                    point_color = (0, 0, 255)  # BGR
                    thickness = 2
                    lineType = 4
                    img = cv2.ellipse(img, ptCenter, axesSize, rotateAngle, startAngle, endAngle, point_color,
                                      thickness,
                                      lineType)
                    img2 = cv2.ellipse(preprocessimg_rgb, ptCenter, axesSize, rotateAngle, startAngle, endAngle,
                                       point_color,
                                       thickness,
                                       lineType)
                    cv2.imshow("ellipse", img)
                    cv2.imshow("ellipse2", img2)

                if e < min_e:
                    min_e = e
                    min_e_r1 = r1
                    min_e_r2 = r2
                    min_e_area = area
        except:
            pass

    print("min_e: ", min_e)
    print("min_e_r1: ", min_e_r1)
    print("min_e_r2: ", min_e_r2)
    print("min_e_area: ", min_e_area)
    cv2.waitKey(0)


# 找出裂缝区域
def find_cracked_img(preprocessedimage):
    # 在分割后的图像找轮廓

    img = preprocessedimage.copy()
    cv_show("img", img)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = []  # 轮廓所围成的面积构成的数组
    for idx in range(len(contours)):
        areas.append(cv2.contourArea(contours[idx]))

    largest_idx = np.argmax(np.array(areas))  # 最大面积的下标

    # 最小外接矩形
    # rect = cv2.minAreaRect(contours[largest_idx])  # 返回  中心点坐标（x,y) width:rect[1][0] height: rect[1][1] 旋转角度θ
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)

    # 画出样品的轮廓
    img = preprocessedimage.copy()
    # 为显示彩色,把灰度图像转换为RGB三通道图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # img_with_rect = cv2.drawContours(img_rgb, [box], 0, (0, 0, 255), 2)
    # all_images.append((img_with_rect,"img_with_rect"))          # 填入列表

    draw_con_img = cv2.drawContours(img_rgb, contours, largest_idx, (170, 170, 170), cv2.FILLED)

    # 显示画出轮廓后的图像
    cv_show("image with contours", draw_con_img)
    # 图像取差集 imagesub = imagesub1 - imagesub2
    imgsub1 = draw_con_img.copy()
    # imgsub1 = cv2.cvtColor(imgsub1,cv2.COLOR_BGR2GRAY)
    imgsub2 = preprocessedimage.copy()
    # blur_image为单通道图像，需要转换为RGB图像
    imgsub2 = cv2.cvtColor(imgsub2, cv2.COLOR_GRAY2BGR)

    # 取差集
    imgsub = cv2.subtract(imgsub1, imgsub2)

    all_images.append((imgsub, "sub"))
    # 显示取差集后图像
    cv_show("image substraction", imgsub)

    # imgsub_gray = cv2.cvtColor(imgsub,cv2.COLOR_BGR2GRAY)
    # cv_show("image sub gray",imgsub_gray)

    # 取差集后对图像进行闭运算操作,填补小孔洞

    # imgsub = cv2.cvtColor(imgsub,cv2.COLOR_GRAY2BGR)
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

    # 在二值图像找出所有轮廓,  过滤网中所有的孔洞的轮廓
    contours, hierarchy = cv2.findContours(image_find_contours_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # frac_cont = judge_is_cracked(contours)
    frac_cont = judge_is_cracked_ellipse(preprocessedimage, image_find_contours, contours)
    #
    # img_with_frac = cv2.drawContours(image_find_contours,frac_cont,-1,(51,153,255),2)
    #

    # cv_show("image with frac contours",img_with_frac)

    # cv2.waitKey(0)


# 对单个图像检测
def single_detection(imagepath):
    rowimg = read_an_gray_image(imagepath)
    preproimg = pre_process(rowimg)
    find_cracked_img(preproimg)
    # laplacian_tranform(preproimg)


# 对文件夹下的所有图像进行检测
def dictory_detection(dictorypath):
    imgslist = read_images(dictorypath)
    # print("一共{}副图像".format(len(imgslist)))
    for i in range(len(imgslist)):
        rowimg = imgslist[i]
        preproimg = pre_process(rowimg)
        find_cracked_img(preproimg)


def main():
    # 对一幅裂缝图像处理
    single_detection(filepath)


if __name__ == "__main__":
    main()
