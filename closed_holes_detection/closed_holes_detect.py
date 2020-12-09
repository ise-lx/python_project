# -*- coding: utf-8 -*-
'''

   @Time    : 2020/10/3 10:18
   @Author  : liuxu
   @File    : closed_holes_detect.py
   @Software: PyCharm

'''

# 检测堵孔
import cv2
import numpy as np
import os
import math
import matplotlib.pylab as plt

# plt.set_cmap('binary')


# 待检测图像路径
filepath = "./closed_holes_images/closed_hole.JPG"
# filepath = "./closed_holes_images/4.jpg"
# filepath = "./closed_holes_images/closed_holes_circle.jpg"
# filepath = "./closed_holes_images/closed_holes_3.jpg"
# filepath = "./unclosed_holes_images/unclosed.JPG"
# filepath = "./unclosed_holes_images/unclosed_2.JPG"
# filepath = "./unclosed_holes_images/unclosed_3.jpg"
# filepath = "./unclosed_holes_images/1.jpg"
# filepath = "./unclosed_holes_images/123.tif"
# filepath = "./unclosed_holes_images/131.tif"
# filepath = "./unclosed_holes_images/135.tif"
# filepath = "./unclosed_holes_images/140.tif"
# filepath = "./unclosed_holes_images/144.tif"
# filepath = "./unclosed_holes_images/150.tif"

# filepath = "./unclosed_holes_images/homomorphic_filtered.png"

# lost_corner_img_dictoryname = "./closed_holes_images"
# unlost_corner_img_dictoryname = "./unclosed_holes_images"

# 设置图像的宽和高
IMAGE_WIDTH = 750
IMAGE_HEIGHT = 750

# 存放图像处理过程的列表
all_images = []


# 求窗口内所有的孔径，返回窗口内所有孔洞的孔径
def cal_all_hole_size(contours):
    radius_list = []
    for i in range(len(contours)):
        temp_con = contours[i]
        (x, y), radius = cv2.minEnclosingCircle(temp_con)
        if radius > 0:
            radius_list.append(radius)
    radius_list.sort(reverse=True)
    fileObject = open('holes_size.txt', 'w')
    for r in radius_list:
        if float(r) < 100.0:
            fileObject.write(str(r))
            fileObject.write('\n')
    return radius_list


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


# 显示图像
def cv_show(name, img):
    cv2.imshow(name, img)
    # cv2.waitKey(0)


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
    all_images.append((grayimage, "grayimage"))  # 填入列表
    resizedimage = row_image_resize(grayimage, IMAGE_WIDTH, IMAGE_HEIGHT)  # resize
    all_images.append((resizedimage, "resizedimage"))  # 填入列表
    cv_show("resizeimg", resizedimage)
    #
    # blur_image = cv2.blur(resizedimage,(3,3))
    # cv_show("blur_img",blur_image)

    # 运用大津算法进行图像分割
    ret, im_th = cv2.threshold(resizedimage, 0, 255, cv2.THRESH_OTSU)

    # 使用全局固定阈值处理
    ret,im_th = cv2.threshold(resizedimage,int(ret+25),255,cv2.THRESH_BINARY)


    cv_show("image after OTSU", im_th)
    all_images.append((im_th, "OTSU"))

    # 闭运算操作 3x3
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # closedimg = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # cv_show("after closed 3x3:",closedimg)
    #
    # 闭运算操作 5x5
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # closedimg = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel, iterations=1)
    #
    # cv_show("after closed 5x5:",closedimg)

    # #中值滤波去除椒盐噪声,卷积核为3x3,中值滤波能消除裂缝,带来不利影响,所以不一定要用中值滤波
    # blur_image = cv2.medianBlur(im_th, 3)
    # #显示中值滤波处理后图像
    # cv_show("image after medianBlur", blur_image)
    # print("image shape:", blur_image.shape)
    # print("image shape 0:", blur_image.shape[0])
    # print("image shape 1:", blur_image.shape[1])

    # return blur_image
    #
    # return closedimg
    return im_th


# 求滑动窗口内孔径的平均值
def cal_mean_hole_size(contours):
    radius_list = []
    for i in range(len(contours)):
        temp_con = contours[i]
        (x, y), radius = cv2.minEnclosingCircle(temp_con)
        if radius > 0:
            radius_list.append(radius)

    mean_radius = np.mean(radius_list)

    print("孔径的平均值为：", mean_radius)


# 占空比计算函数
def cal_duty_cycle_in_rect(cropped_rotated_image, edge_length):


    # 正常区域占空比
    normal_duty_list = []
    # 非正常区域占空比
    unnormal_duty_list = []




    # 中值滤波去除面积微小的区域
    cropped_rotated_image = cv2.medianBlur(cropped_rotated_image, 3)

    print("根据占空比特征检测缺陷")
    # 是否存在堵孔缺陷的标志
    closed_holes_flag = 0
    # print("before shape: ",cropped_rotated_image.shape)
    # print("edge_length: ",edge_length)

    # print("计算窗口内孔洞数量前的图像尺寸：",cropped_rotated_image.shape)
    # print("窗口的长度：",edge_length)
    img = cropped_rotated_image.copy()
    width_0 = img.shape[1]  # numpy的shape方法和
    height_0 = img.shape[0]
    # 减小尺寸后的图像
    mini_img = img[5:(height_0 - 5), 5:(width_0 - 5)]
    all_images.append((mini_img, "mini_img"))  # 填入列表
    print("mini_img shape: ", mini_img.shape)

    # 减小后的尺寸
    width = mini_img.shape[1]  # numpy的shape方法和
    height = mini_img.shape[0]
    # print("width:",width)
    # print("height:",height)

    cv_show("mini:", mini_img)
    th = mini_img.max()  # 因为从二值图像转换RGB-->灰度图,灰度图像也是有两个值th和0
    S = float(edge_length * edge_length)  # 计算滑块的面积
    gray_value_S = S * float(th)  # 当滑块全部落在目标区域内的时候，总的灰度值的和

    # 初始化一个表示孔洞数量的二维数组
    rate_arr = np.zeros(
        [math.ceil((height - edge_length) / 10), math.ceil((width - edge_length) / 10)])  # math.ceil向上取整
    # print("二维数组shape: ",cons_num_arr.shape)

    min_value_rate = 1.0
    # 为提高运算速度，左右和上下相邻两个窗口间隔10px

    black_img = np.zeros_like(mini_img)  # 创建一个同大小的(0,0,0)黑色图像
    white_img = black_img
    white_img_rgb = cv2.cvtColor(white_img, cv2.COLOR_GRAY2BGR)

    mini_img_rgb = cv2.cvtColor(mini_img, cv2.COLOR_GRAY2BGR)
    for row in range(0, int(height - edge_length), 10):
        for col in range(0, int(width - edge_length), 10):
            temp_img = mini_img[int(row):int(row + edge_length), int(col):int(col + edge_length)]
            box = [[col, row], [col + edge_length, row], [col + edge_length, row + edge_length],
                   [col, row + edge_length]]  # 堵孔的矩形框
            box = np.array(box)  # list to ndarray
            # print("temp box: ",box)
            temp_sum = temp_img.sum()
            # print("temp_sum value: ", temp_sum)
            temp_rate = float(temp_sum / gray_value_S)
            rate_arr[int(row / 10)][int(col / 10)] = temp_rate

            # print("temp_sum value / gray_value_S={}".format(temp_sum / gray_value_S))
            if temp_rate < min_value_rate:
                min_value_rate = temp_rate

            if temp_rate < 0.055:
                # mini_img_with_rect = cv2.drawContours(mini_img_rgb, [box], 0, (0, 0, 255), 1)
                cv2.drawContours(white_img, [box], 0, (255, 255, 255), cv2.FILLED)
                unnormal_duty_list.append(temp_rate)
            else:
                normal_duty_list.append(temp_rate)

                # cv_show("mini_img_with_rect: ", mini_img_with_rect)
    print("正常区域占空比列表：",normal_duty_list)
    f1 = open("normal_duty_list.txt",'w')
    f2 = open("unnormal_duty_list.txt",'w')

    for line in normal_duty_list:
        f1.write(str(line)+"\n")
    f1.close()
    for line in unnormal_duty_list:
        f2.write(str(line)+"\n")
    f2.close()

    print("正常区域占空比列表的长度：",len(normal_duty_list))
    print("非正常区域占空比列表：",unnormal_duty_list)
    print("非正常区域占空比列表长度：",len(unnormal_duty_list))


    # 把检测的滑动窗口用轮廓算法连接起来
    contours, _ = cv2.findContours(white_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(white_img_rgb,contours,-1,(0,0,255),thickness=1)
    # cv2.imshow("contours",white_img_rgb)

    cv2.drawContours(mini_img_rgb, contours, -1, (0, 0, 255), thickness=2)
    cv2.imshow("contours", mini_img_rgb)

    print("min_value_rate: ", min_value_rate)
    if min_value_rate < 0.055:
        print("该过滤网存在堵孔缺陷")
    else:
        print("该过滤网不存在堵孔缺陷")

    cv2.imshow("white with rect", white_img)

    return rate_arr

def crap_and_rotate(img, rect):
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # 对图像进行剪裁和旋转至水平
    width = int(rect[1][0])  # 图像的宽度
    height = int(rect[1][1])  # 图像的高度
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))

    cv_show("cropped and rotated! ", warped)
    all_images.append((warped, "wrop&rotate"))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return warped_gray


# 找出区域
def find_closed_holes(preprocessedimage):
    # 在分割后的图像找轮廓

    # brefore_closed_5x5 = preprocessedimage.copy()
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    # closedimg = cv2.morphologyEx(brefore_closed_5x5, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv_show("after closed 5x5:",closedimg)

    img = preprocessedimage.copy()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("contours type: ",type(contours))
    # print("contours: ",contours)
    # print("contours length: ",len(contours))

    # print("轮廓的类型：",type(contours))     # type is list
    areas = []  # 轮廓所围成的面积构成的数组
    for idx in range(len(contours)):
        areas.append(cv2.contourArea(contours[idx]))

    largest_idx = np.argmax(np.array(areas))  # 最大面积的下标

    # 最小外接矩形
    rect = cv2.minAreaRect(contours[largest_idx])  # 返回  中心点坐标（x,y) width:rect[1][0] height: rect[1][1] 旋转角度θ
    # print("rect type:", type(rect))
    # print(rect)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # print("box type",type(box))
    # print("box:", box)

    # 画出样品的轮廓
    img = preprocessedimage.copy()
    # 为显示彩色,把灰度图像转换为RGB三通道图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_with_rect = cv2.drawContours(img_rgb, [box], 0, (0, 0, 255), 2)
    all_images.append((img_with_rect, "img_with_rect"))  # 填入列表

    cv_show("image with rect", img_with_rect)

    new_img_with_rect = crap_and_rotate(img_with_rect, rect)
    cv_show("new_img_with_rect", new_img_with_rect)

    # # 画出样品的轮廓
    # img = preprocessedimage.copy()
    # # 为显示彩色,把灰度图像转换为RGB三通道图像
    # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
    draw_con_img = cv2.drawContours(img_rgb, contours, largest_idx, (255, 0, 0), cv2.FILLED)

    # 显示画出轮廓后的图像
    # cv_show("image with contours", draw_con_img)

    # 图像取差集 imagesub = imagesub1 - imagesub2
    imgsub1 = draw_con_img.copy()
    imgsub2 = preprocessedimage.copy()
    # blur_image为单通道图像，需要转换为RGB图像
    imgsub2 = cv2.cvtColor(imgsub2, cv2.COLOR_GRAY2BGR)

    # 取差集
    imgsub = cv2.subtract(imgsub1, imgsub2)

    all_images.append((imgsub, "sub"))
    # 显示取差集后图像
    cv_show("image substraction", imgsub)

    # 取差集后对图像进行闭运算操作,填补小孔洞
    # 定义矩形结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
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

    hole_size_list = cal_all_hole_size(contours)
    print("holes size:", hole_size_list)

    areas = []
    for idx in range(len(contours)):
        areas.append(cv2.contourArea(contours[idx]))

    # 在差集图像中画出轮廓
    image_with_cons = cv2.drawContours(image_find_contours, contours, -1, (51, 153, 255), 1)

    all_images.append((image_with_cons, "image_with_cons"))

    cv_show("image_with_cons: ", image_with_cons)

    # 对图像进行剪裁和旋转至水平
    width = int(rect[1][0])  # 图像的宽度
    height = int(rect[1][1])  # 图像的高度
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image_with_cons, M, (width, height))

    cv_show("cropped and rotated! ", warped)
    all_images.append((warped, "wrop&rotate"))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    all_images.append((warped_gray, "gray"))


    ###################################占空比与最小生成树检测缺陷#########################################
    # 根据窗口内的占空比
    # 改进后占空比
    arr = cal_duty_cycle_in_rect(warped_gray, int(warped_gray.shape[0] / 5))



# 对单个图像检测
def single_detection(imagepath):
    rowimg = read_an_gray_image(imagepath)
    preproimg = pre_process(rowimg)
    find_closed_holes(preproimg)


# 对文件夹下的所有图像进行检测
def dictory_detection(dictorypath):
    imgslist = read_images(dictorypath)
    # print("一共{}副图像".format(len(imgslist)))
    for i in range(len(imgslist)):
        rowimg = imgslist[i]
        preproimg = pre_process(rowimg)
        find_closed_holes(preproimg)



# main函数
def main():
    # 对一幅堵孔图像处理
    single_detection(filepath)
    cv2.waitKey(0)
    # 对文件夹下所有存在堵孔的图像进行检测
    # dictory_detection(sunken_img_dictoryname)

    # 对文件下所有正常图像进行检测d
    # dictory_detection(unsunken_img_dictoryname)


if __name__ == "__main__":
    main()
