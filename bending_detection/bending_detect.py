import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

filepath = "./bending_image/bianxing_test.jpg"
# filepath = "./bending_image/test3.jpg"
#
# filepath = "./bending_image/test2.JPG"
# filepath = "./bending_image/light_test12.jpg"
# filepath = "./bending_image/normal.jpg" 从
# bending_img_dictoryname = "./bending_images/"
# unbending_img_dictoryname = "./unbending_images"

# 设置图像的宽和高
IMAGE_WIDTH = 650
IMAGE_HEIGHT = 650

# 存放处理过程的所有图像的列表,以(img,title)元组方式添加
all_images = []


# 使用plt显示一张图片
def plt_show_one_pic(plt_title, plt_pic):
    temp_img = plt_pic
    title = plt_title
    plt.imshow(temp_img)
    plt.title(title, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    # plt.show()            # 控制是否显示


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
def read_an_gray_image(filepath):
    return cv2.imread(filepath)


# 读入文件夹下的所有图像,返回一个路径的列表,(相对路径)
def read_images(dictoryname):
    dicnames = []
    for imagename in os.listdir(dictoryname):
        path = dictoryname + "/" + imagename
        image = cv2.imread(path)
        dicnames.append(image)

    return dicnames


# sobel边缘检测
def sobel_edge(img):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad


# 在光照不均匀条件下,全局阈值分割和局部阈值分割都效果不好,所以选用基于边缘检测的图像分割
def based_edge_detection_preprocess(original_img):
    # 先边缘检测再二值化
    # 边缘检测
    edge_img = sobel_edge(original_img)
    all_images.append(("edge img", edge_img))
    ret, im_th = cv2.threshold(edge_img, 0, 255, cv2.THRESH_OTSU)
    all_images.append(("im th", im_th))
    # 闭运算填补细小裂缝

    # 定义矩形结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    close_img = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel, iterations=1)

    return close_img


# #图像预处理函数
def pre_process(rowimage):
    # 显示原图
    # plt_show_one_pic("original image", rowimage)
    all_images.append(("orignial image", rowimage))  # 方便以后显示
    grayimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2GRAY)  # 灰度化
    all_images.append(("gray image", grayimage))

    # plt_show_one_pic("gray image", grayimage)
    resizedimage = row_image_resize(grayimage, IMAGE_WIDTH, IMAGE_HEIGHT)  # resize

    cv_show("resizeimg",resizedimage)


    # 运用大津算法进行图像分割
    ret, im_th = cv2.threshold(resizedimage, 0, 255, cv2.THRESH_OTSU)
    # cv2.imwrite("./unbending_image/im_th.jpg",im_th)

    all_images.append(("im_th", im_th))
    # plt_show_one_pic("OTSU", im_th)
    # plt.show()
    # #中值滤波去除椒盐噪声,卷积核为5x5时
    # blur_image = cv2.medianBlur(im_th, 3)
    # 显示中值滤波处理后图像
    # cv_show("image after medianBlur", blur_image)

    return im_th


# 计算最小外接矩形的四个角的坐标距离轮廓的大小
# 传入轮廓和外接矩形的四个坐标
def cal_distance(img, max_cont, coord_1, coord_2, coord_3, coord_4):
    # 初始化四个距离,取一个较大的数
    global point_1, point_2, point_3, point_4
    coord_1_min_distance = 100000.0
    coord_2_min_distance = 100000.0
    coord_3_min_distance = 100000.0
    coord_4_min_distance = 100000.0
    # print("四个坐标：", coord_1, coord_2, coord_3, coord_4)

    for con_point in max_cont:  # 计算coor_1到max_cont的最短距离
        point_width = con_point[0][0]
        point_height = con_point[0][1]

        # 临时距离
        coord_1_temp_distance = float(np.sqrt((point_width - coord_1[0]) ** 2 + (point_height - coord_1[1]) ** 2))
        coord_2_temp_distance = float(np.sqrt((point_width - coord_2[0]) ** 2 + (point_height - coord_2[1]) ** 2))
        coord_3_temp_distance = float(np.sqrt((point_width - coord_3[0]) ** 2 + (point_height - coord_3[1]) ** 2))
        coord_4_temp_distance = float(np.sqrt((point_width - coord_4[0]) ** 2 + (point_height - coord_4[1]) ** 2))

        # 找到四个角点的坐标
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

    img = cv2.circle(img, (point_1[0][0], point_1[0][1]), 3, (0, 255, 0), 2)
    img = cv2.circle(img, (point_2[0][0], point_2[0][1]), 3, (0, 255, 0), 2)
    img = cv2.circle(img, (point_3[0][0], point_3[0][1]), 3, (0, 255, 0), 2)
    img = cv2.circle(img, (point_4[0][0], point_4[0][1]), 3, (0, 255, 0), 2)
    # print("在轮廓中四个点的坐标为(1-右下；2左下；3-左上；4-右上)：")

    # print(point_1)
    # print(point_2)
    # print(point_3)
    # print(point_4)
    # cv_show("img with circle:",img)
    # plt_show_one_pic("img with 4 points",img)
    return (point_1, point_2, point_3, point_4)


# 找出过滤网的轮廓
def find_contours(preprocessedimage):
    # 画出样品的轮廓
    precessed_img = preprocessedimage.copy()
    black_img = np.ones_like(preprocessedimage)  # 创建一个同大小的(0,0,0)黑色图像
    white_img = black_img * 255  # 创建一个白色图
    white_img_rgb = cv2.cvtColor(white_img, cv2.COLOR_GRAY2BGR)
    precessed_img_rgb = cv2.cvtColor(preprocessedimage, cv2.COLOR_GRAY2BGR)
    # 在二值图像找出所有轮廓
    contours, hierarchy = cv2.findContours(precessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("contours:",contours)

    areas = []
    for idx in range(len(contours)):
        areas.append(cv2.contourArea(contours[idx]))

    # 最大面积的下标
    max_idx = np.argmax(np.array(areas))

    print("max area findContours:", max(areas))
    M = cv2.moments(contours[max_idx])
    print("max area moments:", M['m00'])
    # 最大轮廓
    max_cnt = contours[max_idx]
    # print("contour: ",max_cnt)
    # print("type contour: ",type(max_cnt))
    # print("shape contour: ",max_cnt.shape)
    img_with_max_con = cv2.drawContours(white_img_rgb, contours, max_idx, (0, 0, 0), 1)
    cv2.imwrite("./unbending_image/img_with_max_con.jpg", img_with_max_con)
    all_images.append(("image contours", img_with_max_con))
    # plt_show_one_pic("img with max con", img_with_max_con)

    # cv_show("img with max con", img_with_max_con)
    plt_show_one_pic("img with max con", img_with_max_con)
    # print("black img shape: ",black_img.shape)
    # cv_show("black img",black_img)
    # print("max cont",max_cnt)
    # print("type cont",type(max_cnt))
    # print("cont shape: ",max_cnt.shape)

    # 最小外接矩形
    rect = cv2.minAreaRect(max_cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cord_1 = box[0]
    cord_2 = box[1]
    cord_3 = box[2]
    cord_4 = box[3]
    # img_with_rect = cv2.drawContours(white_img_rgb, [box], 0, (0, 255, 0), 1)
    img_with_rect = white_img_rgb
    # cv_show("img_with_rect", img_with_rect)
    point_1, point_2, point_3, point_4 = cal_distance(img_with_rect, max_cnt, cord_1, cord_2, cord_3, cord_4)
    # cv_show("with points",img_with_cons_points)
    line1 = find_four_edges_contours(img_with_max_con, max_cnt, (point_1, point_2, point_3, point_4))


# 找到四个边所对应的轮廓，并计算出夹角
def find_four_edges_contours(img, cnt, points):
    draw_img = img.copy()
    cv_show("before draw a line", img)

    [rows, cols, vals] = cnt.shape
    print("cnt shape", cnt.shape)
    cord_1 = points[0][0]
    cord_2 = points[1][0]
    cord_3 = points[2][0]
    cord_4 = points[3][0]
    # print("cord1:",cord_1)
    # print("cord2:",cord_2)
    # print("cord3:",cord_3)
    # print("cord4:",cord_4)
    # cv_show("black img",black_img)

    x1 = [cord_1[0], cord_2[0]]  # 第一条边x,y
    y1 = [cord_1[1], cord_2[1]]

    # 二条边x,y
    x2 = [cord_2[0], cord_3[0]]
    y2 = [cord_2[1], cord_3[1]]
    # 三条边x,y
    x3 = [cord_3[0], cord_4[0]]
    y3 = [cord_3[1], cord_4[1]]
    # 四条边x,y
    x4 = [cord_4[0], cord_1[0]]
    y4 = [cord_4[1], cord_1[1]]

    # 从小到大排序
    x1.sort(), y1.sort(), x2.sort(), y2.sort(), x3.sort()
    y3.sort(), x4.sort(), y4.sort()

    # print("x1",x1)
    # print("y1",y1)
    #
    # print("x2",x2)
    # print("y2",y2)
    #
    # print("x3",x3)
    # print("y3",y3)
    #
    # print("x4",x4)
    # print("y4",y4)

    # 第一条边,要用到x
    # 直线：下

    line_1 = []

    # 判断x与y哪个跨度大
    x1_len = math.fabs(x1[0] - x1[1])
    y1_len = math.fabs(y1[0] - y1[1])
    if x1_len > y1_len:
        for index_x in range(x1[0], x1[1]):
            temp_ys = []  # index_x在轮廓的点集上对应的所有y值的列表
            for i in range(rows):
                temp_x = cnt[i][0][0]
                temp_y = cnt[i][0][1]
                if index_x == temp_x:
                    temp_ys.append(temp_y)

            try:
                value_y = max(temp_ys)
                if (value_y > int(y1[0] - 10)) & (value_y < int(y1[1] + 10)):
                    line_1.append([index_x, value_y])
            except:
                pass
        np_line_1 = np.array(line_1)
        # print("np_line_1：",np_line_1)
        line1_k, line1_b = fit_line(np_line_1)
        # print("line1 k:",line1_k)
        # print("line1 b:",line1_b)

        x_1 = [i for i in range(x1[0], x1[1])]
        y_1_pre = line1_k * x_1 + line1_b
        x_1_begin = x_1[0]
        y_1_begin = y_1_pre[0]
        x_1_end = x_1[-1]
        y_1_end = y_1_pre[-1]

        cv2.line(draw_img, (int(x_1_begin), int(y_1_begin)), (int(x_1_end), int(y_1_end)), (0, 0, 255), thickness=1)
        # cv_show("line1",draw_img)

        plt.plot(x_1, y_1_pre)
        # plt.show()            # 显示

    else:  # 当y_len大于x_len时

        line_1 = []
        # 从头开始的话,x对应的y比较多
        for index_y in range(y1[0], y1[1]):
            temp_xs = []  # index_x在轮廓的点集上对应的所有y值的列表
            for i in range(rows):
                temp_x = cnt[i][0][0]
                temp_y = cnt[i][0][1]
                if index_y == temp_y:
                    temp_xs.append(temp_x)

            try:
                value_x = max(temp_xs)
                if (value_x < int(x1[1] + 10)) & (value_x > int(x1[0] - 10)):
                    line_1.append([value_x, index_y])
            except:
                pass
        np_line_1 = np.array(line_1)
        # print("line_2:",np_line_2)
        line1_k, line1_b = fit_line(np_line_1)
        # print("line2 k:",line2_k)
        # print("line2 b:",line2_b)

        # degree = cal_line_angle(line1_k,line2_k)
        # print("两条直线的夹角是：",degree)

        y_1 = [i for i in range(y1[0], y1[1])]

        if line1_k == 0:
            x_1_pre = -line1_b
        else:
            x_1_pre = (y_1 - line1_b) / line1_k

        x_1_begin = x_1_pre[0]
        y_1_begin = y_1[0]
        x_1_end = x_1_pre[-1]
        y_1_end = y_1[-1]

        cv2.line(draw_img, (int(x_1_begin), int(y_1_begin)), (int(x_1_end), int(y_1_end)), (0, 0, 255), thickness=1)

        plt.plot(x_1_pre, y_1)
        # plt.show()

    # 第二条边,要用到x
    # 直线：下
    line_2 = []

    # 判断x与y哪个跨度大
    x2_len = math.fabs(x2[0] - x2[1])
    y2_len = math.fabs(y2[0] - y2[1])
    if x2_len > y2_len:
        for index_x in range(x2[0], x2[1]):
            temp_ys = []  # index_x在轮廓的点集上对应的所有y值的列表
            for i in range(rows):
                temp_x = cnt[i][0][0]
                temp_y = cnt[i][0][1]
                if index_x == temp_x:
                    temp_ys.append(temp_y)

            try:
                value_y = max(temp_ys)
                if (value_y > int(y2[0] - 10)) & (value_y < int(y2[1] + 10)):
                    line_2.append([index_x, value_y])
            except:
                pass
        np_line_2 = np.array(line_2)
        # print("np_line_1：",np_line_1)
        line2_k, line2_b = fit_line(np_line_2)
        # print("line1 k:",line1_k)
        # print("line1 b:",line1_b)

        x_2 = [i for i in range(x2[0], x2[1])]
        y_2_pre = line2_k * x_2 + line2_b

        x_2_begin = x_2[0]
        y_2_begin = y_2_pre[0]
        x_2_end = x_2[-1]
        y_2_end = y_2_pre[-1]

        cv2.line(draw_img, (int(x_2_begin), int(y_2_begin)), (int(x_2_end), int(y_2_end)), (0, 0, 255), thickness=1)
        plt.plot(x_2, y_2_pre)
        # plt.show()            # 显示

    else:  # 当y_len大于x_len时

        line_2 = []
        # 从头开始的话,x对应的y比较多
        for index_y in range(y2[0], y2[1]):
            temp_xs = []  # index_x在轮廓的点集上对应的所有y值的列表
            for i in range(rows):
                temp_x = cnt[i][0][0]
                temp_y = cnt[i][0][1]
                if index_y == temp_y:
                    temp_xs.append(temp_x)

            try:
                value_x = max(temp_xs)
                if (value_x < int(x2[1] + 10)) & (value_x > int(x2[0] - 10)):
                    line_2.append([value_x, index_y])
            except:
                pass
        np_line_2 = np.array(line_2)
        # print("line_2:",np_line_2)
        line2_k, line2_b = fit_line(np_line_2)
        # print("line2 k:",line2_k)
        # print("line2 b:",line2_b)

        # degree = cal_line_angle(line1_k,line2_k)
        # print("两条直线的夹角是：",degree)

        y_2 = [i for i in range(y2[0], y2[1])]

        if line2_k == 0:
            x_2_pre = -line2_b
        else:
            x_2_pre = (y_2 - line2_b) / line2_k

        x_2_begin = x_2_pre[0]
        y_2_begin = y_2[0]
        x_2_end = x_2_pre[-1]
        y_2_end = y_2[-1]

        cv2.line(draw_img, (int(x_2_begin), int(y_2_begin)), (int(x_2_end), int(y_2_end)), (0, 0, 255), thickness=1)
        plt.plot(x_2_pre, y_2)
        # plt.show()

    # 第三条边,要用到x
    # 直线：下
    line_3 = []

    # 判断x与y哪个跨度大
    x3_len = math.fabs(x3[0] - x3[1])
    y3_len = math.fabs(y3[0] - y3[1])
    if x3_len > y3_len:
        for index_x in range(x3[0], x3[1]):
            temp_ys = []  # index_x在轮廓的点集上对应的所有y值的列表
            for i in range(rows):
                temp_x = cnt[i][0][0]
                temp_y = cnt[i][0][1]
                if index_x == temp_x:
                    temp_ys.append(temp_y)

            try:
                value_y = max(temp_ys)
                if (value_y > int(y3[0] - 10)) & (value_y < int(y3[1] + 10)):
                    line_3.append([index_x, value_y])
            except:
                pass
        np_line_3 = np.array(line_3)
        # print("np_line_1：",np_line_1)
        line3_k, line3_b = fit_line(np_line_3)
        # print("line1 k:",line1_k)
        # print("line1 b:",line1_b)

        x_3 = [i for i in range(x3[0], x3[1])]
        y_3_pre = line3_k * x_3 + line3_b

        x_3_begin = x_3[0]
        y_3_begin = y_3_pre[0]
        x_3_end = x_3[-1]
        y_3_end = y_3_pre[-1]

        cv2.line(draw_img, (int(x_3_begin), int(y_3_begin)), (int(x_3_end), int(y_3_end)), (0, 0, 255), thickness=1)

        plt.plot(x_3, y_3_pre)
        # plt.show()            # 显示

    else:  # 当y_len大于x_len时

        line_3 = []
        # 从头开始的话,x对应的y比较多
        for index_y in range(y3[0], y3[1]):
            temp_xs = []  # index_x在轮廓的点集上对应的所有y值的列表
            for i in range(rows):
                temp_x = cnt[i][0][0]
                temp_y = cnt[i][0][1]
                if index_y == temp_y:
                    temp_xs.append(temp_x)

            try:
                value_x = max(temp_xs)
                if (value_x < int(x3[1] + 10)) & (value_x > int(x3[0] - 10)):
                    line_3.append([value_x, index_y])
            except:
                pass
        np_line_3 = np.array(line_3)
        # print("line_2:",np_line_2)
        line3_k, line3_b = fit_line(np_line_3)
        # print("line2 k:",line2_k)
        # print("line2 b:",line2_b)

        # degree = cal_line_angle(line1_k,line2_k)
        # print("两条直线的夹角是：",degree)

        y_3 = [i for i in range(y3[0], y3[1])]

        if line3_k == 0:
            x_3_pre = -line3_b
        else:
            x_3_pre = (y_3 - line3_b) / line3_k

        x_3_begin = x_3_pre[0]
        y_3_begin = y_3[0]
        x_3_end = x_3_pre[-1]
        y_3_end = y_3[-1]
        cv2.line(draw_img, (int(x_3_begin), int(y_3_begin)), (int(x_3_end), int(y_3_end)), (0, 0, 255), thickness=1)

        plt.plot(x_3_pre, y_3)
        # plt.show()

    # 第四条边,要用到x
    # 直线：下
    [rows, cols, vals] = cnt.shape
    line_4 = []

    # 判断x与y哪个跨度大
    x4_len = math.fabs(x4[0] - x4[1])
    y4_len = math.fabs(y4[0] - y4[1])
    if x4_len > y4_len:
        for index_x in range(x4[0], x4[1]):
            temp_ys = []  # index_x在轮廓的点集上对应的所有y值的列表
            for i in range(rows):
                temp_x = cnt[i][0][0]
                temp_y = cnt[i][0][1]
                if index_x == temp_x:
                    temp_ys.append(temp_y)

            try:
                value_y = max(temp_ys)
                if (value_y > int(y4[0] - 10)) & (value_y < int(y4[1] + 10)):
                    line_4.append([index_x, value_y])
            except:
                pass
        np_line_4 = np.array(line_4)
        # print("np_line_1：",np_line_1)
        line4_k, line4_b = fit_line(np_line_4)
        # print("line1 k:",line1_k)
        # print("line1 b:",line1_b)

        x_4 = [i for i in range(x4[0], x4[1])]
        y_4_pre = line4_k * x_4 + line4_b

        x_4_begin = x_4[0]
        y_4_begin = y_4_pre[0]
        x_4_end = x_4[-1]
        y_4_end = y_4_pre[-1]

        cv2.line(draw_img, (int(x_4_begin), int(y_4_begin)), (int(x_4_end), int(y_4_end)), (0, 0, 255), thickness=1)

        plt.plot(x_4, y_4_pre)
        # plt.show()            # 显示

    else:  # 当y_len大于x_len时

        [rows, cols, vals] = cnt.shape

        line_4 = []
        # 从头开始的话,x对应的y比较多
        for index_y in range(y4[0], y4[1]):
            temp_xs = []  # index_x在轮廓的点集上对应的所有y值的列表
            for i in range(rows):
                temp_x = cnt[i][0][0]
                temp_y = cnt[i][0][1]
                if index_y == temp_y:
                    temp_xs.append(temp_x)

            try:
                value_x = max(temp_xs)
                if (value_x < int(x4[1] + 10)) & (value_x > int(x4[0] - 10)):
                    line_4.append([value_x, index_y])
            except:
                pass
        np_line_4 = np.array(line_4)
        # print("line_2:",np_line_2)
        line4_k, line4_b = fit_line(np_line_4)
        # print("line2 k:",line2_k)
        # print("line2 b:",line2_b)

        # degree = cal_line_angle(line1_k,line2_k)
        # print("两条直线的夹角是：",degree)

        y_4 = [i for i in range(y4[0], y4[1])]

        if line4_k == 0:
            x_4_pre = -line4_b
        else:
            x_4_pre = (y_4 - line4_b) / line4_k

        x_4_begin = x_4_pre[0]
        y_4_begin = y_4[0]
        x_4_end = x_4_pre[-1]
        y_4_end = y_4[-1]

        cv2.line(draw_img, (int(x_4_begin), int(y_4_begin)), (int(x_4_end), int(y_4_end)), (0, 0, 255), thickness=1)

        cv2.imwrite("./unbending_image/lines_on_cont.jpg", draw_img)
        plt.plot(x_4_pre, y_4)
        # plt.show()

    plt.show()
    cv_show("line1,2,3,4", draw_img)

    print('*' * 10, '计算四个直线的角度', '*' * 10)
    print()
    print("四条直线的斜率：")
    print('k1:', line1_k)
    print('k2:', line2_k)
    print('k3:', line3_k)
    print('k4:', line4_k)
    degree1 = cal_line_angle(line1_k, line2_k)
    degree2 = cal_line_angle(line2_k, line3_k)
    degree3 = cal_line_angle(line3_k, line4_k)
    degree4 = cal_line_angle(line4_k, line1_k)
    print("四个角度为：")
    print(degree1)
    print(degree2)
    print(degree3)
    print(degree4)

    print('*' * 15, '结束', '*' * 15)


# 根据点最小二乘法拟合出一条曲线
def fit_line(cont_points):
    # print("cont_points:",cont_points)
    output = cv2.fitLine(cont_points, cv2.DIST_HUBER, 0, 0.01, 0.01)  # 最小二乘法
    # print("output:",output)
    k = output[1] / output[0]
    b = output[3] - k * output[2]
    # print("k: ",k)
    # print("b: ",b)

    return k, b


# 计算两个直线的夹角 ,k1,k2分别为两直线的斜率
def cal_line_angle(k1, k2):
    degree = math.fabs(np.arctan((k1 - k2) / (float(1 + k1 * k2))) * 180 / np.pi)
    return degree


# 对单个图像检测
def single_detection(imagepath):
    rowimg = read_an_gray_image(imagepath)
    preproimg = pre_process(rowimg)
    find_contours(preproimg)
    cv2.waitKey(0)

    # pre_sobelimg = based_edge_detection_preprocess(rowimg)
    # find_contours(pre_sobelimg)


# 对文件夹下的所有图像进行检测
def dictory_detection(dictorypath):
    imgslist = read_images(dictorypath)
    # imgslist = read_images(dictorypath)
    print("一共{}副图像".format(len(imgslist)))
    for i in range(len(imgslist)):
        rowimg = imgslist[i]
        preproimg = pre_process(rowimg)
        find_contours(preproimg)


# main函数
def main():
    # 对一幅图像处理
    single_detection(filepath)
    # 对文件夹下所有存在凹陷的图像进行检测
    # dictory_detection(sunken_img_dictoryname)

    # 对文件下所有正常图像进行检测
    # dictory_detection(unsunken_img_dictoryname)

    # 显示所有处理过程中的图像
    # for (title,img) in all_images:
    #     plt_show_one_pic(title,img)

    # plt.show()


if __name__ == "__main__":
    main()
