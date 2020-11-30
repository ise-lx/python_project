# 检测堵孔缺陷加入最小生成树
'''
@Author : liuxu
@Date : 2020/11/26
'''
import cv2
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import datetime

# 待检测图像路径
filepath = "./closed_holes_images/closed_hole.JPG"
# filepath = "./unclosed_holes_images/1.jpg"

# 设置图像的宽和高
IMAGE_WIDTH = 750
IMAGE_HEIGHT = 750

# 所有图像

all_images = []


# coding=utf-8
class Graph(object):
    def __init__(self, maps):
        self.maps = maps
        self.nodenum = self.get_nodenum()
        self.edgenum = self.get_edgenum()

    def get_nodenum(self):
        return len(self.maps)

    def get_edgenum(self):
        count = 0
        for i in range(self.nodenum):
            for j in range(i):
                if self.maps[i][j] > 0 and self.maps[i][j] < 9999:
                    count += 1
        return count

    def kruskal(self):
        res = []
        if self.nodenum <= 0 or self.edgenum < self.nodenum - 1:
            return res
        edge_list = []
        for i in range(self.nodenum):
            for j in range(i, self.nodenum):
                if self.maps[i][j] < 9999:
                    edge_list.append([i, j, self.maps[i][j]])  # 按[begin, end, weight]形式加入
        edge_list.sort(key=lambda a: a[2])  # 已经排好序的边集合

        group = [[i] for i in range(self.nodenum)]
        for edge in edge_list:
            for i in range(len(group)):
                if edge[0] in group[i]:
                    m = i
                if edge[1] in group[i]:
                    n = i
            if m != n:
                res.append(edge)
                group[m] = group[m] + group[n]
                group[n] = []
        return res

    def prim(self):
        res = []
        if self.nodenum <= 0 or self.edgenum < self.nodenum - 1:
            return res
        res = []
        seleted_node = [0]
        candidate_node = [i for i in range(1, self.nodenum)]

        while len(candidate_node) > 0:
            begin, end, minweight = 0, 0, 9999
            for i in seleted_node:
                for j in candidate_node:
                    if self.maps[i][j] < minweight:
                        minweight = self.maps[i][j]
                        begin = i
                        end = j
            res.append([begin, end, minweight])
            seleted_node.append(end)
            candidate_node.remove(end)
        return res


# 使用plt显示一张图片
def plt_show_one_pic(pic):
    temp_img = pic[0]
    title = pic[1]
    plt.imshow(temp_img)
    plt.title(title, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# 显示图像
def cv_show(name, img):
    cv2.imshow(name, img)
    # name=name
    # cv2.waitKey(0)


# 对图像resize处理
def row_image_resize(image, width, height):
    # image输入图像，width，height 期望得到的宽和高
    image = cv2.resize(image, (width, height))
    # print("resized image:", image.shape)
    return image


# #图像预处理函数
def pre_process(rowimage):
    # 显示原图
    # cv_show("row image", rowimage)
    grayimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2GRAY)  # 灰度化
    all_images.append((grayimage, "grayimage"))  # 填入列表
    resizedimage = row_image_resize(grayimage, IMAGE_WIDTH, IMAGE_HEIGHT)  # resize
    all_images.append((resizedimage, "resizedimage"))  # 填入列表
    # cv_show("resizeimg", resizedimage)
    # 运用大津算法进行图像分割
    ret, im_th = cv2.threshold(resizedimage, 0, 255, cv2.THRESH_OTSU)
    # cv_show("image after OTSU", im_th)
    all_images.append((im_th, "OTSU"))
    return im_th



# 不使用for循环求两个矩阵的欧氏距离
def compute_distances_no_loops(mat_1,mat2):
    dists = np.sqrt(-2*np.dot(mat_1, mat2.T) + np.sum(np.square(mat2), axis = 1) +
                    np.transpose([np.sum(np.square(mat_1), axis = 1)]))
    return dists

# 对图像进行旋转和裁剪操作
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

    # cv_show("cropped and rotated! ", warped)
    all_images.append((warped, "wrop&rotate"))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return warped_gray


# 计算两个轮廓之间的最小距离
def cal_two_contours_distance(cont1, cont2):
    min_distance = 100000
    cont1_sq = np.squeeze(cont1)
    cont2_sq = np.squeeze(cont2)
    dismat = compute_distances_no_loops(cont1_sq,cont2_sq)
    dist = np.min(dismat)
    return dist


# 窗口的内的所有轮廓计算距离矩阵
def distance_matrix(contours):
    # 先构造一个空的正方形矩阵(方阵)
    dis_mat = np.ones((len(contours), len(contours))) * 9999

    # 对角线值为0
    for i in range(len(contours)):
        dis_mat[i][i] = 0

    for i in range(int(len(contours) - 1)):
        cont1 = contours[i]
        for j in range(i+1, len(contours)):
            cont2 = contours[j]
            temp_distance = cal_two_contours_distance(cont1, cont2)  # 计算两个轮廓之间的最小距离
            dis_mat[i][j] = temp_distance
            dis_mat[j][i] = temp_distance

    return dis_mat


# 根据距离矩阵生成最小生成树,分为kruskal算法和prim算法两种
def mstreegeneratefrommat(dis_mat):
    graph = Graph(dis_mat)
    graph_kruskal = graph.kruskal()
    # graph_prim = graph.prim()
    print("最小生成树kruskal算法")
    print(graph_kruskal)
    # print("最小生成树prim算法")
    # print(graph_prim)
    return graph_kruskal
    # return graph_prim


# 改进后的占空比计算函数
def cal_duty_cycle_in_rect(cropped_rotated_image, edge_length):
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

    # cv_show("mini:", mini_img)
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
            window_contours, _ = cv2.findContours(temp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print("删除前：", len(window_contours))
            window_contours = [cont for cont in window_contours if cv2.contourArea(cont) > 2]
            print("删除后：", len(window_contours))

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

            if temp_rate < 0.07:
                # mini_img_with_rect = cv2.drawContours(mini_img_rgb, [box], 0, (0, 0, 255), 1)
                dis_mat = distance_matrix(window_contours)
                # print("邻接矩阵：\n", dis_mat)
                tree = mstreegeneratefrommat(dis_mat)
                # print("最小生成树为：\n", tree)
                weights_sum_value = 0.
                for i in range(len(tree)):
                    weights_sum_value += tree[i][2]
                # print("权重的和为：", weights_sum_value)
                weights_sum_value_div_n = weights_sum_value / len(window_contours)
                if weights_sum_value_div_n > 7.8:
                    cv2.drawContours(white_img, [box], 0, (255, 255, 255), cv2.FILLED)
                # print("weights_sum_value_div_n=", weights_sum_value_div_n)
                # print("weights_sum/num=", weights_sum_value / len(window_contours))

                # cv_show("mini_img_with_rect: ", mini_img_with_rect)
                # print("轮廓的数量：", len(window_contours))

    # 把检测的滑动窗口用轮廓算法连接起来
    contours, _ = cv2.findContours(white_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(white_img_rgb,contours,-1,(0,0,255),thickness=1)
    # cv2.imshow("contours",white_img_rgb)

    cv2.drawContours(mini_img_rgb, contours, -1, (0, 0, 255), thickness=2)
    # cv2.imshow("contours", mini_img_rgb)

    print("min_value_rate: ", min_value_rate)
    if min_value_rate < 0.05:
        print("该过滤网存在堵孔缺陷")
    else:
        print("该过滤网不存在堵孔缺陷")

    cv2.imshow("white with rect", white_img)

    return rate_arr


# 找出区域
def detect_closed_holes(preprocessedimage):
    # 在分割后的图像找轮廓

    contours, hierarchy = cv2.findContours(preprocessedimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    # all_images.append((img_with_rect, "img_with_rect"))  # 填入列表

    # cv_show("image with rect", img_with_rect)

    new_img_with_rect = crap_and_rotate(img_with_rect, rect)
    # cv_show("new_img_with_rect", new_img_with_rect)

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
    # cv_show("image substraction", imgsub)

    # 取差集后对图像进行闭运算操作,填补小孔洞

    # 定义矩形结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closedimg = cv2.morphologyEx(imgsub, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv_show("image after close operation", closedimg)

    # 在取差集后的图像找轮廓
    image_find_contours = closedimg.copy()
    # 二值化,要用到阈值函数cv2.threshold()
    # cv2.findContours()要输入二值图像

    # 灰度化
    image_find_contours_gray = cv2.cvtColor(image_find_contours, cv2.COLOR_BGR2GRAY)
    ret, image_find_contours_bin = cv2.threshold(image_find_contours_gray, 0, 255, cv2.THRESH_OTSU)

    # 在二值图像找出所有轮廓,  过滤网中所有的孔洞的轮廓
    contours, hierarchy = cv2.findContours(image_find_contours_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for idx in range(len(contours)):
        areas.append(cv2.contourArea(contours[idx]))

    # 在差集图像中画出轮廓
    image_with_cons = cv2.drawContours(image_find_contours, contours, -1, (51, 153, 255), 1)

    # all_images.append((image_with_cons, "image_with_cons"))

    # cv_show("image_with_cons: ", image_with_cons)

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

    # cv_show("cropped and rotated! ", warped)
    # all_images.append((warped, "wrop&rotate"))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # all_images.append((warped_gray, "gray"))

    # 检测缺陷
    # 根据窗口内的占空比
    arr = cal_duty_cycle_in_rect(warped_gray, int(warped_gray.shape[0] / 5))


# 对单个图像检测
def single_detection(imagepath):
    rowimg = cv2.imread(imagepath)
    preproimg = pre_process(rowimg)
    detect_closed_holes(preproimg)


def main():
    starttime = datetime.datetime.now()
    single_detection(filepath)
    endtime = datetime.datetime.now()
    costtime = (endtime - starttime).seconds
    print("耗费时间：{}s".format(costtime))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
