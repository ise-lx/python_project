import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# 占空比文件
file = open('normal_duty_list.txt','r')
# file = open('unnormal_duty_list.txt','r')
# 最小生成树文件

# file = open('normal_tree_list.txt','r')
line = file.readline()
data = []
while line:
    line = line.strip('\n')
    fline = float(line)
    temp = round(fline,3)

    data.append(temp)
    line = file.readline()

file.close()



print("data:")
print("data length:",len(data))
print(data)

ndata = np.array(data)
print("numpy:",ndata)
print("mean value:",ndata.mean())
print("std value:",ndata.std(ddof=1))


plt.hist(data,color='#242424', alpha=0.35,bins=20) # 设置直方边线颜色为黑色，不透明度为 0.35
plt.xlabel("最小生成树权值")
plt.ylabel("频数")
plt.show()
