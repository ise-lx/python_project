import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os

file = open('holes_size.txt','r')
line = file.readline()
data = []
while line:
    line = line.strip('\n')
    fline = float(line)
    temp = round(fline,2)
    # temp = int(fline)
    if temp > 1:
        data.append(temp)
    line = file.readline()

file.close()

print("data:")
print(data)



plt.hist(data, edgecolor='k', alpha=0.35,bins=25) # 设置直方边线颜色为黑色，不透明度为 0.35
plt.show()
