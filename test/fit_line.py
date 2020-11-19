# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
import cv2



# 使用plt显示一张图片
def plt_show_one_pic(plt_title, plt_pic):
    temp_img = plt_pic
    title = plt_title
    plt.imshow(temp_img)
    plt.title(title, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.show()



# 显示图像
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

x = np.linspace(0, 30, num=50)
y = 0.2 * x + [np.random.random() for _ in range(50)]

g = 3 * x + [2*np.random.random() for _ in range(50)]

y = g
plt.plot(x,y)
# plt.show()



img_black = np.zeros((500,500),np.uint8)
print(img_black)
# cv_show("img_black",img_black)
img_black_rgb = cv2.cvtColor(img_black,cv2.COLOR_GRAY2BGR)
# cv_show("img rgb",img_black_rgb)



y_pre = k *x +b
plt.plot(x,y_pre)
plt.show()


def fit_line(data):
    loc = []
    for i in range(50):
        loc.append((data[0], data[1]))
    data = np.array(loc)
    output = cv2.fitLine(data, cv2.DIST_L2, 0, 0.01, 0.01)
    k = output[1] / output[0]
    b = output[3] - k * output[2]

    return k,b
