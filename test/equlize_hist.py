'''
  -*- coding: utf-8 -*-
  @Time    : 2020/10/17 14:30
  @Author  : liuxu
  @File    : equlize_hist.py
  @Software: PyCharm
  @Theme   : 
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

src_img = "./light_images/light_1.JPG"

# 设置图像的宽和高
IMAGE_WIDTH = 650
IMAGE_WIDTH = 650


# 使用plt显示一张图片
def plt_show_one_pic(title, pic):
    temp_img = pic
    title = title
    plt.imshow(temp_img)
    plt.title(title, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# 使用opencv显示图像
def cv_show(name, img):
    cv2.imshow(name, img)


#
#
#
# img = cv2.imread(src_img,0)
# img = cv2.resize(img,(IMAGE_WIDTH,IMAGE_WIDTH))
# plt_show_one_pic("original image",img)
# cv_show("original image",img)

import logging
import numpy as np


# Homomorphic filter class
class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.

    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H

        .
    """

    def __init__(self, a=0.5, b=1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = 1 / (1 + (Duv / filter_params[0] ** 2) ** filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = np.exp((-Duv / (2 * (filter_params[0]) ** 2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b * H) * I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H=None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter == 'butterworth':
            H = self.__butterworth_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'gaussian':
            H = self.__gaussian_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt)) - 1
        return np.uint8(I)


# End of class HomomorphicFilter

if __name__ == "__main__":
    import cv2

    # Code parameters
    path_out = ''
    img_path = src_img

    # Derived code parameters
    img_path_in = img_path
    img_path_out = path_out + 'filtered.png'

    # Main code
    img = cv2.imread(img_path_in)[:, :, 0]
    homo_filter = HomomorphicFilter(a=0.75, b=1.25)
    img_filtered = homo_filter.filter(I=img, filter_params=[30, 2])
    cv2.imwrite(img_path_out, img_filtered)

#
# print(img.shape)
#
#
#
# def homomorphic_filter(src, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
#     gray = src.copy()
#     if len(src.shape) > 2:
#         gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#     gray = np.float64(gray)
#     rows, cols = gray.shape
#
#     gray_fft = np.fft.fft2(gray)
#     gray_fftshift = np.fft.fftshift(gray_fft)
#     dst_fftshift = np.zeros_like(gray_fftshift)
#     M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
#     D = np.sqrt(M ** 2 + N ** 2)
#     Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
#     dst_fftshift = Z * gray_fftshift
#     dst_fftshift = (h - l) * dst_fftshift + l
#     dst_ifftshift = np.fft.ifftshift(dst_fftshift)
#     dst_ifft = np.fft.ifft2(dst_ifftshift)
#     dst = np.real(dst_ifft)
#     dst = np.uint8(np.clip(dst, 0, 255))
#     return dst
#
#
#
# dst = homomorphic_filter(img)
# cv_show("dst",dst)
# plt_show_one_pic("dst image",dst)
#
#
#
#
# img = dst
#
# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# # for i in range(4):
# #     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
# #     plt.title(titles[i])
# #     plt.xticks([]),plt.yticks([])
# # plt.show()
#
# for i in range(4):
#     cv_show(titles[i],images[i])
#
#
#
#
#
# cv2.waitKey(0)
#
# # img = cv2.medianBlur(img,5)
#
# #
# #
