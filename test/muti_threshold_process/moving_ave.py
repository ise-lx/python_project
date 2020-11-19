import cv2
import numpy as np
from scipy.signal import lfilter

N = 10
b = 0.5

def max_min_value_filter(image, ksize=3, mode=1):
    img = image.copy()
    rows, cols = img.shape
    # if channels == 3:
    #     img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    padding = (ksize-1) // 2
    new_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
    for i in range(rows):
        for j in range(cols):
            roi_img = new_img[i:i+ksize, j:j+ksize].copy()
            min_val, max_val, min_index, max_index = cv2.minMaxLoc(roi_img)
            if mode == 1:
                img[i, j] = max_val
            elif mode == 2:
                img[i, j] = min_val
            else:
                raise Exception("please Select a Mode: max(1) or min(2)")

    return img


def movingthreshold(f, n, k):
    shape = f.shape
    assert n >= 1
    assert 0 < k < 1
    f[1:-1:2, :] = np.fliplr(f[1:-1:2, :])
    f = f.flatten()
    maf = np.ones(n) / n
    res_filter = lfilter(maf, 1, f)
    g = np.array(f > k * res_filter).astype(int)
    g = g.reshape(shape)
    g[1:-1:2, :] = np.fliplr(g[1:-1:2, :])
    g = g * 255

    # max value filter
    # g = max_min_value_filter(g, 3, 2)
    # cv2.blur(g, (3, 3))

    return g


img = cv2.imread('../light_images/light_1.JPG', 0)
res = movingthreshold(img, N, b)
cv2.imwrite('../light_images/light_1_results.jpg', res)


