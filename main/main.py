'''
  -*- coding: utf-8 -*-
  @Time    : 2020/10/31 14:29
  @Author  : liuxu
  @File    : main.py
  @Software: PyCharm
  @Theme   : 
'''

from bending_detection import bending_detect
from fracture_detection import fracture_detect
from closed_holes_detection import closed_holes_detect
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math



def main():
    bending_detect.main()
    fracture_detect.main()
    closed_holes_detect.main()


if __name__ == '__main__':
    main()



