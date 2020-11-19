'''
  -*- coding: utf-8 -*-
  @Time    : 2020/10/19 23:37
  @Author  : liuxu
  @File    : decision_tree.py
  @Software: PyCharm
  @Theme   : 
'''

import math
def cal_ent(value1):
    value2 = abs(1 - value1)
    ent = value1*math.log(value1,2) + value2*math.log(value2,2)
    return -ent



value = 2/3
print(cal_ent(value))


