import scipy.io as scio
import numpy as np


label_5 = './label_5.mat'
speed1500 = './speed1500_0y1.mat'
speed1800 = './speed1800_0y1.mat'

data = scio.loadmat(label_5)
speed1500 = scio.loadmat(speed1500)
print(type(data))
print(data)
label = data['label']
print(label)
print("type label: ",type(label))
print("label shape: ",label.shape)
speed1500_0y1 = speed1500['speed1500_0y1']
print(speed1500_0y1)

print("speed1500 shape: ",speed1500_0y1.shape)
