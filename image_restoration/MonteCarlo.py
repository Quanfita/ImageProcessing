from cmath import sqrt
from email.mime import image
from turtle import distance
import numpy as np
import cv2
import random

def sample(area,point,num):
    rand = np.random.uniform(low=0.0, high=1.0, size=area.shape[:-1])
    distance = np.sum((area - point) ** 2)**.5
    rand[np.where(rand>=.5)] = 1
    rand[np.where(rand<.5)] = 0
    rand[np.where(distance<=20)] = 1
    rand[np.where(distance>20)] = 0
    s = np.sum(rand)
    point = np.sum(area * rand.reshape([*area.shape[:-1],1])) / s
    return point

def monte_carlo(image):
    alpha = .5
    r = 50
    num = 10
    h,w = image.shape[:-1]
    # test_map = np.zeros([h,w],dtype=np.uint8)
    sample_map = np.zeros([h,w,3],dtype=np.uint8)
    Lm = 20
    Lmm = 100
    for i in range(h):
        for j in range(w):
            b_h,b_w = max(0,i-r),max(0,j-r)
            e_h,e_w = min(h,b_h+r),min(w,b_w+r)
            sample_map[i,j] = sample(image[b_h:e_h,b_w:e_w],image[i,j],num)
            image[i,j] = image[i,j] * (1 - alpha) + sample_map[i,j] * alpha
    cv2.imwrite('res.png',image)
    cv2.imwrite('test_map.png',sample_map)

if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(__file__)
    image = cv2.imread(os.path.join(base_dir,'res1.png'))
    monte_carlo(image)