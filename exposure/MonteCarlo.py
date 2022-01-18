from cmath import sqrt
from email.mime import image
import numpy as np
import cv2
import random

def sample(r, num):
    kernel = np.zeros([2*r+1,2*r+1],dtype=np.float32)
    x = random.sample(range(2*r+1), num)
    y = random.sample(range(2*r+1), num)
    for item in zip(x,y):
        kernel[item[0],item[1]] = abs(r - item[0]) + abs(r - item[1])
    kernel *= 1 / np.sum(kernel)
    return kernel

def distance(a,b):
    d = (a - b)**2
    return np.sum(d)**.5

def monte_carlo(image):
    alpha = .5
    r = 50
    num = 10
    kernel = sample(r, num)
    h,w = image.shape[:-1]
    test_map = np.zeros([h,w],dtype=np.uint8)
    tmp = cv2.filter2D(image, -1, kernel)
    Lm = 20
    Lmm = 100
    for i in range(h):
        for j in range(w):
            if distance(image[i,j],tmp[i,j]) > Lm:
                image[i,j] = (1 - alpha)*image[i,j] + alpha*tmp[i,j]
                test_map[i,j] = 255
    test_map = cv2.blur(test_map,(15,15))
    test_map[np.where(test_map>=64)] = 255
    test_map[np.where(test_map<64)] = 0
    cv2.imwrite('tmp.png',tmp)
    cv2.imwrite('diff.png',np.sum(np.abs(image-tmp),axis=-1)/3)
    cv2.imwrite('res.png',image)
    cv2.imwrite('test_map.png',test_map)

if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(__file__)
    image = cv2.imread(os.path.join(base_dir,'res1.png'))
    monte_carlo(image)