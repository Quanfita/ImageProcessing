from hashlib import sha3_224
import cv2
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import cg

import math
import sys
import os


def rotate_img(img, angle):
    row, col = img.shape
    M = cv2.getRotationMatrix2D((row / 2, col / 2), angle, 1)
    res = cv2.warpAffine(img, M, (row, col))
    return res

def get_eight_directions(l_len):    
    L = np.zeros((8, l_len, l_len))
    half_len = (l_len + 1) // 2
    for i in range(8):
        if i == 0 or i == 1 or i == 2 or i == 7:
            for x in range(l_len):
                    y = half_len - int(round((x + 1 - half_len) * math.tan(math.pi * i // 8)))
                    if y >0 and y <= l_len:
                        L[i, x, y - 1 ] = 1
            if i != 7:
                L[i + 4] = np.rot90(L[i])
    L[3] = np.rot90(L[7], 3)
    return L

# compute and get the stroke of the raw img
def get_stroke(img, ks,  dirNum):
    height , width = img.shape[0], img.shape[1]
    img = np.float32(img) / 255.0
    # img = cv2.medianBlur(img, 3)
    #print(img.shape)
    #cv2.imshow('blur', img)
    imX = np.append(np.absolute(img[:, 0 : width - 1] - img[:, 1 : width]), np.zeros((height, 1)), axis = 1)
    imY = np.append(np.absolute(img[0 : height - 1, :] - img[1 : height, :]), np.zeros((1, width)), axis = 0)
    img_gredient = np.sqrt((imX ** 2 + imY ** 2))
    # img_gredient = imX + imY

    kernel_Ref = np.zeros((ks * 2 + 1, ks * 2 + 1))
    kernel_Ref [ks, :] = 1

    response = np.zeros((dirNum, height, width))
    L = get_eight_directions(2 * ks + 1)
    for n in range(dirNum):
        ker = rotate_img(kernel_Ref, n * 180 / dirNum)
        response[n, :, :] = cv2.filter2D(img, -1, ker)

    Cs = np.zeros((dirNum, height, width))
    for x in range(width):
        for y in range(height):
            i = np.argmax(response[:,y,x])
            Cs[i, y, x] = img_gredient[y,x]

    spn = np.zeros((8, img.shape[0], img.shape[1]))

    kernel_Ref = np.zeros((2 * ks + 1, 2 * ks + 1))
    kernel_Ref [ks, :] = 1
    for n in range(width):
        if (ks - n) >= 0:
            kernel_Ref[ks  - n, :] = 1
        if (ks + n)  < ks * 2:
            kernel_Ref[ks + n, :] = 1

    kernel_Ref = np.zeros((2*ks + 1, 2 * ks + 1))
    kernel_Ref [ks, :] = 1

    for i in range(8):
        ker = rotate_img(kernel_Ref, i * 180 / dirNum)
        spn[i]= cv2.filter2D(Cs[i], -1, ker)

    sp = np.sum(spn, axis = 0)
    sp =  (sp - np.min(sp)) / (np.max(sp) - np.min(sp))
    S = 1 - sp

    return S

def pencil_stroke(img):
    h,w = img.shape[:-1]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img,(1024,1024))
    S = get_stroke(img, 3, 8)
    S = S ** 10
    S = (cv2.resize(S,(w,h))*255).astype(np.uint8)
    S[np.where(S>.5*255)] = 255
    S[np.where(S<.5*255)] = 0
    # c_min, c_max = np.min(S[np.where((S>.3*255)&(S<.7*255))]),np.max(S[np.where((S>.3*255)&(S<.7*255))])
    # S[np.where((S>.3*255)&(S<.7*255))] = ((S[np.where((S>.3*255)&(S<.7*255))] - c_min) / (c_max - c_min) * 255).astype(np.uint8)
    return S

if __name__ == '__main__':
    img_path = 'sample.png'
    img = cv2.imread(img_path)
    res = pencil_stroke(img)
    cv2.imwrite('stroke.jpg', res)
