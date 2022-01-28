# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 11:12:14 2018

@author: Quanfita
"""

from sys import flags
import cv2
from matplotlib.pyplot import flag
import numpy as np

def Normal(img1,img2,alpha=.5):
    res = alpha * img1 + (1 - alpha) * img2
    return res.astype(np.uint8)

def Dissolve(img1,img2,alpha=.5):
    h,w = img1.shape[:-1]
    sample = np.random.uniform(size=(h,w))
    sample[np.where(sample>1-alpha)] = 1
    sample[np.where(sample<=1-alpha)] = 0
    sample = sample.reshape([h,w,1]).astype(np.uint8)
    res = img1 * sample + img2 * (1 - sample)
    return res.astype(np.uint8)

def Screen(img1,img2):#滤色
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = 1 - np.multiply((1 - img1),(1 - img2))
    res = (res*255).astype(np.uint8)
    return res

def Multiply(img1,img2):#正片叠底
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = np.multiply(img1,img2)
    res = (res*255).astype(np.uint8)
    return res

def Overlay(img1,img2):#叠加
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    height = img1.shape[0]
    weight = img1.shape[1]
    channels = img1.shape[2]
    res = np.zeros([height,weight,channels],dtype=np.float32)
    
    for row in range(height):            #遍历高
        for col in range(weight):         #遍历宽
            for c in range(channels):     #遍历通道
                if img2[row, col, c] <0.5:
                    res[row, col, c] = 2*img1[row, col, c]*img2[row, col, c]
                else:
                    res[row, col, c] = 1 - 2 * (1 - img1[row, col, c]) * (1 - img2[row, col, c])
    res = (255*res).astype(np.uint8)
    return res

def SoftLight(img1,img2):#柔光
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = np.zeros_like(img1,dtype=np.float32)
    
    # for row in range(height):            #遍历高
    #     for col in range(weight):         #遍历宽
    #         for c in range(channels):     #遍历通道
    #             if img1[row, col, c] <0.5:
    #                 res[row, col, c] = (2 * img1[row, col, c] - 1)*(img2[row, col, c] - img2[row, col, c]**2) + img2[row, col, c]
    #             else:
    #                 res[row, col, c] = (2 * img1[row, col, c] - 1)*(np.sqrt(img2[row, col, c]) - img2[row, col, c]) + img2[row, col, c]
    flags = np.zeros_like(img1)
    flags[np.where(img1<.5)] = 1
    res = flags * ((2 * img1 - 1) * (img2 - img2**2) + img2) + (1 - flags) * ((2 * img1 - 1) * (img2**.5 - img2) + img2)
    res = (255*res).astype(np.uint8)
    return res

def HardLight(img1,img2):#强光
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = np.zeros_like(img1,dtype=np.float32)
    
    # for row in range(height):            #遍历高
    #     for col in range(weight):         #遍历宽
    #         for c in range(channels):     #遍历通道
    #             if img1[row, col, c] <0.5:
    #                 res[row, col, c] = 2*img1[row, col, c]*img2[row, col, c]
    #             else:
    #                 res[row, col, c] = 1 - 2 * (1 - img1[row, col, c])*(1 - img2[row, col, c])
    flags = np.zeros_like(img1)
    flags[np.where(img1<.5)] = 1
    res = flags * 2 * img1 * img2 + (1 - flags) * (1 - 2 * (1 - img1)*(1 - img2))
    res = (255*res).astype(np.uint8)
    return res

def LinearAdd(img1,img2,a=0.5):#线性叠加
    res = a*img1+(1-a)*img2
    return res.astype(np.uint8)

def ColorBurn(img1,img2):#颜色加深
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    tmp = np.zeros(img1.shape,dtype=np.float32)
    res = (img1 + img2 - 1.0) / (img1+0.01)
    res = np.maximum(tmp,res)
    res = (res*255).astype(np.uint8)
    return res

def LinearBurn(img1,img2):#线性加深
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = img1 + img2 - 1.0
    res = np.maximum(0.0,res)
    res = (res*255).astype(np.uint8)
    return res

def ColorDodge(img1,img2):#颜色减淡
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = img2 / (1.0 - img1 + 0.01)
    res = np.maximum(0.0,res)
    res = np.minimum(1.0,res)
    res = (res*255).astype(np.uint8)
    return res

def LinearDodge(img1,img2):#线性减淡
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = img1 + img2
    res = np.minimum(1.0,res)
    res = (res*255).astype(np.uint8)
    return res

def Lighten(img1,img2):#变亮
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = np.zeros_like(img1,dtype=np.float32)
    flags = np.zeros_like(img1)
    flags[np.where(img1>img2)] = 1
    res = flags * img1 + (1 - flags) * img2
    res = (255*res).astype(np.uint8)
    return res

def Darken(img1,img2):#变暗
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = np.zeros_like(img1,dtype=np.float32)
    flags = np.zeros_like(img1)
    flags[np.where(img1<img2)] = 1
    res = flags * img1 + (1 - flags) * img2
    res = (255*res).astype(np.uint8)
    return res

def LighterColor(img1,img2):#浅色
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = np.zeros_like(img1,dtype=np.float32)
    flags = np.zeros_like(img1)
    flags[np.where(np.sum(img1,axis=2)>np.sum(img2,axis=2))] = 1
    res = flags * img1 + (1 - flags) * img2
    res = (255*res).astype(np.uint8)
    return res

def DarkerColor(img1,img2):#深色
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = np.zeros_like(img1,dtype=np.float32)
    flags = np.zeros_like(img1)
    flags[np.where(np.sum(img1,axis=2)<np.sum(img2,axis=2))] = 1
    res = flags * img1 + (1 - flags) * img2
    res = (255*res).astype(np.uint8)
    return res

def VividLight(img1,img2):#亮光
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = np.zeros_like(img1,dtype=np.float32)
    flags = np.zeros_like(img1)
    flags[np.where(img2<=.5)] = 1
    res = flags * (1 - (1 - img1) / (2 * img2 + 0.001)) + (1 - flags) * (img1 / (2 * (1 - img2) + 0.001))
    res = np.clip(res,0,1)
    res = (255*res).astype(np.uint8)
    return res

def LinearLight(img1,img2):#线性光
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = 2 * img1 + img2 - 1.0
    res = np.minimum(1.0,res)
    res = np.maximum(0.0,res)
    res = (res*255).astype(np.uint8)
    return res

def PinLight(img2,img1):#点光
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    height = img1.shape[0]
    weight = img1.shape[1]
    channels = img1.shape[2]
    res = np.zeros([height,weight,channels],dtype=np.float32)
    
    for i in range(height):            #遍历高
        for j in range(weight):         #遍历宽
            for c in range(channels):
                if img1[i,j,c] <= 2 * img2[i,j,c] - 1.0:
                    res[i,j,c] = 2 * img2[i,j,c] - 1.0
                elif 2 * img2[i,j,c] - 1.0 < img1[i,j,c] < 2 * img2[i,j,c]:
                    res[i,j,c] = img1[i,j,c]
                else:
                    res[i,j,c] = 2 * img2[i,j,c]
    res = (255*res).astype(np.uint8)
    return res

def HardMix(img1,img2):#实色混合
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = np.zeros_like(img1,dtype=np.float32)
    res[np.where(img1+img2>=1)] = 1
    res = (255*res).astype(np.uint8)
    return res

def Difference(img1,img2):#差色
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = np.abs(img2-img1)
    res = np.minimum(1.0,res)
    res = np.maximum(0.0,res)
    res = (res*255).astype(np.uint8)
    return res

def Exclusion(img1,img2):#排除
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = (img1 + img2) - (img1*img2)/0.5
    res = np.minimum(1.0,res)
    res = np.maximum(0.0,res)
    res = (res*255).astype(np.uint8)
    return res

def Subtract(img1,img2):#减去
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = img2 - img1
    res = np.minimum(1.0,res)
    res = np.maximum(0.0,res)
    res = (res*255).astype(np.uint8)
    return res

def Divide(img1,img2):#划分
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    res = img2 / (img1+0.01)
    res = np.minimum(1.0,res)
    res = np.maximum(0.0,res)
    res = (res*255).astype(np.uint8)
    return res

if __name__ == '__main__':
    img1 = cv2.imread('./sample/lena.jpg')
    img2 = cv2.imread('./sample/background.jpg')
    img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))
    normal = Normal(img1,img2)
    dissolve = Dissolve(img1,img2)
    screen = Screen(img1,img2)
    overlay = Overlay(img1,img2)
    softlight = SoftLight(img1,img2)
    hardlight = HardLight(img1,img2)
    multiply = Multiply(img1,img2)
    colorburn = ColorBurn(img1,img2)
    linearburn = LinearBurn(img1,img2)
    colordodge = ColorDodge(img1,img2)
    lineardodge = LinearDodge(img1,img2)
    lighten = Lighten(img1,img2)
    darken = Darken(img1,img2)
    darkercolor = DarkerColor(img1,img2)
    lightercolor = LighterColor(img1,img2)
    vividlight = VividLight(img1,img2)
    linearlight = LinearLight(img1,img2)
    pinlight = PinLight(img1,img2)
    hardmix = HardMix(img1,img2)
    difference = Difference(img1,img2)
    exclusion = Exclusion(img1,img2)
    subtract = Subtract(img1,img2)
    divide = Divide(img1,img2)
    cv2.imwrite('./result/normal.jpg',normal)
    cv2.imwrite('./result/dissolve.jpg',dissolve)
    cv2.imwrite('./result/screen.jpg',screen)
    cv2.imwrite('./result/overlay.jpg',overlay)
    cv2.imwrite('./result/softlight.jpg',softlight)
    cv2.imwrite('./result/hardlight.jpg',hardlight)
    cv2.imwrite('./result/multiply.jpg',multiply)
    cv2.imwrite('./result/colorburn.jpg',colorburn)
    cv2.imwrite('./result/linearburn.jpg',linearburn)
    cv2.imwrite('./result/colordodge.jpg',colordodge)
    cv2.imwrite('./result/lineardodge.jpg',lineardodge)
    cv2.imwrite('./result/lighten.jpg',lighten)
    cv2.imwrite('./result/darken.jpg',darken)
    cv2.imwrite('./result/darkercolor.jpg',darkercolor)
    cv2.imwrite('./result/lightercolor.jpg',lightercolor)
    cv2.imwrite('./result/vividlight.jpg',vividlight)
    cv2.imwrite('./result/linearlight.jpg',linearlight)
    cv2.imwrite('./result/pinlight.jpg',pinlight)
    cv2.imwrite('./result/hardmix.jpg',hardmix)
    cv2.imwrite('./result/difference.jpg',difference)
    cv2.imwrite('./result/exclusion.jpg',exclusion)
    cv2.imwrite('./result/subtract.jpg',subtract)
    cv2.imwrite('./result/divide.jpg',divide)
