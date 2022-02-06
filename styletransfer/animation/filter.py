import imp
import cv2
import numpy as np
from math import tanh

s = 9
sigmacolor = 17
sigmaspace = 17

phieq = 20

sigma = 1
phie = 5
tau = 0.981

def qnearest(f):
    f = f / 255 * 100
    if f >= 0 and f < 100.0/16:
            q=0
    elif f >= 100.0/16 and f < 12.5 + 100.0/16:
        q = 12.5
    elif f >= 12.5+100.0/16 and f < 25 + 100.0/16:
        q = 25
    elif f >= 25 + 100.0/16 and f<37.5 + 100.0/16:
        q = 37.5
    elif f >= 37.5 + 100.0/16 and f < 50 + 100.0/16:
        q = 50
    elif f >= 50 + 100.0/16 and f < 62.5 + 100.0/16:
        q = 62.5
    elif f >= 62.5 + 100.0/16 and f < 75 + 100.0/16:
        q = 75
    elif f >= 75 + 100.0/16 and f < 87.5+100.0/16:
        q = 87.5
    elif f >= 93.75:
        q = 100
    return q / 100 * 255

def filter(image):
    m1,n1 = image.shape[:-1]
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L0 = img_lab[:,:,0].copy()
    lab1 = cv2.bilateralFilter(img_lab,s,sigmacolor,sigmaspace)
    rgb1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)
    cv2.imwrite('bilateral.png',rgb1)
    lab2 = cv2.bilateralFilter(img_lab,s,sigmacolor,sigmaspace)
    lab2 = cv2.bilateralFilter(lab2,s,sigmacolor,sigmaspace)
    lab2 = cv2.bilateralFilter(lab2,s,sigmacolor,sigmaspace)
    rgb2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    cv2.imwrite('bilateral_.png',rgb2)
    rgb2 = cv2.bilateralFilter(image,s,sigmacolor,sigmaspace)
    cv2.imwrite('bilateral_rgb.png',rgb2)
    L = lab1[:,:,0].copy()
    quantum = np.zeros(shape=image.shape[:-1])
    for i in range(m1):
        for j in range(n1):
            quantum[i,j]=qnearest(L[i,j])+5*tanh((L[i,j]-qnearest(L[i,j]))/phieq)
    L = quantum
    lab1[:,:,0] = L
    rgb1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)
    return rgb1

if __name__ == '__main__':
    image = cv2.imread('26.jpg')
    # filter(image)
    cv2.imwrite('RTVA.png',filter(image))