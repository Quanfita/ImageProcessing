import cv2
import numpy as np
import math

def zmMinFilterGray(src, r=7):
    '''''最小值滤波，r是滤波器半径'''
    return cv2.erode(src,np.ones((2*r-1,2*r-1)))

def guidedfilter(I, p, r, eps):
    '''''引导滤波，直接参考网上的matlab代码'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r,r))
    m_p = cv2.boxFilter(p, -1, (r,r))
    m_Ip = cv2.boxFilter(I*p, -1, (r,r))
    cov_Ip = m_Ip-m_I*m_p

    m_II = cv2.boxFilter(I*I, -1, (r,r))
    var_I = m_II-m_I*m_I

    a = cov_Ip/(var_I+eps)
    b = m_p-a*m_I

    m_a = cv2.boxFilter(a, -1, (r,r))
    m_b = cv2.boxFilter(b, -1, (r,r))
    return m_a*I+m_b

def get_mean_and_std(img):
	x_mean, x_std = cv2.meanStdDev(img)
	x_mean = np.hstack(np.around(x_mean, 2))
	x_std = np.hstack(np.around(x_std, 2))
	return x_mean, x_std

def color_transfer(sc):
	sc = cv2.cvtColor(sc, cv2.COLOR_BGR2LAB)
	s_mean, s_std = get_mean_and_std(sc)
	t_mean, t_std = np.array([175.15,132.56,114.29]),np.array([45.69,11.05,17.46])
	st = (s_mean / t_mean) ** 1.5
	img_n = ((sc-s_mean)*(t_std/s_std))+t_mean
	np.putmask(img_n, img_n > 255, 255)
	np.putmask(img_n, img_n < 0, 0)
	dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2BGR)
	return dst

def color_matching(img,dark):
    dark = dark / 255
    img = img.copy()
    img = color_transfer(img)
    return img

def clean_noise(img,threshold=32):
    contours,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= threshold:
            cv_contours.append(contour)
    cv2.fillPoly(img, cv_contours, (255, 255, 255))
    return img

def animation(image):
    cv2.imwrite('dark.png',np.min(image,2))
    dark_channel = zmMinFilterGray(np.min(image,2),7)
    cv2.imwrite('dark_channel.png', dark_channel)
    dark_blur = cv2.blur(dark_channel,(15,15))
    dark_blur = cv2.bilateralFilter(dark_blur,0,100,15)
    cv2.imwrite('dark_blur.png', dark_blur)
    # _,dark_mask = cv2.threshold(dark_channel,64,255,cv2.THRESH_BINARY)
    # dark_mask = cv2.GaussianBlur(dark_mask,(15,15),10)
    # dark_mask = clean_noise(dark_mask,128)
    # dark_mask = cv2.GaussianBlur(dark_mask,(15,15),10)
    # cv2.imwrite('dark_mask.png', dark_mask)
    color = image.copy()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # for _ in range(3):
    #     color = cv2.bilateralFilter(color,9,17,17)
    color = cv2.pyrMeanShiftFiltering(color,30,30)
    tone = color_matching(color,dark_blur)
    kernel = np.array([[1,0],
                       [0,-1]])
    threshold = 230
    image = cv2.GaussianBlur(image,(1,1),3)
    res1 = cv2.filter2D(image,-1,kernel)
    res2 = cv2.filter2D(image,-1,kernel.T)
    res = 255 - np.clip(res1+res2,0,255)
    # res[np.where(res<=threshold)] = 0
    res[np.where(res>threshold)] = 255
    ret, binary = cv2.threshold(res,240,255,cv2.THRESH_BINARY)
    _,contours,_ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 32:
            cv_contours.append(contour)
    cv2.fillPoly(res, cv_contours, (255, 255, 255))
    cv2.imwrite('edge.png', res)
    cv2.imwrite('tone.png', tone)
    res = res / 255
    res = res.reshape((*res.shape,1)) * tone
    res = np.clip(res,0,255)
    cv2.imwrite('res.png', res.astype(np.uint8))


if __name__ == '__main__':
    image = cv2.imread('90.jpg')
    animation(image)