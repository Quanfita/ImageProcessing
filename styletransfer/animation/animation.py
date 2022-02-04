import cv2
import numpy as np
import math

def saturation(image, value):
    image = image.copy() / 255
    rgb_max = np.max(image,axis=2)
    rgb_min = np.min(image,axis=2)
    delta = rgb_max - rgb_min
    v = rgb_max + rgb_min
    L = v / 2
    t = np.zeros_like(v,dtype=int)
    t[np.where(L < .5)] = 1
    S = t * delta / (v) + (1 - t) * delta / (2 - v)
    if value > 0:
        t[:,:] = 0
        t[S + value >= 1] = 1
        L = L.reshape((*image.shape[:-1],1))
        S = S.reshape((*image.shape[:-1],1))
        t = t.reshape((*image.shape[:-1],1))
        alpha = (t * S + (1 - t) * (1 - value))
        image = image + (image - L)*alpha
    else:
        alpha = (1 / value -1).reshape((*image.shape[:-1],1))
        L = L.reshape((*image.shape[:-1],1))
        image = L + (image - L) * (1 + alpha)
    return image*255

def contrast(image,value):
    image = image / 255
    mean = np.mean(image)
    # mean = np.mean(image,axis=2).reshape([*image.shape[:-1],1])
    image = ((image - mean) * (1 + value) + mean)*255
    image = np.clip(image,0,255)
    return image.astype(np.uint8)

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


def tonemapping(image,value):
    image = image.copy() * 1.0
    image_blur = cv2.GaussianBlur(image,(5,5),10)
    bias = image - image_blur
    image = image_blur*(1 + value) + bias
    image = image / 255 
    mask_1 = image  < 0 
    mask_2 = image  > 1
    img_out = image * (1-mask_1)
    img_out = img_out * (1-mask_2) + mask_2
    return (img_out*255).astype(np.uint8)

def color_matching(img,dark):
    dark = dark / 255
    img = img.copy()
    img = saturation(img,.5)
    img = contrast(img,.5) * dark.reshape((*dark.shape,1)) + (1 - dark.reshape((*dark.shape,1))) * img
    ligten = img * dark.reshape((*dark.shape,1))
    cv2.imwrite('lighten_area.png',ligten)
    img = tonemapping(img,.5) * (1 - dark.reshape((*dark.shape,1))) + dark.reshape((*dark.shape,1)) * img
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
    contours,_ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 32:
            cv_contours.append(contour)
    print(n)
    cv2.fillPoly(res, cv_contours, (255, 255, 255))
    cv2.imwrite('edge.png', res)
    cv2.imwrite('tone.png', tone)
    res = res / 255
    res = res.reshape((*res.shape,1)) * tone
    cv2.imwrite('res.png', res.astype(np.uint8))


if __name__ == '__main__':
    image = cv2.imread('90.jpg')
    animation(image)