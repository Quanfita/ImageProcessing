import cv2
import numpy as np

def luminance(image,value):
    if value > 0:
        image = image.copy() * (1 / (1 - value))
        image = np.clip(image,0,255)
    else:
        image = image.copy() * (1 + value)
    return image.astype(np.uint8)

def exposure(image,value):
    image = image * 2 ** (1 + value)
    image = np.clip(image,0,255)
    return image

def contrast(image,value):
    image = image / 255
    mean = np.mean(image)
    # mean = np.mean(image,axis=2).reshape([*image.shape[:-1],1])
    image = ((image - mean) * (1 + value) + mean)*255
    image = np.clip(image,0,255)
    return image.astype(np.uint8)

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

def sharpen(image,value):
    img = image.copy() * 1.0
    gauss_out = cv2.GaussianBlur(img, (5,5),10)
    
    # alpha 0 - 5
    alpha = value
    img_out = (img - gauss_out) * alpha + img
    
    img_out = img_out/255.0
    
    # 饱和处理
    mask_1 = img_out  < 0 
    mask_2 = img_out  > 1
    
    img_out = img_out * (1-mask_1)
    img_out = img_out * (1-mask_2) + mask_2
    return (img_out*255).astype(np.uint8)

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

if __name__ == '__main__':
    image = cv2.imread('lena.jpg')
    cv2.imwrite('luminance.png',luminance(image,.5))
    cv2.imwrite('exposure.png',exposure(image,.5))
    cv2.imwrite('contrast.png',contrast(image,.5))
    cv2.imwrite('saturation.png',saturation(image,.5))
    cv2.imwrite('sharpen.png',sharpen(image,.5))
    cv2.imwrite('tonemapping.png',tonemapping(image,.5))