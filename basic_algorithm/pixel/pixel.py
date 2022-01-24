import cv2
import numpy as np

def exposure(image,value):
    if value > 0:
        image = image.copy() * (1 / (1 - value))
        image = np.clip(image,0,255)
    else:
        image = image.copy() * (1 + value)
    return image.astype(np.uint8)

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

def shapen(image,value):
    image_blur = cv2.blur(image,(5,5))
    cv2.imwrite('tmp.png',image_blur)
    bias = image - image_blur
    cv2.imwrite('tmp.png',bias*.5)
    image = image + bias * (1/(1 - value))*0.1
    image = np.clip(image,0,255)
    return image

def tonemapping(image,value):
    image_blur = cv2.blur(image,(5,5))
    bias = image - image_blur
    print(np.max(bias),np.min(bias))
    image = np.clip(image_blur*(1 + value),0,225) + bias
    image = np.clip(image,0,255)
    return image

if __name__ == '__main__':
    image = cv2.imread('lena.jpg')
    cv2.imwrite('exposure.png',exposure(image,.5))
    cv2.imwrite('contrast.png',contrast(image,.5))
    cv2.imwrite('saturation.png',saturation(image,.5))
    cv2.imwrite('shapen.png',shapen(image,.5))