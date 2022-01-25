from re import A
import numpy as np
import cv2

def BGR2RGB(image):
    img = np.zeros_like(image)
    img[:,:,0] = image[:,:,2]
    img[:,:,1] = image[:,:,1]
    img[:,:,2] = image[:,:,0]
    return img

def RGB2BGR(image):
    img = np.zeros_like(image)
    img[:,:,0] = image[:,:,2]
    img[:,:,1] = image[:,:,1]
    img[:,:,2] = image[:,:,0]
    return img

def RGB2GRAY(image):
    R = image[:,:,2]
    G = image[:,:,1]
    B = image[:,:,0]
    return (R*0.299+G*0.587+B*0.114).astype(np.uint8)

def RGB2HSV(image):
    image = image.copy() / 255
    v = np.max(image,axis=2)
    s = np.zeros_like(image[:,:,0])
    s[np.where(v!=0)] = v[np.where(v!=0)] - np.min(image[np.where(v!=0)],axis=2)

    pass

def RGB2LAB(image):
    Xn = 0.950456
    Yn = 1.0
    Zn = 1.088754
    image = image.copy() / 255
    M = np.array([[0.412453,0.357580,0.180423],
                  [0.212671,0.715160,0.072169],
                  [0.019334,0.119193,0.950227]],dtype=np.float32)
    X = np.sum(M[0]*image,axis=2)
    Y = np.sum(M[1]*image,axis=2)
    Z = np.sum(M[2]*image,axis=2)
    t = np.zeros(image.shape[:-1])
    t[np.where(X>(6/29)**3)] = 1
    print(t)
    L = 116*(((Y/Yn)**(1/3))*t + (1 - t)*(((29/6)**2)/3*(Y/Yn)+4/29)) - 16
    a = 500*((((X/Xn)**(1/3))*t + (1 - t)*(((29/6)**2)/3*(X/Xn)+4/29)) - (((Y/Yn)**(1/3))*t + (1 - t)*(((29/6)**2)/3*(Y/Yn)+4/29)))
    b = 200*((((Y/Yn)**(1/3))*t + (1 - t)*(((29/6)**2)/3*(Y/Yn)+4/29)) - (((Z/Zn)**(1/3))*t + (1 - t)*(((29/6)**2)/3*(Z/Zn)+4/29)))
    img = np.zeros(image.shape)
    img[:,:,0] = L
    img[:,:,1] = a
    img[:,:,2] = b
    return (img*255).astype(np.uint8)

def RGB2CMY(image):
    C = 1 - image[:,:,0]/255
    M = 1 - image[:,:,1]/255
    Y = 1 - image[:,:,2]/255
    img = np.zeros_like(image)
    img[:,:,0] = C*100
    img[:,:,1] = M*100
    img[:,:,2] = Y*100
    return img

def RGB2CMYK(image):
    h,w = image.shape[:-1]
    CMY = RGB2CMY(image)/100
    print(CMY)
    K = np.min(CMY,axis=2)
    print(K)
    t = np.where(K<1)
    C = np.zeros(shape=(h,w))
    M = np.zeros(shape=(h,w))
    Y = np.zeros(shape=(h,w))
    C[t] = (CMY[:,:,0][t] - K[t]) / (1 - K[t])
    M[t] = (CMY[:,:,1][t] - K[t]) / (1 - K[t])
    Y[t] = (CMY[:,:,2][t] - K[t]) / (1 - K[t])
    t = np.where(K==1)
    C[t], M[t], Y[t] = 0, 0, 0
    img = np.zeros(shape=(h,w,4))
    img[:,:,0] = C
    img[:,:,1] = M
    img[:,:,2] = Y
    img[:,:,3] = K
    return (img*100).astype(np.uint8)

def CMYK2RGB(image):
    h,w = image.shape[:-1]
    img = np.zeros(shape=(h,w,3))
    image = image.astype(np.uint16)
    img[:,:,0] = 255 * (100 - image[:,:,0]) * (100 - image[:,:,-1]) / 10000
    img[:,:,1] = 255 * (100 - image[:,:,1]) * (100 - image[:,:,-1]) / 10000
    img[:,:,2] = 255 * (100 - image[:,:,2]) * (100 - image[:,:,-1]) / 10000
    return img.astype(np.uint8)

def LAB2RGB(image):
    pass

def HSV2RGB(image):
    pass

if __name__ == '__main__':
    image = np.array([[[13,124,255],[115,82,68],[0,0,0]]],dtype=np.uint8)
    cmy = RGB2CMYK(image)
    print(cmy)
    print(RGB2LAB(image))