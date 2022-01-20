# from PIL import Image, ImageFilter
import cv2
import numpy as np

def ToneMapping(image):
    image_L = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    np_img = np.array(image, dtype=np.float32)
    np_img_L = np.array(image_L, dtype=np.float32)
    delta = 0.0001
    normal_image = np_img / 255
    normal_image_L = np_img_L / 255
    height, width = normal_image.shape[:-1]
    n = height * width
    Lwaver = np.exp(np.sum(np.log(delta + normal_image_L)/n))
    # print(Lwaver)
    lg = np.log(normal_image / Lwaver + 1) / np.log(np.max(normal_image) / Lwaver + 1)
    # lg = lg.reshape([height,width,1]) * normal_image
    lg_min, lg_max = np.min(lg), np.max(lg)
    lg = (lg - lg_min) / (lg_max - lg_min)
    t = lg[:,:,0] * .299 + lg[:,:,1] * .587 + lg[:,:,2] * .114
    print(np.exp(np.sum(np.log(delta + t)/n)))
    return lg

def HSVToneMapping(image):
    image_L = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_t = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    delta = 0.0001
    normal_image_L = image_L / 255
    height, width = normal_image_L.shape
    n = height * width
    Lwaver = np.exp(np.sum(np.log(delta + normal_image_L)/n))
    lg = np.log(image_L / Lwaver + 1) / np.log(np.max(image_L) / Lwaver + 1)
    lg_min, lg_max = np.min(lg), np.max(lg)
    lg = (lg - lg_min) / (lg_max - lg_min)
    image_t[:,:,-1] = (lg*255).astype(np.uint8)
    image_t = cv2.cvtColor(image_t,cv2.COLOR_HSV2BGR)
    return image_t

def LABToneMapping(image):
    image_L = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_t = image.copy()
    image_t = cv2.cvtColor(image_t,cv2.COLOR_BGR2LAB)
    delta = 0.0001
    normal_image_L = image_L / 255
    height, width = normal_image_L.shape
    n = height * width
    Lwaver = np.exp(np.sum(np.log(delta + normal_image_L)/n))
    lg = np.log(image_L / Lwaver + 1) / np.log(np.max(image_L) / Lwaver + 1)
    lg_min, lg_max = np.min(lg), np.max(lg)
    lg = (lg - lg_min) / (lg_max - lg_min)
    image_t[:,:,0] = (lg*255).astype(np.uint8)
    lg = (lg * 255).astype(np.uint8)
    return lg

def NewToneMapping(image):
    image_L = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_t = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    delta = 0.0001
    normal_image_L = image_L / 255
    height, width = normal_image_L.shape
    n = height * width
    Lwaver = np.exp(np.sum(np.log(delta + normal_image_L)/n))
    lg = np.log(image_L / Lwaver + 1) / np.log(np.max(image_L) / Lwaver + 1)
    lg_min, lg_max = np.min(lg), np.max(lg)
    lg = (lg - lg_min) / (lg_max - lg_min)
    image_t[:,:,-1] = (lg*255).astype(np.uint8)
    image_t = cv2.cvtColor(image_t,cv2.COLOR_HSV2BGR)
    image_t = cv2.GaussianBlur(image_t,(11,11),0)
    lg = ((image_t/255) * (normal_image_L.reshape([height,width,1])**.3))
    lg_min, lg_max = np.min(lg), np.max(lg)
    lg = (lg - lg_min) / (lg_max - lg_min)
    # t = lg[:,:,0] * .299 + lg[:,:,1] * .587 + lg[:,:,2] * .114
    # print(np.exp(np.sum(np.log(delta + t)/n)))
    return (lg*255).astype(np.uint8)

def mathmethod(image):
    image_t = image.copy()
    image_t = image_t / 255
    # image_t = (2*image_t**2+7*image_t**.5)/9
    image_t = np.log(9*image_t+1)
    lg_min, lg_max = np.min(image_t), np.max(image_t)
    image_t = (image_t - lg_min) / (lg_max - lg_min)
    image_t = (image_t*255).astype(np.uint8)
    return image_t

def ACESToneMapping(image):
    a = 2.15
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    adj = 5.0
    image_t = image.copy() / 255.0
    image_t *= adj
    res = (image_t * (a * image_t + b)) / (image_t * (c * image_t + d) + e)
    res = (res * 255.0).astype(np.uint8)
    return res

def ReinhardToneMapping(image,value):
    image_t = image.copy() / 255
    MIDDLE_GREY = 1
    image_t *= MIDDLE_GREY / value
    image_t /= (1 + image_t)
    return (image_t * 255.0).astype(np.uint8)

def bloom(image):
    aces_res = ACESToneMapping(image) / 255
    image_t = image.copy() / 255
    kernel = np.array([[1,4,7,4,1],
                        [4,16,26,16,4],
                        [7,26,41,26,7],
                        [4,16,26,16,4],
                        [1,4,7,4,1]],dtype=np.float32)/273
    image_t = cv2.filter2D(image_t,-1,kernel)
    image_t = 1 - (1 - image_t) * (1 - aces_res)
    image_t = (image_t * 255).astype(np.uint8)
    return image_t

def gamma(image,g):
    image_t = image.copy() / 255.0
    image_t = image_t ** g
    image_t = (np.clip(image_t, 0, 1) * 255).astype(np.uint8)
    return image_t

if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(__file__)
    def saveImg(name,image):
        cv2.imwrite(os.path.join(base_dir,name),image)

    image = cv2.imread(os.path.join(base_dir,'test4.png'))

    img = NewToneMapping(image)
    saveImg('res1.png', img)

    lg = ToneMapping(image)
    img = (lg*255).astype(np.uint8)
    saveImg('res2.png', img)

    lg = mathmethod(image)
    saveImg('res7.png',lg)

    img = ReinhardToneMapping(np.array(image),0.1)
    saveImg('res3.png', img)

    img = ACESToneMapping(np.array(image))
    saveImg('res4.png', img)

    img = gamma(np.array(image),.3)
    saveImg('res5.png', img)

    img = bloom(np.array(image))
    saveImg('res6.png', img)