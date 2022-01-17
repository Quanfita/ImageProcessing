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
    lg = lg - lg_min / (lg_max - lg_min)
    t = lg[:,:,0] * .299 + lg[:,:,1] * .587 + lg[:,:,2] * .114
    # print(np.exp(np.sum(np.log(delta + t)/n)))
    return lg

def NewToneMapping(image):
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
    lg = np.log(np_img_L / Lwaver + 1) / np.log(np.max(np_img_L) / Lwaver + 1)
    lg_min, lg_max = np.min(lg), np.max(lg)
    lg = lg - lg_min / (lg_max - lg_min)
    Lbias = np.sum(lg) / n
    alpha = .48
    lg = lg.reshape([height,width,1]) * normal_image**((Lbias - Lwaver)*alpha)
    lg = 1 - (1 - lg) * (1 - normal_image)
    lg_min, lg_max = np.min(lg), np.max(lg)
    lg = lg - lg_min / (lg_max - lg_min)
    t = lg[:,:,0] * .299 + lg[:,:,1] * .587 + lg[:,:,2] * .114
    # print(np.exp(np.sum(np.log(delta + t)/n)))
    return lg

def ACESToneMapping(image):
    a = 2.15
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    adj = 5.0
    image = image / 255.0
    image *= adj
    res = (image * (a * image + b)) / (image * (c * image + d) + e)
    res = (res * 255.0).astype(np.uint8)
    return res

def ReinhardToneMapping(image,value):
    image = image / 255
    MIDDLE_GREY = 1
    image *= MIDDLE_GREY / value
    image /= (1 + image)
    return (image * 255.0).astype(np.uint8)

def bloom(image):
    aces_res = ACESToneMapping(image) / 255
    image = image / 255
    kernel = np.array([[1,4,7,4,1],
                        [4,16,26,16,4],
                        [7,26,41,26,7],
                        [4,16,26,16,4],
                        [1,4,7,4,1]],dtype=np.float32)/273
    image = cv2.filter2D(image,-1,kernel)
    image = 1 - (1 - image) * (1 - aces_res)
    image = (image * 255).astype(np.uint8)
    return image

def gamma(image,g):
    image = image / 255.0
    image = image ** g
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return image

if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(__file__)

    image = cv2.imread(os.path.join(base_dir,'test14.jpg'))

    lg = NewToneMapping(image)
    img = (lg*255).astype(np.uint8)
    cv2.imwrite('res1.png', img)

    lg = ToneMapping(image)
    img = (lg*255).astype(np.uint8)
    cv2.imwrite('res2.png', img)

    img = ReinhardToneMapping(np.array(image),0.1)
    cv2.imwrite('res3.png', img)

    img = ACESToneMapping(np.array(image))
    cv2.imwrite('res4.png', img)

    img = gamma(np.array(image),.3)
    cv2.imwrite('res5.png', img)

    img = bloom(np.array(image))
    cv2.imwrite('res6.png', img)