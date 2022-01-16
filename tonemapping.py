from PIL import Image
import numpy as np

def ToneMapping(image):
    image_L = image.convert('L')
    np_img = np.array(image, dtype=np.float32)
    np_img_L = np.array(image_L, dtype=np.float32)
    delta = 0.0001
    normal_image = np_img / 255
    normal_image_L = np_img_L / 255
    height, width = normal_image.shape[:-1]
    n = height * width
    print(n,np.exp(np.sum(np.log(delta + normal_image_L))/n))
    Lwaver = np.exp(np.sum(np.log(delta + normal_image_L)/n))
    # normal_image_t = alpha / Lwaver * normal_image
    # normal_image_t[np.where(normal_image_t>1)] = 1 - normal_image_t[np.where(normal_image_t>1)]/np.max(normal_image_t)
    # normal_image = np.clip(normal_image_t, 0,1)
    lg = np.log(np_img_L / Lwaver + 1) / np.log(np.max(np_img_L) / Lwaver + 1)
    lg_min, lg_max = np.min(lg), np.max(lg)
    lg = lg - lg_min / (lg_max - lg_min)
    lg = lg.reshape([height,width,1]) * normal_image**.3
    lg_min, lg_max = np.min(lg), np.max(lg)
    lg = lg - lg_min / (lg_max - lg_min)
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

def gamma(image,g):
    image = image / 255.0
    image = image ** g
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return image

image = Image.open('test5.jpg')

lg = ToneMapping(image)
img = Image.fromarray((lg*255).astype(np.uint8))
img.save('res1.png')

img = Image.fromarray(ACESToneMapping(np.array(image)))
img.save('res2.png')

img = Image.fromarray(gamma(np.array(image),.3))
img.save('res3.png')