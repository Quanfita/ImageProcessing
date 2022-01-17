import os
from turtle import width
import cv2
import glob
import numpy as np

base_dir = os.path.dirname(__file__)

name_dict = {'res1':'Ours','res2':'Tone Mapping','res3':'Reinhard','res4':'ACES','res5':'Gamma','res6':'ACES+bloom'}
file_list = glob.glob(os.path.join(base_dir,'*.png'))
name_list = [os.path.basename(item).split('.')[0] for item in file_list]
base_width = 400
base_height = 300
print(name_list)
width = base_width * len(name_list)
height = base_height

def mergeGlobalImage():
    image = np.ones([height+40,width,3],dtype=np.uint8) * 255
    for i,item in enumerate(file_list):
        img = cv2.imread(item)
        img = cv2.resize(img, (base_width,base_height))
        print(i*height,i*width)
        image[0:base_height,i*base_width:(i+1)*base_width] = img
        cv2.putText(image, name_dict.get(name_list[i]), (i*base_width+(base_width - len(name_dict.get(name_list[i]))*20)//2, base_height+30), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
    cv2.imwrite('compare.png',image)

def mergeOriginImage(x=0,y=0,w=base_width,h=base_height):
    image = np.ones([height+40,width,3],dtype=np.uint8) * 255
    for i,item in enumerate(file_list):
        img = cv2.imread(item)
        img = img[y:y+h,x:x+w]
        print(i*height,i*width)
        image[0:base_height,i*base_width:(i+1)*base_width] = img
        cv2.putText(image, name_dict.get(name_list[i]), (i*base_width+(base_width - len(name_dict.get(name_list[i]))*24)//2, base_height+30), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
    cv2.imwrite('compare.png',image)

if __name__ == '__main__':
    mergeOriginImage()