import cv2
import numpy as np 

def Three_element_add(array):
    array0 = array[:]
    array1 = np.append(array[1:],np.array([0]))
    array2 = np.append(array[2:],np.array([0, 0]))
    arr_sum = array0 + array1 + array2
    return arr_sum[:-2]


def VThin(image, array):
    NEXT = 1
    height, width = image.shape[:2]
    for i in range(1,height):
        M_all = Three_element_add(image[i])
        for j in range(1,width):
            if NEXT == 0:
                NEXT = 1
            else:
                M = M_all[j-1] if j<width-1 else 1
                if image[i, j] == 0 and M != 0:
                    a = np.zeros(9,dtype=np.uint8)
                    if height-1 > i and width-1 > j:
                        kernel = image[i - 1:i + 2, j - 1:j + 2]
                        a = np.where(kernel == 255, 1, 0)
                        a = a.reshape(1, -1)[0]
                    NUM = np.array([1,2,4,8,0,16,32,64,128],dtype=np.uint8)
                    sumArr = np.sum(a*NUM)
                    image[i, j] = array[sumArr] * 255
                    if array[sumArr] == 1:
                        NEXT = 0
    return image


def HThin(image, array):
    height, width = image.shape[:2]
    NEXT = 1
    for j in range(1,width):
        M_all = Three_element_add(image[:,j])
        for i in range(1,height):
            if NEXT == 0:
                NEXT = 1
            else:
                M = M_all[i-1] if i < height - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = np.zeros(9,dtype=np.uint8)
                    if height - 1 > i and width - 1 > j:
                        kernel = image[i - 1:i + 2, j - 1:j + 2]
                        a = np.where(kernel == 255, 1, 0)
                        a = a.reshape(1, -1)[0]
                    NUM = np.array([1, 2, 4, 8, 0, 16, 32, 64, 128],dtype=np.uint8)
                    sumArr = np.sum(a * NUM)
                    image[i, j] = array[sumArr] * 255
                    if array[sumArr] == 1:
                        NEXT = 0
    return image


def Xihua(binary, array, num=10):
    binary_image = binary.copy()
    image = cv2.copyMakeBorder(binary_image, 1, 0, 1, 0, cv2.BORDER_CONSTANT, value=0)
    for i in range(num):
        VThin(image, array)
        HThin(image, array)
    return image

array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,\
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,\
         1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

def edge_demo(image):
    blurred = cv2.GaussianBlur(image,(3,3),0)  #高斯降噪，适度
    gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
    #求梯度
    xgrd = cv2.Sobel(gray,cv2.CV_16SC1,1,0)
    ygrd = cv2.Sobel(gray,cv2.CV_16SC1,0,1)

    egde_output = cv2.Canny(xgrd,ygrd,50,150)  #50低阈值，150高阈值
    #egde_output = cv.Canny(gray,50,150)   #都可使用
    cv2.imwrite('canny_edge.jpg',255-egde_output)
    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(egde_output, kernel)
    cv2.imwrite('canny_edge_dilate.jpg',255-img_dilate)
    img_erode = Xihua(255-img_dilate,array)
    cv2.imwrite('canny_edge_erode.jpg',img_erode)

def edge_filter(image):
    kernel1 = np.array([[-3,-1,0,1,3],
                        [-1,0,0,0,1],
                        [0,0,0,0,0],
                        [-1,0,0,0,1],
                        [-3,-1,0,1,3]])
    kernel2 = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])
    kernel3 = np.array([[-1,-2,-1],
                        [0,0,0],
                        [-1,-2,-1]])
    # s1 = np.sum(kernel1)
    res1 = cv2.filter2D(image,-1,kernel2)
    res2 = cv2.filter2D(image,-1,kernel3)
    res = np.clip(res1+res2,0,255)
    res[np.where(res>20)] = 255
    cv2.imwrite('res1.png',255-res)
    res = cv2.filter2D(image,-1,kernel1)
    # res[np.where(res<=100)] = 0
    cv2.imwrite('res2.png',255-res)

def edge(image):
    h,w = image.shape
    image = cv2.resize(image,(w,h))
    sigma = 3
    kernel_size = (0,0)
    L = cv2.GaussianBlur(image, kernel_size, sigma)
    H = (255 - cv2.subtract(image, L)) / 255
    H = H ** 5
    # contours, _ = cv2.findContours((H*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # n = len(contours)  # 轮廓的个数
    # cv_contours = []
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area >= 32:
    #         cv_contours.append(contour)
    H = cv2.resize(H,(w,h))
    # cv2.fillPoly(H, cv_contours, (255, 255, 255))
    cv2.imwrite('res.png',H*255)

if __name__ == '__main__':
    img = cv2.imread('sample.jpg',0)
    image = cv2.imread('sample.jpg')
    edge(img)
    edge_filter(img)
    # edge_demo(image)