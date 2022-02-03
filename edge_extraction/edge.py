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
    kernel1 = np.array([[1,0],
                        [0,-1]])
    kernel2 = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])
    kernel3 = np.array([[-1,0,1],
                        [-1,0,1],
                        [-1,0,1]])
    threshold = 50
    l = []
    res1 = cv2.filter2D(image,-1,kernel2)
    res2 = cv2.filter2D(image,-1,kernel2.T)
    res = np.clip(res1+res2,0,255)
    res[np.where(res>=threshold)] = 255
    l.append(255-res)
    cv2.imwrite('Sobel.png',255-res)
    res1 = cv2.filter2D(image,-1,kernel1)
    res2 = cv2.filter2D(image,-1,kernel1.T)
    res = np.clip(res1+res2,0,255)
    res[np.where(res>=threshold)] = 255
    l.append(255-res)
    cv2.imwrite('Roberts.png',255-res)
    res1 = cv2.filter2D(image,-1,kernel3)
    res2 = cv2.filter2D(image,-1,kernel3.T)
    res = np.clip(res1+res2,0,255)
    res[np.where(res>=threshold)] = 255
    l.append(255-res)
    cv2.imwrite('Prewitt.png',255-res)
    length = len(l)
    l1 = l.pop(0).astype(np.float32)
    res1 = l1.copy()
    res2 = l1.copy()
    for img in l:
        res1 = res1 * img
        res2 = res2 + img
    res1 = (res1/255/255).astype(np.uint8)
    res1[np.where(res1>=threshold)] = 255
    cv2.imwrite('SRP_M.png',res1)
    res2 = (res2 / length).astype(np.uint8)
    res2[np.where(res2>=threshold*3)] = 255
    cv2.imwrite('SRP_A.png',res2)

def edge(image):
    sigma = 3
    kernel_size = (0,0)
    L = cv2.GaussianBlur(image, kernel_size, sigma)
    H = (255 - cv2.subtract(image, L))
    H[H<=250] = 0
    cv2.imwrite('DoG.png',H)

if __name__ == '__main__':
    img = cv2.imread('sample.jpg',0)
    image = cv2.imread('sample.jpg')
    edge(img)
    edge_filter(img)
    # edge_demo(image)