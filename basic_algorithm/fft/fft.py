import cv2
import numpy as np

def np_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift))
    fmax, fmin = np.max(fimg),np.min(fimg)
    fimg = (fimg - fmin) / (fmax - fmin) 
    return fshift, fimg*255

def np_fft_inv(fshift):
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    return iimg

def cv_fft(img):
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    fimg = np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    fmax, fmin = np.max(fimg),np.min(fimg)
    fimg = (fimg - fmin) / (fmax - fmin)
    return dft_shift, fimg*255

def cv_fft_inv(dftshift):
    ishift = np.fft.ifftshift(dftshift)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])
    fmax, fmin = np.max(res),np.min(res)
    res = (res - fmin) / (fmax - fmin)
    return res * 255


if __name__ == '__main__':
    img = cv2.imread('test.jpg', 0)
    fshift, fft = np_fft(img)
    cv2.imwrite('np_fft.png',fft)
    cv2.imwrite('np_fft_inv.png',np_fft_inv(fshift))
    fshift, fft = cv_fft(img)
    cv2.imwrite('cv_fft.png',fft)
    cv2.imwrite('cv_fft_inv.png',cv_fft_inv(fshift))
