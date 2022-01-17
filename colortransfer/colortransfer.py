import cv2
import numpy as np

def get_mean_and_std(img):
	x_mean, x_std = cv2.meanStdDev(img)
	x_mean = np.hstack(np.around(x_mean, 2))
	x_std = np.hstack(np.around(x_std, 2))
	return x_mean, x_std

def color_transfer(sc, dc):
	sc = cv2.cvtColor(sc, cv2.COLOR_BGR2LAB)
	s_mean, s_std = get_mean_and_std(sc)
	dc = cv2.cvtColor(dc, cv2.COLOR_BGR2LAB)
	t_mean, t_std = get_mean_and_std(dc)
	st = (s_mean / t_mean) ** 1.2
	img_n = ((sc-s_mean)*(t_std/s_std))+t_mean
	np.putmask(img_n, img_n > 255, 255)
	np.putmask(img_n, img_n < 0, 0)
	dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2BGR)
	return dst, st

sc = cv2.imread("A.jpg", 1)
dc = cv2.imread("B.jpg", 1)
dst,st = color_transfer(sc, dc)
dst = dst / 255
m = np.mean(dst)
dst = (dst - m) * st + m
dst = (np.clip(dst,0,1) * 255).astype(np.uint8)
cv2.imwrite('r.png',dst)