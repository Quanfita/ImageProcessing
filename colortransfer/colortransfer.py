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

def color_transfer_min(sc, dc):
	sc = cv2.cvtColor(sc, cv2.COLOR_BGR2LAB)
	s_mean, s_std = get_mean_and_std(sc)
	dc = cv2.cvtColor(dc, cv2.COLOR_BGR2LAB)
	t_mean, t_std = get_mean_and_std(dc)
	# st = (s_mean / t_mean) ** 1.2
	s_min,s_max = np.array([np.min(sc[:,:,0]),np.min(sc[:,:,1]),np.min(sc[:,:,2])]), np.array([np.max(sc[:,:,0]),np.max(sc[:,:,1]),np.max(sc[:,:,2])])
	t_min,t_max = np.array([np.min(dc[:,:,0]),np.min(dc[:,:,1]),np.min(dc[:,:,2])]), np.array([np.max(dc[:,:,0]),np.max(dc[:,:,1]),np.max(dc[:,:,2])])
	img_n = ((sc-s_min)*(t_std/s_std))+t_min
	np.putmask(img_n, img_n > 255, 255)
	np.putmask(img_n, img_n < 0, 0)
	dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2BGR)
	return dst

def color_transfer_max(sc, dc):
	sc = cv2.cvtColor(sc, cv2.COLOR_BGR2LAB)
	s_mean, s_std = get_mean_and_std(sc)
	dc = cv2.cvtColor(dc, cv2.COLOR_BGR2LAB)
	t_mean, t_std = get_mean_and_std(dc)
	# st = (s_mean / t_mean) ** 1.2
	s_min,s_max = np.array([np.min(sc[:,:,0]),np.min(sc[:,:,1]),np.min(sc[:,:,2])]), np.array([np.max(sc[:,:,0]),np.max(sc[:,:,1]),np.max(sc[:,:,2])])
	t_min,t_max = np.array([np.min(dc[:,:,0]),np.min(dc[:,:,1]),np.min(dc[:,:,2])]), np.array([np.max(dc[:,:,0]),np.max(dc[:,:,1]),np.max(dc[:,:,2])])
	img_n = ((sc-s_mean)*((t_max-t_min)/(s_max-s_min)))+t_mean
	np.putmask(img_n, img_n > 255, 255)
	np.putmask(img_n, img_n < 0, 0)
	dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2BGR)
	return dst

def color_transfer_nonlinear(sc,dc):
	# sc = cv2.cvtColor(sc, cv2.COLOR_BGR2LAB)
	# dc = cv2.cvtColor(dc, cv2.COLOR_BGR2LAB)
	s_min,s_max = np.array([np.min(sc[:,:,0]),np.min(sc[:,:,1]),np.min(sc[:,:,2])]), np.array([np.max(sc[:,:,0]),np.max(sc[:,:,1]),np.max(sc[:,:,2])])
	t_min,t_max = np.array([np.min(dc[:,:,0]),np.min(dc[:,:,1]),np.min(dc[:,:,2])]), np.array([np.max(dc[:,:,0]),np.max(dc[:,:,1]),np.max(dc[:,:,2])])
	s_mean, s_std = get_mean_and_std(sc)
	t_mean, t_std = get_mean_and_std(dc)
	lg = np.mat([[s_min[0]**2,s_min[0],1],
				  [s_max[0]**2,s_max[0],1],
				  [s_mean[0]**2,s_mean[0],1]])
	lga = np.mat([[t_min[0],s_min[0],1],
				  [t_max[0],s_max[0],1],
				  [t_mean[0],s_mean[0],1]])
	lgb = np.mat([[s_min[0]**2,t_min[0],1],
				  [s_max[0]**2,t_max[0],1],
				  [s_mean[0]**2,t_mean[0],1]])
	lgc = np.mat([[s_min[0]**2,s_min[0],t_min[0]],
				  [s_max[0]**2,s_max[0],t_max[0]],
				  [s_mean[0]**2,s_mean[0],t_mean[0]]])
	aL = np.linalg.det(lga)/np.linalg.det(lg)
	bL = np.linalg.det(lgb)/np.linalg.det(lg)
	cL = np.linalg.det(lgc)/np.linalg.det(lg)
	lg = np.mat([[s_min[1]**2,s_min[1],1],
				  [s_max[1]**2,s_max[1],1],
				  [s_mean[1]**2,s_mean[1],1]])
	lga = np.mat([[t_min[1],s_min[1],1],
				  [t_max[1],s_max[1],1],
				  [t_mean[1],s_mean[1],1]])
	lgb = np.mat([[s_min[1]**2,t_min[1],1],
				  [s_max[1]**2,t_max[1],1],
				  [s_mean[1]**2,t_mean[1],1]])
	lgc = np.mat([[s_min[1]**2,s_min[1],t_min[1]],
				  [s_max[1]**2,s_max[1],t_max[1]],
				  [s_mean[1]**2,s_mean[1],t_mean[1]]])
	aA = np.linalg.det(lga)/np.linalg.det(lg)
	bA = np.linalg.det(lgb)/np.linalg.det(lg)
	cA = np.linalg.det(lgc)/np.linalg.det(lg)
	lg = np.mat([[s_min[2]**2,s_min[2],1],
				  [s_max[2]**2,s_max[2],1],
				  [s_mean[2]**2,s_mean[2],1]])
	lga = np.mat([[t_min[2],s_min[2],1],
				  [t_max[2],s_max[2],1],
				  [t_mean[2],s_mean[2],1]])
	lgb = np.mat([[s_min[2]**2,t_min[2],1],
				  [s_max[2]**2,t_max[2],1],
				  [s_mean[2]**2,t_mean[2],1]])
	lgc = np.mat([[s_min[2]**2,s_min[2],t_min[2]],
				  [s_max[2]**2,s_max[2],t_max[2]],
				  [s_mean[2]**2,s_mean[2],t_mean[2]]])
	aB = np.linalg.det(lga)/np.linalg.det(lg)
	bB = np.linalg.det(lgb)/np.linalg.det(lg)
	cB = np.linalg.det(lgc)/np.linalg.det(lg)
	a = np.array([aL,aA,aB])
	b = np.array([bL,bA,bB])
	c = np.array([cL,cA,cB])
	img_n = a*sc**2 + b*sc + c
	np.putmask(img_n, img_n > 255, 255)
	np.putmask(img_n, img_n < 0, 0)
	# dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2BGR)
	return dst
	

def color_transfer_maxmin(sc,dc):
	sc = cv2.cvtColor(sc, cv2.COLOR_BGR2LAB)
	dc = cv2.cvtColor(dc, cv2.COLOR_BGR2LAB)
	s_min,s_max = np.array([np.min(sc[:,:,0]),np.min(sc[:,:,1]),np.min(sc[:,:,2])]), np.array([np.max(sc[:,:,0]),np.max(sc[:,:,1]),np.max(sc[:,:,2])])
	t_min,t_max = np.array([np.min(dc[:,:,0]),np.min(dc[:,:,1]),np.min(dc[:,:,2])]), np.array([np.max(dc[:,:,0]),np.max(dc[:,:,1]),np.max(dc[:,:,2])])
	img_n = (sc - s_min)/(s_max - s_min) * (t_max - t_min) + t_min
	np.putmask(img_n, img_n > 255, 255)
	np.putmask(img_n, img_n < 0, 0)
	dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2BGR)
	return dst

sc = cv2.imread("A.jpg", 1)
dc = cv2.imread("B.jpg", 1)
dst,st = color_transfer(sc, dc)
cv2.imwrite('r.png',dst)
dst = dst / 255
m = np.mean(dst)
dst = (dst - m) * st + m
dst = (np.clip(dst,0,1) * 255).astype(np.uint8)
cv2.imwrite('rr.png',dst)
dst = color_transfer_min(sc, dc)
cv2.imwrite('r_min.png',dst)
dst = color_transfer_max(sc, dc)
cv2.imwrite('r_max.png',dst)
dst = color_transfer_nonlinear(sc, dc)
cv2.imwrite('r_n.png',dst)
# dst = dst / 255
# m = np.mean(dst)
# dst = (dst - m) * st + m
# dst = (np.clip(dst,0,1) * 255).astype(np.uint8)
# cv2.imwrite('r.png',dst)