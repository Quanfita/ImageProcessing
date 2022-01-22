from statistics import mean
import cv2
import numpy as np
import scipy.linalg

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
	st = (s_mean / t_mean) ** 1.5
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

def cramer(array,constant) -> list:
	A = np.linalg.det(array)
	Ais = []
	for i in range(array.shape[-1]):
		Ai = array.copy()
		Ai[:,i] = constant
		Ais.append(Ai)
	return [np.linalg.det(Ai)/A for Ai in Ais]

def color_transfer_nonlinear(sc,dc):
	# sc = sc / 255
	# dc = dc / 255
	# sc = sc.astype(np.uint64)
	# dc = dc.astype(np.uint64)
	sc = cv2.cvtColor(sc, cv2.COLOR_BGR2LAB).astype(np.uint64)
	dc = cv2.cvtColor(dc, cv2.COLOR_BGR2LAB).astype(np.uint64)
	s_min,s_max = np.array([np.min(sc[:,:,0]),np.min(sc[:,:,1]),np.min(sc[:,:,2])]), np.array([np.max(sc[:,:,0]),np.max(sc[:,:,1]),np.max(sc[:,:,2])])
	t_min,t_max = np.array([np.min(dc[:,:,0]),np.min(dc[:,:,1]),np.min(dc[:,:,2])]), np.array([np.max(dc[:,:,0]),np.max(dc[:,:,1]),np.max(dc[:,:,2])])
	s_mean, _ = get_mean_and_std(sc)
	t_mean, _ = get_mean_and_std(dc)
	s_mean = s_mean.astype(np.uint8)
	t_mean = t_mean.astype(np.uint8)
	# s_mean = np.array([np.median(sc[:,:,i]) for i in range(3)])
	# s_mean_max = np.array([np.median(sc[:,:,i][np.where(sc[:,:,i]>=s_mean[i])]) for i in range(3)])
	# s_mean_min = np.array([np.median(sc[:,:,i][np.where(sc[:,:,i]<s_mean[i])]) for i in range(3)])
	# t_mean = np.array([np.median(dc[:,:,i]) for i in range(3)])
	# t_mean_max = np.array([np.median(dc[:,:,i][np.where(dc[:,:,i]>=t_mean[i])]) for i in range(3)])
	# t_mean_min = np.array([np.median(dc[:,:,i][np.where(dc[:,:,i]<t_mean[i])]) for i in range(3)])
	t = [len(sc[:,:,0][np.where(sc[:,:,0]>=s_mean[0])]),len(sc[:,:,1][np.where(sc[:,:,1]>=s_mean[1])]),len(sc[:,:,2][np.where(sc[:,:,2]>=s_mean[2])])]
	s_mean_max = np.array([np.sum(sc[:,:,0][np.where(sc[:,:,0]>=s_mean[0])])/t[0], np.sum(sc[:,:,1][np.where(sc[:,:,1]>=s_mean[1])])/t[1],np.sum(sc[:,:,1][np.where(sc[:,:,1]>=s_mean[1])])/t[2]],dtype=np.uint8)
	t = [len(sc[:,:,0][np.where(sc[:,:,0]<s_mean[0])]),len(sc[:,:,1][np.where(sc[:,:,1]<s_mean[1])]),len(sc[:,:,2][np.where(sc[:,:,2]<s_mean[2])])]
	s_mean_min = np.array([np.sum(sc[:,:,0][np.where(sc[:,:,0]<s_mean[0])])/t[0], np.sum(sc[:,:,0][np.where(sc[:,:,0]<s_mean[0])])/t[1], np.sum(sc[:,:,0][np.where(sc[:,:,0]<s_mean[0])])/t[2]],dtype=np.uint8)
	print(s_min,s_mean_min,s_mean,s_mean_max,s_max)
	t = [len(dc[:,:,0][np.where(dc[:,:,0]>=t_mean[0])]),len(dc[:,:,1][np.where(dc[:,:,1]>=t_mean[1])]),len(dc[:,:,2][np.where(dc[:,:,2]>=t_mean[2])])]
	t_mean_max = np.array([np.sum(dc[:,:,0][np.where(dc[:,:,0]>=t_mean[0])])/t[0], np.sum(dc[:,:,1][np.where(dc[:,:,1]>=t_mean[1])])/t[1],np.sum(dc[:,:,1][np.where(dc[:,:,1]>=t_mean[1])])/t[2]],dtype=np.uint8)
	t = [len(dc[:,:,0][np.where(dc[:,:,0]<t_mean[0])]),len(dc[:,:,1][np.where(dc[:,:,1]<t_mean[1])]),len(dc[:,:,2][np.where(dc[:,:,2]<t_mean[2])])]
	t_mean_min = np.array([np.sum(dc[:,:,0][np.where(dc[:,:,0]<t_mean[0])])/t[0], np.sum(dc[:,:,0][np.where(dc[:,:,0]<t_mean[0])])/t[1], np.sum(dc[:,:,0][np.where(dc[:,:,0]<t_mean[0])])/t[2]],dtype=np.uint8)
	print(t_min,t_mean_min,t_mean,t_mean_max,t_max)
	# a = np.array([[s_min[0]**4,s_min[0]**3,s_min[0]**2,s_min[0],1],
	# 			  [s_mean_min[0]**4,s_mean_min[0]**3,s_mean_min[0]**2,s_mean_min[0],1],
	# 			  [s_mean[0]**4,s_mean[0]**3,s_mean[0]**2,s_mean[0],1],
	# 			  [s_mean_max[0]**4,s_mean_max[0]**3,s_mean_max[0]**2,s_mean_max[0],1],
	# 			  [s_max[0]**4,s_max[0]**3,s_max[0]**2,s_max[0],1]])
	# b = np.array([t_min[0],t_mean_min[0],t_mean[0],t_mean_max[0],t_max[0]])
	# aL,bL,cL,dL,eL = cramer(a,b)
	# print(aL,bL,cL,dL,eL)
	# a = np.array([[s_min[1]**4,s_min[1]**3,s_min[1]**2,s_min[1],1],
	# 			  [s_mean_min[1]**4,s_mean_min[1]**3,s_mean_min[1]**2,s_mean_min[1],1],
	# 			  [s_mean[1]**4,s_mean[1]**3,s_mean[1]**2,s_mean[1],1],
	# 			  [s_mean_max[1]**4,s_mean_max[1]**3,s_mean_max[1]**2,s_mean_max[1],1],
	# 			  [s_max[1]**4,s_max[1]**3,s_max[1]**2,s_max[1],1]])
	# b = np.array([t_min[1],t_mean_min[1],t_mean[1],t_mean_max[1],t_max[1]])
	# aA,bA,cA,dA,eA = cramer(a,b)
	# a = np.array([[s_min[2]**4,s_min[2]**3,s_min[2]**2,s_min[2],1],
	# 			  [s_mean_min[2]**4,s_mean_min[2]**3,s_mean_min[2]**2,s_mean_min[2],1],
	# 			  [s_mean[2]**4,s_mean[2]**3,s_mean[2]**2,s_mean[2],1],
	# 			  [s_mean_max[2]**4,s_mean_max[2]**3,s_mean_max[2]**2,s_mean_max[2],1],
	# 			  [s_max[2]**4,s_max[2]**3,s_max[2]**2,s_max[2],1]])
	# b = np.array([t_min[2],t_mean_min[2],t_mean[2],t_mean_max[2],t_max[2]])
	# aB,bB,cB,dB,eB = cramer(a,b)

	a = np.array([[s_mean_min[0]**2,s_mean_min[0],1],
				  [s_mean[0]**2,s_mean[0],1],
				  [s_mean_max[0]**2,s_mean_max[0],1]])
	b = np.array([t_mean_min[0],t_mean[0],t_mean_max[0]])
	aL,bL,cL = cramer(a,b)
	a = np.array([[s_mean_min[1]**2,s_mean_min[1],1],
				  [s_mean[1]**2,s_mean[1],1],
				  [s_mean_max[1]**2,s_mean_max[1],1]])
	b = np.array([t_mean_min[1],t_mean[1],t_mean_max[1]])
	aA,bA,cA = cramer(a,b)
	a = np.array([[s_mean_min[2]**2,s_mean_min[2],1],
				  [s_mean[2]**2,s_mean[2],1],
				  [s_mean_max[2]**2,s_mean_max[2],1]])
	b = np.array([t_mean_min[2],t_mean[2],t_mean_max[2]])
	aB,bB,cB = cramer(a,b)

	# lg = np.mat([[s_min[0]**2,s_min[0],1],
	# 			  [s_max[0]**2,s_max[0],1],
	# 			  [s_mean[0]**2,s_mean[0],1]])
	# lga = np.mat([[t_min[0],s_min[0],1],
	# 			  [t_max[0],s_max[0],1],
	# 			  [t_mean[0],s_mean[0],1]])
	# lgb = np.mat([[s_min[0]**2,t_min[0],1],
	# 			  [s_max[0]**2,t_max[0],1],
	# 			  [s_mean[0]**2,t_mean[0],1]])
	# lgc = np.mat([[s_min[0]**2,s_min[0],t_min[0]],
	# 			  [s_max[0]**2,s_max[0],t_max[0]],
	# 			  [s_mean[0]**2,s_mean[0],t_mean[0]]])
	# aL = np.linalg.det(lga)/np.linalg.det(lg)
	# bL = np.linalg.det(lgb)/np.linalg.det(lg)
	# cL = np.linalg.det(lgc)/np.linalg.det(lg)
	# lg = np.mat([[s_min[1]**2,s_min[1],1],
	# 			  [s_max[1]**2,s_max[1],1],
	# 			  [s_mean[1]**2,s_mean[1],1]])
	# lga = np.mat([[t_min[1],s_min[1],1],
	# 			  [t_max[1],s_max[1],1],
	# 			  [t_mean[1],s_mean[1],1]])
	# lgb = np.mat([[s_min[1]**2,t_min[1],1],
	# 			  [s_max[1]**2,t_max[1],1],
	# 			  [s_mean[1]**2,t_mean[1],1]])
	# lgc = np.mat([[s_min[1]**2,s_min[1],t_min[1]],
	# 			  [s_max[1]**2,s_max[1],t_max[1]],
	# 			  [s_mean[1]**2,s_mean[1],t_mean[1]]])
	# aA = np.linalg.det(lga)/np.linalg.det(lg)
	# bA = np.linalg.det(lgb)/np.linalg.det(lg)
	# cA = np.linalg.det(lgc)/np.linalg.det(lg)
	# lg = np.mat([[s_min[2]**2,s_min[2],1],
	# 			  [s_max[2]**2,s_max[2],1],
	# 			  [s_mean[2]**2,s_mean[2],1]])
	# lga = np.mat([[t_min[2],s_min[2],1],
	# 			  [t_max[2],s_max[2],1],
	# 			  [t_mean[2],s_mean[2],1]])
	# lgb = np.mat([[s_min[2]**2,t_min[2],1],
	# 			  [s_max[2]**2,t_max[2],1],
	# 			  [s_mean[2]**2,t_mean[2],1]])
	# lgc = np.mat([[s_min[2]**2,s_min[2],t_min[2]],
	# 			  [s_max[2]**2,s_max[2],t_max[2]],
	# 			  [s_mean[2]**2,s_mean[2],t_mean[2]]])
	# aB = np.linalg.det(lga)/np.linalg.det(lg)
	# bB = np.linalg.det(lgb)/np.linalg.det(lg)
	# cB = np.linalg.det(lgc)/np.linalg.det(lg)
	a = np.array([aL,aA,aB])
	b = np.array([bL,bA,bB])
	c = np.array([cL,cA,cB])
	# d = np.array([dL,dA,dB])
	# e = np.array([eL,eA,eB])
	# img_n = a*sc**4 + b*sc**3 + c*sc**2 + d*sc + e
	img_n = a*sc**2 + b*sc + c
	print(a,b,c)
	print(np.max(img_n),np.min(img_n))
	img_n = np.clip(img_n,0,255).astype(np.uint8)
	# mi,ma = np.min(img_n),np.max(img_n)
	# img_n = (img_n - mi) / (ma - mi) * 255
	dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2BGR)
	# dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
	# dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_HSV2RGB)
	# dst = cv2.convertScaleAbs(img_n)
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

def color_transfer_svd(sc, dc):
	sc = cv2.cvtColor(sc, cv2.COLOR_BGR2RGB)
	dc = cv2.cvtColor(dc, cv2.COLOR_BGR2RGB)
	h,w,_ = sc.shape
	sc = (sc.copy() / 255).reshape([-1,3]).T
	dc = (dc.copy() / 255).reshape([-1,3]).T
	mean_s = np.mean(sc, 1)
	mean_t = np.mean(dc, 1)
	cov_s = np.cov(sc)
	cov_t = np.cov(dc)
	U_s, A_s, _ = np.linalg.svd(cov_s)
	U_t, A_t, _ = np.linalg.svd(cov_t)
	rgbh_s = np.concatenate([sc,np.ones(shape=(1,sc.shape[-1]))])
	T_t = np.eye(4)
	T_t[0:3,3] = mean_t
	T_s = np.eye(4)
	T_s[0:3,3] = -mean_s
	R_t = scipy.linalg.block_diag(U_t,1)
	R_s = scipy.linalg.block_diag(np.linalg.inv(U_s),1)
	S_t = scipy.linalg.block_diag(np.diag(A_t)**.5,1)
	S_s = scipy.linalg.block_diag(np.reciprocal(np.diag(A_s)**.5),1)
	S_s[np.isinf(S_s)] = 0
	S_t[np.isinf(S_t)] = 0
	T_m = T_t @ R_t @ S_t @ S_s @ R_s @ T_s
	rgbh_e = T_m @ rgbh_s
	rgb_e = rgbh_e[0:3,:].T
	rgb_e = np.clip(rgb_e,0,1)
	rgb_e = (rgb_e.reshape([h,w,3])*255).astype(np.uint8)
	return cv2.cvtColor(cv2.convertScaleAbs(rgb_e), cv2.COLOR_RGB2BGR)

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
dst = color_transfer_maxmin(sc, dc)
cv2.imwrite('r_m.png',dst)
dst = color_transfer_svd(sc, dc)
cv2.imwrite('r_svd.png',dst)
# dst = dst / 255
# m = np.mean(dst)
# dst = (dst - m) * st + m
# dst = (np.clip(dst,0,1) * 255).astype(np.uint8)
# cv2.imwrite('r.png',dst)