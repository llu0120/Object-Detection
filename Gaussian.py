#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 21:26:57 2019

@author: LuLienHsi
"""
#%%
import numpy as np
import cv2
import math 
import os
from scipy.stats import multivariate_normal
#%%
os.chdir(r'/Users/LuLienHsi/Desktop/UCSD_Documents/2019_Winter/ECE276A_Sensing&EstimationRobotics/ECE276A_HW1/parameters') #进入指定的目录

true_mean = np.load('true_mean.npy')
dark_mean = np.load('dark_mean.npy')
bright_mean = np.load('bright_mean.npy')
false_mean = np.load('false_mean.npy')
blue_mean = np.load('blue_mean.npy')

true_cov = np.load('true_cov.npy')
dark_cov = np.load('dark_cov.npy')
bright_cov = np.load('bright_cov.npy')
false_cov = np.load('false_cov.npy')
blue_cov = np.load('blue_cov.npy')
#%%
def Gaussian(x,mean,cov,k):
    
    det_cov = np.linalg.det(cov)
    inverse_cov = np.linalg.pinv(cov) 
    diff_xmean = x-mean
    
    e_term = np.exp(-0.5*np.dot(np.dot((diff_xmean,inverse_cov),(np.array([diff_xmean]).T))))

    front_term = 1/(math.sqrt(det_cov*((2*math.pi)**k)))
    return front_term * e_term


#%%
#single Gaussian and BDR
os.chdir(r'/Users/LuLienHsi/Desktop/UCSD_Documents/2019_Winter/ECE276A_Sensing&EstimationRobotics/ECE276A_HW1/trainset') #进入指定的目录
prior_true = 0.7
prior_darkblue = 0.15
prior_brightblue = 0.15

prior_blue = 0.25
prior_nonblue = 0.75

img = cv2.imread('42.png')
#mask_img = np.zeros(shape=(800,1200),dtype = np.uint8)
#for x in range(0,len(img)):#(len(img_test[0])):
#    for y in range(0,len(img[0])):
#        Ptrue_conditional = Gaussian(img[x][y], true_mean, true_cov,3)
#        Pdarkblue_conditional = Gaussian(img[x][y], dark_mean, dark_cov,3)
#        Pbrightblue_conditional = Gaussian(img[x][y], bright_mean, bright_cov,3)
#        Pfalse_conditional = Gaussian(img[x][y], false_mean, false_cov,3)
#        Pblue_conditional = Gaussian(img[x][y], blue_mean, blue_cov,3)
#        
        #BDR
#        Pblue = Pblue_conditional*prior_blue
#        Pnonblue = Pfalse_conditional*prior_nonblue
#        Ptrue = Ptrue_conditional*prior_true
#        Pdarkblue = Pdarkblue_conditional*prior_darkblue
#        Pbrightblue = Pbrightblue_conditional*prior_brightblue
#        
#        if Pblue > Pnonblue:
#            if max(Ptrue,Pdarkblue,Pbrightblue) == Ptrue:
#                mask_img[x][y] = 1
#            elif max(Ptrue,Pdarkblue,Pbrightblue) != Ptrue: 
#                mask_img[x][y] = 0
#        elif Pnonblue > Pblue: 
#            mask_img[x][y] = 0

#mask_img = np.zeros(shape=(800,1200),dtype = np.uint8)
        #Pcondition = [self.Gaussian(img, self.true_mean, self.true_cov,3),self.Gaussian(img, self.dark_mean, self.dark_cov,3),self.Gaussian(img, self.bright_mean, self.bright_cov,3),self.Gaussian(img, self.false_mean, self.false_cov,3),self.Gaussian(img, self.blue_mean, self.blue_cov,3)]
        #P = [Pcondition[4]*self.prior_blue,Pcondition[3]*self.prior_nonblue,Pcondition[0]*self.prior_true,Pcondition[1]*self.prior_darkblue,Pcondition[2]*self.prior_brightblue]
        
#Ptrue_conditional = Gaussian(img, true_mean, true_cov,3)
#Pdarkblue_conditional = Gaussian(img, dark_mean, dark_cov,3)       
#Pbrightblue_conditional = Gaussian(img, bright_mean, bright_cov,3)
#Pfalse_conditional = Gaussian(img, false_mean, false_cov,3)
#Pblue_conditional = Gaussian(img, blue_mean, blue_cov,3)
        

Ptrue_conditional = multivariate_normal.pdf(img, true_mean, true_cov)
Pdarkblue_conditional = multivariate_normal.pdf(img,dark_mean, dark_cov)
Pbrightblue_conditional = multivariate_normal.pdf(img, bright_mean, bright_cov)
Pfalse_conditional = multivariate_normal.pdf(img, false_mean, false_cov)
Pblue_conditional = multivariate_normal.pdf(img, blue_mean, blue_cov)     

Pblue = Pblue_conditional*prior_blue
Pnonblue = Pfalse_conditional*prior_nonblue        
Ptrue = Ptrue_conditional*prior_true
Pdarkblue = Pdarkblue_conditional*prior_darkblue
Pbrightblue = Pbrightblue_conditional*prior_brightblue
        
classify1 = Pblue > Pnonblue
classify1 = np.reshape(classify1,(1,800*1200))
classify2 = Ptrue > Pdarkblue 
classify2 = np.reshape(classify2,(1,800*1200))
classify3 = Ptrue > Pbrightblue
classify3 = np.reshape(classify3,(1,800*1200))

mask_img = np.zeros((1,(800*1200)),dtype = np.uint8)


for i in range(len(classify1[0])):
    if classify1[0][i] == True and classify2[0][i] == True and classify3[0][i] == True:
        mask_img[0][i] = 0
    else:
        mask_img[0][i] = 1
mask_img = np.reshape(mask_img,(800,1200))


#%%
import matplotlib.pyplot as plt
plt.imshow(mask_img, cmap=plt.cm.binary)
plt.show()

#%%
np.save('mask_img',mask_img)






