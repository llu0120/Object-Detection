#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:33:47 2019

@author: LuLienHsi
"""

import numpy as np
import cv2
import os
os.chdir(r'/Users/LuLienHsi/Desktop/UCSD_Documents/2019_Winter/ECE276A_Sensing&EstimationRobotics/ECE276A_HW1/trainset') #进入指定的目录
true_rgbdata = []
false_rgbdata = []
darkblue_rgbdata = []
brightblue_rgbdata = []
blue_rgbdata = []

np.array(true_rgbdata)
np.array(false_rgbdata)
np.array(darkblue_rgbdata)
np.array(brightblue_rgbdata)
np.array(blue_rgbdata)

for i in range(1,72):#1,47
    mask= np.load('mask_%d.npy' %i)
    dark_mask = np.load('mask_%d_darkblue.npy' %i)
    bright_mask = np.load('mask_%d_brightblue.npy' %i)
    img_train = cv2.imread('%d.png'%i)
    x = len(img_train)
    y = len(img_train[1])
    for i in range(x):
        for j in range(y): 
            if mask[i][j] == True:
                true_rgbdata.append(img_train[i,j])
            if dark_mask[i][j] == True:
                darkblue_rgbdata.append(img_train[i,j])
            if bright_mask[i][j] == True:
                brightblue_rgbdata.append(img_train[i,j])
            if mask[i][j] == False and dark_mask[i][j] == False and bright_mask[i][j] == False: 
                false_rgbdata.append(img_train[i,j])
#%%
blue_rgbdata.extend(true_rgbdata)  
blue_rgbdata.extend(darkblue_rgbdata)
blue_rgbdata.extend(brightblue_rgbdata)


#%%
#caculate mean & standard deviation for r,g,b (true)
#caculate mean & standard deviation for r,g,b (false)
os.chdir(r'/Users/LuLienHsi/Desktop/UCSD_Documents/2019_Winter/ECE276A_Sensing&EstimationRobotics/ECE276A_HW1/parameters') #进入指定的目录

true_mean = np.mean(true_rgbdata, axis = 0) 
np.save('true_mean',true_mean)
dark_mean = np.mean(darkblue_rgbdata, axis = 0) 
np.save('dark_mean',dark_mean)
bright_mean = np.mean(brightblue_rgbdata, axis = 0) 
np.save('bright_mean',bright_mean)
false_mean = np.mean(false_rgbdata, axis = 0) 
np.save('false_mean',false_mean)
blue_mean = np.mean(blue_rgbdata, axis = 0)
np.save('blue_mean',blue_mean)

true_cov = np.diag(np.var(true_rgbdata, axis = 0))
np.save('true_cov',true_cov)
dark_cov = np.diag(np.var(darkblue_rgbdata, axis = 0))
np.save('dark_cov',dark_cov)
bright_cov = np.diag(np.var(brightblue_rgbdata, axis = 0))
np.save('bright_cov',bright_cov)
false_cov = np.diag(np.var(false_rgbdata,axis = 0))
np.save('false_cov',false_cov)
blue_cov = np.diag(np.var(blue_rgbdata,axis = 0))
np.save('blue_cov',blue_cov)

#%% 
print(true_mean)
print(dark_mean)
print(bright_mean)
print(false_mean)
print(blue_mean)

print(true_cov)
print(dark_cov)
print(bright_cov)
print(false_cov)
print(blue_cov)