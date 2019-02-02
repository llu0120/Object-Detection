#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 22:41:01 2019

@author: LuLienHsi
"""
import numpy as np
import cv2
import os

os.chdir(r'/Users/LuLienHsi/Desktop/UCSD_Documents/2019_Winter/ECE276A_Sensing&EstimationRobotics/ECE276A_HW1/trainset') #进入指定的目录
mask_img = np.load('mask_img.npy')
img = cv2.imread('42.png')

boxes = []
contours, hierarchy = cv2.findContours(mask_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
#cv2.drawContours(img,contours,-1,(0,0,255),3) 

#cnt = contours[0]
#area = cv2.contourArea(cnt)
#perimeter = cv2.arcLength(cnt,True)
#epsilon = 0.1*cv2.arcLength(cnt,True)
#approx = cv2.approxPolyDP(cnt,epsilon,True)
for i in range(0,len(contours)): 
    x, y, w, h = cv2.boundingRect(contours[i])  
    x1 = x 
    x2 = x + w
    y1 = y
    y2 = y + h
    cnt = contours[i]
    if h/w < 1.4 or h/w > 3: 
        continue
    if  w/h > 0.4 and w/h < 0.6 and w*h < 800000:     
        boxes.append([x1,y1,x2,y2])
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) 
    elif h/w > 0.4 and h/w < 0.6 and w*h < 800000:
        boxes.append([x1,y1,x2,y2])
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) 
    elif w*h > 1000:
        if w*h < 2000:
            boxes.append([x1,y1,x2,y2])
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) 
        elif w*h>800000:
            continue
    else:    
        rect = cv2.minAreaRect(i)
#                if rect < 1000: 
#                    continue
        box = np.int0(cv2.boxPoints(rect))
        box = box.tolist()
        box.append([x1,y1,x2,y2])
#                rect = cv2.minAreaRect(cnt)
#                if rect < 1000: 
#                    continue
#                else: 
#                    box = cv2.boxPoints(rect)
#                    box = np.int0(box)
#                    cv2.drawContours(img,[box],0,(0,255,0),2)
#                    boxes.append(box[1][0],box[1][1],box[3][0],box[1])
#        rect = cv2.minAreaRect(cnt)
#        if rect < 1000: 
#            continue
#        else: 
#            box = cv2.boxPoints(rect)
#            box = np.int0(box)
#            cv2.drawContours(img,[box],0,(0,255,0),2)
#            print(box)
            #for i in box:
            #    perimeter = cv2.arcLength(cnt,True)
            #    if perimeter > 100:
            #    #boxes.append(box[])
boxes = sorted(boxes, key=lambda x: x[1])
print(boxes)
cv2.imshow("img", img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

