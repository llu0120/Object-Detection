#!/usr/bin/env python
# coding: utf-8

# In[4]:


cd /Users/LuLienHsi/Desktop/UCSD_Documents/2019_Winter/ECE276A_Sensing&EstimationRobotics/ECE276A_HW1/trainset


# In[13]:


from roipoly import RoiPoly
from matplotlib import pyplot as plt
import cv2
import matplotlib
import numpy as np
import imageio

get_ipython().run_line_magic('matplotlib', 'notebook')
matplotlib.use('TkAgg')
img = imageio.imread('67.png')
#img = cv2.imread('67.png')
plt.imshow(img)
my_roi = RoiPoly(color='b') # draw new ROI in blue color


# In[14]:


plt.imshow(img)
my_roi.display_roi()


# In[394]:


plt.imshow(img)
my_roi1 = RoiPoly(color='r') # draw new ROI in red color


# In[317]:


plt.imshow(img)
my_roi.display_roi()


# In[318]:


plt.imshow(img)
#my_roi2 = RoiPoly(color='g') # draw new ROI in red color


# In[400]:


plt.imshow(img)
my_roi.display_roi()
#my_roi1.display_roi()
#my_roi2.display_roi()


# In[15]:


img_one = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = my_roi.get_mask(img_one) #+ my_roi1.get_mask(img_one) #+ my_roi2.get_mask(img_one)
plt.imshow(mask)


# In[16]:


import numpy as np
np.save('mask_67_darktblue',mask)


# In[102]:


np.load('mask_26.npy')

