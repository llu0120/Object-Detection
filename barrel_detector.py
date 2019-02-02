'''
ECE276A WI19 HW1
Blue Barrel Detector
'''
import os,cv2
import numpy as np
import math
from scipy.stats import multivariate_normal
class BarrelDetector():
    def __init__(self):
        '''
        Initilize your blue barrel detector with the attributes you need
			eg. parameters of your classifier
		'''
        self.prior_true = 0.34
        self.prior_darkblue = 0.33
        self.prior_brightblue = 0.33
        self.prior_blue = 0.5
        self.prior_nonblue = 0.5
        #71 without other
#        self.true_mean = [135.81946161,  75.43086253,  28.56619635]
#        self.dark_mean = [71.31109289, 56.08157004, 46.77927952]
#        self.bright_mean = [153.79783246, 139.73896499, 113.01294801]
#        self.false_mean = [ 89.54187225, 102.19346057, 110.55725679]
#        self.blue_mean = [139.25425343, 102.10773845,  66.56672277]
#        self.true_cov = [[2623.02246207,    0.,            0.        ],
#                         [   0.,         1323.37972428,    0.        ],
#                         [   0.,            0.,          879.33526435]]
#        self.dark_cov = [[934.90956379,   0.,           0.        ],
#                         [  0.,         515.2550793,    0.        ],
#                         [  0.,           0.,         456.54263351]]
#        self.bright_cov = [[1433.05018711,    0.,            0.        ],
#                           [   0.,         1693.47066258,    0.        ],
#                           [   0.,            0.,         2330.16814329]]
#        self.false_cov = [[3704.48614411,    0.,            0.        ],
#                          [   0.,         3418.78242249,    0.        ],
#                          [   0.,            0.,         3546.65905645]]
#        self.blue_cov = [[2402.28210907,    0.,            0.        ],
#                         [   0.,         2543.54893177,    0.        ],
#                         [   0.,            0.,         3164.9792454 ]]
#       75---> segmentation perfect
        self.true_mean = [162.47715197,  96.92821078,  43.94440053]
        self.dark_mean = [71.31109289, 56.08157004, 46.77927952]
        self.bright_mean = [153.79783246, 139.73896499, 113.01294801]
        self.false_mean = [ 88.99102282, 101.82667983, 109.28615979]
        self.blue_mean = [154.25979013, 109.9775994,   68.84860029]
        self.true_cov = [[3916.35808505,    0.,           0.        ],
                         [   0.,         2716.35239984,    0.        ],
                         [   0.,            0.,         2288.95291542]]
        self.dark_cov = [[934.90956379,   0.,           0.        ],
                         [  0.,         515.2550793,    0.        ],
                         [  0.,           0.,         456.54263351]]
        self.bright_cov = [[1433.05018711,    0.,            0.        ],
                           [   0.,         1693.47066258,    0.        ],
                           [   0.,            0.,         2330.16814329]]
        self.false_cov = [[3587.1514229,     0.,            0.        ],
                          [   0.,         3311.2278334,     0.        ],
                          [   0.,            0.,         3459.35294767]]
        self.blue_cov = [[3284.73579487,    0.,            0.        ],
                         [   0.,         2806.38033207,    0.        ],
                         [   0.,            0.,         3290.41175657]]
        #66 
#        self.true_mean = [125.10485467,  70.67048176,  30.93339098]
#        self.dark_mean = [70.44399427, 55.87152787, 46.76423336]
#        self.bright_mean = [153.61364672, 139.52922924, 112.75451711]
#        self.false_mean = [88.32621069, 100.92730403, 109.47819877]
#        self.blue_mean = [135.00409623, 103.41077814,  72.35397203]
#        
#        self.true_cov = [[2502.79670168,    0.,            0.        ],
#                         [   0.,         1292.70034661,    0.        ],
#                         [   0.,            0.,          927.6760101 ]]
#        self.dark_cov = [[908.5840175,    0.,           0.        ],
#                         [  0.,         513.49123299,   0.        ],
#                         [  0.,           0.,         456.80833641]]
#        self.bright_cov = [[1417.02029292,    0.,            0.        ],
#                           [   0.,         1672.38479407,    0.        ],
#                           [   0.,            0.,         2297.7165868 ]]
#        self.false_cov = [[3765.16041786,    0.,            0.        ],
#                          [   0.,         3474.13141904,    0.        ],
#                          [   0.,            0.,         3617.03943398]]
#        self.blue_cov = [[2375.54007235,    0.,            0.        ],
#                         [   0.,         2696.46675801,    0.        ],
#                         [   0.,            0.,         3160.52328492]]
        
    def Gaussian(self,x,mean,cov,k):
        det_cov = np.linalg.det(cov)
        inverse_cov = np.linalg.pinv(cov) 
        diff_xmean = x-mean
    
        e_term = np.exp(-0.5*np.dot(np.dot(diff_xmean,inverse_cov),(np.array([diff_xmean]).T)))
        front_term = 1/(math.sqrt(det_cov*((2*math.pi)**k)))
        return front_term * e_term
    
    def segment_image(self, img):
        '''
			Calculate the segmented image using a classifier
			eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		# YOUR CODE HERE
        mask_img = np.zeros(shape=(800,1200),dtype = np.uint8)
        
        Ptrue_conditional = multivariate_normal.pdf(img, self.true_mean, self.true_cov)
        Pdarkblue_conditional = multivariate_normal.pdf(img,self.dark_mean, self.dark_cov)
        Pbrightblue_conditional = multivariate_normal.pdf(img, self.bright_mean, self.bright_cov)
        Pfalse_conditional = multivariate_normal.pdf(img, self.false_mean, self.false_cov)
        Pblue_conditional = multivariate_normal.pdf(img, self.blue_mean, self.blue_cov)
        Pblue = Pblue_conditional*self.prior_blue
        Pnonblue = Pfalse_conditional*self.prior_nonblue
        Ptrue = Ptrue_conditional*self.prior_true
        Pdarkblue = Pdarkblue_conditional*self.prior_darkblue
        Pbrightblue = Pbrightblue_conditional*self.prior_brightblue
        
        classify1 = Pblue > Pnonblue
        classify1 = np.reshape(classify1,(1,800*1200))
        classify2 = Ptrue > Pdarkblue
        classify2 = np.reshape(classify2,(1,800*1200))
        classify3 = Ptrue > Pbrightblue
        classify3 = np.reshape(classify3,(1,800*1200))
        
        mask_img = np.zeros((1,(800*1200)),dtype = np.uint8)
        for i in range(len(classify1[0])):
           if classify1[0][i] == True and classify3[0][i] == True and classify3[0][i] == True:
               mask_img[0][i] = 1
           else:
               mask_img[0][i] = 0
        mask_img = np.reshape(mask_img,(800,1200))
        #for x in range(0,len(img)):#(len(img_test[0])):
        #    for y in range(0,len(img[0])):
                #Pcondition = [self.Gaussian(img[x][y], self.true_mean, self.true_cov,3),self.Gaussian(img[x][y], self.dark_mean, self.dark_cov,3),self.Gaussian(img[x][y], self.bright_mean, self.bright_cov,3),self.Gaussian(img[x][y], self.false_mean, self.false_cov,3),self.Gaussian(img[x][y], self.blue_mean, self.blue_cov,3)]
                #Ptrue_conditional = self.Gaussian(img[x][y], self.true_mean, self.true_cov,3)
                #Pdarkblue_conditional = self.Gaussian(img[x][y], self.dark_mean, self.dark_cov,3)
                #Pbrightblue_conditional = self.Gaussian(img[x][y], self.bright_mean, self.bright_cov,3)
                #Pfalse_conditional = self.Gaussian(img[x][y], self.false_mean, self.false_cov,3)
                #Pblue_conditional = self.Gaussian(img[x][y], self.blue_mean, self.blue_cov,3)
               
                #BDR
                #P = [Pcondition[4]*self.prior_blue,Pcondition[3]*self.prior_nonblue,Pcondition[0]*self.prior_true,Pcondition[1]*self.prior_darkblue,Pcondition[2]*self.prior_brightblue]
                #Pblue = Pblue_conditional*self.prior_blue
                #Pnonblue = Pfalse_conditional*self.prior_nonblue
               
                #Ptrue = Ptrue_conditional*self.prior_true
                #Pdarkblue = Pdarkblue_conditional*self.prior_darkblue
                #Pbrightblue = Pbrightblue_conditional*self.prior_brightblue
                
          #      if P[0] > P[1]:#Pblue > Pnonblue:
          #          if max(P[2],P[3],P[4]) == P[2]:
          #              mask_img[x][y] = 1
          #          elif max(P[2],P[3],P[4]) != P[2]:#max(Ptrue,Pdarkblue,Pbrightblue) != Ptrue: 
          #              mask_img[x][y] = 0
          #      elif P[1] > P[0]:#Pnonblue > Pblue: 
          #          mask_img[x][y] = 0
        
        return mask_img
    def get_bounding_box(self, img):
        '''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		# YOUR CODE HERE
        boxes = []
        kernel = np.ones((5,5),np.uint8)
        mask_img = self.segment_image(img)
        mask_img = cv2.erode(mask_img,kernel)
        mask_img = cv2.dilate(mask_img,kernel,iterations = 3)
        contours, hierarchy = cv2.findContours(mask_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
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
            if h/w < 1.4 or h/w > 3: 
                continue
            if  w/h > 0.4 and w/h < 0.6:     
                boxes.append([x1,y1,x2,y2])
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) 
            elif h/w > 0.4 and h/w < 0.6:
                boxes.append([x1,y1,x2,y2])
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) 
            elif w*h > 1000:
                boxes.append([x1,y1,x2,y2])
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) 
            else:    
                rect = cv2.minAreaRect(i)
#                if rect < 1000: 
#                    continue
                box = np.int0(cv2.boxPoints(rect))
                box = box.tolist()
                box.append([x1,y1,x2,y2])
#                    box = np.int0(box)
#                    cv2.drawContours(img,[box],0,(0,255,0),2)
#                    boxes.append(box[1][0],box[1][1],box[3][0],box[3][1])
        #cv2.imshow("img", img) 
        #cv2.waitKey(0) 
        #cv2.destroyAllWindows()
        boxes = sorted(boxes, key=lambda x: x[1])
        return boxes
#if __name__ == '__main__':
#	folder = "trainset"
#    my_detector = BarrelDetector()
    
	#for filename in os.listdir(folder):
	# read one test image
#    img = cv2.imread('46.png')
#	cv2.imshow('image', img)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()
	#Display results:
	#(1) Segmented images
#    mask_img = my_detector.segment_image(img)
    #(2) Barrel bounding box
#    boxes = my_detector.get_bounding_box(img)
#    cv2.imshow('image',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope


