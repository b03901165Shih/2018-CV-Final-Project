import cv2
import numpy as np
from matplotlib import pyplot as plt


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               flags = 2)

def find_disp(imgl, imgr):
    # Initiate SIFT detector
    sift = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(imgl,None)
    kp2, des2 = sift.detectAndCompute(imgr,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Initialize lists
    list_kp1 = []
    list_kp2 = []
    #match_list = []
    # For each match...
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        #if(abs(y1-y2)<=1.01 and x1>x2 and len(match_list)<70):
        #    match_list.append(mat)
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))
    #img3 = cv2.drawMatches(imgl,kp1,imgr,kp2,match_list, None,**draw_params)
    #plt.imshow(img3),plt.show()
    list_kp1 = np.array(list_kp1)
    list_kp2 = np.array(list_kp2)
    max_disp = 0
    for i in range(list_kp1.shape[0]):
        if(abs(list_kp1[i,1]-list_kp2[i,1])<=1.01):
            max_disp = np.maximum(list_kp1[i,0]-list_kp2[i,0], max_disp)
    return max_disp


def find_recify_shift(imgl, imgr):
    # Initiate SIFT detector
    sift = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(imgl,None)
    kp2, des2 = sift.detectAndCompute(imgr,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Initialize lists
    list_kp1 = []
    list_kp2 = []
    #match_list = []
    # For each match...
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        #if(abs(y1-y2)<=0.75 and x1-x2<=64 and x2>x1 and len(match_list)<70):
        #    match_list.append(mat)
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))
    #img3 = cv2.drawMatches(imgl,kp1,imgr,kp2,match_list, None,**draw_params)
    #plt.imshow(img3),plt.show()
    list_kp1 = np.array(list_kp1)
    list_kp2 = np.array(list_kp2)
    max_disp = 0; count = 0
    for i in range(list_kp1.shape[0]):
        if(abs(list_kp1[i,1]-list_kp2[i,1])<=0.75 and list_kp2[i,0]-list_kp1[i,0]<=64):	#<64: prevent outlier
            count += 1
            max_disp = np.maximum(list_kp2[i,0]-list_kp1[i,0], max_disp)
    print('Counted match:',count)
    return max_disp

#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
#plt.imshow(img3),plt.show()

def rectify_by_shift(imgl, imgr):
    rows, cols, ch = imgl.shape
    rect_shift = np.rint(find_recify_shift(imgl, imgr)).astype(np.int32)
    if(rect_shift==0):
        return imgl, imgr
    print('Rectify Shift:', rect_shift)
    #Translation
    M = np.float32([[1,0,-(rect_shift+6)],[0,1,0]])
    dst = cv2.warpAffine(imgr,M,(cols,rows))
    #cv2.imwrite('data_pre/Real_shift/TL'+str(i)+'_rectified.bmp',dst)
    #cv2.imwrite('data_pre/Real_shift/TR'+str(i)+'_rectified.bmp',imgr)
    return imgl, dst

'''
for i in range(10):
    img1 = cv2.imread('data/Real/TL'+str(i)+'.bmp',0)  
    img2 = cv2.imread('data/Real/TR'+str(i)+'.bmp',0) 
    rectify_by_shift(img1, img2)
'''	

'''
for i in range(10):
    img1 = cv2.imread('data_pre/Real_shift/TL'+str(i)+'_rectified.bmp',0)  
    img2 = cv2.imread('data_pre/Real_shift/TR'+str(i)+'_rectified.bmp',0) 
    print('#'+str(i)+' Real disp:',find_disp(img1, img2))

for i in range(10):
    img1 = cv2.imread('data/Synthetic/TL'+str(i)+'.png',0)  
    img2 = cv2.imread('data/Synthetic/TR'+str(i)+'.png',0) 
    print('#'+str(i)+' Synthetic reverse disp:',find_recify_shift(img1, img2))
    print('#'+str(i)+' Synthetic disp:',find_disp(img1, img2))
'''
	