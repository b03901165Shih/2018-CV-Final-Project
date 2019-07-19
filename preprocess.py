import os
import sys
import numpy as np
import random
import re
import cv2
import glob
import h5py
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

def flip(img,axes):
    if (axes == 0) :
        #vertical flip
        return cv2.flip( img, 0 )
    elif(axes == 1):
        #horizontal flip
        return cv2.flip( img, 1 )
    elif(axes == -1):
        #both direction
        return cv2.flip( img, -1 ) 

def readPFM(file):
    file = open(file, 'rb')
    header = file.readline().rstrip()
    header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def augmentPFM(img, option): 
    output = np.zeros(shape=img.shape)
    if(option==0):	#upside-down
        output = flip(img,0)
    elif(option==3):# left-right
        output = flip(img,1)   
    else:
        output = img
    return output

def augment(img_path, option): 
    img = cv2.imread(img_path)
    output = np.zeros(shape=img.shape)
    if(option==0):	# upside-down
        output = flip(img,0)
    elif(option==1): #brightness, sharpness...
        img = Image.open(img_path)
        output = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.7,1.3))#Brightness
        output = ImageEnhance.Color(output).enhance(np.random.uniform(0.7,1.3))#Color
        output = ImageEnhance.Contrast(output).enhance(np.random.uniform(0.7,1.3))#Contrast
        output = ImageEnhance.Sharpness(output).enhance(np.random.uniform(0.7,1.3))#Sharpness
    elif(option==2):	# Noise+Blur
        noise1 = np.random.randn(img.shape[0], img.shape[1], img.shape[2])*np.random.randint(10,20)
        blurk = np.random.randint(3,6)
        output = cv2.blur(img,(blurk,blurk))+noise1
    elif(option==3):# left-right
        output = flip(img,1)        
    else:
        output = img
    # to gray (Assume BGR)
    if(option!=1 and output.shape[-1]==3):
        weightB = np.random.uniform(0.1,0.4)
        weightG = np.random.uniform(0.5,0.6)
        output = 0.114 * output[:,:,0] + 0.587 * output[:,:,1] + 0.299 * output[:,:,2]
    if(option==1):
        output = output.convert('L')
    return np.array(output)

	
def getTrain():
    NEGATIVE_LOW = 2
    NEGATIVE_HIGH = 15
    win_r = 5
    cal = 0
    currPath='./training/'
    X1_train = []
    X2_train = []
    y_train  = []
    if(True):
            #grpLeft = f1.create_group("left")
            #grpRightPos = f1.create_group("rightPos")
            #grpRightNeg = f1.create_group("rightNeg")
            for k in range(20):
                if(k==10):
                    continue
                randomNumber = str(k)
                #option = np.random.randint(0,5)
                for option in range(4,5):
                    print('\nImage :',k,'; option:',option)
                    imgL = augment(currPath+'image_0/TL'+randomNumber+'.png',option)
                    imgR = augment(currPath+'image_1/TR'+randomNumber+'.png',option)
                    H, W = imgL.shape
                    stride = (H>1024 or W>1024)*5+(not(H>1024 or W>1024))*1
                    start  = (stride==5)*np.random.randint(0,5)+(stride!=5)*0
                    imgGround = augmentPFM(readPFM(currPath+'disp_noc/TLD'+randomNumber+'.pfm'),option)
                    #plt.imshow(np.concatenate([imgL, imgR, imgGround],1));plt.show()
                    imgL = (cv2.copyMakeBorder(imgL,win_r,win_r,win_r,win_r,cv2.BORDER_REPLICATE).astype('float32'))/255-0.5
                    imgR = (cv2.copyMakeBorder(imgR,win_r,win_r,win_r,win_r,cv2.BORDER_REPLICATE).astype('float32'))/255-0.5
                    print('Row imgL.shape:',imgL.shape)
                    for j in range(start,H,stride):
                        print('Row index:',j, end='\r')
                        for i in range(start,W,stride):
                            if(imgGround[j,i]==np.inf):
                                continue
                            #Left Patch positive and Negative
                            o_positive = 0#random.randint(-1,1)
                            sign = random.choice([-1,1])
                            o_negative = sign * random.randint(NEGATIVE_LOW, NEGATIVE_HIGH)
                            imgLPatch = imgL[j:j+2*win_r+1, i:i+2*win_r+1]
                            disp = (option==3)*(-int(np.rint(imgGround[j,i])))+(option!=3)*(int(np.rint(imgGround[j,i])))
                            if i+o_positive-disp >= 0 and i+o_negative-disp>=0 and i+o_positive-disp < W and i+o_negative-disp < W:
                                imgRPatchPos = imgR[j:j+2*win_r+1,i+o_positive-disp:i+o_positive-disp+2*win_r+1]
                                imgRPatchNeg = imgR[j:j+2*win_r+1,i+o_negative-disp:i+o_negative-disp+2*win_r+1]
                                #if(option==3 and i >100 and j>100):
                                #    plt.imshow(np.concatenate([imgLPatch, imgRPatchPos],1)); plt.show()
                                X1_train.append(imgLPatch)#grpLeft.create_dataset(name=str(cal), data=imgLPatch)
                                X1_train.append(imgLPatch)
                                X2_train.append(imgRPatchPos)#grpRightPos.create_dataset(str(cal), data=imgRPatchPos)
                                X2_train.append(imgRPatchNeg)#grpRightNeg.create_dataset(str(cal), data=imgRPatchNeg)
                                y_train.append(1);y_train.append(0)
                                cal=cal +1
    print('\nCal:',cal)
    X1 = np.array(X1_train)
    X2 = np.array(X2_train)
    y = np.array(y_train)
    return (X1,X2,y)


'''
(X1_train, X2_train, y_train) = getTrain()

print(X1_train.shape)
print(X2_train.shape)
print(y_train.shape)

for i in range(12550,12580):
    print(y_train[i])
    plt.imshow(np.concatenate([X1_train[i], X2_train[i]],1)); plt.show()
'''

''''
if __name__=='__main__':
    NEGATIVE_LOW = 4
    NEGATIVE_HIGH = 10
    cal = 0
    currPath='./training/'
    with h5py.File("trainPatches.hdf5", "w") as f1:
            grpLeft = f1.create_group("left")
            grpRightPos = f1.create_group("rightPos")
            grpRightNeg = f1.create_group("rightNeg")
            for k in range(10):
                print('\nImage :',k)
                randomNumber = str(k)
                imgL = cv2.imread(currPath+'image_0/TL'+randomNumber+'.png')/255#((f2['left/'+randomNumber][()]))
                imgR = cv2.imread(currPath+'image_1/TR'+randomNumber+'.png')/255#((f2['right/'+randomNumber][()]))
                imgGround = readPFM(currPath+'disp_noc/TLD'+randomNumber+'.pfm')
                print('Row imgL.shape:',imgL.shape)
                for j in range(0,imgL.shape[0],2):
                    print('Row index:',j, end='\r')
                    for i in range(0,imgL.shape[1],2):
                        o_positive = random.randint(-1,1)
                        sign = random.choice([-1,1])
                        o_negative = sign * random.randint(NEGATIVE_LOW, NEGATIVE_HIGH)
                        if j-5>=0 and i-5>=0 and j+6<imgL.shape[0] and i+6<imgL.shape[1]:
                            imgLPatch = imgL[j-5:j+6, i-5:i+6]
                            disp = int(float(imgGround[j,i]))
                            #print(disp, i, j)
                            if np.sum(np.array(imgLPatch))==0 or disp==0:
                                continue
                            if i+o_positive-disp+6 < imgL.shape[1] and i+o_positive-disp-5>=0 and i+o_negative-disp-5 >= 0 and i+o_negative-disp+6 < imgL.shape[1]:
                                imgRPatchPos = imgR[j-5:j+6,i+o_positive-disp-5:i+o_positive-disp+6]
                                #plt.imshow(np.concatenate([imgLPatch, imgRPatchPos],1)); plt.show()
                                imgRPatchNeg = imgR[j-5:j+6,i+o_negative-disp-5:i+o_negative-disp+6]
                                #grpLeft.create_dataset(name=str(cal), data=imgLPatch)
                                #grpRightPos.create_dataset(str(cal), data=imgRPatchPos)
                                #grpRightNeg.create_dataset(str(cal), data=imgRPatchNeg)
                                cal=cal +1
    print('\nCal:',cal)
'''