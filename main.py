import numpy as np
import argparse
import cv2
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import matplotlib.pyplot as plt
import chainer
import mcnet
from util import writePFM
from img_rectify import *
from filters import GuidedFilter


parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL0.pfm', type=str, help='left disparity map')
parser.add_argument('--setting', default= -1, type=int)

np.random.seed(2019)

#USE_GUIDED is useful only when USE_GRAPHCUT is True
USE_GRAPHCUT 	= False
USE_GUIDED   	= False
#Priority: USE_MCCNN_SELF> USE_MCCNN
USE_MCCNN	 	= False
USE_MCCNN_SELF	= False

if(USE_GRAPHCUT):
    import maxflow as mf
if(USE_MCCNN_SELF):
    from keras.models import load_model
    from keras import optimizers

cost_alpha = 0.9#0.78
cost_tau1 = 10
cost_tau2 = 2
cost_window = 0
guide_r   = -10
'''
if(USE_MCCNN_SELF):
    guide_r   = 5
elif(USE_MCCNN):
    guide_r   = 15
if(USE_GRAPHCUT):
    guide_r   = 30
'''
gudie_eps = 0.01
bilateral_sigmas = [30,0.18]
bilateral_r  = 30
#Graph cut
truncate_cost_window = 0	# for graph cut subpixel truncate cost
GC_cost_window = 20
spatial_r = 20
cheat_r   = 10
#Guided cost
GC_guide_r    = 10
GC_guide_eps  = (0.01)**2
#smoothness
lambda_s=-10
'''
if(USE_MCCNN):
    lambda_s  = 1
else:
    lambda_s  = 10
'''
tau_dis = 1
gamma = 10
eps = 0.01
#MCCNN cost
cost_mctau = 0.5

def computeDisp(Il, Ir, max_disp, model):
    h, w, ch = Il.shape
    labels = np.zeros((h, w)).astype('uint8')
    max_cost = ((1-cost_alpha)*cost_tau1+cost_alpha*cost_tau2)*3
    #================================================================================================
    #                                                                                               #
    #================================================================================================
    # >>> Cost computation
    # TODO: Compute matching cost from Il and Ir
    set_img_shift = img_shift(Ir, max_disp, True)
    cost_set = np.zeros((max_disp+1, Ir.shape[0], Ir.shape[1])).astype('uint8')
    for i in range(max_disp+1):
        print('Calculating cost for disparity =', i, 'out of',max_disp)
        if(USE_MCCNN_SELF):
            cost_set[i] = ((1-predict_cost_MCCNN(Il, set_img_shift[i], model))*255).astype('uint8')
        elif(USE_MCCNN):
            batch = chainer.dataset.concat_examples([(np.moveaxis(Il.astype('float32'), -1, 0), np.moveaxis(set_img_shift[i].astype('float32'), -1, 0), np.array([max_disp]))], 0)
            with chainer.no_backprop_mode():
                dst = np.minimum(cost_mctau,chainer.cuda.to_cpu(model(*batch)[0,0].array))
                cost_set[i] = ((dst/cost_mctau)*255).astype('uint8')
        else:
            res = truncated_cost_computation(Il, set_img_shift[i], cost_alpha, cost_tau1, cost_tau2, cost_window)
            cost_set[i] = (((res)/max_cost)*255).astype('uint8')
    #================================================================================================
    #                                                                                               #
    #================================================================================================
    # >>> Cost aggregation
    # TODO: Refine cost by aggregate nearby costs
    guide_image = cv2.ximgproc.createGuidedFilter(Il, guide_r, gudie_eps)
    for i in range(max_disp+1):
        cost_set[i] = guide_image.filter(cost_set[i])
    #================================================================================================
    #                                                                                               #
    #================================================================================================
    # >>> Disparity optimization
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    disparity_left = np.argmin(cost_set, axis=0)
    #plt.imshow(disparity_left, cmap='gray')
    #plt.show()
    #================================================================================================
    #                                                                                               #
    #================================================================================================
    # >>> Disparity refinement
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    set_img_shift = img_shift(Il, max_disp, False)
    cost_set = np.zeros((max_disp+1, Ir.shape[0], Ir.shape[1])).astype('uint8')
    guide_image = cv2.ximgproc.createGuidedFilter(Ir, guide_r, gudie_eps)
    for i in range(max_disp+1):
        print('Calculating cost for disparity =', i, 'out of',max_disp)
        if(USE_MCCNN_SELF):
            dst = ((1-predict_cost_MCCNN( set_img_shift[i], Ir, model))*255).astype('uint8')
        elif(USE_MCCNN):
            batch = chainer.dataset.concat_examples([(np.moveaxis(set_img_shift[i].astype('float32'), -1, 0), np.moveaxis(Ir.astype('float32'), -1, 0), np.array([max_disp]))], 0)
            with chainer.no_backprop_mode():
                dst = np.minimum(cost_mctau,chainer.cuda.to_cpu(model(*batch)[0,0].array))
                dst = ((dst/cost_mctau)*255).astype('uint8')
        else:
            res = truncated_cost_computation(Ir, set_img_shift[i], cost_alpha, cost_tau1, cost_tau2, cost_window)
            dst = (((res)/max_cost)*255).astype('uint8')
        cost_set[i] = guide_image.filter(dst)
    disparity_right = np.argmin(cost_set, axis=0)
    #plt.imshow(disparity_right, cmap='gray')
    #plt.show()
    left_labels = post_processing(Il, Ir, disparity_left, disparity_right, bilateral_sigmas, bilateral_r, max_disp, True)
    if(USE_GRAPHCUT):
        left_labels = GraphCut_LocalExpansion(Il, Ir, max_disp, left_labels, model, True)
        #=================================================================================================
	    # Right image post processing
	    #=================================================================================================
        right_labels = post_processing(Il, Ir, disparity_left, disparity_right, bilateral_sigmas, bilateral_r, max_disp, False)
        right_labels = GraphCut_LocalExpansion(Il, Ir, max_disp, right_labels, model, False)
        labels = post_processing_3D(Il, Ir, left_labels, right_labels, bilateral_sigmas, bilateral_r, max_disp)
    else:
        labels = left_labels
    return labels

def predict_cost_MCCNN(imgL, imgR, model):  
    win_r = 5
    H, W, _ = imgL.shape
    cost = np.zeros((H,W))
    imgL_pad = (cv2.copyMakeBorder(imgL,win_r,win_r,win_r,win_r,cv2.BORDER_REPLICATE).astype('float32'))/255-0.5
    imgR_pad = (cv2.copyMakeBorder(imgR,win_r,win_r,win_r,win_r,cv2.BORDER_REPLICATE).astype('float32'))/255-0.5
    #print('Inference row number:',i,end='\r')
    imgLPatches = np.zeros((W*H, 2*win_r+1, 2*win_r+1, 1))
    imgRPatches = np.zeros((W*H, 2*win_r+1, 2*win_r+1, 1))
    for i in range(0,H,1):
        for j in range(0,W,1):
            imgLPatches[j+i*W,:,:,0] = cv2.cvtColor(imgL_pad[i:i+2*win_r+1, j:j+2*win_r+1], cv2.COLOR_BGR2GRAY)
            imgRPatches[j+i*W,:,:,0] = cv2.cvtColor(imgR_pad[i:i+2*win_r+1, j:j+2*win_r+1], cv2.COLOR_BGR2GRAY)
    cost[:,:] = (model.predict([imgLPatches,imgRPatches],batch_size = 1024, verbose=1)[:,0]).reshape((H,W))
    #print('cost ['+str(i)+','+str(j)+']:',model.predict([imgLPatch[None],imgRPatch[None]], verbose=0)); input()
    return cost

def post_processing(Il, Ir, disparity_left, disparity_right, bilateral_sigmas, bilateral_r, max_disp, left=True):
    # Occlusion check and fix (LR consistency)
    occlusion = np.zeros((Ir.shape[0], Ir.shape[1])).astype('uint8')
    if(left):
        for i in range(Ir.shape[0]):
            for j in range(Ir.shape[1]):
                if(abs(disparity_left[i,j] - disparity_right[i,j-disparity_left[i,j]])>1 or j < disparity_left[i,j]):
                    occlusion[i,j] = 1
        # Hole filling
        disparity_left_left  = disparity_left.copy()
        disparity_left_right = disparity_left.copy()
        for i in range(Ir.shape[0]):
            left_valid = -1
            for j in range(Ir.shape[1]):
                if(occlusion[i,j]==1 and left_valid != -1):
                    disparity_left_left[i,j] = left_valid
                elif(occlusion[i,j]==0):
                    left_valid = disparity_left[i,j]
                else:
                    disparity_left_left[i,j] = max_disp+1
        for i in range(Ir.shape[0]):
            right_valid = -1
            for j in range(Ir.shape[1]-1,-1,-1):
                if(occlusion[i,j]==1 and right_valid != -1):
                    disparity_left_right[i,j] = right_valid
                elif(occlusion[i,j]==0):
                    right_valid = disparity_left[i,j]
                else:
                    disparity_left_right[i,j] = max_disp+1
        disparity_left = np.minimum(disparity_left_left, disparity_left_right)
        # Weighted median filter
        labels = color_bilateral_median(Il, disparity_left, occlusion, bilateral_sigmas, bilateral_r, max_disp).astype('uint8')
    else:
        for i in range(Ir.shape[0]):
            for j in range(Ir.shape[1]):
                if(j+disparity_left[i,j]>disparity_left.shape[1]-1 or abs(disparity_right[i,j] - disparity_left[i,j+disparity_left[i,j]])>1):
                    occlusion[i,j] = 1
        # Hole filling
        disparity_right_left  = disparity_right.copy()
        disparity_right_right = disparity_right.copy()
        for i in range(Ir.shape[0]):
            left_valid = -1
            for j in range(Ir.shape[1]):
                if(occlusion[i,j]==1 and left_valid != -1):
                    disparity_right_left[i,j] = left_valid
                elif(occlusion[i,j]==0):
                    left_valid = disparity_right[i,j]
                else:
                    disparity_right_left[i,j] = max_disp+1
        for i in range(Ir.shape[0]):
            right_valid = -1
            for j in range(Ir.shape[1]-1,-1,-1):
                if(occlusion[i,j]==1 and right_valid != -1):
                    disparity_right_right[i,j] = right_valid
                elif(occlusion[i,j]==0):
                    right_valid = disparity_right[i,j]
                else:
                    disparity_right_right[i,j] = max_disp+1
        disparity_right = np.minimum(disparity_right_left, disparity_right_right)
        # Weighted median filter
        labels = color_bilateral_median(Ir, disparity_right, occlusion, bilateral_sigmas, bilateral_r, max_disp).astype('uint8')        
    # Apply two median filter at the end
    for i in range(2):
        labels = cv2.medianBlur(labels,3)
    return labels

def post_processing_3D(Il, Ir, left_labels, right_labels, bilateral_sigmas, bilateral_r, max_disp):
    H,W,_ = Il.shape
    disparity_left  = label_to_img(left_labels,  max_disp)
    disparity_right = label_to_img(right_labels, max_disp)
    disp_floor = np.floor(disparity_left+0.001).astype('int32')
    disp_res = disparity_left-disp_floor
    #plt.imshow(np.concatenate([disparity_left,disparity_right]),cmap='gray')
    #plt.show()
    occlusion = np.zeros((Ir.shape[0], Ir.shape[1])).astype('uint8')
    for i in range(H):
         for j in range(W):
            interp_disp = (1-disp_res[i,j])*(disparity_right[i,j-disp_floor[i,j]])+disp_res[i,j]*(disparity_right[i,j-(disp_floor[i,j]+1)])
            if(abs(disparity_left[i,j] - interp_disp)>1 or j < disp_floor[i,j]+1):
                occlusion[i,j] = 1
    #plt.imshow(occlusion,cmap='gray')
    #plt.show()
    # Hole filling
    label_left_left  = left_labels.copy()
    label_left_right = left_labels.copy()
    for i in range(H):
        left_valid = np.array([0,0,0])
        for j in range(W):
            if(occlusion[i,j]==1 and not np.array_equal(left_valid, np.array([0,0,0]))):
                label_left_left[i,j] = left_valid
            elif(occlusion[i,j]==0):
                left_valid = left_labels[i,j]
            else:
                label_left_left[i,j] = [0,0,max_disp+1]
    for i in range(H):
        right_valid = np.array([0,0,0])
        for j in range(W-1,-1,-1):
            if(occlusion[i,j]==1 and not np.array_equal(right_valid, np.array([0,0,0]))):
                label_left_right[i,j] = right_valid
            elif(occlusion[i,j]==0):
                right_valid = left_labels[i,j]
            else:
                label_left_right[i,j] = [0,0,max_disp+1]
    disparity_left_left  = label_to_img(label_left_left,  max_disp)
    disparity_left_right = label_to_img(label_left_right, max_disp)
    #plt.imshow(disparity_left_left,cmap='gray')
    #plt.show()
    #plt.imshow(disparity_left_right,cmap='gray')
    #plt.show()
    disparity_left = np.minimum(disparity_left_left, disparity_left_right)
    #plt.imshow(disparity_left,cmap='gray')
    #plt.show()
    # Weighted median filter
    labels = color_bilateral_median(Il, disparity_left, occlusion, bilateral_sigmas, bilateral_r, max_disp)
    for i in range(2):
        labels = cv2.medianBlur(labels.astype('float32'),3)
    return labels

def compute_mccnn_cost_self(Il, Ir, max_disp, model, left = True):    
    h, w, ch = Il.shape
    if(left):
        set_img_shift = img_shift_subpixel(Ir, max_disp, True).astype(np.float32)
        cost_set = np.zeros((max_disp*10+1, Ir.shape[0], Ir.shape[1])).astype(np.float32)
        for i in range(max_disp*10+1): 
            print('Compute cost for image#',str(i),' out of',str(max_disp*10),end='\r')		
            cost_set[i] = ((1-predict_cost_MCCNN(Il, set_img_shift[i], model))*255).astype('uint8')
    else:
        set_img_shift = img_shift_subpixel(Il, max_disp, False).astype(np.float32)
        cost_set = np.zeros((max_disp*10+1, Ir.shape[0], Ir.shape[1])).astype(np.float32)
        for i in range(max_disp*10+1):
            print('Compute cost for image#',str(i),' out of',str(max_disp*10),end='\r')		
            cost_set[i] = ((1-predict_cost_MCCNN(set_img_shift[i], Ir,  model))*255).astype('uint8')
    print('')
    return cost_set


def compute_mccnn_cost(Il, Ir, max_disp, model, left = True):    
    h, w, ch = Il.shape
    Il_32 = Il.copy().astype(np.float32)
    Ir_32 = Ir.copy().astype(np.float32)
    if(left):
        set_img_shift = img_shift_subpixel(Ir, max_disp, True).astype(np.float32)
        cost_set = np.zeros((max_disp*10+1, Ir.shape[0], Ir.shape[1])).astype(np.float32)
        for i in range(max_disp*10+1):                
            batch = chainer.dataset.concat_examples([(np.moveaxis(Il_32, -1, 0), np.moveaxis(set_img_shift[i], -1, 0), np.array([max_disp]))], 0)
            print('Compute cost for image#',str(i),' out of',str(max_disp*10),end='\r')
            with chainer.no_backprop_mode():
                cost_set[i] = np.minimum(cost_mctau,chainer.cuda.to_cpu(model(*batch)[0,0].array))
    else:
        set_img_shift = img_shift_subpixel(Il, max_disp, False).astype(np.float32)
        cost_set = np.zeros((max_disp*10+1, Ir.shape[0], Ir.shape[1])).astype(np.float32)
        for i in range(max_disp*10+1):
            batch = chainer.dataset.concat_examples([(np.moveaxis(set_img_shift[i], -1, 0), np.moveaxis(Ir_32, -1, 0), np.array([max_disp]))], 0)
            print('Compute cost for image#',str(i),' out of',str(max_disp*10),end='\r')
            with chainer.no_backprop_mode():
                cost_set[i] = np.minimum(cost_mctau,chainer.cuda.to_cpu(model(*batch)[0,0].array))
    print('')
    return cost_set
    
	
# use cost from cost volume filtering for now
def compute_cost(Il, Ir, max_disp ,left = True):
    #guide_r = 1
    #guide_eps = 0.1
    max_cost = ((1-cost_alpha)*cost_tau1+cost_alpha*cost_tau2)*3
    h, w, ch = Il.shape
    if(left):
        set_img_shift = img_shift_subpixel(Ir, max_disp, True)
        cost_set = np.zeros((max_disp*10+1, Ir.shape[0], Ir.shape[1]))
        for i in range(max_disp*10+1):
            print('Compute cost for image#',str(i),' out of',str(max_disp*10),end='\r')
            cost_set[i] = truncated_cost_computation(Il, set_img_shift[i], cost_alpha, cost_tau1, cost_tau2, truncate_cost_window)
        #guide_image = cv2.ximgproc.createGuidedFilter(Il, guide_r, guide_eps)
    else:
        set_img_shift = img_shift_subpixel(Il, max_disp, False)
        cost_set = np.zeros((max_disp*10+1, Ir.shape[0], Ir.shape[1]))
        for i in range(max_disp*10+1):
            print('Compute cost for image#',str(i),' out of',str(max_disp*10),end='\r')
            cost_set[i] = truncated_cost_computation(Ir, set_img_shift[i], cost_alpha, cost_tau1, cost_tau2, truncate_cost_window)
        #guide_image = cv2.ximgproc.createGuidedFilter(Ir, guide_r, guide_eps)
    #for i in range(max_disp+1):
    #    cost_set[i] = (guide_image.filter(((cost_set[i]/max_cost)*255).astype('uint8')))*max_cost/255
    print('')
    return cost_set

#edge padding (to the left or the right)
def img_padding(img, max_disp, left = True):
    pad_img = np.zeros((img.shape[0],img.shape[1]+max_disp+1,3)).astype('uint8')
    if(left):
        pad_img[:img.shape[0], max_disp+1:] = img
        for i in range(max_disp+1):
            pad_img[:img.shape[0], i, :] = img[:,0]
    else:	# pad to the right
        pad_img[:img.shape[0], :img.shape[1]] = img
        for i in range(max_disp+1):
            pad_img[:img.shape[0], img.shape[1]+i, :] = img[:,-1]
    return pad_img

def img_shift(img, max_disp, left = True):
    set_img = np.zeros((max_disp+1,img.shape[0],img.shape[1],3)).astype('uint8')
    pad_img = img_padding(img, max_disp, left)
    if(left):
        for i in range(max_disp+1):
            set_img[max_disp-i] = pad_img[:,i+1:i+img.shape[1]+1]
    else:
        for i in range(max_disp+1):
            set_img[i] = pad_img[:,i:i+img.shape[1]]
    return set_img

#i = 4.3 -> index = 43
def img_shift_subpixel(img, max_disp, left = True):
    set_img = np.zeros((max_disp*10+1,img.shape[0],img.shape[1],3)).astype('uint8')
    pad_img = cv2.copyMakeBorder(img, 0, 0, max_disp, max_disp, cv2.BORDER_REPLICATE)
    if(left):
        for i in range(max_disp*10+1):
            dst = cv2.warpAffine(pad_img,np.float32([[1,0,0.1*i],[0,1,0]]),(pad_img.shape[1],pad_img.shape[0]))
            set_img[i] = dst[:,max_disp:-max_disp]
    else:
        for i in range(max_disp*10+1):
            dst = cv2.warpAffine(pad_img,np.float32([[1,0,-0.1*i],[0,1,0]]),(pad_img.shape[1],pad_img.shape[0]))
            set_img[i] = dst[:,max_disp:-max_disp]
    return set_img

def weight_pq(pixel_diff, r):
    vec = np.zeros(pixel_diff.shape)
    for i in range(3):
        vec[:,:,i] = np.exp(-np.abs(pixel_diff[:,:,i])/r)
    return vec[:,:,0]*vec[:,:,1]*vec[:,:,2]

def color_bilateral_median(guide_img, img, occlude_mask, sigmas, pads, max_disp):
    img_width  = img.shape[0]
    img_height = img.shape[1]
    guide_img_pad = np.zeros((img_width+2*pads, img_height+2*pads, 3), dtype = img.dtype)
    for i in range(3):
        guide_img_pad[:,:,i] = np.pad(guide_img[:,:,i], pads, 'edge')
    img_pad = np.pad(img[:,:], pads, 'edge')
    img_result = img.copy()
    # Build spatial filter
    spatial_filter = np.zeros((2*pads+1,2*pads+1))
    for i in range(2*pads+1):
        for j in range(2*pads+1):
            spatial_filter[i,j] = Gs(i-pads, j-pads, sigmas[0])
    spatial_filter = spatial_filter.reshape((2*pads+1)**2)
    for i in range(img_width):
        if(i%100==0):
            print('Bilateral: Row index',i,' done out of',img_width)
        for j in range(img_height):
            if(occlude_mask[i,j]==0):
                continue
            guide_img_vec = (guide_img_pad[i+pads, j+pads]-guide_img_pad[i:i+2*pads+1,j:j+2*pads+1,:].reshape(((2*pads+1)**2,3)))/max_disp
            tmp = color_Gr(guide_img_vec, sigmas[1])*spatial_filter
            tmp = (tmp/sum(tmp))
            #weighted_patch = np.multiply(tmp,(w_img_pad[i:i+2*pads+1,j:j+2*pads+1])).reshape(((2*pads+1)**2))
            img_result[i,j] = find_median((img_pad[i:i+2*pads+1,j:j+2*pads+1]).reshape(((2*pads+1)**2)), tmp)
    return img_result

# return three color Gr
def color_Gr(pixel_diff, sigma_r):
    vec = np.zeros((pixel_diff.shape[0],3))
    for i in range(3):
        vec[:,i] = np.exp(-(pixel_diff[:,i])**2/(2*sigma_r**2))#*np.power(10.000,20)
    return vec[:,0]*vec[:,1]*vec[:,2]

# return space filter Gs
def Gs(i, j, sigma_s):
    return np.exp(-(i**2+j**2)/(2*sigma_s**2))

def find_median(values, weights):
    order = np.argsort(np.array([values, weights]), axis=-1)
    sorted_arr = np.array([values, weights])[:,order[0,:]]
    cdf = np.add.accumulate(sorted_arr[1,:])
    med_ind = np.argmax(cdf >= cdf[-1]/2)
    return (sorted_arr[0,med_ind])#(int)(sorted[1,med_ind])

def distance_weight_compute(img1, window_size, gamma, spatial_r):  
    img1_pad = cv2.copyMakeBorder(img1,window_size, window_size, window_size, window_size, cv2.BORDER_CONSTANT).astype('float32')
    # Build spatial filter
    spatial_filter = np.zeros((2*window_size+1,2*window_size+1))
    for i in range(2*window_size+1):
        for j in range(2*window_size+1):
            spatial_filter[i,j] = Gs(i-window_size, j-window_size, spatial_r)
    w_pq = np.zeros((img1.shape[0],img1.shape[1],2*window_size+1,2*window_size+1))
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            w_pq[i,j,:,:] = np.multiply(spatial_filter,weight_pq((img1_pad[i:i+2*window_size+1,j:j+2*window_size+1]-img1_pad[i+window_size,j+window_size]),gamma))
            w_pq[i,j,:,:] = w_pq[i,j,:,:]/np.sum(w_pq[i,j,:,:])
    return w_pq       

# (window_size*2+1)**2
# gamma = 10
def truncated_cost_computation(img1, img2, alpha, tau1, tau2, window_size):
    if(window_size!=0):
        img1_pad = cv2.copyMakeBorder(img1,window_size, window_size, window_size, window_size, cv2.BORDER_CONSTANT).astype('float32')
        img2_pad = cv2.copyMakeBorder(img2,window_size, window_size, window_size, window_size, cv2.BORDER_CONSTANT).astype('float32')
    else:
        img1_pad = img1.astype('float32')
        img2_pad = img2.astype('float32')
    img1_pad = np.sum(img1_pad,-1)/3
    img2_pad = np.sum(img2_pad,-1)/3
    # calculate gradient for left and right image
    grad_img1 = cv2.filter2D(img1_pad, cv2.CV_32F, np.array([[-0.5,0,0.5]]))
    grad_img2 = cv2.filter2D(img2_pad, cv2.CV_32F, np.array([[-0.5,0,0.5]]))
    # calculate truncated cost per pixel
    diff_img      = np.abs(img1_pad-img2_pad)
    diff_grad_img = np.abs(grad_img1-grad_img2)
    diff_img      = np.where(diff_img>tau1, tau1, diff_img)
    diff_grad_img = np.where(diff_grad_img>tau2, tau2, diff_grad_img)
    # summation over a window
    kernel = np.ones((2*window_size+1,2*window_size+1))
    if(window_size!=0):
        AD  = cv2.filter2D(diff_img,      cv2.CV_32F, kernel)[window_size:-window_size,window_size:-window_size]
        GAD = cv2.filter2D(diff_grad_img, cv2.CV_32F, kernel)[window_size:-window_size,window_size:-window_size]
    else:
        AD = diff_img; GAD = diff_grad_img
    return (((1-alpha)*AD+alpha*GAD)/((2*window_size+1)**2))*3

#def post_processing_3D(Il, Ir, label_left, label_right, bilateral_sigmas, bilateral_r, max_disp, left=True):
#    return labels

#(np.rint(np.dot(pix_labels[:,i,j,:],[i,j,1]))).astype('int32')
def init_label_candidates(rand_vec, rand_disp, max_disp):
    K,h,w,c = rand_vec.shape
    norm_vecs = np.zeros((K,h,w,3))
    for i in range(3):
        norm_vecs[:,:,:,i] = rand_vec[:,:,:,i]/np.sqrt(np.sum(rand_vec**2,-1))
    pix_labels = np.zeros((K,h,w,3))
    pix_labels[:,:,:,0] = -norm_vecs[:,:,:,0]/norm_vecs[:,:,:,2]
    pix_labels[:,:,:,1] = -norm_vecs[:,:,:,1]/norm_vecs[:,:,:,2]
    for k in range(K):
        for i in range(h):
            for j in range(w):
                pix_labels[k,i,j,2] = np.dot(norm_vecs[k,i,j,:],[i,j,rand_disp[k,i,j]])/norm_vecs[k,i,j,2]
    return pix_labels

# select Kr labels from all r_size*r_size region
# only output selected 
def label_to_region_proposal_one(Kr, K_want, pix_labels, r_size):
    K,H,W,c = pix_labels.shape
    pix_labels_pad = np.zeros((K,H+r_size,W+r_size,c))
    for k in range(K):
        pix_labels_pad[k] = cv2.copyMakeBorder(pix_labels[k], 0, r_size, 0, r_size, cv2.BORDER_REPLICATE)
    H_region = np.ceil(H/r_size).astype('int32')
    W_region = np.ceil(W/r_size).astype('int32')
    r_j = np.zeros((16*Kr, H_region, W_region, 3))
    for h in range(H_region):
        for w in range(W_region):
            x_ind = h*r_size; y_ind = w*r_size
            np_choice = np.random.choice(r_size**2,Kr,False)
            ind = Kr*(4*(h%4)+(w%4))
            r_j[ind:ind+Kr,h,w]=(pix_labels_pad[0,x_ind:x_ind+r_size,y_ind:y_ind+r_size,:].reshape(-1,c))[np_choice]
    # expand to r_size**2 local patch
    ret_r_j = np.zeros((H_region*r_size,W_region*r_size,c))
    ret_r_j = cv2.resize(r_j[K_want], (0,0) , fx =r_size, fy = r_size, interpolation=cv2.INTER_NEAREST)
    ret_r_j = cv2.copyMakeBorder(ret_r_j, r_size, r_size, r_size, r_size, cv2.BORDER_REPLICATE)
    ret_r_j_enlarge = np.zeros((H_region*r_size,W_region*r_size,c))
    # expand to r_size**2*(3*3) local patch
    print('Processing #'+str(K_want)+' proposal out of',16*Kr,end='\r')
    for h in range(H):
        for w in range(W):
            window = (ret_r_j[h:h+2*r_size+1,w:w+2*r_size+1,:].reshape((-1,c)))
            window_copy = window.copy()
            ret_r_j_enlarge[h,w,:] = window_copy[(np.logical_or(window[:,0],window[:,1],window[:,2])!=0).argmax(axis=0)]
    print('')
    return ret_r_j_enlarge[:H,:W,:]

def create_mask(g_j):
    K,H,W,c = g_j.shape
    mask = np.zeros((K,H,W))
    for k in range(K):
        mask[k] = np.logical_and(np.logical_and((g_j[k,:,:,0]==0),(g_j[k,:,:,1]==0)),(g_j[k,:,:,2]==0))
    return mask

def data_cost_img(cost_set, w_pq, label_map, window_size, max_disp):
    H,W,c = label_map.shape
    d_cost_img = np.zeros((H,W))
    for i in range(H):
        if(i%100==0):
            print('Row '+str(i)+' in progress out of '+str(H),'...', end='\r')
        top = np.max([0,i-window_size]);  down  = np.min([H-1, i+window_size])
        for j in range(W):
            left = np.max([0,j-window_size]); right = np.min([W-1, j+window_size])
            width = right-left+1; height = down-top+1
            inds = np.indices((height,width))
            inds = np.moveaxis(np.stack((inds[0]+top, inds[1]+left),0),0,-1)
            new_inds = np.concatenate([inds, np.ones((height,width,1))], -1)
            disparity_map = np.rint(np.dot(new_inds,label_map[i,j])*10).astype('int32').reshape((-1))
            disparity_map[disparity_map>max_disp*10] = max_disp*10
            disparity_map[disparity_map<0] = 0
            ind_dis = inds.reshape((-1,2))
            cost_map = cost_set[disparity_map[:],ind_dis[:,0],ind_dis[:,1]]
            #if(i==20 and j < 100 and j >80):
            #    print('disparity_map:', disparity_map)
            #    print('ind_dis:', ind_dis)
            #    print('cost_map:', cost_map); input()
            d_cost_img[i,j] = np.sum(np.multiply(cost_map, w_pq[i,j, top-i+window_size:down-i+window_size+1, left-j+window_size:right-j+window_size+1].reshape((-1))))
    return d_cost_img

def data_cost_img_mask(cost_set, w_pq, label_map, window_size, max_disp, label_mask):
    H,W,c = label_map.shape
    d_cost_img = np.zeros((H,W))
    for i in range(H):
        if(i%100==0):
            print('Row '+str(i)+' in progress out of '+str(H),'...', end='\r')
        top = np.max([0,i-window_size]);  down  = np.min([H-1, i+window_size])
        for j in range(W):
            if(label_mask[i,j]==1):
                d_cost_img[i,j] = np.inf
                continue
            left = np.max([0,j-window_size]); right = np.min([W-1, j+window_size])
            width = right-left+1; height = down-top+1
            inds = np.indices((height,width))
            inds = np.moveaxis(np.stack((inds[0]+top, inds[1]+left),0),0,-1)
            new_inds = np.concatenate([inds, np.ones((height,width,1))], -1)
            disparity_map = np.rint(np.dot(new_inds,label_map[i,j])*10).astype('int32').reshape((-1))
            disparity_map[disparity_map>max_disp*10] = max_disp*10
            disparity_map[disparity_map<0] = 0
            ind_dis = inds.reshape((-1,2))
            cost_map = cost_set[disparity_map[:],ind_dis[:,0],ind_dis[:,1]]
            d_cost_img[i,j] = np.sum(np.multiply(cost_map, w_pq[i,j, top-i+window_size:down-i+window_size+1, left-j+window_size:right-j+window_size+1].reshape((-1))))
    return d_cost_img

#Il_pad, cost_set_pad = padding window_size on four sides
def data_cost_img_guide(cost_set_pad, Il_pad, label_map, window_size, max_disp):
    H,W,c = label_map.shape
    d_cost_img = np.zeros((H,W))	# pad r_size on four sides
    inds = np.indices((2*window_size+1,2*window_size+1))
    for i in range(H):
        if(i%100==0):
            print('Row '+str(i)+' in progress out of '+str(H+1),'...', end='\r')
        for j in range(W):
            top  = i; left = j; #+((r_size*3-1)>>1)
            new_inds = np.moveaxis(np.stack((inds[0]+top-window_size, inds[1]+left-window_size),0),0,-1)
            new_inds = np.concatenate([new_inds, np.ones((2*window_size+1,2*window_size+1,1))], -1)
            disparity_map = np.rint(np.dot(new_inds,label_map[i,j])*10).astype('int32').reshape((-1))
            disparity_map[disparity_map>max_disp*10] = max_disp*10
            disparity_map[disparity_map<0] = 0
            ind_dis = np.moveaxis(np.stack((inds[0]+top, inds[1]+left),0),0,-1).reshape((-1,2))
            cost_map = cost_set_pad[disparity_map[:],ind_dis[:,0],ind_dis[:,1]].astype('float32')
            #guide_filter = cv2.ximgproc.createGuidedFilter(Il_pad[top:top+2*window_size+1,left:left+2*window_size+1], GC_guide_r, GC_guide_eps)
            guide_filter  = GuidedFilter(Il_pad[top:top+2*window_size+1,left:left+2*window_size+1], radius=GC_guide_r, epsilon=GC_guide_eps)
            cost_g = guide_filter.filter(((cost_map.reshape((2*window_size+1, 2*window_size+1))/cost_mctau)))*cost_mctau
            d_cost_img[i,j] = cost_g[window_size, window_size]
    return d_cost_img

def data_cost_img_mask_guide(cost_set_pad, Il_pad, K_want, r_size, label_map, window_size, max_disp, label_mask):
    H,W,c = label_map.shape
    d_cost_img = np.zeros((H+(((r_size*3-1)*3)>>1),W+(((r_size*3-1)*3)>>1)))	# pad (r_size*3-1)/2 on four sides (right/bottom: extra (r_size*3-1)/2 pixel)
    gwindow_r = ((r_size*3-1)>>1)+window_size
    label_map_pad  = cv2.copyMakeBorder(label_map,  0, 2*gwindow_r, 0, 2*gwindow_r, cv2.BORDER_REPLICATE)
    label_mask_pad = cv2.copyMakeBorder(label_mask, 0, 2*gwindow_r, 0, 2*gwindow_r, cv2.BORDER_REPLICATE)
    col =  K_want%4; row = (int)(K_want/4)
    col_index = np.floor((W-r_size*col)/(4*r_size)).astype('int32')
    row_index = np.floor((H-r_size*row)/(4*r_size)).astype('int32')
    inds = np.indices((2*gwindow_r+1,2*gwindow_r+1))
    for i in range(row_index+1):
        print('Row '+str(i)+' in progress out of '+str(row_index+1),'...', end='\r')
        for j in range(col_index+1):
            cen_x = ((r_size-1)>>1)+r_size*row+4*r_size*i ; top  = cen_x# - ((r_size-1)>>1) 
            cen_y = ((r_size-1)>>1)+r_size*col+4*r_size*j ; left = cen_y# - ((r_size-1)>>1) 
            if((i==row_index or j==col_index) and ((cen_x-((3*r_size-1)>>1))>H-1 or (cen_y-((3*r_size-1)>>1))>W-1 or label_mask_pad[cen_x, cen_y]==1)):
                continue
            new_inds = np.moveaxis(np.stack((inds[0]+top-gwindow_r, inds[1]+left-gwindow_r),0),0,-1)
            new_inds = np.concatenate([new_inds, np.ones((2*gwindow_r+1,2*gwindow_r+1,1))], -1)
            # cen-2 for avoiding H, W boundary case (r_size should be of size higher than 5)
            disparity_map = np.rint(np.dot(new_inds,label_map_pad[cen_x,cen_y])*10).astype('int32').reshape((-1))
            disparity_map[disparity_map>max_disp*10] = max_disp*10
            disparity_map[disparity_map<0] = 0
            ind_dis = np.moveaxis(np.stack((inds[0]+top, inds[1]+left),0),0,-1).reshape((-1,2))
            cost_map = cost_set_pad[disparity_map[:],ind_dis[:,0],ind_dis[:,1]].astype('float32')
            #guide_filter = cv2.ximgproc.createGuidedFilter(Il_pad[top:top+2*gwindow_r+1,left:left+2*gwindow_r+1], GC_guide_r, GC_guide_eps)
            guide_filter  = GuidedFilter(Il_pad[top:top+2*gwindow_r+1,left:left+2*gwindow_r+1], radius=GC_guide_r, epsilon=GC_guide_eps)
            cost_g = guide_filter.filter(((cost_map.reshape((2*gwindow_r+1, 2*gwindow_r+1))/cost_mctau)))*cost_mctau
            d_cost_img[top:top+3*r_size,left:left+3*r_size] = cost_g[gwindow_r-((3*r_size)>>1):gwindow_r+((3*r_size)>>1)+1,gwindow_r-((3*r_size)>>1):gwindow_r+((3*r_size)>>1)+1]
    d_cost = d_cost_img[((r_size*3-1)>>1):-((r_size*3-1)), ((r_size*3-1)>>1):-((r_size*3-1))]
    d_cost[label_mask==1] = np.inf
    return d_cost

'''
r_size = 25
if(USE_GUIDED):
    gwindow_r = ((r_size*3-1)>>1)+GC_cost_window
    Il_pad = cv2.copyMakeBorder(Il, gwindow_r, gwindow_r*2, gwindow_r, gwindow_r*2, cv2.BORDER_REPLICATE)
    cost_set_pad = np.zeros((10*max_disp+1, cost_set.shape[1]+3*gwindow_r, cost_set.shape[2]+3*gwindow_r))
    for i in range(10*max_disp+1):
        cost_set_pad[i] = cv2.copyMakeBorder(cost_set[i], gwindow_r, 2*gwindow_r, gwindow_r, 2*gwindow_r, cv2.BORDER_REPLICATE)
else:
    Il_pad = 0 ; cost_set_pad = 0 ;

for i in range(16):
    K_want = i
    print('5*5 prop Current i =',i,'; Total iteration =',16,'; Outer iteration #'+str(iter))
    label_map = label_to_region_proposal_one(1, i, ref_labels[None], r_size)
    label_mask = create_mask(label_map[None])[0]
    d_cost = data_cost_img_mask_guide(cost_set_pad, Il_pad, GC_guide_r, GC_guide_eps, K_want, r_size, label_map, window_size, max_disp, label_mask)
    plt.imshow(d_cost)
    plt.show()
'''

## shift = [0,2]-> right; [1,1]-> down (for ind_shift and shift)
## shift = [0,1]-> right; [1,0]-> down (for w_shift)
## w_shift: same but for the weight (you can have different shift for value and weight)
def img_s_cost(ref_label_map, prop_label_map, w_pq, shift, ind_shift, w_shift, window_size):
    H,W,c = prop_label_map.shape
    prop_label_map_shift = cv2.copyMakeBorder(prop_label_map, 0, 1, 1, 1, cv2.BORDER_CONSTANT).astype('float32')
    prop_label_map_shift = prop_label_map_shift[shift[0]:shift[0]+H, shift[1]:shift[1]+W]
    inds = np.moveaxis(np.indices((H,W)),0,-1)
    inds = np.concatenate((inds, np.ones((H,W,1))),-1)
    inds_shift = cv2.copyMakeBorder(inds, 0, 1, 1, 1, cv2.BORDER_CONSTANT).astype('float32')
    inds_shift = inds_shift[ind_shift[0]:ind_shift[0]+H, ind_shift[1]:ind_shift[1]+W]
    #compute cost
    cost_img1 = np.abs(np.sum(np.multiply(inds, ref_label_map),-1)-np.sum(np.multiply(inds, prop_label_map_shift),-1))
    cost_img2 = np.abs(np.sum(np.multiply(inds_shift, ref_label_map),-1)-np.sum(np.multiply(inds_shift, prop_label_map_shift),-1))
    cost_img = lambda_s*np.multiply(np.maximum(w_pq[:,:,window_size+w_shift[0],window_size+w_shift[1]], eps),np.minimum(cost_img1+cost_img2, tau_dis))
    return cost_img
    

#p/q: 3D vector (location) , fp/fq: 3D label
def s_cost_one(p, q, fp, fq, weight):
    return lambda_s*np.max([weight, eps])*np.min([np.abs(np.dot(p,fp)-np.dot(p,fq))+np.abs(np.dot(q,fp)-np.dot(q,fq)), tau_dis])

# lambda = 20, g for graph(need to set data term first)
# label_mask == 0 means have a valid value
def smooth_cost(w_pq, g, ref_label_map, prop_label_map, label_mask, window_size):
    H,W,c = prop_label_map.shape
    #=======================================================================================================#
    self_cost_right 	= img_s_cost(ref_label_map, ref_label_map,  w_pq, [0,2], [0,2], [0,1], window_size)
    cost_center_wright 	= img_s_cost(ref_label_map, prop_label_map, w_pq, [0,1], [0,2], [0,1], window_size)
    cost_right 			= img_s_cost(ref_label_map, prop_label_map, w_pq, [0,2], [0,2], [0,1], window_size)
    rever_cost_right 	= img_s_cost(prop_label_map, ref_label_map, w_pq, [0,2], [0,2], [0,1], window_size)
    #=======================================================================================================#
    self_cost_down  	= img_s_cost(ref_label_map, ref_label_map,  w_pq, [1,1], [1,1], [1,0], window_size)
    cost_center_wdown 	= img_s_cost(ref_label_map, prop_label_map, w_pq, [0,1], [1,1], [1,0], window_size)
    cost_down 			= img_s_cost(ref_label_map, prop_label_map, w_pq, [1,1], [1,1], [1,0], window_size)
    rever_cost_down 	= img_s_cost(prop_label_map, ref_label_map, w_pq, [1,1], [1,1], [1,0], window_size)
    '''#=======================================================================================================#
    self_cost_ldown  	= img_s_cost(ref_label_map, ref_label_map,  w_pq, [1,0], [1,0], [1,-1], window_size)
    cost_center_wldown 	= img_s_cost(ref_label_map, prop_label_map, w_pq, [0,1], [1,0], [1,-1], window_size)
    cost_ldown 			= img_s_cost(ref_label_map, prop_label_map, w_pq, [1,0], [1,0], [1,-1], window_size)
    rever_cost_ldown 	= img_s_cost(prop_label_map, ref_label_map, w_pq, [1,0], [1,0], [1,-1], window_size)
    #=======================================================================================================#
    self_cost_rdown  	= img_s_cost(ref_label_map, ref_label_map,  w_pq, [1,2], [1,2], [1,1], window_size)
    cost_center_wrdown 	= img_s_cost(ref_label_map, prop_label_map, w_pq, [0,1], [1,2], [1,1], window_size)
    cost_rdown 			= img_s_cost(ref_label_map, prop_label_map, w_pq, [1,2], [1,2], [1,1], window_size)
    rever_cost_rdown 	= img_s_cost(prop_label_map, ref_label_map, w_pq, [1,2], [1,2], [1,1], window_size)
    #=======================================================================================================#'''
    #s_cost = np.zeros((H,W,3,3)) #edge weight from this node out
    #i, j : pivot
    for i in range(H):
        if(i%100==0):
            print('Row '+str(i)+' in progress out of '+str(H),'...', end='\r')
        for j in range(W):
            id1 = i*W+j
            #continue condition: if(i+ik<0 or i+ik>H-1 or j+jk<0 or j+jk>W-1 or (ik==0 and jk==0))
            #Right direction (ik =0; jk =1)
            ik = 0 ; jk = 1;
            id2 = id1+W*ik+jk
            if(i+ik<0 or i+ik>H-1 or j+jk<0 or j+jk>W-1):
                assert(True)
            elif(np.array_equal(ref_label_map[i,j], ref_label_map[i+ik,j+jk])):
                if(label_mask[i,j]==0):
                    g.add_edge(id1, id2, rever_cost_right[i,j], rever_cost_right[i,j])
                elif(label_mask[i+ik,j+jk]==0):
                    g.add_edge(id1, id2, cost_right[i,j], cost_right[i,j])
            else: # f0 and f1 has different label -> auxiliary node!!
                node_id = g.add_nodes(1)
                g.add_tedge(node_id, 0, self_cost_right[i,j])
                if(label_mask[i,j]==0):
                    g.add_edge(id1, node_id, cost_center_wright[i,j], cost_center_wright[i,j])                
                    g.add_edge(node_id, id2, rever_cost_right[i,j], rever_cost_right[i,j])
                elif(label_mask[i+ik,j+jk]==0):
                    g.add_edge(id1, node_id, cost_right[i,j], cost_right[i,j])
                    g.add_edge(node_id, id2, np.inf, np.inf)
            #Down direction (ik =1; jk =0)
            ik = 1 ; jk = 0;
            id2 = id1+W*ik+jk
            if(i+ik<0 or i+ik>H-1 or j+jk<0 or j+jk>W-1):
                assert(True)
            elif(np.array_equal(ref_label_map[i,j], ref_label_map[i+ik,j+jk])):
                if(label_mask[i,j]==0):
                    g.add_edge(id1, id2, rever_cost_down[i,j], rever_cost_down[i,j])
                elif(label_mask[i+ik,j+jk]==0):
                    g.add_edge(id1, id2, cost_down[i,j], cost_down[i,j])
            else: # f0 and f1 has different label -> auxiliary node!!
                node_id = g.add_nodes(1)
                g.add_tedge(node_id, 0, self_cost_down[i,j])
                if(label_mask[i,j]==0):
                    g.add_edge(id1, node_id, cost_center_wdown[i,j], cost_center_wdown[i,j])
                    g.add_edge(node_id, id2, rever_cost_down[i,j], rever_cost_down[i,j])
                elif(label_mask[i+ik,j+jk]==0):
                    g.add_edge(id1, node_id, cost_down[i,j], cost_down[i,j])
                    g.add_edge(node_id, id2, np.inf, np.inf)
            '''
            #Right Down direction (ik =1; jk =1)
            ik = 1 ; jk = 1;
            id2 = id1+W*ik+jk
            if(i+ik<0 or i+ik>H-1 or j+jk<0 or j+jk>W-1):
                assert(True)
            elif(np.array_equal(ref_label_map[i,j], ref_label_map[i+ik,j+jk])):
                if(label_mask[i,j]==0):
                    g.add_edge(id1, id2, rever_cost_rdown[i,j], rever_cost_rdown[i,j])
                elif(label_mask[i+ik,j+jk]==0):
                    g.add_edge(id1, id2, cost_rdown[i,j], cost_rdown[i,j])
            else: # f0 and f1 has different label -> auxiliary node!!
                node_id = g.add_nodes(1)
                g.add_tedge(node_id, 0, self_cost_rdown[i,j])
                if(label_mask[i,j]==0):
                    g.add_edge(id1, node_id, cost_center_wrdown[i,j], cost_center_wrdown[i,j])
                    g.add_edge(node_id, id2, rever_cost_rdown[i,j], rever_cost_rdown[i,j])
                elif(label_mask[i+ik,j+jk]==0):
                    g.add_edge(id1, node_id, cost_rdown[i,j], cost_rdown[i,j])
                    g.add_edge(node_id, id2, np.inf, np.inf)
            #Left Down direction (ik =1; jk =-1)
            ik = 1 ; jk = -1;
            id2 = id1+W*ik+jk
            if(i+ik<0 or i+ik>H-1 or j+jk<0 or j+jk>W-1):
                assert(True)
            elif(np.array_equal(ref_label_map[i,j], ref_label_map[i+ik,j+jk])):
                if(label_mask[i,j]==0):
                    g.add_edge(id1, id2, rever_cost_ldown[i,j], rever_cost_ldown[i,j])
                elif(label_mask[i+ik,j+jk]==0):
                    g.add_edge(id1, id2, cost_ldown[i,j], cost_ldown[i,j])
            else: # f0 and f1 has different label -> auxiliary node!!
                node_id = g.add_nodes(1)
                g.add_tedge(node_id, 0, self_cost_ldown[i,j])
                if(label_mask[i,j]==0):
                    g.add_edge(id1, node_id, cost_center_wldown[i,j], cost_center_wldown[i,j])
                    g.add_edge(node_id, id2, rever_cost_ldown[i,j], rever_cost_ldown[i,j])
                elif(label_mask[i+ik,j+jk]==0):
                    g.add_edge(id1, node_id, cost_ldown[i,j], cost_ldown[i,j])
                    g.add_edge(node_id, id2, np.inf, np.inf)
            '''
    print('')
    return g

def build_graph(g, d_cost_source, d_cost_sink, nodeids):
    #g.add_grid_edges(nodeids, s_cost[:,:,0,0], structure=np.array([[1, 0, 0],[0, 0, 0],[0, 0, 0]]), symmetric=False)
    g.add_grid_tedges(nodeids, d_cost_sink, d_cost_source)
    return g

#data_cost_img_mask(cost_set, w_pq, prop_label_map, window_size, max_disp, label_mask)
#cost_set_pad, Il_pad, K_want, r_size: only for guided
def binary_fusion(cost_set, data_cost_source, cost_set_pad, Il_pad, K_want, r_size, w_pq, ref_labels, label_map, label_mask, window_size, max_disp):
    _,H,W = cost_set.shape
    data_cost_sink = np.zeros((H,W))
    #d_cost[:,:,0] = data_cost_source#data_cost_img(cost_set, w_pq, ref_labels, window_size, max_disp)
    if(USE_GUIDED):
        data_cost_sink = data_cost_img_mask_guide(cost_set_pad, Il_pad, K_want, r_size, label_map, window_size, max_disp, label_mask)
    else:
        data_cost_sink = data_cost_img_mask(cost_set, w_pq, label_map, window_size, max_disp, label_mask)
    g = mf.GraphFloat()
    nodeids = g.add_grid_nodes((H, W))
    g = build_graph(g, data_cost_source, data_cost_sink, nodeids)
    g = smooth_cost(w_pq, g, ref_labels, label_map, label_mask, window_size)
    print('Max flow capacity:', g.maxflow())
    sgm = g.get_grid_segments(nodeids)
    ref_labels[sgm] = label_map[sgm]	#replace True sgm to label map
    data_cost_source[sgm] = data_cost_sink[sgm]
    return ref_labels, data_cost_source#(ref_labels, sgm, g)

#for visulization
def label_to_img(label_map, max_disp):
    H,W,c = label_map.shape
    inds = np.moveaxis(np.indices((H,W)),0,-1)
    inds = np.concatenate((inds, np.ones((H,W,1))),-1)
    ret_img = np.sum(np.multiply(inds, label_map),-1)
    ret_img[ret_img>max_disp] = max_disp
    ret_img[ret_img<0] = 0
    return ret_img

def label_to_energy(cost_set, w_pq, Il, label_map, window_size, max_disp):
    H,W,c = label_map.shape
    d_cost = 0 ; s_cost = 0;
    # compute for data energy
    if(USE_GUIDED):
        Il_pad = cv2.copyMakeBorder(Il, window_size, window_size+1, window_size, window_size+1, cv2.BORDER_REPLICATE)
        cost_set_pad = np.zeros((10*max_disp+1, cost_set.shape[1]+2*window_size+1, cost_set.shape[2]+2*window_size+1))
        for i in range(10*max_disp+1):
            cost_set_pad[i] = cv2.copyMakeBorder(cost_set[i], window_size, window_size+1, window_size, window_size+1, cv2.BORDER_REPLICATE)
        d_cost = np.sum(data_cost_img_guide(cost_set_pad, Il_pad, label_map, window_size, max_disp))
    else:
        d_cost = np.sum(data_cost_img(cost_set, w_pq, label_map, window_size, max_disp))
    # compute for smoothness energy
    self_cost_right = img_s_cost(label_map, label_map, w_pq, [0,2], [0,2], [0,1], window_size)[:,:-1]
    self_cost_down  = img_s_cost(label_map, label_map, w_pq, [1,1], [1,1], [1,0], window_size)[:-1,:]
    #self_cost_ldown = img_s_cost(label_map, label_map, w_pq, [1,0], [1,0], [1,-1], window_size)[:-1,1:]
    #self_cost_rdown = img_s_cost(label_map, label_map, w_pq, [1,2], [1,2], [1,1], window_size)[:-1,:-1]
    s_cost = np.sum(self_cost_right)+np.sum(self_cost_down)#+np.sum(self_cost_ldown)+np.sum(self_cost_rdown)
    return (d_cost, s_cost)


###############################################################################
## Purturbation + choose next label for all pixels
# return perturbed 3D labels
def perturb_labels(cands_label, ref_labels, r_d, r_n, max_disp):
    Ks,H,W,C = cands_label.shape
    unit_norms = np.zeros(shape=cands_label.shape)
    disp_recover = np.zeros((Ks,H,W))
    unit_norms[:,:,:,2] = 1/(1+cands_label[:,:,:,0]**2+cands_label[:,:,:,1]**2)
    unit_norms[:,:,:,0]	= unit_norms[:,:,:,2]*cands_label[:,:,:,0]
    unit_norms[:,:,:,1]	= unit_norms[:,:,:,2]*cands_label[:,:,:,1]
    for i in range(Ks):
        disp_recover[i] = label_to_img(cands_label[i], max_disp)
    ##perturbation begins
    disp_recover += np.random.uniform(-r_d,r_d, (Ks,H,W))
    disp_recover[disp_recover>max_disp] = max_disp
    disp_recover[disp_recover<0] = 0
    unit_norms += r_n*(np.random.rand(Ks,H,W,3)-0.5)
    new_cands = init_label_candidates(unit_norms, disp_recover, max_disp)
    return np.concatenate([ref_labels[None,:,:,:],new_cands],0)

## Linear regression in a local window for good initial solution
def disp_to_labels(disp_map, cost_set, w_pq, window_size):
    _,H,W = cost_set.shape
    d_cost_img = np.zeros((H,W))
    new_labels = np.zeros((H,W,3))
    for i in range(H):
        print('Row '+str(i)+' in progress out of '+str(H),'...', end='\r')
        top = np.max([0,i-window_size]);  down  = np.min([H-1, i+window_size])
        for j in range(W):
            left = np.max([0,j-window_size]); right = np.min([W-1, j+window_size])
            width = right-left+1; height = down-top+1
            inds = np.indices((height,width))
            inds = np.moveaxis(np.stack((inds[0]+top, inds[1]+left),0),0,-1)
            A = np.concatenate([inds, np.ones((height,width,1))], -1).reshape((-1,3))
            w= w_pq[i,j, top-i+GC_cost_window:down-i+GC_cost_window+1, left-j+GC_cost_window:right-j+GC_cost_window+1].reshape((-1))
            d = disp_map[top:down+1, left:right+1].reshape((-1))
            A = np.matmul(np.diag(w),A); d = np.matmul(np.diag(w),d);
            new_labels[i,j,:] = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.transpose(),A)),A.transpose()),d)
    return new_labels


###############################################################################

def GraphCut_LocalExpansion(Il, Ir, max_disp, init_labels, model, left=True):
    #max_disp = 15
    #scale_factor = 16
    #Il = cv2.imread(left_path)
    #Ir = cv2.imread(right_path)
    (H,W,c) = Il.shape
    rand_disp = np.random.uniform(0,max_disp,(1,H,W))
    rand_vec = np.random.rand(1,H,W,3)-0.5
    ref_labels = init_label_candidates(rand_vec, rand_disp, max_disp)[0]
    print('Compute weight...')
    if(left):
        w_pq = distance_weight_compute(Il,GC_cost_window,gamma,spatial_r)
        if(USE_MCCNN_SELF):
            cost_set = compute_mccnn_cost_self(Il, Ir, max_disp, model, True)
        elif(USE_MCCNN):
            cost_set = compute_mccnn_cost(Il, Ir, max_disp, model, True)
        else:
            cost_set = compute_cost(Il, Ir, max_disp, True)
    else:
        w_pq = distance_weight_compute(Ir,GC_cost_window,gamma,spatial_r)
        if(USE_MCCNN_SELF):
            cost_set = compute_mccnn_cost_self(Il, Ir, max_disp, model, False)
        elif(USE_MCCNN):
            cost_set = compute_mccnn_cost(Il, Ir, max_disp, model, False)
        else:
            cost_set = compute_cost(Il, Ir, max_disp, False)
    cost_set = cost_set-np.min(cost_set,0)
    r_d = (max_disp)/4#(max_disp)/2
    r_n = 1/4#1
    ########################################################################################################
    ## Attempt to modify this to LocalExp version
    ## Warning: Only an attempt
    ##########################################################
    #img = cv2.imread(init_path)/scale_factor
    #disp_map = img[:,:,0]
    #plt.imshow(init_labels, cmap='gray')
    #plt.show()
    cheat_labels = disp_to_labels(init_labels, cost_set, w_pq, cheat_r)
    ref_labels = cheat_labels
    #plt.imshow(label_to_img(ref_labels, max_disp), cmap='gray')
    #plt.show()
    if(USE_GUIDED):
        Il_pad = cv2.copyMakeBorder(Il, GC_cost_window, GC_cost_window+1, GC_cost_window, GC_cost_window+1, cv2.BORDER_REPLICATE)
        cost_set_pad = np.zeros((10*max_disp+1, cost_set.shape[1]+2*GC_cost_window+1, cost_set.shape[2]+2*GC_cost_window+1))
        for i in range(10*max_disp+1):
            cost_set_pad[i] = cv2.copyMakeBorder(cost_set[i], GC_cost_window, GC_cost_window+1, GC_cost_window, GC_cost_window+1, cv2.BORDER_REPLICATE)
        data_cost_source = data_cost_img_guide(cost_set_pad, Il_pad, ref_labels, GC_cost_window, max_disp)
    else:
        Il_pad = 0 ; cost_set_pad = 0 ;
        data_cost_source = data_cost_img(cost_set, w_pq, ref_labels, GC_cost_window, max_disp)
    '''d_cost, s_cost = label_to_energy(cost_set, w_pq, Il, ref_labels, GC_cost_window, max_disp)
    print('Data       Energy:', d_cost)
    print('Smoothness Energy:', s_cost)
    print('Total      Energy:', d_cost+s_cost)'''
    for iter in range(3):
        ##########################################################
        # Region 5*5 propagation (no random)
        r_size = 5
        if(USE_GUIDED):
            gwindow_r = ((r_size*3-1)>>1)+GC_cost_window
            Il_pad = cv2.copyMakeBorder(Il, gwindow_r, gwindow_r*2, gwindow_r, gwindow_r*2, cv2.BORDER_REPLICATE)
            cost_set_pad = np.zeros((10*max_disp+1, cost_set.shape[1]+3*gwindow_r, cost_set.shape[2]+3*gwindow_r))
            for i in range(10*max_disp+1):
                cost_set_pad[i] = cv2.copyMakeBorder(cost_set[i], gwindow_r, gwindow_r*2, gwindow_r, gwindow_r*2, cv2.BORDER_REPLICATE)
        else:
            Il_pad = 0 ; cost_set_pad = 0 ;
        for i in range(16):
            print('5*5 prop Current i =',i,'; Total iteration =',16,'; Outer iteration #'+str(iter))
            label_map = label_to_region_proposal_one(1, i, ref_labels[None], r_size)
            label_mask = create_mask(label_map[None])[0]
            ref_labels,data_cost_source = binary_fusion(cost_set, data_cost_source, cost_set_pad, Il_pad, i, r_size, w_pq, ref_labels, label_map, label_mask, GC_cost_window, max_disp)
        '''print('Compute energy...')
        d_cost, s_cost = label_to_energy(cost_set, w_pq, Il, ref_labels, GC_cost_window, max_disp)
        print('Data       Energy:', d_cost)
        print('Smoothness Energy:', s_cost)
        print('Total      Energy:', d_cost+s_cost) 
        plt.imshow(label_to_img(ref_labels, max_disp), cmap='gray')
        plt.show()'''
        # Region 5*5 refinement (add noise)
        local_r_d = r_d; local_r_n = r_n;
        for j in range(1):
            for i in range(16):
                print('5*5 refine Current i =',i,'; Total iteration =',16,'; Outer iteration #'+str(iter))
                pix_labels = perturb_labels(ref_labels[None], ref_labels, local_r_d, local_r_n, max_disp)[1]
                label_map = label_to_region_proposal_one(1, i, pix_labels[None], 5)
                label_mask = create_mask(label_map[None])[0]
                ref_labels,data_cost_source = binary_fusion(cost_set, data_cost_source, cost_set_pad, Il_pad, i, r_size, w_pq, ref_labels, label_map, label_mask, GC_cost_window, max_disp)
            local_r_d = np.max([local_r_d*0.5, 0.5]); local_r_n = np.max([local_r_n*0.5, 0.05])
        '''print('Compute energy...')
        d_cost, s_cost = label_to_energy(cost_set, w_pq, Il, ref_labels, GC_cost_window, max_disp)
        print('Data       Energy:', d_cost)
        print('Smoothness Energy:', s_cost)
        print('Total      Energy:', d_cost+s_cost)
        plt.imshow(label_to_img(ref_labels, max_disp), cmap='gray')
        plt.show()''' 
        ##########################################################
        ##########################################################
        # Region 15*15 propagation (no random)
        r_size = 15
        if(USE_GUIDED):
            gwindow_r = ((r_size*3-1)>>1)+GC_cost_window
            Il_pad = cv2.copyMakeBorder(Il, gwindow_r, gwindow_r*2, gwindow_r, gwindow_r*2, cv2.BORDER_REPLICATE)
            cost_set_pad = np.zeros((10*max_disp+1, cost_set.shape[1]+3*gwindow_r, cost_set.shape[2]+3*gwindow_r))
            for i in range(10*max_disp+1):
                cost_set_pad[i] = cv2.copyMakeBorder(cost_set[i], gwindow_r, gwindow_r*2, gwindow_r, gwindow_r*2, cv2.BORDER_REPLICATE)
        else:
            Il_pad = 0 ; cost_set_pad = 0 ;
        for i in range(16):
            print('15*15 prop Current i =',i,'; Total iteration =',16,'; Outer iteration #'+str(iter))
            label_map = label_to_region_proposal_one(1, i, ref_labels[None], r_size)
            label_mask = create_mask(label_map[None])[0]
            ref_labels,data_cost_source = binary_fusion(cost_set, data_cost_source, cost_set_pad, Il_pad, i, r_size, w_pq, ref_labels, label_map, label_mask, GC_cost_window, max_disp)
        '''print('Compute energy...')
        d_cost, s_cost = label_to_energy(cost_set, w_pq, Il, ref_labels, GC_cost_window, max_disp)
        print('Data       Energy:', d_cost)
        print('Smoothness Energy:', s_cost)
        print('Total      Energy:', d_cost+s_cost) 
        plt.imshow(label_to_img(ref_labels, max_disp), cmap='gray')
        plt.show()''' 
        ##########################################################
        ##########################################################
        # Region 25*25 propagation (no random)
        r_size = 25
        if(USE_GUIDED):
            gwindow_r = ((r_size*3-1)>>1)+GC_cost_window
            Il_pad = cv2.copyMakeBorder(Il, gwindow_r, gwindow_r*2, gwindow_r, gwindow_r*2, cv2.BORDER_REPLICATE)
            cost_set_pad = np.zeros((10*max_disp+1, cost_set.shape[1]+3*gwindow_r, cost_set.shape[2]+3*gwindow_r))
            for i in range(10*max_disp+1):
                cost_set_pad[i] = cv2.copyMakeBorder(cost_set[i], gwindow_r, gwindow_r*2, gwindow_r, gwindow_r*2, cv2.BORDER_REPLICATE)
        else:
            Il_pad = 0 ; cost_set_pad = 0 ;
        for i in range(16):
            print('25*25 prop Current i =',i,'; Total iteration =',16,'; Outer iteration #'+str(iter))
            label_map = label_to_region_proposal_one(1, i, ref_labels[None], r_size)
            label_mask = create_mask(label_map[None])[0]
            ref_labels,data_cost_source = binary_fusion(cost_set, data_cost_source, cost_set_pad, Il_pad, i, r_size, w_pq, ref_labels, label_map, label_mask, GC_cost_window, max_disp)
        '''print('Compute energy...')
        d_cost, s_cost = label_to_energy(cost_set, w_pq, Il, ref_labels, GC_cost_window, max_disp)
        print('Data       Energy:', d_cost)
        print('Smoothness Energy:', s_cost)
        print('Total      Energy:', d_cost+s_cost)
        plt.imshow(label_to_img(ref_labels, max_disp), cmap='gray')
        plt.show() '''
        ##########################################################
        r_d = np.max([r_d*0.5, 0.5])
        r_n = np.max([r_n*0.5, 0.05])
        #disp_img = label_to_img(ref_labels, max_disp)
        #cv2.imwrite(storePath+'_'+str(iter)+'.png', np.uint8(disp_img * scale_factor))
    #np.save(storePath+'.npy', ref_labels)
    return ref_labels

	
def initGPUModel(path):
    chainer.config.train = False
    chainer.set_debug(False)
    chainer.using_config('use_cudnn', 'auto')
    # Load MC-CNN pre-trained models from
    # kitti_fast, kitti_slow, kitti2015_fast, kitti2015_slow, mb_fast, mb_slow
    model = mcnet.MCCNN_pretrained(path)
    # Make a specified GPU current
    chainer.cuda.get_device_from_id(0).use()
    model.to_gpu()  # Copy the model to the GPU
    return model
   
###############################################################################

# model_options:
# kitti_slow, kitti2015_slow, mb_slow
def midbury():
    args = parser.parse_args()
    print('args.output :',args.output)
    print('args.setting:',args.setting)
    model_path = 'mccnn/mb_slow'
    model = 0
	#===================================================
    #=              Setting Parameters                 =
	#===================================================
    global guide_r
    global lambda_s
    global bilateral_sigmas
    global bilateral_r
    global mf
    setting = 2	#real data
    if(args.setting!=-1):
        setting = min(max(0,args.setting),3)
    bilateral_sigmas = [30,0.18]
    bilateral_r  = 30
    setParamters(setting)
    if(USE_GRAPHCUT):
        import maxflow as mf
        guide_r   = 10
        if(USE_MCCNN):
            lambda_s  = 0.5
        else:
            lambda_s  = 10
    if(USE_MCCNN_SELF):
        from keras.models import load_model
        from keras import optimizers
        guide_r   = 10
        model = load_model('my_mccnn_new4.h5')
    elif(USE_MCCNN):
        guide_r   = 10
        model = initGPUModel(model_path)
	#===================================================
    #=              Rectify and Run                    =
	#===================================================
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp, model)
    cv2.imwrite('result/tsukuba.png', np.uint8(labels * scale_factor))
    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp, model)
    cv2.imwrite('result/venus.png', np.uint8(labels * scale_factor))
    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp, model)
    cv2.imwrite('result/teddy.png', np.uint8(labels * scale_factor))
    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp, model)
    cv2.imwrite('result/cones.png', np.uint8(labels * scale_factor))

#setting 0: USE_MCCNN
#setting 1: USE_MCCNN+USE_GRAPHCUT
#setting 2: USE_MCCNN_SELF
#setting 3: USE_MCCNN_SELF+USE_GRAPHCUT
def setParamters(setting):
    global USE_GRAPHCUT
    global USE_MCCNN
    global USE_MCCNN_SELF
    if(setting==0 or setting==1):
        USE_MCCNN=True
    if(setting==2 or setting==3):
        USE_MCCNN_SELF=True
    if(setting==1 or setting==3):
        USE_GRAPHCUT=True
    print('=================Settings=================')
    print('=== Setting:        ', setting)
    print('=== USE_MCCNN:      ', USE_MCCNN)
    print('=== USE_MCCNN_SELF: ', USE_MCCNN_SELF)
    print('=== USE_GRAPHCUT:   ', USE_GRAPHCUT)
    print('==========================================')

def main():
    model_path = 'mccnn/mb_slow'
    model = 0
    args = parser.parse_args()
    print('args.output :',args.output)
    print('args.setting:',args.setting)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    H,W,_ = img_left.shape
    #===================================================
    #=              Setting Parameters                 =
    #===================================================
    global guide_r
    global lambda_s
    global bilateral_sigmas
    global bilateral_r
    global mf
    if(find_recify_shift(img_left, img_right)==0):	
        #synthetic data
        setting = 2
    else:
        setting = 0	#MTK real data
        bilateral_sigmas = [5,0.18]
        bilateral_r  = 5
    img_left, img_right = rectify_by_shift(img_left, img_right)
    max_disp = np.rint(find_disp(img_left, img_right)).astype('int32')+5#62
    # User defined setting
    if(args.setting!=-1):
        setting = min(max(0,args.setting),3)
    setParamters(setting)
    if(USE_MCCNN_SELF):
        from keras.models import load_model
        from keras import optimizers
        guide_r   = 15
        model = load_model('my_mccnn_new4.h5')
    elif(USE_MCCNN):
        guide_r   = 15
        model = initGPUModel(model_path)
    if(USE_GRAPHCUT):
        import maxflow as mf
        guide_r   = 30
        if(USE_MCCNN):
            lambda_s  = 1
        else:
            lambda_s  = 10
    #===================================================
    #=              Rectify and Run                    =
    #===================================================
    scale_factor = 255/max_disp#6
    print('max_disp: ',max_disp)
    #print('Save disp image to :', '.'+args.output.split('.')[1]+'.png')
    tic = time.time()
    disp = computeDisp(img_left, img_right, max_disp, model)
    toc = time.time()
    #print('Save disp image to :','.'+args.output.split('.')[1]+'.png')
    #cv2.imwrite('.'+args.output.split('.')[1]+'.png', np.uint8(disp * scale_factor))
    writePFM(args.output, disp.astype('float32'))
    print('Elapsed time: %f sec.' % (toc - tic))


if __name__ == '__main__':
    #main()
    midbury()