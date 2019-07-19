import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_matching(img_l,img_r): #right matches left
    if len(img_l.shape)>2:
        for c in range(img_l.shape[2]):
            match_histogram(img_l[:,:,c],img_r[:,:,c])
    else:
        match_histogram(img_l[:,:],img_r[:,:])

def match_histogram(y_l,y_r):
    hist,bins = np.histogram(y_l[:,:].flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    hist_r,bins=np.histogram(y_r[:,:].flatten(),256,[0,256])
    cdf_r =hist_r.cumsum()
    cdf_r_normalized=cdf_r/cdf_r.max()
    mapping_func=map_transform(cdf_normalized,cdf_r_normalized)
    for i in range(y_r.shape[0]):
        for j in range(y_r.shape[1]):
            y_r[i,j]=mapping_func[y_r[i,j]]

def map_transform(cdf,cdf_r):
    func=list()
    for i in range(256):
        for j in range(len(cdf)):
            if (cdf[j+1]>=cdf_r[i]):
                if((cdf_r[i]-cdf[j])>cdf[j+1]-cdf_r[i]):
                    func.append(j+1)
                else:
                    func.append(j)
                break
    return func

def plot_hist(img,img2):
    #plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.hist(img2.flatten(),256,[0,256], color = 'g')
    plt.xlim([0,256])
    plt.legend(('1','2'), loc = 'r')
    plt.show()

'''
if __name__ == "__main__":
    in_path = './data/Synthetic/'
    out_path = './data_pre/Synthetic/'
    for i in range(10):
        img_l=cv2.imread(in_path+'TL'+str(i)+'.png')
        img_r=cv2.imread(in_path+'TR'+str(i)+'.png')
        histogram_matching(img_r,img_l)
        cv2.imwrite(out_path+'TL'+str(i)+'.png',img_l)
        cv2.imwrite(out_path+'TR'+str(i)+'.png',img_r)
'''