import cv2
from os.path import join
from os import listdir
import json
import numpy as np
from skimage.measure import compare_ssim, compare_psnr
from functools import partial
from copy import deepcopy as dp
import math
import warnings
from tqdm import tqdm
from createTable import saveFile
warnings.filterwarnings("ignore")    
class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-1]
        bwindex = []
        for ch in range(C):
            x = X[..., ch]
            y = Y[..., ch]
            index = self.index_fn(x, y)
            if not math.isinf(index):  bwindex.append(index)
            else: bwindex.append(40)
        return bwindex
def compare_ncc(x, y):
    return np.mean((x-np.mean(x)) * (y-np.mean(y))) / (np.std(x) * np.std(y)) 
def ssq_error(correct, estimate):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate**2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate**2)
    else:
        alpha = 0.
    return np.sum((correct - alpha*estimate) ** 2)
def local_error(correct, estimate, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N, C = correct.shape
    ssq = total = 0.
    for c in range(C):
        for i in range(0, M - window_size + 1, window_shift):
            for j in range(0, N - window_size + 1, window_shift):
                correct_curr = correct[i:i+window_size, j:j+window_size, c]
                estimate_curr = estimate[i:i+window_size, j:j+window_size, c]
                ssq += ssq_error(correct_curr, estimate_curr)
                total += np.sum(correct_curr**2)
    return ssq / total
def computeIndex(image1,image2,test = False):
    ssim,psnr,ncc,lmse,distance = [],[],[],[],[]
    cal_bwssim = Bandwise(partial(compare_ssim, data_range=1))
    cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=1))
    for i in range(1):
        Y = image1#[i]#.transpose(1,2,0)[:,:,::-1]
        X = image2#[i]#.transpose(1,2,0)[:,:,::-1]    
        ssim.append(np.mean(cal_bwssim(dp(Y), dp(X))))
        psnr.append(np.mean(cal_bwpsnr(dp(Y), dp(X))))
        lmse.append(local_error(Y, X, 20, 10))
        ncc.append(compare_ncc(Y, X))
        distance.append(np.linalg.norm(X - Y))
    if not test: return list(map(np.mean,(ssim,psnr,ncc,lmse,distance)))
    else: return list(map(np.mean,(ssim,psnr,ncc,lmse,distance))),[ssim,psnr,ncc,lmse,distance]
def compute(originalPath,RLresult):
    saveDic = {}
    for dataName in ["real20","Li","Postcard","Object","Wild"]:
        inPath =join(originalPath,dataName)
        data = []
        for file in tqdm(listdir(join(inPath,"T"))):
            Bimage = cv2.imread(join(inPath,"blend",file)).astype(np.float32)/255.
            Timage = cv2.imread(join(inPath,"T",file)).astype(np.float32)/255.
            Dimage = cv2.imread(join(RLresult,dataName,"SIRR",file)).astype(np.float32)/255.
            Eimage = cv2.imread(join(RLresult,dataName,"RL",file)).astype(np.float32)/255.
            imageList = [Bimage,Dimage,Eimage]
            result = []
            for im in imageList:
                ssim,psnr,ncc,lmse,distance = computeIndex(Timage,im)
                result.append([ssim,psnr,ncc,lmse,distance])
            data.append(result)
        data = np.sum(np.array(data),axis=0)/len(data)
        index = {}
        indexName = ["ssim","psnr","ncc","lmse","distance"]
        names  = ["Input","Original","Current"]
        for i in range(len(names)):
            index[names[i]] = {indexName[k]:data[i][k] for k in range(len(indexName))}
        saveDic[dataName] = index
        print(saveDic[dataName]["Current"])
        dataNum1 = [20,20,199,200,55]
    saveFile(["real 20","Li et al.","Postcard","Object","Wild"],
             dataNum1,saveDic,"./","flag string","result")