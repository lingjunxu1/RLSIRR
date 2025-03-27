import random
import os
import numpy as np
from torchvision import transforms
import torch
import math
from PIL import Image
def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def ConfusionMatrix(num_classes, pres, gts):
    def __get_hist(pre, gt):
        pre = pre.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        pre[pre >= 0.5] = 1
        pre[pre < 0.5] = 0
        gt[gt >= 0.5] = 1
        gt[gt < 0.5] = 0
        mask = (gt >= 0) & (gt < num_classes)
        label = num_classes * gt[mask].astype(int) + pre[mask].astype(int)
        hist = np.bincount(label, minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist

    cm = np.zeros((num_classes, num_classes))
    for lt, lp in zip(gts, pres):
        cm += __get_hist(lt.flatten(), lp.flatten())
    return cm
def get_score(confusionMatrix):
    precision = np.diag(confusionMatrix) / (confusionMatrix.sum(axis=0) + np.finfo(np.float32).eps)
    recall = np.diag(confusionMatrix) / (confusionMatrix.sum(axis=1) + np.finfo(np.float32).eps)
    f1score = 2 * precision * recall / ((precision + recall) + np.finfo(np.float32).eps)
    iou = np.diag(confusionMatrix) / (
            confusionMatrix.sum(axis=1) + confusionMatrix.sum(axis=0) - np.diag(confusionMatrix) + np.finfo(
        np.float32).eps)
    po = np.diag(confusionMatrix).sum() / (confusionMatrix.sum() + np.finfo(np.float32).eps)
    pe = (confusionMatrix[0].sum() * confusionMatrix[0:2][0].sum() + confusionMatrix[1].sum() * confusionMatrix[0:2][
        1].sum()) / confusionMatrix.sum() ** 2 + np.finfo(np.float32).eps
    kc = (po - pe) / (1 - pe + np.finfo(np.float32).eps)
    return precision, recall, f1score, iou, kc
class DataTransform:
    def __init__(self,SIRR,USSFNetID,errNetID):
        self.model = SIRR(USSFNetID,errNetID)
    def setData(self,x1,Gt):
        with torch.no_grad():
            x2 = self.model.inference(x1)
            distance_x1 = torch.mean(torch.square(x1-Gt),axis=1,keepdim=True)
            distance_x2 = torch.mean(torch.square(x2-Gt),axis=1,keepdim=True)
            lab = torch.where(distance_x1<distance_x2,1,0).float()
            return x2.detach(),lab.detach()
class TrainDataTransform:
    def __init__(self,SIRR,USSFNetID,errNetID,batchsize):
        self.model = SIRR(USSFNetID,errNetID)
        self.num = batchsize//24
    def setData(self,x1,Gt):
        with torch.no_grad():
            x2 = [self.model.inference(x1[i*24:(i+1)*24,:,:,:]) for i in range(self.num)]
            x2 = torch.cat(x2,dim=0)
            distance_x1 = torch.mean(torch.square(x1-Gt),axis=1,keepdim=True)
            distance_x2 = torch.mean(torch.square(x2-Gt),axis=1,keepdim=True)
            lab = torch.where(distance_x1<distance_x2,1,0).float()
            return x2.detach(),lab.detach()
def saveTensorImage(image_tensor, savePath, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    Image.fromarray(image_numpy.astype(np.uint8)).save(savePath)
class Smooth():
    def __init__(self,K=30):
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1).cuda()
        self.filter2D = torch.nn.AvgPool2d(kernel_size=61, stride=1, padding=30).cuda()
        self.K = K
    def stake(self,x1,x2,pre,timestep,thre = 0.5):
        with torch.no_grad():
            if timestep==0: 
                otherMask = torch.where(pre>=thre,1,0).to(torch.float32)
                otherImage = x1
                sirrImage = x2
            else:
                otherMask = torch.where(pre>=thre,0,1).to(torch.float32)
                otherImage = x2
                sirrImage = x1
            otherW = otherMask.clone().detach()
            for i in range(1,self.K+1):
                mask_ = -self.maxpool(-otherMask)
                otherW=torch.where(mask_!=otherMask,i/self.K,otherW)
                otherMask = mask_.clone().detach()
            otherW = self.filter2D(otherW)
            sirrW = torch.ones(otherMask.shape,device=otherW.device)-otherW
            otherW = torch.repeat_interleave(otherW ,3 ,dim=1)
            sirrW = torch.repeat_interleave(sirrW ,3 ,dim=1)
            return (otherW*otherImage+sirrImage*sirrW).detach()
class PPOSettings:
    def __init__(self,policyHid = 1024):
        self.policy_feature_dim = 480
        self.glance_size = 224
        self.state_dim = self.policy_feature_dim * math.ceil(self.glance_size / 32) * math.ceil(self.glance_size / 32)
        self.policy_action_dim = 3
        self.policy_hidden_state_dim = policyHid
        self.policy_conv = True
        self.gpu = True
        self.gamma = 0.99
        self.policy_lr = 0.0003
class Tools:
    def __init__(self,SIRR,USSFNetID,errNetID,K=30,batchsize=1):
        if batchsize>20: self.DataTransform = TrainDataTransform(SIRR,USSFNetID,errNetID,batchsize)
        else: self.DataTransform = DataTransform(SIRR,USSFNetID,errNetID)
        self.smooth = Smooth(K)
        self.NormalizeForms = transforms.Normalize(mean=[0.5], std=[0.5])