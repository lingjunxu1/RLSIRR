from MSP import MSP
import torch
import torch.nn as nn
from dataLoader.getloader import getTrainingData
from torch.utils.data import DataLoader
from utils import Tools,ConfusionMatrix,get_score
import random
from os.path import join
from time import time
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3" 
parser = argparse.ArgumentParser()
parser.add_argument("--datadirSyn", type=str, help="JPEGImages path")
parser.add_argument("--realData", type=str, help="real data path")
parser.add_argument("--savePath", type=str, help="ckpts save path")
args = parser.parse_args()
datadir_syn = args.datadirSyn
device_ids = [0,1]
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
SIRRID = 1
lr = 0.0005
batch_size = 36
episodeLength = 5
epochs = 20
saveDir = args.savePath
if __name__=="__main__":
    model = MSP(episodeLength,device,lr,device_ids)#
    model.printNetwork()
    criterion_ce = nn.BCELoss()
    dataloader_train = getTrainingData(batch_size,datadir_syn,args.realData)
    allStep = len(dataloader_train)*epochs
    tools = Tools(device_ids[0],SIRRID,K=16,batchsize=batch_size)
    step = 0
    for epoch in range(epochs):
        print("epoch:",epoch)
        for index,(x1,Gt,_) in enumerate(dataloader_train):
            time1 = time()
            step+=1
            x1, Gt = x1.to(device), Gt.to(device)
            for timestep in range(episodeLength):
                x2,lab = tools.DataTransform.setData(x1,Gt) #x2和lab已经detach了
                pre = model.models[timestep](tools.NormalizeForms(x1), tools.NormalizeForms(x2))
                loss = criterion_ce(pre, lab)
                model.optimizers[timestep].zero_grad()
                loss.backward()
                model.optimizers[timestep].step()
                cm = ConfusionMatrix(2, pre, lab)
                precision, recall, f1, iou, kc = get_score(cm)
                print('%d/%d, Pre:%f, Rec:%f, F1:%f, IoU:%f, KC:%f, loss:%f' % (index, len(dataloader_train), precision[1], recall[1], f1[1], iou[1], kc, loss.cpu().item()))
                if random.random()> 0.95-(step/allStep)*0.85:
                    x1 = tools.smooth.stake(x1,x2,pre,timestep).clone().detach()
                else: x1 = x2.clone().detach()
            if step%2000==0: model.saveModel(join(saveDir,str(step)))
            time2 = time()
            print(time2-time1,"--"*40)
        model.saveModel(join(saveDir,str(step)))