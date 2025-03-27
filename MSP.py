
from USSFNet.USSFCNet import USSFCNet
import torch
from torch import optim
import os
class MSP():
    def __init__(self,number,device,lr,ParallelIDs=None,initckpt = None):
        self.models = [USSFCNet(in_ch=3, out_ch=1, ratio=0.5).to(device) for i in range(number)]
        if initckpt!=None:
            for i in range(len(self.models)): 
                ckps = torch.load(initckpt, map_location='cuda:0')
                self.models[i].load_state_dict({k.replace('module.',''):v for k,v in ckps.items()})
        if ParallelIDs!=None: 
            for i in range(number): self.models[i] = torch.nn.DataParallel(self.models[i], device_ids=ParallelIDs).to(device)
        for i in range(number): self.models[i] = self.models[i].train()
        self.optimizers = [optim.Adam(self.models[i].parameters(),lr, weight_decay=0.0005) for i in range(number)]
    def loadckpt(self,ckptdir,eval = True):
        if ".pth" in ckptdir:
            for i in range(len(self.models)): 
                ckps = torch.load(ckptdir, map_location='cuda:0')
                self.models[i].load_state_dict({k.replace('module.',''):v for k,v in ckps.items()})#,strict=False)
        else:
            for i in range(len(self.models)): 
                ckps = torch.load(os.path.join(ckptdir,f"{i}.pth"), map_location='cuda:0')
                self.models[i].load_state_dict({k.replace('module.',''):v for k,v in ckps.items()})#,strict=False)
        if eval:
            for i in range(len(self.models)): self.models[i] = self.models[i].eval()
    def saveModel(self,savePath):
        if not os.path.exists(savePath): os.makedirs(savePath)
        for i in range(len(self.models)):
            torch.save(self.models[i].state_dict(),os.path.join(savePath,f"{i}.pth"), _use_new_zipfile_serialization=False)
    
    def printNetwork(self):
        num_params = 0
        for net in self.models:
            for param in net.parameters():
                num_params += param.numel()
        print('Total number of parameters: %d,%.3fMb' % (num_params, num_params / (1024 * 1024)))
    def __len__(self):
        return len(self.models)