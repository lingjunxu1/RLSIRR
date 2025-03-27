import warnings
warnings.filterwarnings("ignore")
from .models.errnet_model import ERRNetModel
import torchvision.transforms as transforms
from .setopts import options
import torch
import numpy as np
from PIL import Image
from .util.index import *
class SIRR:
    def __init__(self,USSFID,errNetId):
        self.USSFID = USSFID
        self.errNetId = errNetId
        self.device = torch.device(f'cuda:{errNetId}')
        self.deviceUSSF = torch.device(f'cuda:{USSFID}')
        opts = options(errNetId)
        opts.gpu_ids = [errNetId]
        self.model = ERRNetModel(errNetId)
        self.model.initialize(opts)
        #self.model = torch.nn.DataParallel(self.model,device_ids=[2,3])
        self.printNetWork()
    def inference(self,torchData):
        if self.USSFID!=self.errNetId: inputData = torchData.to(self.device)
        else: inputData = torchData
        output = self.model.eval(inputData)
        if self.USSFID!=self.errNetId: output = output.to(self.deviceUSSF)
        return output.detach()
    def printNetWork(self):
        params = sum(p.numel() for p in self.model.net_i.parameters())
        vgg_params = sum(p.numel() for p in self.model.vgg.parameters())
        num_params = vgg_params+params
        print("ErrNet Number of parameters:{:.0f}, {:.4f}Mb".format(num_params,num_params*4/(1024**2)))


