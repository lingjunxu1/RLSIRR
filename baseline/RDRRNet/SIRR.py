from .model import buildModel
from PIL import Image
import torch
import thop
class SIRR:
    def __init__(self,USSFID,diviceId):
        self.net,self.net_Det = buildModel(diviceId)
        self.USSFID = USSFID
        self.diviceId = diviceId
        self.device = torch.device(f'cuda:{diviceId}')
        self.deviceUSSF = torch.device(f'cuda:{USSFID}')
        self.printNetWork()
    def inference(self,torchData):
        if self.USSFID!=self.diviceId: inputData = torchData.to(self.device)
        else: inputData = torchData
        
        sparse_out = self.net_Det(inputData)
        #flops1, params1 = thop.profile(self.net_Det, inputs=(inputData,))  
        #print("net_Det {}GFlops {}M".format(flops1/1000**3,params1/1000**2))
        output = self.net(inputData, sparse_out)
        #flops1, params1 = thop.profile(self.net, inputs=(inputData,sparse_out))  
        #print("net {}GFlops {}M".format(flops1/1000**3,params1/1000**2))
        if self.USSFID!=self.diviceId: output = output.to(self.deviceUSSF)
        return output.detach()
    def printNetWork(self):
        T_params = sum(p.numel() for p in self.net.parameters())
        R_params = sum(p.numel() for p in self.net_Det.parameters())
        num_params = T_params+R_params
        print("Number of parameters:{:.0f}, {:.4f}Mb".format(num_params,num_params*4/(1024**2)))