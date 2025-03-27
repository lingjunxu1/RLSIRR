from .model import buildModel
import torch
from torch.autograd import Variable
import thop
class SIRR:
    def __init__(self,USSFID,diviceId):
        
        self.USSFID = USSFID
        self.diviceId = diviceId
        self.device = torch.device(f'cuda:{diviceId}')
        self.deviceUSSF = torch.device(f'cuda:{USSFID}')
        self.model,self.vgg = buildModel(self.device)
        self.printNetWork()
    def inference(self,torchData):
        if self.USSFID!=self.diviceId: inputData = torchData.to(self.device)
        else: inputData = torchData
        with torch.no_grad():
            output, _, _ = self.model(inputData,self.vgg(inputData))
            #flops1, params1 = thop.profile(self.vgg, inputs=(inputData,))  
            #print("vgg {}GFlops {}M".format(flops1/1000**3,params1/1000**2))
            #flops1, params1 = thop.profile(self.model, inputs=(inputData,self.vgg(inputData),))  
            #print("self.model {}GFlops {}M".format(flops1/1000**3,params1/1000**2))
            
        if self.USSFID!=self.diviceId: output = output.to(self.deviceUSSF)
        return output.detach()
    def printNetWork(self):
        model_params = sum(p.numel() for p in self.model.parameters())
        vgg_params   = sum(p.numel() for p in self.vgg.parameters())
        num_params = model_params+vgg_params
        print("Number of parameters:{:.0f}, {:.4f}Mb".format(num_params,num_params*4/(1024**2)))