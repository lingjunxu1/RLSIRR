import torch
from .model import buildModel
import thop
#from PIL import Image
#import numpy as np
#def tensor2im(image_tensor, imtype=np.uint8):
#    image_tensor = image_tensor.detach()
#    image_numpy = image_tensor[0].cpu().float().numpy()
#    image_numpy = np.clip(image_numpy, 0, 1)
#    if image_numpy.shape[0] == 1:
#        image_numpy = np.tile(image_numpy, (3, 1, 1))
#    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
#    # image_numpy = image_numpy.astype(imtype)
#    return image_numpy
class SIRR:
    def __init__(self,USSFID,diviceId):
        self.model = buildModel(diviceId)
        self.USSFID = USSFID
        self.diviceId = diviceId
        self.device = torch.device(f'cuda:{diviceId}')
        self.deviceUSSF = torch.device(f'cuda:{USSFID}')
        self.model._eval()
        self.printNetWork()
    def inference(self,torchData):
        if self.USSFID!=self.diviceId: inputData = torchData.to(self.device)
        else: inputData = torchData
        with torch.no_grad():
            out_l1, out_r1 = self.model.net_i1(self.model.hyper_column(inputData))
            #flops1, params1 = thop.profile(self.model.net_i1, inputs=(self.model.hyper_column(inputData),))  
            #print("net_i1 {}GFlops {}M".format(flops1/1000**3,params1/1000**2))
            input_i, input_j = out_l1.detach(), out_r1.detach()
            output, _ = self.model.net_i2(self.model.hyper_column(input_i), self.model.hyper_column(input_j))
            #flops1, params1 = thop.profile(self.model.net_i2, inputs=(self.model.hyper_column(input_j),))  
            #print("net_i2 {}GFlops {}M".format(flops1/1000**3,params1/1000**2))
        #torch.save(output,"./temp.pth")
        #out_l2 = tensor2im(output)
        #Image.fromarray(out_l2.astype(np.uint8)).save("./ans.png")
        #exit(0)
        if self.USSFID!=self.diviceId: output = output.to(self.deviceUSSF)
        return output.detach()
    def printNetWork(self):
        net_i1_params = sum(p.numel() for p in self.model.net_i1.parameters())
        net_i2_params = sum(p.numel() for p in self.model.net_i2.parameters())
        vgg_params = sum(p.numel() for p in self.model.vgg.parameters())
        num_params = net_i1_params+net_i2_params+vgg_params
        print("Number of parameters:{:.0f}, {:.4f}Mb".format(num_params,num_params*4/(1024**2)))