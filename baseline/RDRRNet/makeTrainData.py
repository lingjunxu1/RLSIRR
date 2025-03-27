from SIRR import SIRR
import os
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
import time
import numpy as np
kk = 1
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0")
class dataSet(Dataset):
    def __init__(self,inPath,filenames,blendName="blend",Tname="T"):
        super(dataSet).__init__()
        self.inPath = inPath
        self.filenames = filenames
        self.filenames.sort()
        self.transfroms = transforms.ToTensor()
    def __getitem__(self,index):
        filename = self.filenames[index]
        blend = Image.open(os.path.join(self.inPath,"blend",filename)).convert("RGB")
        blend = self.transfroms(blend)
        return blend,filename
    def __len__(self):
        return len(self.filenames)
        pass
def saveTensorImage(image_tensor, savePath, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    Image.fromarray(image_numpy.astype(np.uint8)).convert("RGB").save(savePath)
