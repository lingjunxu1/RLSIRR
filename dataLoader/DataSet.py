from torch.utils import data
from os import listdir
from os.path import join
from torchvision import transforms
from PIL import Image
class DataSet(data.Dataset):
    def __init__(self,rootPath,blendName = "blend",GtName = "T"):
        self.images = listdir(join(rootPath,blendName))
        self.root = rootPath
        self.blendName = blendName
        self.GtName = GtName
        self.Transforms = transforms.ToTensor()
    def __getitem__(self,index):
        image = self.images[index]
        blend = Image.open(join(self.root,self.blendName,image))
        Gt    = Image.open(join(self.root,self.GtName,image))
        blend = self.Transforms(blend)
        Gt = self.Transforms(Gt)
        return blend,Gt,image
    def __len__(self):
        return len(self.images)