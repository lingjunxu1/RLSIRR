from .models.arch.dsrnet import DSRNet
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
from .models.losses import Vgg19
import random
import os
def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_torch()

def dsrnet_s(in_channels=3, out_channels=3, width=32):
    enc_blks = [2, 2, 2]
    middle_blk_num = 4
    dec_blks = [2, 2, 2]

    return DSRNet(in_channels, out_channels, width=width,
                  middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks,
                  dec_blk_nums=dec_blks,
                  shared_b=False)


def dsrnet_l(in_channels=3, out_channels=3, width=64):
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    return DSRNet(in_channels, out_channels, width=width,
                  middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks,
                  dec_blk_nums=dec_blks,
                  shared_b=True)



def buildModel(device):
    vgg = Vgg19(requires_grad=False).to(device)
    network = dsrnet_l(3,3)
    state_dict = torch.load("./baseline/DSRNetl/ckpts")
    network.load_state_dict(state_dict)
    network.eval()
    return network.to(device),vgg.to(device)
    
def align(x1):
    h, w = x1.height, x1.width
    h, w = h // 32 * 32, w // 32 * 32
    x1 = x1.resize((w, h))
    return x1
def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy
