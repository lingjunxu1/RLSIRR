from .networks.NAFNet_arch import NAFNet_wDetHead, NAFNetLocal
from .networks.network_RefDet import RefDet,RefDetDual
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.image as img
import random
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

def buildModel(device):
    net = NAFNet_wDetHead(img_channel= 3, width=32, middle_blk_num=1,
                      enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1], global_residual=False,
                     drop_flag = False,  drop_rate=0.4,
                          concat = True, merge_manner = 0)
    net_Det = RefDet(backbone='efficientnet-b3',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=False,
                 has_se=False,
                 num_of_layers=6,
                 expansion = 4)
    
    checkpoint = torch.load('baseline/RDRRNet/ckpts/RR.pth')
    net.load_state_dict(checkpoint, strict=True)
    checkpoint1 = torch.load('baseline/RDRRNet/ckpts/RD.pth')
    net_Det.load_state_dict(checkpoint1, strict=True)
    net_Det.to(device)
    net.to(device)
    
    net.eval()
    net_Det.eval()
    return net,net_Det
