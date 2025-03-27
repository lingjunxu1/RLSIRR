import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
import itertools
from collections import OrderedDict

from ..util.util import set_opt_param,get_model_list
from .networks import init_weights
from .losses import Vgg19
from . import arch

from .base_model import BaseModel
from PIL import Image
from os.path import join
from calflops import calculate_flops
import thop
def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy


class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        
        gradx = (img[...,1:,:] - img[...,:-1,:]).abs().sum(dim=1, keepdim=True)
        grady = (img[...,1:] - img[...,:-1]).abs().sum(dim=1, keepdim=True)

        gradX[...,:-1,:] += gradx
        gradX[...,1:,:] += gradx
        gradX[...,1:-1,:] /= 2

        gradY[...,:-1] += grady
        gradY[...,1:] += grady
        gradY[...,1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge


class ERRNetBase(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
               
        input = data
        self.input = input
        
        self.input_edge = self.edge_map(self.input)

        self.issyn = False
        self.aligned = True
        
       
    def eval(self, data, savedir=None, suffix=None, pieapp=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

        with torch.no_grad():
            self.forward()
            return self.output_i

class ERRNetModel(ERRNetBase):
    def name(self):
        return 'errnet'
        
    def __init__(self,gpuID):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device(f"cuda:{gpuID}" if torch.cuda.is_available() else "cpu")

    def _eval(self):
        self.net_i.eval()

    def _train(self):
        self.net_i.train()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        in_channels = 3
        self.vgg = None
        if opt.hyper:
            self.vgg = Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472
        self.net_i = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)
        init_weights(self.net_i, init_type=opt.init_type) # using default initialization as EDSR
        self.edge_map = EdgeMap(scale=1).to(self.device)
        if opt.resume:
            self.load(self, opt.resume_epoch)
        if opt.no_verbose is False:
            self.print_network()
    def forward(self):
        # without edge
        input_i = self.input
        if self.vgg is not None:
            #flops1, params1 = thop.profile(self.vgg, inputs=(self.input,))  
            #print("vgg {}GFlops {}M".format(flops1/1000**3,params1/1000**2))
            hypercolumn = self.vgg(self.input)
            _, C, H, W = self.input.shape
            hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in hypercolumn]
            input_i = [input_i]
            input_i.extend(hypercolumn)
            input_i = torch.cat(input_i, dim=1)
        
        output_i = self.net_i(input_i)
        self.output_i = output_i
        #flops1, params1 = thop.profile(self.net_i, inputs=(input_i,))  
        #print("net_i {}GFlops {}M".format(flops1/1000**3,params1/1000**2))
        return output_i
    def optimize_parameters(self):
        self._train()
        self.forward()
        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    @staticmethod
    def load(model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        state_dict = None

        if icnn_path is None:
            model_path = get_model_list(model.save_dir, model.name(), epoch=resume_epoch)
            state_dict = torch.load(model_path)
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            model.net_i.load_state_dict(state_dict['icnn'])
            if model.isTrain:
                model.optimizer_G.load_state_dict(state_dict['opt_g'])
        else:
            state_dict = torch.load(icnn_path)
            model.net_i.load_state_dict(state_dict['icnn'])
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            # if model.isTrain:
            #     model.optimizer_G.load_state_dict(state_dict['opt_g'])
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(), 
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        return state_dict


class NetworkWrapper(ERRNetBase):
    # You can use this class to wrap other module into our training framework (\eg BDN module)
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _eval(self):
        self.net.eval()

    def _train(self):
        self.net.train()

    def initialize(self, opt, net):
        BaseModel.initialize(self, opt)
        self.net = net.to(self.device)
        self.edge_map = EdgeMap(scale=1).to(self.device)
    def state_dict(self):
        state_dict = self.net.state_dict()
        return state_dict
