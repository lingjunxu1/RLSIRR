from .models.twostage_ytmt_model_use import TwoStageYTMTNetModel as TwoStageYTMTNetModel
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
import random
import os
def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_torch()
class Opts:
    def __init__(self):
        self.batchSize=1
        self.checkpoints_dir='./baseline/YTMT/checkpoints'
        self.debug=False
        self.debug_eval=False
        self.display_freq=100
        self.display_id=0
        self.display_port=8097
        self.display_single_pane_ncols=0
        self.display_winsize=256
        self.eval_freq=1
        self.fineSize='224,224'
        self.finetune=False
        self.fixed_lr=0
        self.gan_type='rasgan'
        self.gpu_ids=[0]
        self.graph=False
        self.high_gamma=1.3
        self.high_sigma=5
        self.hyper=True
        self.icnn_path='./baseline/YTMT/checkpoints/ytmt_uct_sirs/ytmt_uct_sirs_68_077_00595364.pt'
        self.if_align=True
        self.inet='ytmt_ucs_old'
        self.init_lr=0.01
        self.init_type='edsr'
        self.isTrain=False
        self.lambda_gan=0.01
        self.lambda_vgg=0.1
        self.loadSize='224,336,448'
        self.low_gamma=1.3
        self.low_sigma=2
        self.lr=0.0001
        self.max_dataset_size=None
        self.model='twostage_ytmt_model'
        self.nEpochs=60
        self.nThreads=8
        self.name='ytmt_uct_sirs_test'
        self.no_flip=False
        self.no_html=False
        self.no_log=True
        self.no_verbose=False
        self.print_freq=100
        self.r_pixel_weight=1.0
        self.real20_size=420
        self.resize_or_crop='resize_and_crop'
        self.resume=True
        self.resume_epoch=None
        self.save_epoch_freq=1
        self.save_freq=1
        self.seed=2018
        self.select=None
        self.serial_batches=False
        self.start_now=False
        self.supp_eval=False
        self.testr=False
        self.tv_type=None
        self.unaligned_loss='vgg'
        self.update_html_freq=1000
        self.verbose=False
        self.vgg_layer=31
        self.wd=0
        self.which_model_D='disc_vgg'



def buildModel(deviceID):
    opt = Opts()
    opt.isTrain = False
    #cudnn.benchmark = True
    opt.no_log = True
    opt.display_id = 0
    opt.verbose = False
    model = TwoStageYTMTNetModel(deviceID)  # models.__dict__[self.opt.model]()
    model.initialize(opt)
    return model
def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy