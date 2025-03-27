
from .datasets_pairs import my_dataset, my_dataset_eval,my_dataset_wTxt,FusionDataset
from torch.utils.data import Dataset, ConcatDataset
from .image_folder import read_fns
from .reflect_dataset_for_fusion import CEILDataset
import os
import torch
from torch.utils.data import DataLoader
class opts:
    def __init__(self,path):
        self.training_data_path = path
        self.training_data_path_Txt1 = [os.listdir(os.path.join(self.training_data_path,"blend"))]
        self.low_A = 2
        self.high_A = 5
        self.low_sigma =2
        self.high_sigma=5
        self.low_gamma =1.3
        self.high_gamma=1.3
        self.Crop_patches = 320
        self.low_beta = 1.3
        self.syn_mode = 3
        self.high_beta = 3
        self.fusion_ratio = 0.7
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
def getTrainingData(BATCH_SIZE,datadir_syn,datadir_real,fix_sampleA=100000, Crop_patches=320):
    args = opts(datadir_real)
    rootA = datadir_real
    rootA_txt1_list = args.training_data_path_Txt1
    train_Pre_dataset_list = []
    for idx_dataset in range(len(rootA_txt1_list)):
        train_Pre_dataset = my_dataset_wTxt(rootA, rootA_txt1_list[idx_dataset],
                                            crop_size=Crop_patches,
                                            fix_sample_A=fix_sampleA,
                                            regular_aug=str2bool)  # threshold_size =  args.threshold_size
        train_Pre_dataset_list.append(train_Pre_dataset)
    train_pre_datasets = ConcatDataset(train_Pre_dataset_list)
    train_dataset_syn = CEILDataset(
        datadir_syn, os.listdir(datadir_syn), size=None,
        enable_transforms=True,
        low_sigma=args.low_sigma, high_sigma=args.high_sigma,
        low_gamma=args.low_gamma, high_gamma=args.high_gamma,crop_size=args.Crop_patches, mode=args.syn_mode,
                 low_A=args.low_A, high_A=args.high_A,
                low_beta=args.low_beta, high_beta=args.high_beta)
    train_sets = FusionDataset([train_dataset_syn, train_pre_datasets], fusion_ratios=[args.fusion_ratio, 1.0 - args.fusion_ratio])
    train_loader = DataLoader(dataset=train_sets, batch_size=BATCH_SIZE,
                              num_workers= 4 ,shuffle=True)
    print('len(train_loader):', len(train_loader))
    return train_loader