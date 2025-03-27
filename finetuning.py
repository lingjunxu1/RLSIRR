from MSP import MSP
from dataLoader.getloader import getTrainingData
from utils import *
from ppo import PPO,Memory
import torch
import os
from time import time
import torch.nn as nn
from os.path import join
from utils import *
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3" 
parser = argparse.ArgumentParser()
parser.add_argument("--datadirSyn", type=str, help="Berkeley path")
parser.add_argument("--realData", type=str, help="real data path")
parser.add_argument("--savePath", type=str, help="ckpts save path")
parser.add_argument("--segModelPath", type=str, help="trained seg ckpts save path")
parser.add_argument("--rlModelPath", type=str, help="trained rl ckpt save path")
parser.add_argument( "--policyHid", type=int, help="Policy network hidden layer dimension",default=1024)
args = parser.parse_args()

device_ids = [0,1]
device = device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
SIRRID = 1
episodeLength = 5
epochs = 10
batch_size = 8
step = 0
trainedCkpt = args.segModelPath
saveDir =args.savePath
RLckpt = args.rlModelPath
if __name__=="__main__":
    seed_torch()
    tools = Tools(device_ids[0],SIRRID,K=16,batchsize=batch_size)
    model = MSP(episodeLength,device,0.00001,None,None)
    model.loadckpt(trainedCkpt,False)
    settings = PPOSettings(args.policyHid)
    memory = Memory()
    policy = PPO(settings.policy_feature_dim, settings.state_dim, 
                 settings.policy_action_dim, settings.policy_hidden_state_dim, 
                 settings.policy_conv, settings.gpu, gamma=settings.gamma, 
                 lr=settings.policy_lr,device = device).to(device)
    policy = PPO(settings.policy_feature_dim, settings.state_dim, settings.policy_action_dim, settings.policy_hidden_state_dim, settings.policy_conv, settings.gpu, gamma=settings.gamma, lr=settings.policy_lr,device = device)
    policy.evalModel(RLckpt)
    dataloader_train = getTrainingData(batch_size,args.datadirSyn,args.realData)
    criterion_ce = nn.BCELoss()
    for epoch in range(epochs):
        print("epoch:",epoch)
        for index,(x1,Gt,_) in enumerate(dataloader_train):
            time1 = time()
            step+=1
            x1, Gt = x1.to(device), Gt.to(device)
            stop = torch.zeros(x1.shape[0]).to(device=device)
            ans = x1.clone().detach()
            sumReward = 0
            print(f"{index}/{len(dataloader_train)}")
            for timestep in range(episodeLength):
                x2,lab = tools.DataTransform.setData(x1,Gt) #x2和lab已经detach了
                pre,hypercolumn = model.models[timestep](tools.NormalizeForms(x1), tools.NormalizeForms(x2),True)
                loss = criterion_ce(pre, lab)
                model.optimizers[timestep].zero_grad()
                loss.backward()
                model.optimizers[timestep].step()
                cm = ConfusionMatrix(2, pre, lab)
                precision, recall, f1, iou, kc = get_score(cm)
                print('%d/%d, Pre:%f, Rec:%f, F1:%f, IoU:%f, KC:%f, loss:%f' % (index, len(dataloader_train), precision[1], recall[1], f1[1], iou[1], kc, loss.cpu().item()))
                action = policy.select_action(hypercolumn,memory,training=False,restart_batch = timestep==0)
                if timestep==0 and action[0].cpu().item()==0:
                    x1 = x2.clone()
                    continue
                ussf = tools.smooth.stake(x1,x2,pre,timestep).detach()
                if action[0].cpu().item()==0:break
                if action[0].cpu().item()==1: x1 = x2.clone()
                elif action[0].cpu().item()==2: x1 = ussf.clone()
            if step%2000==0: model.saveModel(join(saveDir,str(step)))
            time2 = time()
            print(time2-time1,"--"*40)