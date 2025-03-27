from MSP import MSP
from dataLoader.getloader import getTrainingData
from utils import *
from ppo import PPO,Memory
import torch
import os
import time
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
parser = argparse.ArgumentParser()
parser.add_argument("--datadirSyn", type=str, help="Berkeley path")
parser.add_argument("--realData", type=str, help="real data path")
parser.add_argument("--savePath", type=str, help="ckpts save path")
parser.add_argument("--segModelPath", type=str, help="trained seg ckpts save path")
parser.add_argument( "--policyHid", type=int, help="Policy network hidden layer dimension",default=1024)
args = parser.parse_args()
 
device_ids = [0,1]
device = device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
SIRRID = 1
episodeLength = 5
epochs = 10
batch_size = 8
step = 0
savedir = args.savePath
trainedCkpt = args.segModelPath
if not os.path.exists(savedir): os.makedirs(savedir)
if __name__=="__main__":
    seed_torch()
    tools = Tools(device_ids[0],SIRRID,K=16,batchsize=batch_size)
    model = MSP(episodeLength,device,0.001,None,None)
    model.loadckpt(trainedCkpt)
    settings = PPOSettings(args.policyHid)
    memory = Memory()
    policy = PPO(settings.policy_feature_dim, settings.state_dim, 
                 settings.policy_action_dim, settings.policy_hidden_state_dim, 
                 settings.policy_conv, settings.gpu, gamma=settings.gamma, 
                 lr=settings.policy_lr,device = device).to(device)
    policy.train()
    dataloader_train = getTrainingData(batch_size,args.datadirSyn,args.realData)
    torch.save(policy.policy.state_dict(),os.path.join(savedir,f"{step}.pth"), _use_new_zipfile_serialization=False)
    for epoch in range(epochs):
        print("epoch:",epoch)
        for index,(x1,Gt,_) in enumerate(dataloader_train):
            time1 = time.time()
            step+=1
            x1, Gt = x1.to(device), Gt.to(device)
            stop = torch.zeros(x1.shape[0]).to(device=device)
            ans = x1.clone().detach()
            sumReward = 0
            print(f"{index}/{len(dataloader_train)}")
            for timestep in range(episodeLength):
                with torch.no_grad():
                    x2,lab = tools.DataTransform.setData(x1,Gt)
                    pre,hypercolumn = model.models[timestep](tools.NormalizeForms(x1), tools.NormalizeForms(x2),eval=True)
                action = policy.select_action(hypercolumn.detach(), memory, restart_batch=timestep==0, training=True)
                
                stop = torch.where(action==0,1,stop)
                action = torch.where(stop==1,0,action)
                ussf = tools.smooth.stake(x1,x2,pre,timestep).detach()
                ans_ = ans.clone().detach()
                for i in range(x1.shape[0]):
                    if action[i]==1: 
                        ans_[i,:,:,:] = x2[i,:,:,:]
                        x1[i,:,:,:] = x2[i,:,:,:]
                    elif action[i]==2: 
                        ans_[i,:,:,:] = ussf[i,:,:,:]
                        x1[i:,:,:] = ussf[i,:,:,:]
                distance_ans = torch.mean(torch.square(ans-Gt),dim=(1,2,3)).unsqueeze(0)
                distance_ans_ = torch.mean(torch.square(ans_-Gt),dim=(1,2,3)).unsqueeze(0)
                reward = (distance_ans - distance_ans_)*255
                memory.rewards.append(reward.detach())
                print("reward:{:.4f}  {},{},{}".format(torch.mean(reward).cpu().item(),len(torch.where(action==0)[0]),len(torch.where(action==1)[0]),len(torch.where(action==2)[0])))
                sumReward = sumReward*settings.gamma+torch.mean(reward).cpu().item()
                ans = ans_.clone().detach()
            print("allReward:{:.4f}".format(sumReward),end = " ")
            policy.update(memory)
            memory.clear_memory()
            time2 = time.time()
            print(f"{time2-time1}",'-'*20)
            if step%500==0:
                torch.save(policy.policy.state_dict(),os.path.join(savedir,f"{step}.pth"), _use_new_zipfile_serialization=False)