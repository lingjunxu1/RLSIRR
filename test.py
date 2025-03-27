from MSP import MSP
from dataLoader.testset import testset
from torch.utils.data import DataLoader
from utils import *
from ppo import PPO,Memory
from tqdm import tqdm
import torch
import os
import importlib
import argparse
from compare import compute
parser = argparse.ArgumentParser()
parser.add_argument("--SIRR", type=str, help="Test data path")
parser.add_argument("--maxStep", type=int, help="max step")
parser.add_argument("--segModel", type=str, help="trained segmentation model")
parser.add_argument("--rLModel", type=str, help="trained RL model")
parser.add_argument("--dataPath", type=str, help="Test data path")
parser.add_argument( "--resultDir", type=str, help="relult save path")
parser.add_argument( "--policyHid", type=int, help="Policy network hidden layer dimension",default=1024)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5" 
device_ids = [1]
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
errNetId = 0
testPathDic = {
    "real20"  :os.path.join(args.dataPath,"real20"),
    "Li"      :os.path.join(args.dataPath,"Li"),
    "Postcard":os.path.join(args.dataPath,"Postcard"),
    "Object"  :os.path.join(args.dataPath,"Object"),
    "Wild"    :os.path.join(args.dataPath,"Wild")
}
episodeLength = args.maxStep
trainedModel = args.segModel
RLckpt = args.rLModel
resultDir = args.resultDir
for name in testPathDic: 
    if not os.path.exists(os.path.join(resultDir,name,"RL")): os.makedirs(os.path.join(resultDir,name,"RL"))
    if not os.path.exists(os.path.join(resultDir,name,"SIRR")): os.makedirs(os.path.join(resultDir,name,"SIRR"))
if __name__=="__main__":
    seed_torch()#固定随机种子使实验可重复
    SIRR = importlib.import_module("baseline.ERRNet.SIRR")
    model = MSP(episodeLength,device,0.001,None,None)
    model.loadckpt(trainedModel)#此处设置模型为eval模式了
    settings = PPOSettings(args.policyHid)
    memory = Memory()
    policy = PPO(settings.policy_feature_dim, settings.state_dim, settings.policy_action_dim, settings.policy_hidden_state_dim, settings.policy_conv, settings.gpu, gamma=settings.gamma, lr=settings.policy_lr,device = device)
    policy.evalModel(RLckpt)
    tools = Tools(SIRR.SIRR,device_ids[0],errNetId,K=30)
    for name in testPathDic:
        print(name)
        dataloader = DataLoader(  testset(testPathDic[name]),    batch_size=1,shuffle=False,num_workers=1)
        for index,(x1,Gt,imageName) in tqdm(enumerate(dataloader)):
            x1, Gt = x1.to(device), Gt.to(device)
            for timestep in range(episodeLength):
                with torch.no_grad():
                    x2,_ = tools.DataTransform.setData(x1,Gt) #x2和lab已经detach了
                    if timestep==0: saveTensorImage(x2,os.path.join(resultDir,name,"SIRR",imageName[0]))
                    pre,hypercolumn = model.models[timestep](tools.NormalizeForms(x1), tools.NormalizeForms(x2),eval=True)
                    action = policy.select_action(hypercolumn,memory,training=False,restart_batch = timestep==0)
                    ussf = tools.smooth.stake(x1,x2,pre,timestep).detach()
                    if timestep==0 and action[0].cpu().item()==0:
                        x1 = x2.clone()
                        continue
                    if timestep==4 and action[0].cpu().item()!=0:
                        x1 = ussf.clone()
                        continue
                    if action[0].cpu().item()==0:break
                    if action[0].cpu().item()==1: x1 = x2.clone()
                    elif action[0].cpu().item()==2: x1 = ussf.clone()
            saveTensorImage(x1,os.path.join(resultDir,name,"RL",imageName[0]))
            memory.clear_memory()
    compute(args.dataPath,args.resultDir)
    