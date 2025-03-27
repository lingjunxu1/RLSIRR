from .model import buildModel,get_weight_bias
import numpy as np
import torch
import tensorflow.compat.v1 as tf
class SIRR:
    def __init__(self,USSFID,diviceId):
        self.sess,transmission_layer, reflection_layerb4, reflection_layer,self.input,self.vgg_path = buildModel()
        self.fetch_list = [transmission_layer, reflection_layerb4, reflection_layer]
        self.USSFID = USSFID
        self.diviceId = diviceId
        self.device = torch.device("cuda:{}".format(USSFID))
        self.printNetWork()
    def inference(self,torchData):
        image_numpy = torchData.cpu().float().numpy()
        image_numpy = image_numpy.transpose(0,2,3,1)[:,:,:,::-1] #将(bs,c,w,h) rgb转为(bs,w,h,c) bgr  
        output_image_t, _, _=self.sess.run(self.fetch_list,feed_dict={self.input:image_numpy})
        #np.save("./temp.npy",output_image_t)
        ans = torch.tensor(output_image_t[:,:,:,::-1].transpose(0,3,1,2).copy() ,dtype=torch.float32).to(self.device)
        return ans.clone().detach()
    def printNetWork(self):
        para_num = sum([np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
        vgg_num = sum([np.prod(get_weight_bias(self.vgg_path['layers'][0], index)[0].shape.as_list()) for index in [0,2,5,7,10,12,14,16,19,21,23,25,28,30]])
        parameters = para_num+vgg_num
        print("Number of parameters:{:.0f}, {:.4f}Mb".format(parameters,parameters*4/(1024**2)))